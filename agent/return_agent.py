import os
import sqlite3
import tomllib
import json
import random
import string
from datetime import datetime, date
from pathlib import Path
from typing import Any

from langchain.llms.base import LLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

from openai import OpenAI
from rag.retriever import RAGRetriever


SETTINGS = tomllib.loads(Path("settings.toml").read_text(encoding="utf-8"))


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    )


class OllamaLLM(LLM):
    """A thin LangChain LLM wrapper that calls the project's OpenAI client.

    Implemented as a pydantic model so LangChain can instantiate it safely.
    """

    client: OpenAI
    model: str
    temperature: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "temperature": self.temperature}

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        return r.choices[0].message.content.strip()


class ReturnAgent:
    def __init__(self):
        self.client = _make_client()
        self.rag = RAGRetriever(k=4)

        model = SETTINGS["general"]["model"]
        temp = SETTINGS["general"].get("temperature", 0)
        # Instantiate OllamaLLM using keyword args so pydantic/LangChain BaseModel init works
        self.llm = OllamaLLM(client=self.client, model=model, temperature=temp)

        # register tools
        tools = [
            Tool(
                name="consultar_estado_pedido",
                func=self._consultar_estado_pedido,
                description="Verifica si un pedido existe y devuelve estado y fecha. Entrada: tracking ID (string).",
            ),
            Tool(
                name="verificar_elegibilidad_producto",
                func=self._verificar_elegibilidad_producto,
                description=(
                    "Evalúa si un producto es retornable según el pedido y políticas internas. "
                    "Entrada: order_id (tracking), sku"
                ),
            ),
            Tool(
                name="consultar_politicas",
                func=self._consultar_politicas,
                description=(
                    "Recupera fragmentos de políticas de devolución desde el índice RAG para una categoría o tema dado."
                ),
            ),
            Tool(
                name="generar_etiqueta_devolucion",
                func=self._generar_etiqueta_devolucion,
                description=(
                    "Genera una etiqueta de devolución simulada (RMA y URL). Entrada: order_id, sku, carrier"
                ),
            ),
        ]

        # initialize a simple react agent
        # enable handle_parsing_errors so the executor will pass parsing errors back
        # to the agent (allowing the agent to retry) and set verbose for debug logs
        self.agent_executor = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )

    # --- Tools implementation ---
    def _connect_db(self) -> sqlite3.Connection:
        # database file created by create_db.py: ecomarket.db
        return sqlite3.connect("ecomarket.db")

    def _consultar_estado_pedido(self, order_id: str) -> str:
        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("SELECT tracking, status, eta, last_update FROM orders WHERE tracking = ?", (order_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return json.dumps({"exists": False, "status": None, "date": None})
        tracking, status, eta, last_update = row
        # prefer last_update or eta
        date_str = last_update or eta
        return json.dumps({"exists": True, "status": status, "date": date_str})

    def _verificar_elegibilidad_producto(self, order_id: str, sku: str) -> str:
        # Basic eligibility rules:
        # - Order must exist and be delivered
        # - Product must be marked returnable in products table
        # - Days since delivery must be within policy_window (default 30)
        policy_window = SETTINGS.get("policies", {}).get("return_window_days", 30)

        conn = self._connect_db()
        cur = conn.cursor()
        cur.execute("SELECT status, last_update, eta FROM orders WHERE tracking = ?", (order_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return json.dumps({"eligible": False, "reason": "Pedido no encontrado"})
        status, last_update, eta = row
        # determine delivery date
        delivered_date = None
        if status and status.lower().startswith("entreg"):
            delivered_date = last_update or eta

        # product info
        cur.execute("SELECT sku, name, category, returnable FROM products WHERE sku = ?", (sku,))
        prow = cur.fetchone()
        conn.close()
        if not prow:
            return json.dumps({"eligible": False, "reason": "Producto no encontrado"})

        _, name, category, returnable = prow
        # normalize returnable
        try:
            returnable_bool = bool(json.loads(str(returnable).lower())) if isinstance(returnable, str) else bool(returnable)
        except Exception:
            returnable_bool = True if str(returnable).lower() in ("true", "1", "t") else False

        if not returnable_bool:
            return json.dumps({"eligible": False, "reason": f"El producto '{name}' no es retornable según su categoría ({category})."})

        if not delivered_date:
            # not delivered yet
            return json.dumps({"eligible": False, "reason": "El pedido aún no ha sido entregado; no es posible iniciar la devolución."})

        # parse delivered_date as ISO-like
        try:
            delivered_dt = datetime.fromisoformat(delivered_date)
        except Exception:
            try:
                delivered_dt = datetime.strptime(delivered_date, "%Y-%m-%d")
            except Exception:
                delivered_dt = None

        if delivered_dt:
            days = (datetime.now() - delivered_dt).days
            if days > int(policy_window):
                return json.dumps({"eligible": False, "reason": f"La devolución excede el plazo de {policy_window} días (han pasado {days} días)."})

        # otherwise eligible
        return json.dumps({"eligible": True, "reason": "El producto cumple con las condiciones de devolución."})

    def _consultar_politicas(self, categoria: str) -> str:
        docs = self.rag.get_relevant_chunks(categoria)
        if not docs:
            return "No se han encontrado fragmentos de política para la categoría indicada."
        out = []
        for d in docs:
            src = d.metadata.get("source", "source")
            out.append(f"[{src}] {d.page_content.strip()}")
        return "\n\n".join(out)

    def _generar_etiqueta_devolucion(self, order_id: str, sku: str, carrier: str = "EcoShip") -> str:
        # Simula la creación de una etiqueta devolviendo un RMA y una URL
        rma = "RMA-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        # Simulated URL - not hosted, but useful for UI and demo
        label_url = f"https://labels.ecomarket.test/{rma}.pdf"
        return json.dumps({"label_url": label_url, "rma": rma})

    # --- run agent ---
    def run(self, prompt: str) -> str:
        """Run the LangChain agent with the user's prompt and return a string response.

        The agent has tools available; final output is returned as text. We keep the response
        friendly by prefacing the model with the system prompt for returns from settings.toml.
        """
        # Construct an instruction that asks the agent to use the tools and answer warmly.
        system_prompt = SETTINGS.get("prompts", {}).get("return_policy", "")
        # Combine system and user into a single prompt the LLM wrapper will accept
        full_prompt = f"{system_prompt}\n\nUSER:\n{prompt}\n\nRespond warmly and, when appropriate, return JSON with fields 'action' and 'details'."

        try:
            result = self.agent_executor.run(full_prompt)
        except Exception as e:
            result = f"Error ejecutando el agente: {e}"
        # ensure string
        return str(result)
