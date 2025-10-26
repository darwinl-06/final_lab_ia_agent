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
        # Return structured Python dict so orchestration logic can use fields deterministically
        if not row:
            return {"exists": False, "status": None, "date": None}
        tracking, status, eta, last_update = row
        # prefer last_update or eta
        date_str = last_update or eta
        return {"exists": True, "status": status, "date": date_str}

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
            return {"eligible": False, "reason": "Pedido no encontrado"}
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
            return {"eligible": False, "reason": "Producto no encontrado"}

        _, name, category, returnable = prow
        # normalize returnable
        try:
            returnable_bool = bool(json.loads(str(returnable).lower())) if isinstance(returnable, str) else bool(returnable)
        except Exception:
            returnable_bool = True if str(returnable).lower() in ("true", "1", "t") else False

        if not returnable_bool:
            return {"eligible": False, "reason": f"El producto '{name}' no es retornable según su categoría ({category})."}

        if not delivered_date:
            # not delivered yet
            return {"eligible": False, "reason": "El pedido aún no ha sido entregado; no es posible iniciar la devolución."}

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
                return {"eligible": False, "reason": f"La devolución excede el plazo de {policy_window} días (han pasado {days} días)."}

        # otherwise eligible
        return {"eligible": True, "reason": "El producto cumple con las condiciones de devolución."}

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
        return {"label_url": label_url, "rma": rma}

    # --- Intent extraction and deterministic pipeline ---
    def extract_intent(self, user_message: str) -> dict:
        """
        Use the LLM only to extract intent and simple fields from the user's message.

        Returns a dict with keys:
          - accion: "solicitar_devolucion" | "estado_pedido" | "politicas" | "otro"
          - tracking_id: string or None
          - sku: string or None

        IMPORTANT: this function must return a Python dict. The LLM is instructed to
        output ONLY a JSON object with those keys.
        """
        prompt = (
            "Extrae la intención del siguiente mensaje y devuelve SOLO un JSON válido con las claves "
            "{\"accion\", \"tracking_id\", \"sku\"}. Los valores posibles para 'accion' son: "
            "\"solicitar_devolucion\", \"estado_pedido\", \"politicas\", \"otro\". "
            "Si no hay tracking o sku en el mensaje, devuelve null para esos campos. "
            "NO incluyas texto adicional fuera del JSON.\n\n"
            f"MENSAJE: {user_message}\n\nRESPUESTA JSON:"
        )
        try:
            raw = self.llm._call(prompt)
            # parse only the first JSON object in the response
            parsed = json.loads(raw)
            # normalize keys
            accion = parsed.get("accion") if isinstance(parsed.get("accion"), str) else parsed.get("action")
            tracking = parsed.get("tracking_id") if parsed.get("tracking_id") is not None else parsed.get("tracking")
            sku = parsed.get("sku")
            return {"accion": accion or "otro", "tracking_id": tracking, "sku": sku}
        except Exception:
            # fallback: simple heuristic parsing (do not call business logic)
            import re

            tracking_match = re.search(r"(TRK[- ]?\d+)", user_message, re.IGNORECASE)
            sku_match = re.search(r"(SKU[- ]?\d+)", user_message, re.IGNORECASE)
            action = "otro"
            low = user_message.lower()
            if "devol" in low or "devolver" in low or "retornar" in low:
                action = "solicitar_devolucion"
            elif "estado" in low or "tracking" in low or "trk" in low:
                action = "estado_pedido"
            elif "polit" in low or "política" in low or "politica" in low:
                action = "politicas"
            return {"accion": action, "tracking_id": tracking_match.group(1) if tracking_match else None, "sku": sku_match.group(1) if sku_match else None}

    def run_return_pipeline(self, tracking_id: str, sku: str) -> dict:
        """
        Deterministic orchestration for return flow.

        Business rules (comments for developers):
        - If exists is False -> stop and return rejected with motivo and mensaje_usuario.
        - If eligible is False -> do not generate label; return reason exactly as provided by tool.
        - Only if eligible is True -> call generar_etiqueta_devolucion and return aprobado with the exact rma and label_url.
        """
        # Ensure we do not modify tool outputs: call tools and use the returned dicts/strings as-is
        estado = self._consultar_estado_pedido(tracking_id)
        # estado is expected to be a dict with keys: exists, status, date
        if not isinstance(estado, dict) or not estado.get("exists"):
            return {
                "status": "rechazado",
                "motivo": f"El pedido {tracking_id} no existe o no fue encontrado.",
                "mensaje_usuario": "No encontramos tu pedido. Verifica el número o revisa tu correo de confirmación.",
                "pedido_status": estado.get("status") if isinstance(estado, dict) else None,
            }

        # Pedido existe: include pedido_status
        pedido_status = estado.get("status")

        eleg = self._verificar_elegibilidad_producto(tracking_id, sku)
        # eleg expected to be dict with keys eligible (bool) and reason (string)
        if not isinstance(eleg, dict) or not eleg.get("eligible"):
            # fetch policies (tool may return a string)
            polit = None
            try:
                polit = self._consultar_politicas(sku if sku else "general")
            except Exception:
                polit = None
            return {
                "status": "rechazado",
                "motivo": eleg.get("reason") if isinstance(eleg, dict) else str(eleg),
                "politicas": polit,
                "mensaje_usuario": "Tu producto no es elegible para devolución. Te explico por qué y qué alternativas tienes.",
                "pedido_status": pedido_status,
            }

        # eligible == True -> generate etiqueta
        etiqueta = self._generar_etiqueta_devolucion(tracking_id, sku)
        # etiqueta expected to be dict with keys rma and label_url
        return {
            "status": "aprobado",
            "rma": etiqueta.get("rma") if isinstance(etiqueta, dict) else None,
            "label_url": etiqueta.get("label_url") if isinstance(etiqueta, dict) else None,
            "mensaje_usuario": "Tu devolución fue aprobada. Usa esta etiqueta y entrega el paquete en el punto indicado.",
            "pedido_status": pedido_status,
        }

    def finalize_user_response(self, decision: dict) -> str:
        """
        Convert the decision dict from run_return_pipeline into a human-friendly message.

        MUST NOT modify critical fields (rma, label_url, motivo). Only wrap them in text.
        """
        status = decision.get("status")
        if status == "aprobado":
            rma = decision.get("rma")
            label = decision.get("label_url")
            pedido_status = decision.get("pedido_status")
            lines = [
                "Tu devolución ha sido aprobada.",
            ]
            if pedido_status:
                lines.append(f"Estado del pedido: {pedido_status}.")
            if rma:
                lines.append(f"Número RMA: {rma}")
            if label:
                lines.append(f"Descarga tu etiqueta aquí: {label}")
            lines.append(decision.get("mensaje_usuario", ""))
            return "\n".join(lines)

        # rechazado or others
        motivo = decision.get("motivo")
        pedido_status = decision.get("pedido_status")
        polit = decision.get("politicas")
        lines = ["Lo siento — no podemos aprobar tu devolución."]
        if pedido_status:
            lines.append(f"Estado del pedido: {pedido_status}.")
        if motivo:
            # use motivo exactly as provided
            lines.append(f"Motivo: {motivo}")
        if polit:
            lines.append("Políticas relacionadas:")
            lines.append(str(polit))
        lines.append(decision.get("mensaje_usuario", ""))
        return "\n".join(lines)

    # --- run agent ---
    def run(self, prompt: str) -> str:
        """Run the LangChain agent with the user's prompt and return a string response.

        The agent has tools available; final output is returned as text. We keep the response
        friendly by prefacing the model with the system prompt for returns from settings.toml.
        """
        # Construct an instruction that asks the agent to use the tools and answer warmly.
        system_prompt = SETTINGS.get("prompts", {}).get("return_policy", "")
        # Orchestrator: extract intent, then run deterministic pipelines for return flows.
        intent = self.extract_intent(prompt)

        accion = intent.get("accion")
        tracking = intent.get("tracking_id")
        sku = intent.get("sku")

        if accion == "solicitar_devolucion":
            # require tracking and sku
            if not tracking or not sku:
                missing = []
                if not tracking:
                    missing.append("número de tracking")
                if not sku:
                    missing.append("SKU del producto")
                return f"Por favor proporciona {', '.join(missing)} para tramitar la devolución."

            decision = self.run_return_pipeline(tracking, sku)
            return self.finalize_user_response(decision)

        elif accion == "estado_pedido":
            if not tracking:
                return "Por favor proporciona el número de tracking para consultar el estado del pedido."
            estado = self._consultar_estado_pedido(tracking)
            if not isinstance(estado, dict) or not estado.get("exists"):
                return "No encontramos tu pedido. Verifica el tracking e inténtalo de nuevo."
            return f"Estado del pedido {tracking}: {estado.get('status')} (fecha: {estado.get('date')})."

        elif accion == "politicas":
            tema = sku or "general"
            polit = self._consultar_politicas(tema)
            return f"Fragmentos de políticas relevantes:\n{polit}"

        else:
            # fallback: ask for clarification so the LLM doesn't invent business logic
            return "No estoy seguro de cómo ayudarte con eso — ¿puedes reformular tu solicitud indicando si quieres consultar un estado, solicitar una devolución o ver políticas?"
