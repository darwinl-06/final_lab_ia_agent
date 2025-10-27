import os
import sqlite3
import tomllib
import json
import random
import string
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional

try:
    from langgraph.graph import StateGraph, END
except Exception:
    # Minimal local fallback implementation for environments without langgraph
    # This provides just enough behaviour for tests and local runs. For
    # production, install the real `langgraph` package.
    END = object()

    class StateGraph:
        def __init__(self, _typed):
            self._nodes = {}
            self._edges = {}
            self._conditional = {}
            self._entry = None
            self._finish = END

        def add_node(self, name, fn):
            self._nodes[name] = fn

        def add_edge(self, src, dst):
            self._edges[src] = dst

        def add_conditional_edges(self, src, decision_fn):
            self._conditional[src] = decision_fn

        def set_entry_point(self, name):
            self._entry = name

        def set_finish_point(self, finish):
            self._finish = finish

        def compile(self):
            return self

        def invoke(self, state):
            cur = self._entry
            # loop until END or no node
            while cur is not None and cur is not END:
                fn = self._nodes.get(cur)
                if not fn:
                    break
                res = fn(state)
                if res is END:
                    break
                # conditional edge
                if cur in self._conditional:
                    cur = self._conditional[cur](state)
                    continue
                # simple edge
                cur = self._edges.get(cur)
            return state

from openai import OpenAI
from rag.retriever import RAGRetriever


SETTINGS = tomllib.loads(Path("settings.toml").read_text(encoding="utf-8"))


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    )


class OllamaLLM:

    def __init__(self, client: OpenAI, model: str, temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    def _call(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        return r.choices[0].message.content.strip()


class ReturnState(TypedDict):
    tracking_id: str
    sku: str
    pedido: Optional[dict]
    elegibilidad: Optional[dict]
    etiqueta: Optional[dict]
    decision: Optional[dict]


class ReturnAgent:
    def __init__(self):
        self.client = _make_client()
        self.rag = RAGRetriever(k=4)

        model = SETTINGS["general"]["model"]
        temp = SETTINGS["general"].get("temperature", 0)
        # Instantiate the small LLM wrapper with configured model and temperature
        self.llm = OllamaLLM(client=self.client, model=model, temperature=temp)

        

    # --- Tools implementation ---
    def _connect_db(self) -> sqlite3.Connection:
        # database file created by create_db.py: ecomarket.db
        return sqlite3.connect("ecomarket.db")

    def _consultar_estado_pedido(self, order_id: str) -> dict:
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

    def _verificar_elegibilidad_producto(self, order_id: str, sku: str) -> dict:
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

    def _consultar_politicas(self, categoria: str) -> dict:
        docs = self.rag.get_relevant_chunks(categoria)
        if not docs:
            return {"found": False, "message": "No se han encontrado fragmentos de política para la categoría indicada."}
        out = []
        for d in docs:
            src = d.metadata.get("source", "source")
            out.append(f"[{src}] {d.page_content.strip()}")
        return {"found": True, "message": "\n\n".join(out)}

    def _generar_etiqueta_devolucion(self, order_id: str, sku: str, carrier: str = "EcoShip") -> dict:
        rma = "RMA-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
        label_url = f"https://labels.ecomarket.test/{rma}.pdf"
        return {"label_url": label_url, "rma": rma}

    def extract_intent(self, user_message: str) -> dict:
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
        system_prompt = SETTINGS.get("prompts", {}).get("return_policy", "")
        decision_json = json.dumps(decision, ensure_ascii=False, indent=2)

        # Prepare simple placeholder mapping from common keys into the system prompt.
        mapping = {
            "rma": decision.get("rma", ""),
            "label_url": decision.get("label_url", ""),
            "name": decision.get("name", ""),
            "sku": decision.get("sku", ""),
            "tracking": decision.get("tracking_id", ""),
            "motivo": decision.get("motivo", ""),
            # provide the whole decision as CONTEXTO
            "context": decision_json,
        }

        # Replace placeholders in the prompt. Support both {{key}} and {key} styles.
        filled_prompt = system_prompt
        for k, v in mapping.items():
            safe = "" if v is None else str(v)
            filled_prompt = filled_prompt.replace("{{" + k + "}}", safe)
            filled_prompt = filled_prompt.replace("{" + k + "}", safe)

        prompt = filled_prompt + "\n\n" + decision_json
        try:
            response = self.llm._call(prompt)
            if response and response.strip():
                return response
        except Exception:
            # fall through to deterministic fallback
            pass

        # deterministic fallback (previous behavior)
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

        motivo = decision.get("motivo")
        pedido_status = decision.get("pedido_status")
        polit = decision.get("politicas")
        lines = ["Lo siento — no podemos aprobar tu devolución."]
        if pedido_status:
            lines.append(f"Estado del pedido: {pedido_status}.")
        if motivo:
            lines.append(f"Motivo: {motivo}")
        if polit:
            lines.append("Políticas relacionadas:")
            lines.append(str(polit))
        lines.append(decision.get("mensaje_usuario", ""))
        return "\n".join(lines)

    # --- LangGraph nodes and graph builder ---
    def consultar_estado_pedido_node(self, state: ReturnState):
        tracking = state.get("tracking_id")
        pedido = self._consultar_estado_pedido(tracking)
        state["pedido"] = pedido
        if not isinstance(pedido, dict) or not pedido.get("exists"):
            state["decision"] = {
                "status": "rechazado",
                "motivo": f"El pedido {tracking} no existe o no fue encontrado.",
                "mensaje_usuario": "No encontramos tu pedido. Verifica el número o revisa tu correo de confirmación.",
                "pedido_status": pedido.get("status") if isinstance(pedido, dict) else None,
            }
            return END
        return "verificar_elegibilidad"

    def verificar_elegibilidad_node(self, state: ReturnState):
        tracking = state.get("tracking_id")
        sku = state.get("sku")
        eleg = self._verificar_elegibilidad_producto(tracking, sku)
        state["elegibilidad"] = eleg
        if not isinstance(eleg, dict) or not eleg.get("eligible"):
            return "consultar_politicas"
        return "generar_etiqueta"

    def consultar_politicas_node(self, state: ReturnState):
        sku = state.get("sku") or "general"
        polit = self._consultar_politicas(sku)
        state["decision"] = {
            "status": "rechazado",
            "motivo": state.get("elegibilidad", {}).get("reason") if state.get("elegibilidad") else "Producto no elegible",
            "politicas": polit,
            "mensaje_usuario": "Tu producto no es elegible para devolución. Te explico por qué y qué alternativas tienes.",
            "pedido_status": state.get("pedido", {}).get("status") if state.get("pedido") else None,
        }
        return END

    def generar_etiqueta_node(self, state: ReturnState):
        tracking = state.get("tracking_id")
        sku = state.get("sku")
        etiqueta = self._generar_etiqueta_devolucion(tracking, sku)
        state["etiqueta"] = etiqueta
        state["decision"] = {
            "status": "aprobado",
            "rma": etiqueta.get("rma") if isinstance(etiqueta, dict) else None,
            "label_url": etiqueta.get("label_url") if isinstance(etiqueta, dict) else None,
            "mensaje_usuario": "Tu devolución fue aprobada. Usa esta etiqueta y entrega el paquete en el punto indicado.",
            "pedido_status": state.get("pedido", {}).get("status") if state.get("pedido") else None,
        }
        return END

    def build_return_graph(self):
        graph = StateGraph(ReturnState)

        # When adding nodes, wrap them to allow simple console tracing when desired.
        def _wrap_node(name, fn):
            def _wrapped(state):
                print(f"[AGENT TRACE] entering node: {name} | tracking={state.get('tracking_id')} sku={state.get('sku')}")
                return fn(state)
            return _wrapped

        graph.add_node("consultar_estado_pedido", _wrap_node("consultar_estado_pedido", self.consultar_estado_pedido_node))
        graph.add_node("verificar_elegibilidad", _wrap_node("verificar_elegibilidad", self.verificar_elegibilidad_node))
        graph.add_node("consultar_politicas", _wrap_node("consultar_politicas", self.consultar_politicas_node))
        graph.add_node("generar_etiqueta", _wrap_node("generar_etiqueta", self.generar_etiqueta_node))

        # Flujo: del estado al verificador de elegibilidad
        graph.add_edge("consultar_estado_pedido", "verificar_elegibilidad")

        # Condicional desde verificar_elegibilidad a politicas o generar_etiqueta
        graph.add_conditional_edges(
            "verificar_elegibilidad",
            lambda state: "consultar_politicas" if not (state.get("elegibilidad") and state.get("elegibilidad").get("eligible")) else "generar_etiqueta",
        )

        graph.set_entry_point("consultar_estado_pedido")
        graph.set_finish_point(END)

        return graph.compile()

    # --- run agent ---
    def run(self, prompt: str) -> str:
        """Run the agent: interpret intent, execute the LangGraph return flow, and render a response.

        The deterministic business flow lives in a LangGraph StateGraph. The LLM is used only
        to extract the intent and for optional final textual polishing in `finalize_user_response`.
        """
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

            # Build initial graph state
            state: ReturnState = {
                "tracking_id": tracking,
                "sku": sku,
                "pedido": None,
                "elegibilidad": None,
                "etiqueta": None,
                "decision": None,
            }

            graph = self.build_return_graph()
            final_state = graph.invoke(state)
            decision = final_state.get("decision") or {"status": "rechazado", "motivo": "Error interno"}
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
