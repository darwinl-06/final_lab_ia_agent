# agent.py
"""
Agente para gestionar solicitudes de devolución usando LangChain y el RAG existente.
Define las herramientas externas que usa el agente y un punto de entrada `handle_request`.

Herramientas implementadas:
- consultar_estado_pedido(order_id)
- verificar_elegibilidad_producto(order_id, sku)
- consultar_politicas(categoria)
- generar_etiqueta_devolucion(order_id, sku, carrier)

El agente utiliza `rag.RAGRetriever` para obtener fragmentos cuando sea necesario.
"""
from __future__ import annotations
import csv
from datetime import datetime
import os
import random
from typing import Dict, Any

from rag.retriever import RAGRetriever

# LangChain imports (lightweight usage)
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
# Chat models were moved to langchain-community. Prefer that import when available
try:
    from langchain_community.chat_models import ChatOpenAI
except Exception:
    # fall back to original import for older environments to avoid breaking
    from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Config
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ORDERS_FILE = os.path.join(DATA_DIR, "orders.csv")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products.csv")

# Simple CSV loaders (semicolon-separated as in repo)
def _load_orders() -> list[Dict[str, str]]:
    out = []
    with open(ORDERS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            # normalize keys (strip BOM and whitespace) and lowercase
            clean = {}
            for k, v in r.items():
                if k is None:
                    continue
                key = k.strip().lstrip('\ufeff').lower()
                clean[key] = v.strip() if isinstance(v, str) else v
            out.append(clean)
    return out

def _load_products() -> list[Dict[str, str]]:
    out = []
    with open(PRODUCTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            clean = {}
            for k, v in r.items():
                if k is None:
                    continue
                key = k.strip().lstrip('\ufeff').lower()
                clean[key] = v.strip() if isinstance(v, str) else v
            out.append(clean)
    return out

# Instantiate retriever
rag = RAGRetriever(k=4)

# Tool implementations

def consultar_estado_pedido(order_id: str) -> Dict[str, Any]:
    """Busca un pedido por tracking/order_id y devuelve existencia, estado y fecha."""
    orders = _load_orders()
    for o in orders:
        if o.get("tracking") == order_id:
            # parse eta/last_update
            date_str = o.get("eta") or o.get("last_update")
            try:
                date = datetime.fromisoformat(date_str)
            except Exception:
                date = None
            return {
                "exists": True,
                "status": o.get("status", "unknown"),
                "date": date,
                "carrier": o.get("carrier", ""),
                "tracking_url": o.get("tracking_url", "")
            }
    return {"exists": False, "status": None, "date": None}


def verificar_elegibilidad_producto(order_id: str, sku: str) -> Dict[str, Any]:
    """Reglas simples de elegibilidad basadas en 'returnable' y días desde entrega.
    Usa RAG para recuperar políticas relevantes si el caso requiere explicación.
    """
    orders = _load_orders()
    products = _load_products()

    # find order
    order = next((o for o in orders if o.get("tracking") == order_id), None)
    if not order:
        return {"eligible": False, "reason": "Pedido no encontrado"}

    # find product
    product = next((p for p in products if p.get("sku") == sku), None)
    if not product:
        return {"eligible": False, "reason": "SKU no encontrado"}

    # basic rules: product returnable flag
    returnable = product.get("returnable", "false").lower() in ("true", "1", "yes")

    # compute days since delivery if order status is 'Entregado'
    status = order.get("status", "").lower()
    if status == "entregado":
        try:
            eta = datetime.fromisoformat(order.get("eta"))
            days = (datetime.now() - eta).days
        except Exception:
            days = None
    else:
        days = None

    if not returnable:
        # Ask RAG for policy snippet to explain
        ctx = rag.get_relevant_chunks(f"Devoluciones para categoria {product.get('category')}")
        policy_text = "\n\n".join([d.page_content for d in ctx]) if ctx else "La política indica que este producto no es retornable."
        return {"eligible": False, "reason": f"Producto no retornable. {policy_text}"}

    # If returnable, check days limit (example: 30 days)
    if days is not None and days > 30:
        return {"eligible": False, "reason": f"Plazo de devolución vencido ({days} días desde la entrega)."}

    # Otherwise eligible
    return {"eligible": True, "reason": "Elegible para devolución"}


def consultar_politicas(categoria: str) -> str:
    """Devuelve texto de políticas relevantes consultando el RAG."""
    docs = rag.get_relevant_chunks(f"Política de devoluciones categoria {categoria}")
    if not docs:
        return "No se encontraron políticas relevantes."
    return "\n\n".join([f"[{d.metadata.get('source','source')}] {d.page_content}" for d in docs])


def generar_etiqueta_devolucion(order_id: str, sku: str, carrier: str) -> Dict[str, str]:
    """Genera una etiqueta simulada con RMA y URL.
    En un sistema real aquí se llamaría al servicio de transportista.
    """
    rma = f"RMA-{random.randint(100000,999999)}"
    label_url = f"https://labels.example.com/{rma}.pdf"
    return {"label_url": label_url, "rma": rma}


# LangChain Tool wrappers
TOOLS = [
    Tool.from_function(consultar_estado_pedido, name="consultar_estado_pedido", description="Verifica si un pedido existe y devuelve estado, fecha y carrier. Entrada: order tracking id."),
    Tool.from_function(verificar_elegibilidad_producto, name="verificar_elegibilidad_producto", description="Verifica si un SKU es elegible para devolución en un pedido. Entrada: order_id, sku."),
    Tool.from_function(consultar_politicas, name="consultar_politicas", description="Recupera políticas de devolución desde el índice RAG. Entrada: categoria o tema."),
    Tool.from_function(generar_etiqueta_devolucion, name="generar_etiqueta_devolucion", description="Genera una etiqueta de devolución simulada. Entrada: order_id, sku, carrier."),
]

# Agent initializer

def create_agent() -> Any:
    """Crea un agent executor de LangChain usando un modelo de chat local/remote.
    Retorna el executor listo para recibir una pregunta en lenguaje natural.
    """
    # model selection: re-use settings from app.py if available via env
    model_name = os.getenv("AI_MODEL", "gpt-4o-mini")
    # Try to create a real LLM-backed agent. Prefer the `app.client` OpenAI
    # instance if available (the project already configures it for Ollama/local);
    # otherwise fall back to ChatOpenAI. If both fail, provide a SimpleAgent.
    try:
        # try to reuse app.client if present
        import app as app_module
        client = getattr(app_module, "client", None)
    except Exception:
        client = None

    # If we have an `app.client`, create a thin LangChain LLM wrapper that calls it
    try:
        if client is not None:
            from langchain.llms.base import LLM

            class AppClientLLM(LLM):
                """LLM wrapper that delegates to the project's `app.client` (OpenAI client).
                This allows using local Ollama endpoints already configured in app.py.
                """
                model_name = "app.client"

                def _call(self, prompt: str, stop=None) -> str:
                    # Build a single-turn chat completion using the app client
                    try:
                        model_name_local = os.getenv("AI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
                        r = client.chat.completions.create(
                            model=model_name_local,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                        )
                        return r.choices[0].message.content.strip()
                    except Exception as e:
                        raise

                @property
                def _identifying_params(self):
                    return {"model": "app.client"}

                @property
                def _llm_type(self):
                    return "app_client"

            llm = AppClientLLM()
            agent = initialize_agent(
                tools=TOOLS,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
            )
            return agent

        # else fall back to ChatOpenAI
        llm = ChatOpenAI(model=model_name, temperature=0)
        agent = initialize_agent(
            tools=TOOLS,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )
        return agent
    except Exception:
        # Fallback simple agent (no LLM) — supports a few common intents.
        class SimpleAgent:
            def run(self, query: str) -> str:
                q = query.lower()
                # Try to extract tracking and sku
                import re
                trk = None
                sku = None
                m = re.search(r"trk[-_ ]?\d{4}", q)
                if m:
                    trk = m.group(0).upper()
                m2 = re.search(r"sku[-_ ]?\d{3}", q)
                if m2:
                    sku = m2.group(0).upper()

                if "return" in q or "devol" in q:
                    if trk and sku:
                        st = consultar_estado_pedido(trk)
                        elig = verificar_elegibilidad_producto(trk, sku)
                        if not st.get("exists"):
                            return f"No se encontró el pedido {trk}."
                        if not elig.get("eligible"):
                            return f"El producto {sku} no es elegible: {elig.get('reason')}"
                        # generate label using carrier from order
                        carrier = st.get("carrier") or "eco"
                        lab = generar_etiqueta_devolucion(trk, sku, carrier)
                        return (
                            f"Pedido {trk} encontrado (estado: {st.get('status')}).\n"
                            f"Producto {sku} elegible para devolución. RMA: {lab.get('rma')}. \nDescargar etiqueta: {lab.get('label_url')}"
                        )
                    return "Necesito el tracking del pedido y el SKU para procesar una devolución."

                # fallback to RAG policy query
                if "política" in q or "politica" in q or "policy" in q:
                    # attempt to find category
                    mcat = re.search(r"categoria[: ]?(\w+)", q)
                    cat = mcat.group(1) if mcat else "general"
                    pol = consultar_politicas(cat)
                    return f"Políticas relevantes para {cat}:\n\n{pol}"

                # default help
                return (
                    "Puedo ayudar con consultas de devolución: indicar 'devolución' más el tracking y SKU. "
                    "Ejemplo: 'Quiero devolver el pedido TRK-0004 producto SKU-004'"
                )

        return SimpleAgent()


# High-level handler

def handle_request(query: str) -> str:
    """Recibe una pregunta en lenguaje natural, ejecuta el agente y devuelve una respuesta formateada."""
    agent = create_agent()
    try:
        result = agent.run(query)
    except Exception as e:
        return f"Error interno al procesar la solicitud: {e}"
    # LangChain agent returns text; we can post-process for clarity
    return result


if __name__ == "__main__":
    # quick CLI for manual testing
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Pregunta en lenguaje natural")
    args = ap.parse_args()
    print(handle_request(args.q))
