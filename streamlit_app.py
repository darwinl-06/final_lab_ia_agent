import streamlit as st
import time
import re
import json
import html
from typing import List, Dict


def query_agent(prompt: str) -> str:
    """Lazy-load agent and query it. Return a string (may be JSON)."""
    try:
        from agent import handle_request
    except Exception as e:
        return f"Error al cargar el agente: {e}"
    return handle_request(prompt)


def ensure_history():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts {prompt, response, time}


def append_history(prompt: str, response: str):
    st.session_state.history.insert(0, {"prompt": prompt, "response": response, "ts": time.time()})


def ok_copy_html(text: str) -> str:
    js = f'''<button id="copybtn">Copiar</button>
<script>
const text = `{html.escape(text)}`;
const btn = document.getElementById('copybtn');
btn.addEventListener('click', () => {{
  navigator.clipboard.writeText(text).then(()=>{{ btn.innerText='Copiado'; setTimeout(()=>btn.innerText='Copiar',1500)}})
}});
</script>
'''
    return js


st.set_page_config(page_title="Agente Devoluciones - EcoMarket", page_icon="📦", layout="centered")

st.title("📦 Agente de Devoluciones — EcoMarket")
st.markdown("Escribe tu consulta en lenguaje natural. El agente puede verificar estado de pedido, elegibilidad de producto, consultar políticas y generar etiquetas de devolución.")

ensure_history()

col_main = st.container()

with col_main:
    with st.form(key="agent_form"):
        prompt_box = st.text_area("Pregunta / Solicitud", height=140, key="_prompt_box",
                                 placeholder="Ej: Quiero devolver el pedido TRK-0004, producto SKU-004")
        # ejemplos rápidos (ahora en el formulario)
        examples = ["-- elige un ejemplo --",
                    "Quiero devolver el pedido TRK-0004, el producto es el SKU-004",
                    "¿Cuál es la política de devoluciones para productos de la categoría 'hogar'?",
                    "Genera etiqueta para TRK-0004, SKU-004 usando EcoShip",
                    "Quiero devolver TRK-0001 SKU-001. ¿Es posible?"]
        sel = st.selectbox("Ejemplos rápidos", examples)
        if sel != examples[0]:
            prompt_box = sel

        submit = st.form_submit_button("Enviar")

    if submit:
        prompt = prompt_box.strip()
        if not prompt:
            st.warning("Escribe una pregunta o selecciona un ejemplo antes de enviar.")
        else:
            with st.spinner("Procesando con el agente..."):
                response = query_agent(prompt)
                time.sleep(0.2)
            append_history(prompt, response)

            # Mostrar respuesta inmediatamente debajo del formulario
            st.markdown("**Respuesta del agente:**")
            # intentar parsear JSON para mostrar bonito
            try:
                parsed = json.loads(response)
                st.json(parsed)
            except Exception:
                st.markdown(response)

    st.markdown("---")
    st.subheader("Historial")
    if not st.session_state.history:
        st.info("No hay interacciones todavía.")
    else:
        for i, item in enumerate(st.session_state.history[:20]):
            with st.expander(f"{item['prompt']}", expanded=(i==0)):
                st.markdown(f"**Respuesta:**")
                resp = item['response']
                # detect JSON
                try:
                    parsed = json.loads(resp)
                    st.json(parsed)
                except Exception:
                    st.markdown(resp)
                # detect URLs (simple)
                urls = re.findall(r"https?://\S+", resp)
                if urls:
                    for u in urls:
                        st.markdown(f"[Abrir enlace]({u})")

# sidebar and last-response removed per request

st.markdown("---")
st.caption("Interfaz para comunicación con el agente de devoluciones de EcoMarket.")
