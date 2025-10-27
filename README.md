# Proyecto Final: Implementación de un Agente de IA para Automatización de Tareas

**Caso de Estudio:**

El objetivo sigue siendo el mismo: transformar el asistente de atención al cliente de EcoMarket en una herramienta proactiva. La tarea específica a automatizar es el proceso de devolución de productos, que ahora incluirá la verificación de elegibilidad y la generación de la etiqueta.

---

## Fase 1: Diseño de la Arquitectura del Agente

### 1. Definición general del agente

El Agente de Devoluciones EcoMarket es una extensión del sistema RAG desarrollado en el Taller 2. Su propósito es automatizar el proceso de devolución de productos, combinando la recuperación de conocimiento (políticas de devolución y documentación interna) con herramientas que ejecutan acciones específicas, como verificar la elegibilidad y generar etiquetas de envío.

Este agente actúa como un intermediario inteligente entre el usuario y la base de conocimiento de EcoMarket: interpreta la intención del cliente, decide si debe consultar la documentación (vía RAG) o ejecutar una herramienta, y entrega una respuesta clara, contextualizada y útil.

### 2. Herramientas del agente

El agente dispone de las siguientes herramientas:

| Herramienta | Descripción | Entradas | Salidas |
|--------------|-------------|-----------|----------|
| **`consultar_estado_pedido(order_id)`** | Verifica en la base de pedidos (del Taller 1) si el pedido existe, su fecha y estado actual. | ID del pedido | `{exists: bool, status: str, date: datetime}` |
| **`verificar_elegibilidad_producto(order_id, sku)`** | Evalúa si un producto puede ser devuelto según políticas internas. | ID del pedido, SKU | `{eligible: bool, reason: str}` |
| **`consultar_politicas(categoria)`** | Recupera fragmentos de las políticas de devolución desde el índice RAG. | Categoría o tema | Texto explicativo |
| **`generar_etiqueta_devolucion(order_id, sku, carrier)`** | Crea una etiqueta de devolución simulada con número RMA y enlace de descarga. | ID del pedido, SKU, transportadora | `{label_url: str, rma: str}` |

Estas herramientas se diseñaron como simulaciones funcionales, pero el agente puede escalar para conectarse con sistemas reales (CRM o módulos logísticos de EcoMarket).

---

### 3. Selección del marco de trabajo (framework)

Para el desarrollo del agente se seleccionó LangGraph como marco principal, por su enfoque moderno, modularidad y excelente integración con ecosistemas de orquestación de agentes basados en modelos de lenguaje.

LangGraph es una evolución natural del ecosistema de LangChain, diseñada para crear flujos de agentes más estructurados, trazables y robustos, basados en el paradigma de grafos de estados. Este enfoque permite definir de forma clara cómo se comunican los componentes del sistema (nodos, herramientas, memoria, razonamiento y acciones), ofreciendo mayor control sobre la lógica interna del agente.

**Justificación técnica**

**1. Integración fluida con el sistema RAG existente:**
El Taller 2 introdujo un sistema RAG para la recuperación de políticas e información relevante de EcoMarket.
LangGraph permite integrar fácilmente esta capa de recuperación dentro del flujo del agente como un nodo de búsqueda independiente, manteniendo coherencia técnica con los componentes previos y garantizando reutilización del código base.

**2. Arquitectura orientada a grafos de decisión (state graphs):**
A diferencia del modelo secuencial de LangChain, LangGraph permite definir explícitamente los estados y transiciones del agente, lo que mejora la trazabilidad y el control del flujo lógico.
Esto encaja perfectamente con el diseño del proyecto, donde el agente debe alternar entre consultar el RAG, ejecutar herramientas como `verificar_elegibilidad_producto`, o `generar la respuesta final`.

**3. Transparencia y control del razonamiento:**
LangGraph conserva el paradigma Thought → Action → Observation → Answer, pero con mayor visibilidad sobre los pasos intermedios y el estado actual del grafo.
Esto facilita la depuración, la observabilidad y los análisis éticos del comportamiento del sistema, especialmente al tratar con decisiones automatizadas en procesos sensibles como devoluciones o reclamos.

**4. Escalabilidad y modularidad:**
Cada nodo del grafo puede representar una herramienta, un modelo LLM o un componente de decisión, lo que permite escalar fácilmente el sistema a tareas más complejas (reclamos, cambios, actualizaciones de pedidos).
Además, LangGraph soporta de forma nativa la integración con frameworks de despliegue como Streamlit, Gradio y FastAPI, facilitando la construcción de una aplicación web funcional para la fase de demostración.

**5. Ecosistema moderno y mantenido por LangChainAI:**
Aunque LangGraph proviene del mismo equipo detrás de LangChain, su diseño más reciente soluciona limitaciones del framework anterior en términos de rendimiento, trazabilidad y flexibilidad.
Proporciona soporte nativo para arquitecturas avanzadas de agentes como ReAct, Plan-and-Execute, Tool-calling agents y multi-agent collaboration, sin requerir estructuras complejas ni dependencias adicionales.

Su diseño basado en grafos garantiza un desarrollo más claro, reproducible y adaptable, asegurando una integración coherente con los desarrollos previos del proyecto.

---

### 4. Diseño del flujo de trabajo

El flujo del agente se basa en tres etapas: comprensión, decisión y ejecución. Primero, el agente interpreta la intención del usuario y determina si la solicitud requiere información o acción. Luego, decide si debe consultar el sistema RAG para obtener políticas relevantes o invocar una herramienta específica, como `consultar_estado_pedido`, `verificar_elegibilidad_producto` o `generar_etiqueta_devolucion`. Finalmente, combina los resultados obtenidos con la generación de lenguaje natural para entregar una respuesta clara, contextualizada y coherente con las políticas de EcoMarket. Este diseño garantiza un proceso lógico, modular y fácilmente escalable.

---
### 5. Representación visual del flujo
 

<img width="862" height="1715" alt="Mermaid Chart - Create complex, visual diagrams with text -2025-10-18-013730" src="https://github.com/user-attachments/assets/e978e683-473e-48f1-a38e-521dc429f8a9" />

---


## Fase 2: Implementación del Agente

### 1. Desarrollo e implementación

Para desarrollar el agente de IA, se utilizó LangGraph como framework principal, integrando un sistema RAG (Retrieval-Augmented Generation) que permite consultar políticas de devolución almacenadas en una base vectorial. El agente combina capacidades de recuperación de información con herramientas específicas que simulan operaciones reales del sistema EcoMarket. Se implementó un flujo basado en grafos de estados que permite al agente tomar decisiones inteligentes sobre cuándo consultar documentación, ejecutar herramientas o generar respuestas finales, manteniendo coherencia contextual en cada interacción.


### Estructura del proyecto

Este repositorio está organizado para separar claramente responsabilidades: orquestación del agente, herramientas, RAG, tests y la interfaz Streamlit.

Estructura principal:

- `app.py` — punto de entrada (opcional) para integraciones o pruebas rápidas.
- `create_db.py` — script para crear y poblar la base de datos SQLite (`ecomarket.db`) con datos de ejemplo.
- `requirements.txt` — dependencias Python necesarias para ejecutar el proyecto.
- `settings.toml` — configuración (modelo, prompts, políticas).
- `streamlit_app.py` — interfaz web para interactuar con el agente en modo demo.
- `agent/return_agent.py` — implementación principal del agente de devoluciones: extracción de intención, orquestación determinista (grafo), llamadas a herramientas y renderizado de la respuesta final.
- `data/` — CSVs de ejemplo (`orders.csv`, `products.csv`) usados por `create_db.py`.
- `rag/ingest.py` — script para indexar documentos/políticas en el vector store (RAG).
- `rag/retriever.py` — interfaz de alto nivel para recuperar fragmentos de políticas desde el vector store.


### Cómo usar este proyecto (rápido)

1. Clona el repositorio y sitúate en la carpeta del proyecto.
2. Crea y activa un entorno virtual (recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Instala dependencias:

```powershell
pip install -r requirements.txt
```

4. Configura variables de entorno necesarias (opcional según uso de LLM local):

- `OPENAI_API_KEY` — clave API o valor `ollama` si usas Ollama en local.
- `OPENAI_BASE_URL` — URL base del servicio LLM (por defecto `http://localhost:11434/v1`).

### Crear la base de datos (create_db.py)

El proyecto incluye `create_db.py` que crea una base de datos SQLite (`ecomarket.db`) y la rellena con datos de ejemplo contenidos en `data/orders.csv` y `data/products.csv`.

Para crear la DB de ejemplo ejecuta:

```powershell
python .\create_db.py
```

Al finalizar tendrás `ecomarket.db` en la raíz del proyecto y el agente podrá consultar pedidos y productos.

### Ingest (RAG)

El directorio `rag/` contiene las utilidades para indexar fragmentos de políticas en el vector store. Antes de ejecutar el agente con búsqueda de políticas, indexa tus documentos:

```powershell
python .\rag\ingest.py
```

Notas:
- `rag/ingest.py` usa la configuración por defecto del `rag/retriever.py`. Revisa y ajusta credenciales o parámetros si estás usando un servicio de vector store remotos.
- Tras ejecutar el `ingest`, el vector store (por ejemplo `chroma/`) contendrá los vectores necesarios para las consultas de políticas.

### Interfaz  Streamlit

Puedes ejecutar la interfaz web con Streamlit para probar el agente de forma interactiva:

```powershell
streamlit run .\streamlit_app.py
```

La aplicación te permite enviar mensajes, ver la interpretación de intención y observar la ejecución de las herramientas (consulta de pedido, verificación de elegibilidad, generación de etiqueta y recuperación de políticas).

### Consideraciones sobre el LLM

- Por defecto la configuración apunta a un modelo local (`phi3:mini` en `settings.toml`). Si quieres conectar OpenAI u otro servicio, define `OPENAI_API_KEY` y ajusta `OPENAI_BASE_URL`.
- El agente está diseñado para usar el LLM solo en tareas no deterministas (p. ej. redactar la respuesta final usando `finalize_user_response`) y para extracción de intención. Las decisiones del negocio (elegibilidad, verificación, rma) se toman con lógica determinista en Python y usando las salidas de las herramientas.

---

## Fase 3 — Análisis Crítico y Propuestas de Mejora

### 1. Análisis de Seguridad y Ética

Al otorgarle a un agente de IA la capacidad de tomar acciones autónomas, como procesar devoluciones o generar etiquetas, surgen nuevos riesgos éticos y de seguridad que deben abordarse cuidadosamente.

1. **Riesgos de ejecución indebida o no autorizada:**
Si el agente no valida correctamente la identidad del usuario o la elegibilidad del producto, podría ejecutar acciones indebidas (como aprobar una devolución fraudulenta).
**Mitigación:** Implementar un sistema de autenticación seguro y verificar los datos antes de cada acción. Las herramientas del agente deben incluir validaciones estrictas de entrada y salida.

2. **Privacidad y manejo de datos sensibles:**
El agente procesa información personal (nombre, dirección, historial de pedidos), lo que implica un riesgo de exposición o uso indebido de datos.
**Mitigación:** Aplicar principios de data minimization, encriptar la información sensible y limitar el acceso del agente solo a los datos estrictamente necesarios para cumplir su función.

3. **Riesgo de sesgos y decisiones injustas:**
Si el agente fue entrenado o afinado con datos sesgados, podría tomar decisiones discriminatorias (por ejemplo, priorizar ciertos tipos de clientes).
**Mitigación:** Revisar periódicamente los datos de entrenamiento y auditar las decisiones del agente para detectar sesgos o comportamientos no deseados.

4. **Responsabilidad y trazabilidad:**
Cuando un agente actúa de forma autónoma, surge la pregunta de quién es responsable de sus decisiones.
**Mitigación:** Cada acción del agente debe quedar registrada con metadatos (hora, decisión tomada, justificación, herramienta usada) para garantizar la trazabilidad y auditoría posterior.

---

### 2. Monitoreo y Observabilidad

Para garantizar un funcionamiento confiable, el agente debe contar con mecanismos de monitoreo y observabilidad que permitan detectar fallos, comportamientos anómalos o errores en la ejecución de herramientas.

**Propuestas de observabilidad:**

1. **Registro (logging) detallado:** Implementar un logger. Cada interacción debe registrar el prompt, herramientas utilizadas, resultado y estado (éxito/falla).

2. **Sistema de alertas:** Notificaciones al equipo cuando el agente comete errores repetidos, devuelve información incoherente o excede límites operativos.

3. **Panel de control (dashboard):** Métricas de uso, tasa de éxito, tiempos de respuesta y logs de actividad, permitiendo análisis continuo de desempeño.

4. **Pruebas controladas:** Mantener entornos de simulación (sandbox) antes del despliegue en producción para prevenir impactos reales.

---

### 3. Propuestas de Mejora

A partir del funcionamiento actual y de los hallazgos en las pruebas, se plantean las siguientes líneas de mejora para evolucionar el sistema:

1. **Implementar nuevas funcionalidades para ampliar el rango de acción:**
Extender las capacidades del agente más allá del proceso de devoluciones, permitiéndole realizar tareas complementarias como gestionar reclamos, procesar cambios de producto o consultar inventario. Esto consolidaría su rol como asistente integral dentro del ecosistema de atención al cliente.

2. **Integración con el CRM de EcoMarket:**
Conectar el agente directamente con el sistema de gestión de clientes (CRM) para actualizar información, registrar interacciones y mantener una vista unificada del historial de cada usuario.
Esto fortalecería la personalización del servicio y mejoraría la coherencia entre los distintos canales de atención.

3. **Aprendizaje continuo mediante retroalimentación supervisada:**
Incorporar un ciclo de feedback donde las correcciones humanas o las validaciones del equipo de soporte se utilicen para refinar las decisiones del agente.
De esta manera, el modelo evoluciona con base en la experiencia real, mejorando su precisión y adaptabilidad a nuevos escenarios.

4. **Implementación de un enfoque Human-in-the-Loop:**
Integrar una capa de supervisión humana para las decisiones críticas o de alto impacto.
Este enfoque equilibra autonomía y control, asegurando que los agentes actúen con responsabilidad, transparencia y alineación con las políticas éticas y operativas de la empresa.