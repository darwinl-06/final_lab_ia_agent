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

Para el desarrollo del agente se seleccionó **LangChain** como marco principal, por su robustez, madurez y compatibilidad directa con el sistema RAG implementado previamente en el Taller 2.

LangChain es actualmente uno de los frameworks más completos para la **orquestación de agentes y cadenas de razonamiento** con modelos de lenguaje. Ofrece un ecosistema modular que permite integrar modelos LLM, herramientas externas, memorias, bases vectoriales y trazabilidad de decisiones dentro de un mismo entorno de ejecución.

#### Justificación técnica

1. **Integración fluida con el sistema RAG existente:**  El Taller 2 fue desarrollado sobre LangChain para la gestión de embeddings, almacenamiento vectorial y recuperación de información. Continuar con el mismo framework garantiza **consistencia técnica**, evita retrabajo y permite reutilizar directamente los componentes ya probados (retriever, chain, y prompt templates).

2. **Arquitectura orientada a herramientas (Tools):**  LangChain proporciona un modelo nativo para registrar funciones externas como *Tools*, permitiendo al agente decidir **cuándo** y **cómo** utilizarlas.  
   Este enfoque encaja perfectamente con la lógica del proyecto, donde el agente debe invocar acciones concretas como `verificar_elegibilidad_producto` o `generar_etiqueta_devolucion`, de manera controlada y trazable.

3. **Transparencia en el razonamiento del agente:**  Permite visualizar cada paso del proceso de pensamiento (*Thought → Action → Observation → Final Answer*), lo cual facilita la depuración, auditoría y análisis ético del comportamiento del sistema. Esta trazabilidad es clave en aplicaciones con decisiones automatizadas sobre clientes.

4. **Madurez, soporte y escalabilidad:**  LangChain es ampliamente utilizado en la industria, cuenta con una comunidad activa y soporte continuo para múltiples modelos (OpenAI, Anthropic, Mistral, Cohere, entre otros).  
   Además, ofrece integraciones nativas con frameworks de despliegue como **Streamlit**, **Gradio** y **FastAPI**, lo que simplifica el paso a la fase de demostración e implementación.

5. **Compatibilidad con arquitecturas de agentes avanzadas:**  Su ecosistema incluye controladores como **ReAct**, **ConversationalAgent**, **Plan-and-Execute** y **Tool-calling agents**, lo que permite evolucionar el proyecto hacia comportamientos más complejos sin cambiar la base tecnológica.

Aunque se evaluó el uso de **LlamaIndex**, este framework está más orientado a la **gestión y consulta estructurada de conocimiento** (document retrieval) que a la planificación de acciones con herramientas.  
LangChain, en cambio, ofrece una **mayor flexibilidad en la toma de decisiones del agente**, mejor documentación y una integración más directa con pipelines multi-herramienta.

Por estas razones, **LangChain** resulta ser la opción más apropiada, asegurando una arquitectura coherente, escalable y fácilmente integrable con los desarrollos previos del proyecto.

---

### 4. Diseño del flujo de trabajo

El flujo del agente se basa en tres etapas: comprensión, decisión y ejecución. Primero, el agente interpreta la intención del usuario y determina si la solicitud requiere información o acción. Luego, decide si debe consultar el sistema RAG para obtener políticas relevantes o invocar una herramienta específica, como `consultar_estado_pedido`, `verificar_elegibilidad_producto` o `generar_etiqueta_devolucion`. Finalmente, combina los resultados obtenidos con la generación de lenguaje natural para entregar una respuesta clara, contextualizada y coherente con las políticas de EcoMarket. Este diseño garantiza un proceso lógico, modular y fácilmente escalable.

---
### 5. Representación visual del flujo
 

<img width="862" height="1715" alt="Mermaid Chart - Create complex, visual diagrams with text -2025-10-18-013730" src="https://github.com/user-attachments/assets/e978e683-473e-48f1-a38e-521dc429f8a9" />


