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