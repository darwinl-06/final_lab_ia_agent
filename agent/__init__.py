from .return_agent import ReturnAgent


_AGENT: ReturnAgent | None = None


def handle_request(prompt: str) -> str:
    """Top-level entry used by Streamlit. Lazily instantiates the ReturnAgent.

    Returns a string response (agent output)."""
    global _AGENT
    if _AGENT is None:
        _AGENT = ReturnAgent()
    return _AGENT.run(prompt)


def handle_return(order_id: str, sku: str, carrier: str = "EcoShip") -> str:
    """Deterministic return flow: verify order, check eligibility, consult policies and generate label.

    Returns a JSON string with structured fields to avoid hallucinations.
    """
    global _AGENT
    if _AGENT is None:
        _AGENT = ReturnAgent()
    return _AGENT.perform_return_flow(order_id, sku, carrier)
