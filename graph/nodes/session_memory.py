"""
Session memory is largely handled automatically by LangGraph's PostgresSaver
checkpointer (configured in graph.py). This node trims history so we don't
exceed context window limits.
"""
from graph.state import SupportBotState
from observability.logging import get_logger
from config import settings


async def session_memory_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="session_memory")

    history = state.get("session_history") or []
    max_turns = settings.MAX_SESSION_TURNS

    if len(history) > max_turns * 2:
        history = history[-(max_turns * 2):]
        log.info("session_history_trimmed", kept_turns=max_turns)
    else:
        log.info("session_memory_loaded", num_messages=len(history))

    return {"session_history": history}
