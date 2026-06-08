"""
Output validation: two independent LangGraph nodes run in parallel.

  faithfulness_node  — Groq LLM-as-judge: are response claims grounded in context?
  completeness_node  — Groq LLM-as-judge: did we answer all sub-queries?

Both fan out from execution and converge at validation_merge.
"""
from graph.state import SupportBotState
from metrics.faithfulness import score_faithfulness
from metrics.completeness import score_completeness
from observability.logging import get_logger
from config import settings


# ── Node A: Faithfulness ──────────────────────────────────────────────────────

async def faithfulness_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="faithfulness")
    context = state.get("retrieved_context", [])

    if not context:
        log.info("faithfulness_skipped", reason="no_context")
        return {"faithfulness_score": 1.0}

    score = await score_faithfulness(
        query=state["scrubbed_query"],
        response=state["raw_response"],
        context=context,
    )
    log.info("faithfulness_complete", score=round(score, 3))
    return {"faithfulness_score": score}


# ── Node B: Completeness ──────────────────────────────────────────────────────

async def completeness_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="completeness")
    score = await score_completeness(
        intent=state["intent"],
        sub_queries=state["sub_queries"],
        response=state["raw_response"],
    )
    log.info("completeness_complete", score=round(score, 3))
    return {"completeness_score": score}


# ── Node C: Validation merge ──────────────────────────────────────────────────

async def validation_merge_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="validation_merge")

    faithfulness = state.get("faithfulness_score", 1.0)
    completeness = state.get("completeness_score", 1.0)
    passed = (
        faithfulness >= settings.FAITHFULNESS_THRESHOLD
        and completeness >= settings.COMPLETENESS_THRESHOLD
    )

    if not passed:
        log.warning(
            "validation_failed",
            faithfulness=round(faithfulness, 3),
            completeness=round(completeness, 3),
        )
    else:
        log.info(
            "validation_passed",
            faithfulness=round(faithfulness, 3),
            completeness=round(completeness, 3),
        )

    return {
        "validation_passed": passed,
        "final_response": state["raw_response"],
    }
