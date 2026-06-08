from pathlib import Path
from langchain_groq import ChatGroq
from resilience.retry import llm_retry
from config import settings

_PROMPT = Path("prompts/v1/completeness_judge.txt").read_text()

_judge = ChatGroq(model=settings.GROQ_MODEL, temperature=0)


@llm_retry
async def score_completeness(
    intent: str,
    sub_queries: list[str],
    response: str,
) -> float:
    """
    LLM-as-judge metric: did the response address all sub-queries?
    Returns a float in [0.0, 1.0].
    """
    sub_queries_text = "\n".join(f"- {q}" for q in sub_queries) if sub_queries else "- (single question)"

    prompt = _PROMPT.format(
        intent=intent,
        sub_queries=sub_queries_text,
        response=response,
    )

    result = await _judge.ainvoke(prompt)

    try:
        # Pull the first decimal from the response (Llama sometimes adds prose)
        import re
        match = re.search(r"\d*\.?\d+", result.content.strip())
        if not match:
            return 0.5
        score = float(match.group(0))
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.5
