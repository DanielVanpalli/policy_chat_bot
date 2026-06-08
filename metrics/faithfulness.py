"""
Custom Groq-based faithfulness scorer — replaces the Ragas/OpenAI scorer
used in the reference project so the whole stack stays single-provider.

Approach: ask llama-3.1-8b-instant to judge what fraction of the response's
factual claims are directly supported by the retrieved context.
"""
import re
from pathlib import Path
from langchain_groq import ChatGroq
from resilience.retry import llm_retry
from config import settings

_PROMPT = Path("prompts/v1/faithfulness_judge.txt").read_text()

_judge = ChatGroq(model=settings.GROQ_MODEL, temperature=0)


@llm_retry
async def score_faithfulness(
    query: str,
    response: str,
    context: list[str],
) -> float:
    """
    LLM-as-judge metric: are the response's claims grounded in the context?
    Returns a float in [0.0, 1.0].
    """
    context_text = "\n\n---\n\n".join(context) if context else "(no context)"

    prompt = _PROMPT.format(
        query=query,
        context=context_text,
        response=response,
    )

    result = await _judge.ainvoke(prompt)

    match = re.search(r"\d*\.?\d+", result.content.strip())
    if not match:
        return 0.5
    try:
        score = float(match.group(0))
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.5
