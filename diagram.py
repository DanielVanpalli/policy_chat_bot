"""Generate a PNG of the LangGraph workflow using the built-in draw_mermaid_png()."""

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END

from graph.state import SupportBotState
from graph.nodes.safety_gate import pii_scrub_node, attack_detect_node, safety_merge_node
from graph.nodes.query_intelligence import query_intelligence_node
from graph.nodes.session_memory import session_memory_node
from graph.nodes.context_retrieval import context_retrieval_node
from graph.nodes.execution import (
    generate_flash_node,
    generate_pro_node,
    generate_subquery_node,
    merge_subqueries_node,
    route_execution,
)
from graph.nodes.output_validation import faithfulness_node, completeness_node, validation_merge_node
from graph.nodes.cache_store import cache_store_node


def _route_after_safety(state: SupportBotState) -> str:
    return END if state.get("is_attack") else "query_intelligence"


def build_graph_for_diagram():
    g = StateGraph(SupportBotState)

    g.add_node("pii_scrub", pii_scrub_node)
    g.add_node("attack_detect", attack_detect_node)
    g.add_node("safety_merge", safety_merge_node)
    g.add_node("query_intelligence", query_intelligence_node)
    g.add_node("session_memory", session_memory_node)
    g.add_node("context_retrieval", context_retrieval_node)
    g.add_node("generate_flash", generate_flash_node)
    g.add_node("generate_pro", generate_pro_node)
    g.add_node("generate_subquery", generate_subquery_node)
    g.add_node("merge_subqueries", merge_subqueries_node)
    g.add_node("faithfulness", faithfulness_node)
    g.add_node("completeness", completeness_node)
    g.add_node("validation_merge", validation_merge_node)
    g.add_node("cache_store", cache_store_node)

    g.set_entry_point("pii_scrub")
    g.set_entry_point("attack_detect")
    g.add_edge("pii_scrub", "safety_merge")
    g.add_edge("attack_detect", "safety_merge")

    g.add_conditional_edges("safety_merge", _route_after_safety)

    g.add_edge("query_intelligence", "session_memory")
    g.add_edge("session_memory", "context_retrieval")

    g.add_conditional_edges(
        "context_retrieval",
        route_execution,
        {
            "generate_flash": "generate_flash",
            "generate_pro": "generate_pro",
        },
    )

    for exec_node in ("generate_flash", "generate_pro"):
        g.add_edge(exec_node, "faithfulness")
        g.add_edge(exec_node, "completeness")

    g.add_edge("generate_subquery", "merge_subqueries")
    g.add_edge("merge_subqueries", "faithfulness")
    g.add_edge("merge_subqueries", "completeness")

    g.add_edge("faithfulness", "validation_merge")
    g.add_edge("completeness", "validation_merge")

    g.add_edge("validation_merge", "cache_store")
    g.add_edge("cache_store", END)

    return g.compile()


if __name__ == "__main__":
    compiled = build_graph_for_diagram()
    png_bytes = compiled.get_graph().draw_mermaid_png()
    out = "workflow.png"
    with open(out, "wb") as f:
        f.write(png_bytes)
    print(f"Diagram saved → {out}")
