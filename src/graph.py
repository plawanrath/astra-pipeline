#  Build a fan-out / fan-in graph with LangGraph’s StateGraph builder.
from langgraph.graph import StateGraph
from .agents import (
    collector, filter, sentiment, topics, merge, reporter
)


def build_graph():
    """
    Compile and return a runnable LangGraph.

    The graph:
        collector → filter
                      ↘︎              ↘︎
                 sentiment       topics
                      ↘︎              ↘︎
                           merge
                             ↓
                           report
    """
    builder = StateGraph(dict)                  # no strict schema for now

    # ── nodes ───────────────────────────────────────────────────────────────
    builder.add_node("collector", collector)
    builder.add_node("filter",    filter)
    builder.add_node("sentiment", sentiment)
    builder.add_node("topics",    topics)
    builder.add_node("merge",     merge)
    builder.add_node("report",    reporter)

    # ── edges ───────────────────────────────────────────────────────────────
    builder.add_edge("collector", "filter")
    builder.add_edge("filter", "sentiment")
    builder.add_edge("filter", "topics")
    builder.add_edge("sentiment", "merge")
    builder.add_edge("topics",    "merge")
    builder.add_edge("merge",     "report")

    builder.set_entry_point("collector")

    return builder.compile()      # returns a RunnableGraph with .invoke()