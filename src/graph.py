#  Build a fan-out / fan-in graph with LangGraph’s StateGraph builder.
from langgraph.graph import StateGraph
from .agents import (
    collector, filter,
    sentiment5, sentiment3, topics,
    merge, reporter
)

def build_graph():
    g = StateGraph(dict)

    # register nodes
    g.add_node("collector",  collector)
    g.add_node("filter",     filter)
    g.add_node("sentiment5", sentiment5)
    g.add_node("sentiment3", sentiment3)
    g.add_node("topics",     topics)
    g.add_node("merge",      merge)
    g.add_node("report",     reporter)

    # sequential edges  (no parallel writes ⇒ no InvalidUpdateError)
    g.add_edge("collector",  "filter")
    g.add_edge("filter",     "sentiment5")
    g.add_edge("sentiment5", "sentiment3")
    g.add_edge("sentiment3", "topics")
    g.add_edge("topics",     "merge")
    g.add_edge("merge",      "report")

    g.set_entry_point("collector")
    return g.compile()
