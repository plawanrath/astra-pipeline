#  Build a fan-out / fan-in graph with LangGraph’s StateGraph builder.
from langgraph.graph import StateGraph
from .agents import (
    collector, location_inference, filter,
    sentiment5, sentiment3, topics,
    merge, reporter
)

def build_graph():
    g = StateGraph(dict)

    g.add_node("collector",          collector)
    g.add_node("location_inference", location_inference)
    g.add_node("filter",             filter)
    g.add_node("sentiment5",         sentiment5)
    g.add_node("sentiment3",         sentiment3)
    g.add_node("topics",             topics)
    g.add_node("merge",              merge)
    g.add_node("report",             reporter)

    g.add_edge("collector",          "location_inference")
    g.add_edge("location_inference", "filter")
    g.add_edge("filter",             "sentiment5")
    g.add_edge("sentiment5",         "sentiment3")
    g.add_edge("sentiment3",         "topics")
    g.add_edge("topics",             "merge")
    g.add_edge("merge",              "report")

    g.set_entry_point("collector")
    return g.compile()
