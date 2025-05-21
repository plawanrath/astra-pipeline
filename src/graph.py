# src/graph.py
from langgraph import Graph, Agent
from agents import collector, filter, sentiment, topics, merge, reporter

def build_graph():
    g = Graph("pipeline-fanout-fanin")

    # ── nodes ───────────────────────────────────────────────────────────────
    g.add(Agent("collector", collector))
    g.add(Agent("filter",    filter))
    g.add(Agent("sentiment", sentiment))
    g.add(Agent("topics",    topics))
    g.add(Agent("merge",     merge))
    g.add(Agent("report",    reporter))

    # ── edges ───────────────────────────────────────────────────────────────
    g.connect("collector", "filter")

    # fan-out
    g.connect("filter", "sentiment")
    g.connect("filter", "topics")

    # fan-in
    g.connect("sentiment", "merge")
    g.connect("topics",    "merge")

    # final step
    g.connect("merge", "report")

    g.set_entrypoint("collector")
    return g
