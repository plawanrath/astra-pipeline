# agents/reporter.py
"""
Final contextual Markdown report – runs after merge.
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
from collections import Counter
import pandas as pd
from llm_abstraction import get_client

PROMPT_TMPL = Path("prompts/agent3.txt").read_text()

def run(context: dict) -> dict:
    if "merged_df" not in context:
        logging.warning("Reporter invoked before merge complete – skipping")
        return context

    cfg  = context["config"]
    df   = context["merged_df"]
    llm  = get_client(cfg["model"])

    # ── stats ───────────────────────────────────────────────────────────────
    sent_dist = df["sentiment_score"].value_counts().to_dict()

    all_topics = []
    for t in df["topics"]:
        if isinstance(t, list):
            all_topics.extend(t)
        elif isinstance(t, str):
            try: all_topics.extend(json.loads(t))
            except Exception: pass
    top_topics = Counter(all_topics).most_common(10)

    by_loc = {}
    if "location" in df.columns:
        by_loc = df.groupby("location")["sentiment_score"].mean()\
                   .round(2).to_dict()

    # ── prompt & generate ───────────────────────────────────────────────────
    prompt = PROMPT_TMPL.format(
        total_posts      = len(df),
        sentiment_json   = json.dumps(sent_dist),
        top_topics_json  = json.dumps(top_topics),
        demo_json        = json.dumps(by_loc),
        threshold_note   = cfg.get("sentiment_threshold", "N/A")
    )

    tic = time.perf_counter()
    md  = llm.generate(prompt, temperature=0.0)
    logging.info("Reporter LLM %.2fs", time.perf_counter() - tic)

    Path("reports").mkdir(exist_ok=True)
    out = "reports/report.md"
    Path(out).write_text(md)
    logging.info("Saved report → %s", out)

    context["report_path"] = out
    return context
