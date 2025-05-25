# agents/reporter.py
"""
Final contextual Markdown report – runs after merge.
"""
# reporter.py – adds macro P/R/F1 comparison
from __future__ import annotations
import json, logging, time
from collections import Counter
from pathlib import Path
import pandas as pd
from ..llm_abstraction import get_client

PROMPT_TMPL = Path("prompts/agent3.txt").read_text()

def _macro(y_true: pd.Series, y_pred: pd.Series):
    lbls = ["positive", "neutral", "negative"]
    prec = rec = f1 = 0
    for lab in lbls:
        tp = ((y_true == lab) & (y_pred == lab)).sum()
        fp = ((y_true != lab) & (y_pred == lab)).sum()
        fn = ((y_true == lab) & (y_pred != lab)).sum()
        p  = tp / (tp + fp) if tp + fp else 0
        r  = tp / (tp + fn) if tp + fn else 0
        f  = 2*p*r/(p+r) if p + r else 0
        prec += p; rec += r; f1 += f
    n = len(lbls)
    return {"precision": round(prec/n,3),
            "recall":    round(rec/n,3),
            "f1":        round(f1/n,3)}

def run(state: dict) -> dict | None:
    if "merged_df" not in state:
        return None                        # ← no write, avoids root collision ✅

    df  = state["merged_df"]
    cfg = state["config"]
    llm = get_client(cfg["model"])

    gt      = df["label"].str.lower()
    mapped5 = df["score5"].map({-2:"negative",-1:"negative",0:"neutral",1:"positive",2:"positive"})
    pred3   = df["score3"]

    stats = {
        "macro_5pt": _macro(gt, mapped5),
        "macro_3pt": _macro(gt, pred3),
    }

    top_topics = Counter(t for lst in df["topics"] for t in lst or []).most_common(10)

    prompt = PROMPT_TMPL.format(
        total_posts=len(df),
        sentiment_json=json.dumps(stats),
        top_topics_json=json.dumps(top_topics),
        demo_json="{}",
        threshold_note="n/a"
    )

    tic = time.perf_counter()
    md  = llm.generate(prompt, temperature=0.0)
    Path("reports").mkdir(exist_ok=True)
    report_path = "reports/report.md"
    Path(report_path).write_text(md)
    logging.info("Report generated (%.2fs)", time.perf_counter() - tic)

    return {"report_path": report_path}