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

def _macro(y_true: pd.Series, y_pred: pd.Series) -> dict:
    lbls = ["positive", "neutral", "negative"]
    prec = rec = f1 = 0
    for lab in lbls:
        tp = ((y_true == lab) & (y_pred == lab)).sum()
        fp = ((y_true != lab) & (y_pred == lab)).sum()
        fn = ((y_true == lab) & (y_pred != lab)).sum()
        p  = tp / (tp + fp) if tp + fp else 0
        r  = tp / (tp + fn) if tp + fn else 0
        f  = 2*p*r/(p+r) if p+r else 0
        prec += p; rec += r; f1 += f
    n = len(lbls)
    return {"precision": round(prec/n,3),
            "recall":    round(rec/n,3),
            "f1":        round(f1/n,3)}

def run(state: dict) -> dict | None:
    if "merged_df" not in state:
        return None

    df  = state["merged_df"]
    cfg = state["config"]
    llm = get_client(cfg["model"])

    # ground truth & predictions
    gt      = df["label"].str.lower()
    mapped5 = df["score5"].map({-2:"negative",-1:"negative",0:"neutral",1:"positive",2:"positive"})
    pred3   = df["score3"]

    overall = {
        "macro_5pt": _macro(gt, mapped5),
        "macro_3pt": _macro(gt, pred3),
    }

    # ── per-location metrics ────────────────────────────────────────────────
    loc_rows = []
    for loc, grp in df.groupby("location_inferred"):
        gt_loc   = grp["label"].str.lower()
        m5_loc   = grp["score5"].map({-2:"negative",-1:"negative",0:"neutral",1:"positive",2:"positive"})
        p3_loc   = grp["score3"]
        loc_rows.append({
            "location": loc,
            **{f"{k}_5pt": v for k,v in _macro(gt_loc, m5_loc).items()},
            **{f"{k}_3pt": v for k,v in _macro(gt_loc, p3_loc).items()},
        })
    loc_df = pd.DataFrame(loc_rows).sort_values("location")

    # top topics for context
    def _iter_topics(series):
        for item in series:
            if isinstance(item, list):
                yield from item
            else:
                continue

    top_topics = Counter(_iter_topics(df["topics"])).most_common(10)

    # build Markdown comparison table
    table_md = loc_df.to_markdown(index=False, floatfmt=".3f")

    prompt = PROMPT_TMPL.format(
        total_posts=len(df),
        sentiment_json=json.dumps(overall),
        top_topics_json=json.dumps(top_topics),
        demo_json=table_md,           # embed the table
        threshold_note="n/a",
    )

    tic = time.perf_counter()
    md  = llm.generate(prompt, temperature=0.0)

    report_path = "reports/report.md"
    Path("reports").mkdir(exist_ok=True)
    Path(report_path).write_text(md)
    logging.info("Report generated (%.2fs)", time.perf_counter() - tic)

    new_state = state.copy()
    new_state["report_path"] = report_path
    return new_state
