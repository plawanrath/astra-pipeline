# agents/sentiment.py
"""
Agent 1 – 5-point sentiment analysis
reads  context["filtered_posts"]   writes context["sentiment_df"]
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
import pandas as pd
from ..llm_abstraction import get_client
from ..utils.json_utils import safe_extract

PROMPT_TMPL = Path("prompts/agent1.txt").read_text()

def run(state: dict) -> dict:
    rows  = state["filtered_posts"]
    cfg   = state["config"]
    batch = int(cfg.get("batch_size", 1)) or 1
    llm   = get_client(cfg["model"])

    out, tic = [], time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = [
            {
                "post_id": r["post_id"],
                "text":    r["content"],
                "location_raw": r.get("location", ""),
                "location_inferred": r.get("location_inferred", "")
            }
            for r in rows[i : i + batch]
        ]
        prompt = PROMPT_TMPL.replace("{{posts_json}}", json.dumps(chunk))
        try:
            parsed = safe_extract(llm.generate(prompt, temperature=0.3))
            out.extend(parsed)
        except Exception as e:
            logging.warning("Sent-5 batch failed (%s); default score 0", e)
            out.extend({"post_id": r["post_id"], "score": 0} for r in chunk)

    df = pd.DataFrame(out).rename(columns={"score": "score5"}).astype({"score5": "int"})
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/sentiment5.csv", index=False)
    logging.info("Sentiment-5 done (%d rows, %.2fs)", len(df), time.perf_counter() - tic)

    new_state = state.copy()
    new_state["sent5"] = df
    return new_state
