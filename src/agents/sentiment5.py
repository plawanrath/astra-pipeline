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

def _run_hf(rows, llm):
    scored = []
    for r in rows:
        try:
            resp  = json.loads(llm.generate(r["content"]))   # {"score": int}
            score = int(resp["score"])
        except Exception as e:
            logging.warning("HF Sent-5 fail %s – %s", r["post_id"], e)
            score = 0
        scored.append({"post_id": r["post_id"], "score5": score})
    return scored

def _run_openai(rows, llm, batch):
    out = []
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
            parsed = safe_extract(llm.generate(prompt, temperature=0.0))
            out.extend(parsed)
        except Exception as e:
            logging.warning("OpenAI Sent-5 batch fail (%s)", e)
            out.extend({"post_id": r["post_id"], "score": 0} for r in chunk)
    # rename "score"→"score5"
    return [{"post_id": d["post_id"], "score5": int(d["score"])} for d in out]

def run(state: dict) -> dict:
    rows  = state["filtered_posts"]
    cfg   = state["config"]
    llm   = get_client(cfg["model"])
    tic   = time.perf_counter()

    if getattr(llm, "name", "") == "hf-sentiment":
        out = _run_hf(rows, llm)
    else:
        batch = int(cfg.get("batch_size", 1)) or 1
        out   = _run_openai(rows, llm, batch)

    df = pd.DataFrame(out)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/sentiment5.csv", index=False)
    logging.info("Sentiment-5 done (%d rows, %.2fs)", len(df), time.perf_counter()-tic)

    new_state = state.copy()
    new_state["sent5"] = df
    return new_state
