# sentiment3.py – 3-class agent
from __future__ import annotations
import json, logging, time
from pathlib import Path
import pandas as pd
from ..llm_abstraction import get_client
from ..utils.json_utils import safe_extract

PROMPT_TMPL = Path("prompts/agent1b.txt").read_text()
VALID = {"positive", "neutral", "negative"}

def _norm(label, post_id):
    label = (label or "neutral").lower()
    if label not in VALID:
        label = "neutral"
    return {"post_id": post_id, "score3": label}

def run(state: dict) -> dict:
    rows  = state["filtered_posts"]
    cfg   = state["config"]
    batch = int(cfg.get("batch_size", 1)) or 1
    llm   = get_client(cfg["model"])

    out, tic = [], time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk  = rows[i : i + batch]
        prompt = PROMPT_TMPL.replace("{{posts_json}}", json.dumps(chunk))
        try:
            parsed = safe_extract(llm.generate(prompt, temperature=0.3))
            out.extend(_norm(lab.get("label") if isinstance(lab, dict) else lab, c["post_id"])
                        for lab, c in zip(parsed, chunk))
        except Exception as e:
            logging.warning("Sent-3 batch failed (%s); default neutral", e)
            out.extend({"post_id": c["post_id"], "score3": "neutral"} for c in chunk)

    df = pd.DataFrame(out)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/sentiment3.csv", index=False)
    logging.info("Sentiment-3 done (%d rows, %.2fs)", len(df), time.perf_counter() - tic)

    new_state = state.copy()
    new_state["sent3"] = df
    return new_state
