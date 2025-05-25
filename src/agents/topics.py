# agents/topics.py
"""
Agent 2 – batched topic-extraction
  • Reads  context["filtered_posts"]
  • Writes context["topics_df"]

Batch size is taken from cfg["batch_size"] (default 1 = legacy behaviour).
Prompt template must contain the literal token {{posts_json}}.
"""

from __future__ import annotations
import json, logging, time
from pathlib import Path
import pandas as pd
from ..llm_abstraction import get_client
from ..utils.json_utils import safe_extract

PROMPT_TMPL = Path("prompts/agent2.txt").read_text()

def _norm(rec, post_id):
    topics = rec.get("topics") if isinstance(rec, dict) else []
    if not isinstance(topics, list):
        topics = []
    return {"post_id": post_id, "topics": topics}

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
            parsed = safe_extract(llm.generate(prompt, temperature=0.0))
            out.extend(_norm(r, c["post_id"]) for r, c in zip(parsed, chunk))
        except Exception as e:
            logging.warning("Topics batch error (%s); default []", e)
            out.extend({"post_id": c["post_id"], "topics": []} for c in chunk)

    df = pd.DataFrame(out)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/topics.csv", index=False)
    logging.info("Topics done (%d rows, %.2fs)", len(df), time.perf_counter() - tic)

    new_state = state.copy()
    new_state["topics"] = df
    return new_state
