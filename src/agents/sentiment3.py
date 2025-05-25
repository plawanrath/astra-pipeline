# sentiment3.py – 3-class agent
from __future__ import annotations
import json, logging, time
from pathlib import Path
import pandas as pd
from ..llm_abstraction import get_client
from ..utils.json_utils import safe_extract

PROMPT_TMPL = Path("prompts/agent1b.txt").read_text()
MAP_3 = {-1: "negative", 0: "neutral", 1: "positive"}

def _run_hf(rows, llm):
    out = []
    for r in rows:
        try:
            resp = json.loads(llm.generate(r["content"]))   # {"score": -1/0/1}
            label = MAP_3.get(resp.get("score"), "neutral")
        except Exception as e:
            logging.warning("HF Sent-3 fail %s – %s", r["post_id"], e)
            label = "neutral"
        out.append({"post_id": r["post_id"], "score3": label})
    return out

def _run_openai(rows, llm, batch):
    results = []
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
            results.extend(parsed)
        except Exception as e:
            logging.warning("OpenAI Sent-3 batch fail (%s)", e)
            results.extend({"post_id": r["post_id"], "label": "neutral"} for r in chunk)

    # normalise fieldname→score3
    return [{"post_id": d["post_id"], "score3": d.get("label", "neutral").lower()}
            for d in results]

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
    df.to_csv("data/sentiment3.csv", index=False)
    logging.info("Sentiment-3 done (%d rows, %.2fs)", len(df), time.perf_counter()-tic)

    new_state = state.copy()
    new_state["sent3"] = df
    return new_state
