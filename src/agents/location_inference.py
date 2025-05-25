from __future__ import annotations
import json, logging, time
from pathlib import Path
import pandas as pd
from ..llm_abstraction import get_client
from ..utils.json_utils import safe_extract

PROMPT_TMPL = Path("prompts/agent_loc.txt").read_text()

def run(state: dict) -> dict:
    rows  = state["posts"]            # raw collector output
    cfg   = state["config"]
    llm   = get_client(cfg["model"])
    batch = int(cfg.get("batch_size", 1)) or 1

    enriched, tic = [], time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        # build minimal JSON for LLM (post_id + hints)
        hint_list = [
            {
                "post_id": r["post_id"],
                "location_raw": r.get("location", ""),
                "text": r.get("content", "")[:120],  # snippet, keep prompt short
            }
            for r in chunk
        ]
        prompt = PROMPT_TMPL.replace("{{posts_json}}", json.dumps(hint_list))
        try:
            parsed = safe_extract(llm.generate(prompt, temperature=0.0))
            loc_map = {p["post_id"]: p["location_inferred"] for p in parsed}
        except Exception as e:
            logging.warning("Location batch failed (%s); default 'Unknown'", e)
            loc_map = {r["post_id"]: "Unknown" for r in chunk}

        for r in chunk:
            r["location_inferred"] = loc_map.get(r["post_id"], "Unknown")
            enriched.append(r)

    logging.info("Location inference done (%d rows, %.2fs)",
                 len(enriched), time.perf_counter() - tic)

    new_state = state.copy()
    new_state["posts"] = enriched          # replace with enriched list
    return new_state
