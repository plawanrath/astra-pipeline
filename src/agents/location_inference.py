from __future__ import annotations
import json, logging, time
from pathlib import Path
import pandas as pd
from ..llm_abstraction import get_client
from ..utils.json_utils import safe_extract

PROMPT_TMPL = Path("prompts/agent_loc.txt").read_text()

def _ensure_list(parsed):
    """
    The LLM *should* return a JSON array, but sometimes it outputs
    multiple standalone objects or a single object.
    • If we got a dict  → wrap in list
    • If we got a str   → try to split on '}{' pattern
    • Else return as-is
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, str):
        # attempt to split concatenated objects like {…}{…}{…}
        objs = []
        buff = ""
        depth = 0
        for ch in parsed:
            if ch == "{":
                depth += 1
            if depth:
                buff += ch
            if ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        objs.append(json.loads(buff))
                    except Exception:
                        pass
                    buff = ""
        if objs:
            return objs
    raise ValueError("Could not coerce LLM output to list of dicts")

def run(state: dict) -> dict:
    rows  = state["posts"]
    cfg   = state["config"]
    llm   = get_client(cfg["model"])
    batch = int(cfg.get("batch_size", 1)) or 1

    enriched, tic = [], time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = [
            {
                "post_id": r["post_id"],
                "location_raw": r.get("location", ""),
                "text": r.get("content", "")[:120],
            }
            for r in rows[i : i + batch]
        ]
        prompt = PROMPT_TMPL.replace("{{posts_json}}", json.dumps(chunk))

        try:
            raw    = llm.generate(prompt, temperature=0.3)
            parsed = _ensure_list(safe_extract(raw))
            loc_map = {p["post_id"]: p["location_inferred"] for p in parsed
                       if isinstance(p, dict)}
        except Exception as e:
            logging.warning("Location batch failed (%s); default 'Unknown'", e)
            loc_map = {r["post_id"]: "Unknown" for r in chunk}

        for r in chunk:
            r_full = next(p for p in rows if p["post_id"] == r["post_id"])
            r_full["location_inferred"] = loc_map.get(r["post_id"], "Unknown")
            enriched.append(r_full)

    logging.info("Location inference done (%d rows, %.2fs)",
                 len(enriched), time.perf_counter() - tic)

    new_state = state.copy()
    new_state["posts"] = enriched
    return new_state