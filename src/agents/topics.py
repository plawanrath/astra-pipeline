# agents/topics.py
"""
Agent 2 – batched topic-extraction
  • Reads  context["filtered_posts"]
  • Writes context["topics_df"]

Batch size is taken from cfg["batch_size"] (default 1 = legacy behaviour).
Prompt template must contain the literal token {{posts_json}}.
"""

from __future__ import annotations
import json, logging, re, time
from pathlib import Path
import pandas as pd
from ..llm_abstraction import get_client

PROMPT_TMPL = Path("prompts/agent2.txt").read_text()
JSON_RE     = re.compile(r"\[.*\]|\{.*\}", re.S)     # first JSON obj / array

def _safe_parse(text: str):
    m = JSON_RE.search(text.strip())
    if not m:
        raise ValueError("No JSON found")
    return json.loads(m.group())

def run(context: dict) -> dict:
    cfg   = context["config"]
    llm   = get_client(cfg["model"])
    rows  = context["filtered_posts"]
    batch = int(cfg.get("batch_size", 1)) or 1

    out, tic = [], time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk   = rows[i : i + batch]
        prompt  = PROMPT_TMPL.replace("{{posts_json}}", json.dumps(chunk))
        try:
            resp   = llm.generate(prompt, temperature=0.0)
            parsed = _safe_parse(resp)
            out.extend(parsed)
        except Exception as e:
            logging.warning("Topic batch failed (%s). Filling blanks.", e)
            for r in chunk:
                out.append({"post_id": r["post_id"], "topics": []})

    df = pd.DataFrame(out)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/topics.csv", index=False)
    logging.info(
        "Topics done (%d posts, %.2fs)", len(df), time.perf_counter() - tic
    )
    context["topics_df"] = df
    return context
