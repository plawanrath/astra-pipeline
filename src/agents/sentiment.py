# agents/sentiment.py
"""
Agent 1 – 5-point sentiment analysis
reads  context["filtered_posts"]   writes context["sentiment_df"]
"""
from __future__ import annotations
import json, logging, re, time
from pathlib import Path
import pandas as pd
from ..llm_abstraction import get_client

PROMPT_TMPL = Path("prompts/agent1.txt").read_text()
JSON_RE     = re.compile(r"\[.*\]|\{.*\}", re.S)     # first JSON array/object

def _safe_parse(text: str):
    """Return Python obj for first JSON blob in text, else raise."""
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
            logging.warning("Sentiment batch failed (%s); falling back per-item", e)
            for r in chunk:
                out.append({"post_id": r["post_id"], "score": 0})

    df = pd.DataFrame(out).rename(columns={"score": "sentiment_score"})
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/sentiment_scores.csv", index=False)
    logging.info("Sentiment done (%d posts, %.2fs)", len(df), time.perf_counter() - tic)
    context["sentiment_df"] = df
    return context
