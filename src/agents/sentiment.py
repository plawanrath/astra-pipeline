# agents/sentiment.py
"""
Agent 1 – 5-point sentiment analysis
reads  context["filtered_posts"]   writes context["sentiment_df"]
"""
import json, logging, time
from pathlib import Path
import pandas as pd
from llm_abstraction import get_client

PROMPT_TMPL = Path("prompts/agent1.txt").read_text()

def run(context: dict) -> dict:
    cfg   = context["config"]
    llm   = get_client(cfg["model"])
    rows  = context["filtered_posts"]

    tic   = time.perf_counter()
    scored = []

    for row in rows:
        prompt = PROMPT_TMPL.format(post_id=row["post_id"],
                                    content=row["content"])
        try:
            resp  = llm.generate(prompt)
            score = json.loads(resp)["score"]
        except Exception as e:
            logging.error("Sentiment JSON error %s – %s", row["post_id"], e)
            score = 0

        scored.append({"post_id": row["post_id"], "sentiment_score": score})

    df = pd.DataFrame(scored)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/sentiment_scores.csv", index=False)

    logging.info("Sentiment: %d rows (%.2fs)",
                 len(df), time.perf_counter() - tic)

    context["sentiment_df"] = df
    return context
