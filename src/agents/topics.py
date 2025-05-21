# agents/topics.py
"""
Agent 2 – topic extraction
reads  context["filtered_posts"]   writes context["topics_df"]
"""
import json, logging, time
from pathlib import Path
import pandas as pd
from llm_abstraction import get_client

PROMPT_TMPL = Path("prompts/agent2.txt").read_text()

def run(context: dict) -> dict:
    cfg   = context["config"]
    llm   = get_client(cfg["model"])
    rows  = context["filtered_posts"]

    tic   = time.perf_counter()
    out   = []

    for row in rows:
        prompt = PROMPT_TMPL.format(post_id=row["post_id"],
                                    content=row["content"])
        try:
            resp   = llm.generate(prompt)
            topics = json.loads(resp)["topics"]
            if not isinstance(topics, list):
                topics = []
        except Exception as e:
            logging.error("Topic JSON error %s – %s", row["post_id"], e)
            topics = []

        out.append({"post_id": row["post_id"], "topics": topics})

    df = pd.DataFrame(out)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/topics.csv", index=False)

    logging.info("Topics: %d rows (%.2fs)",
                 len(df), time.perf_counter() - tic)

    context["topics_df"] = df
    return context
