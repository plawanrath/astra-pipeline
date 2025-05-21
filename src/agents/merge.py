# agents/merge.py
"""
Idempotent merge node – waits until both sentiment_df & topics_df exist,
then merges them with filtered_posts → merged_df.
"""
import logging
from pathlib import Path
import pandas as pd

def run(context: dict) -> dict:
    # already merged?
    if "merged_df" in context:
        return context

    ready = all(k in context for k in ("filtered_posts",
                                       "sentiment_df",
                                       "topics_df"))
    if not ready:
        return context                     # second branch not finished yet

    base = pd.DataFrame(context["filtered_posts"])
    df   = base.merge(context["sentiment_df"], on="post_id", how="left")\
               .merge(context["topics_df"],    on="post_id", how="left")

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/merged.csv", index=False)
    logging.info("Merge: created merged_df with %d rows", len(df))

    context["merged_df"] = df
    return context
