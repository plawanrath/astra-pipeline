# agents/merge.py
"""
Idempotent merge node – waits until both sentiment_df & topics_df exist,
then merges them with filtered_posts → merged_df.
"""
import logging
from pathlib import Path
import pandas as pd

def run(state: dict) -> dict | None:
    # Wait until prerequisites are available
    if not all(k in state for k in ("filtered_posts", "sent5", "sent3", "topics")):
        return None

    base = pd.DataFrame(state["filtered_posts"])
    df = (
        base.merge(state["sent5"],  on="post_id", how="left")
            .merge(state["sent3"],  on="post_id", how="left")
            .merge(state["topics"], on="post_id", how="left")
    )

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/merged.csv", index=False)
    logging.info("Merge: %d rows", len(df))

    # keep everything and add merged_df so downstream nodes still see 'config'
    new_state = state.copy()
    new_state["merged_df"] = df
    return new_state
