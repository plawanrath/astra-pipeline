# agents/filter.py
"""
Post-Collection Filtering Module
-- filters by location / age range and persists filtered_posts.
"""
import logging, time
from pathlib import Path
import pandas as pd

def run(context: dict) -> dict:
    cfg   = context["config"]
    rows  = context["posts"]                # list[dict] from collector
    tic   = time.perf_counter()

    df = pd.DataFrame(rows)

    # ── apply CLI-driven filters ────────────────────────────────────────────
    if cfg.get("location"):
        df = df[df["location"] == cfg["location"]]

    if cfg.get("age_range"):                # e.g. "18-25"
        try:
            lo, hi = map(int, cfg["age_range"].split('-'))
            df = df[(df["age"] >= lo) & (df["age"] <= hi)]
        except Exception:
            logging.warning("Bad --age-range format, skipping age filter")

    # ── save & log ──────────────────────────────────────────────────────────
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/filtered_posts.csv", index=False)

    logging.info("Filter: %d → %d rows (%.2fs)",
                 len(rows), len(df), time.perf_counter() - tic)

    context["filtered_posts"] = df.to_dict("records")
    return context
