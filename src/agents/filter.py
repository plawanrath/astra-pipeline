import logging, time
from pathlib import Path
import pandas as pd

def run(context: dict) -> dict:
    cfg  = context["config"]
    df   = pd.DataFrame(context["posts"])
    tic  = time.perf_counter()

    loc  = cfg.get("location")
    if loc and "location" in df.columns:
        df = df[df["location"].fillna("").str.contains(loc, case=False)]

    if cfg.get("age_range") and "age" in df.columns:
        lo, hi = map(int, cfg["age_range"].split("-"))
        df = df[(df["age"] >= lo) & (df["age"] <= hi)]

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/filtered_posts.csv", index=False)
    logging.info("Filter kept %d rows (%.2fs)", len(df), time.perf_counter() - tic)
    context["filtered_posts"] = df.to_dict("records")
    return context
