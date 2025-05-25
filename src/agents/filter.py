import logging, time
from pathlib import Path
import pandas as pd

def run(state: dict) -> dict:
    cfg  = state["config"]
    df   = pd.DataFrame(state["posts"])
    tic  = time.perf_counter()

    loc_filter = cfg.get("location")
    if loc_filter:
        df = df[df["location_inferred"]
                  .fillna("")
                  .str.contains(loc_filter, case=False)]

    if cfg.get("age_range") and "age" in df.columns:
        lo, hi = map(int, cfg["age_range"].split("-"))
        df = df[(df["age"] >= lo) & (df["age"] <= hi)]

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/filtered_posts.csv", index=False)
    logging.info("Filter kept %d rows (%.2fs)", len(df), time.perf_counter() - tic)

    new_state = state.copy()
    new_state["filtered_posts"] = df.to_dict("records")
    return new_state
