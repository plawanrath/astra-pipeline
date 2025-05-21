# agents/collector.py  (now supports --file-path)
import json, csv, time, logging
from pathlib import Path
import pandas as pd

def _map_airline(df):
    return pd.DataFrame({
        "post_id"  : df["tweet_id"],
        "user_id"  : df.get("name"),          # may be NaN
        "timestamp": df["tweet_created"],
        "content"  : df["text"],
        "location" : df.get("tweet_location"),
        "label"    : df.get("airline_sentiment")
    })

def _map_reddit(jsl):
    rows = []
    for raw in jsl:
        rows.append({
            "post_id"  : raw["id"],
            "user_id"  : raw.get("author"),
            "timestamp": pd.to_datetime(raw["created_utc"], unit="s", utc=True),
            "content"  : raw["body"],
            "location" : raw.get("subreddit")
        })
    return pd.DataFrame(rows)

def _map_geocov(jsl):
    rows = []
    for raw in jsl:
        cc = (raw.get("place") or {}).get("country_code") or raw.get("country_code")
        rows.append({
            "post_id"  : raw["id"],
            "timestamp": raw["created_at"],
            "content"  : raw["text"],
            "location" : cc
        })
    return pd.DataFrame(rows)

MAPPERS = {"airline": _map_airline, "reddit": _map_reddit, "geocov19": _map_geocov}

def run(context: dict) -> dict:
    cfg = context["config"]
    fp  = cfg.get("file_path")
    tic = time.perf_counter()

    if not fp:
        raise RuntimeError("--file-path is required for offline mode")

    dtype = cfg.get("dataset_type") or Path(fp).stem.lower()
    if "airline" in dtype:
        df = _map_airline(pd.read_csv(fp))
    elif "reddit" in dtype:
        df_raw = [json.loads(l) for l in Path(fp).open()]
        df = _map_reddit(df_raw)
    elif "geocov" in dtype:
        df_raw = [json.loads(l) for l in Path(fp).open()]
        df = _map_geocov(df_raw)
    else:
        raise ValueError(f"Unrecognised dataset-type {dtype}")

    if cfg.get("max_rows"):
        df = df.head(cfg["max_rows"])

    df.to_csv("data/raw_posts.csv", index=False)
    logging.info("Collector loaded %d rows from %s in %.2fs", len(df), fp, time.perf_counter() - tic)
    context["posts"] = df.to_dict("records")
    return context
