# agents/collector.py
import pandas as pd, logging, json, re
from pathlib import Path
from tweepy import Client as TweepyClient

def parse_query(query: str) -> dict:
    # extremely bare-bones example
    loc = re.search(r"in (\w+)$", query)
    return {
        "keywords": query.split(),
        "location": loc.group(1) if loc else None,
        "timespan": None        # extend as needed
    }

def run(context: dict) -> dict:
    cfg   = context["config"]
    query = cfg["query"]
    params = parse_query(query)

    # demo Twitter scrape via v2 API ― replace w/ snscrape for no-auth path
    twitter = TweepyClient(bearer_token=cfg["TWITTER_BEARER"])
    tweets  = twitter.search_recent_tweets(query=query, max_results=50).data or []

    rows = [{
        "post_id": t.id,
        "user_id": t.author_id,
        "timestamp": t.created_at.isoformat(),
        "content": t.text,
        "location": params["location"] or None,
        "age": None,            # unknown for Twitter
    } for t in tweets]

    Path("data").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("data/raw_posts.csv", index=False)
    logging.info("Collector: saved %d rows", len(rows))

    context["posts"] = rows           # pass DataFrame-like list forward
    return context                    # LangGraph node must return full ctx
