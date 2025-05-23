import os, click, logging, json
from dotenv import load_dotenv
load_dotenv()    

from .graph import build_graph

@click.command()
@click.option("--query", required=False, help="Free-text query (ignored if --file-path).")
@click.option("--file-path", type=click.Path(), help="Local CSV / JSONL to ingest instead of scraping.")
@click.option("--dataset-type", type=click.Choice(["airline", "reddit", "geocov19"]), help="Force a column mapper.")
@click.option("--model", default="gpt-4")
@click.option("--location")
@click.option("--age-range")
@click.option("--sentiment-threshold")
@click.option("--max-rows", default=None, type=int, help="Subset first N rows for quick runs.")
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Posts per LLM call for sentiment/topic agents.",
)
def run(**kwargs):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ctx = {"config": {**kwargs,
                      "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                      "TWITTER_BEARER": os.getenv("TWITTER_BEARER")}}
    graph = build_graph()
    graph.invoke(ctx)

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
