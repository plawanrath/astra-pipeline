# src/cli.py
import os, click, json, logging
from graph import build_graph

@click.command()
@click.option("--query", required=True)
@click.option("--model", default="gpt-4o-mini")
@click.option("--location")             
@click.option("--age-range")
@click.option("--sentiment-threshold")
def run(**kwargs):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Build a context dict passed through every agent
    context = {
        "config": {
            **kwargs,
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "TWITTER_BEARER": os.getenv("TWITTER_BEARER"),
        }
    }

    graph = build_graph()
    final_ctx = graph.invoke(context)      # synchronous; returns last node ctx
    md_path = "reports/report.md"
    logging.info("Pipeline complete → %s", md_path)

if __name__ == "__main__":
    run()   # pylint: disable=no-value-for-parameter
