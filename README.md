# astra-pipeline

## Prereqs
- `.env` file in project root with `OPENAI_API_KEY` for ChatGPT
- If using model from HuggingFace then `HF_MODEL` in `.env`

## Datasets for report generation

All raw files sit in ./data/ once downloaded
```
data/
├── Tweets.csv                 # airline
├── reddit_50k.jsonl           # reddit
└── geocov19_100k.jsonl        # geocov19
```

**US-Airline Sentiment (smoke test)**
```
# one-time download (needs Kaggle API token in ~/.kaggle/kaggle.json)
kaggle datasets download -d crowdflower/twitter-airline-sentiment \
  -f Tweets.csv -p data/ --unzip --quiet

# run ASTRA:
docker run --rm -v $PWD/data:/app/data -v $PWD/reports:/app/reports astra \
  --file-path data/Tweets.csv --dataset-type airline --model gpt-4.1
```

- Intermediate data will be generated in data/
- report will be found in reports/
```