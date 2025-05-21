# astra-pipeline

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
  --file-path data/Tweets.csv --dataset-type airline --model gpt-3.5
```

**Pushshift Reddit sample (medium)**
```
# pull 50 k comments with 🤗 Datasets (no creds needed)
python - <<'PY'
from datasets import load_dataset
ds = load_dataset('pushshift/reddit', 'comments', split='train[:50_000]')
ds.to_json('data/reddit_50k.jsonl')
PY

docker run --rm -v $PWD/data:/app/data -v $PWD/reports:/app/reports astra \
  --file-path data/reddit_50k.jsonl --dataset-type reddit \
  --model local --max-rows 20000     # trims run time if desired
```

**GeoCoV19 (large + geo demo)**
```
# download tweet-ID list (≈110 MB)
curl -L -o ids.txt \
  https://crisisnlp.qcri.org/data/GeoCOV19/IDs-Feb1_May1_2020.txt

# hydrate first 100 k IDs (needs Twitter Bearer token)
pip install twarc
export BEARER_TOKEN=xxxxxxxx
twarc2 hydrate ids.txt --limit 100000 > data/geocov19_100k.jsonl

docker run --rm -v $PWD/data:/app/data -v $PWD/reports:/app/reports astra \
  --file-path data/geocov19_100k.jsonl --dataset-type geocov19 \
  --model gpt-4 --location "US"
```