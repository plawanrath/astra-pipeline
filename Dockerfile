###############################################################################
# ASTRA  –  Agentic Sentiment-Topic Reporting Architecture
# Updated: adds Kaggle CLI, Hugging-Face ‚datasets‘ + Twarc for tweet hydration.
###############################################################################

FROM python:3.10-slim AS astra

# ── runtime env ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── OS-level deps ─────────────────────────────────────────────────────────────
# ▸ build-essential : builds wheels for ujson / orjson if they appear
# ▸ git, curl       : used by pip vcs installs & data fetch helpers
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── create unprivileged user ─────────────────────────────────────────────────
ARG UID=1001
RUN adduser --disabled-password --gecos "" --uid ${UID} astra
USER astra
WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────────────────
COPY --chown=astra:astra requirements.txt .
COPY --chown=astra:astra .env .env
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    # extra libs for offline datasets / hydration
    pip install \
        kaggle              \
        datasets[community] \
        twarc               \
        textblob            # needed by some Kaggle CSVs

# ── project code & prompts ───────────────────────────────────────────────────
COPY --chown=astra:astra src/      ./src/
COPY --chown=astra:astra prompts/  ./prompts/
COPY --chown=astra:astra README.md .

# ── runtime data dirs (bind-mounted by docker run) ───────────────────────────
RUN mkdir -p /app/data /app/reports

# ── Kaggle CLI expects ~/.kaggle/kaggle.json (mount it at runtime) ───────────
ENV PATH="/home/astra/.local/bin:$PATH"

# ── default entrypoint ───────────────────────────────────────────────────────
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]

###############################################################################
# Build:   docker build -t astra .
# Run :   docker run --rm                       \
#           -v $PWD/data:/app/data             \
#           -v $PWD/reports:/app/reports       \
#           -v $HOME/.kaggle:/home/astra/.kaggle:ro \
#           astra --file-path data/Tweets.csv --dataset-type airline
###############################################################################
