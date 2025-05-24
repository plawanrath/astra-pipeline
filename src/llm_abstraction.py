"""
Unified LLM interface for ASTRA
– handles both openai-python <1.0 and ≥1.0
– fall-back to Hugging-Face local / remote models
"""

from __future__ import annotations
import os, time, logging, json, random
from abc import ABC, abstractmethod
from typing import Dict
from openai import RateLimitError
from httpx import HTTPStatusError  
from packaging import version
import time, re

BUCKET_RE = re.compile(r"(requests|tokens) per (minute|day)", re.I)

def _retry_after_to_seconds(raw: str | None) -> float | None:
    if not raw:
        return None
    raw = raw.strip().lower()
    try:
        if raw.endswith("ms"):
            return float(raw[:-2]) / 1000.0
        if raw.endswith("s"):
            return float(raw[:-1])
        return float(raw)
    except ValueError:
        return None

def _classify_rate_limit(resp) -> tuple[str, float | None]:
    if resp is None:
        return "unknown", None
    h = {k.lower(): v for k, v in resp.headers.items()}
    now = time.time()
    if "x-ratelimit-reset-requests" in h:
        reset = float(h["x-ratelimit-reset-requests"]) - now
        return ("RPM" if reset < 120 else "RPD", max(reset, 0.0))
    if "x-ratelimit-reset-tokens" in h:
        reset = float(h["x-ratelimit-reset-tokens"]) - now
        return ("TPM" if reset < 120 else "TPD", max(reset, 0.0))
    try:
        msg = resp.json().get("error", {}).get("message", "")
    except Exception:
        msg = resp.text
    m = BUCKET_RE.search(msg)
    if m:
        bucket = "R" if m.group(1).lower() == "requests" else "T"
        period = "M" if m.group(2).lower().startswith("minute") else "D"
        return f"{bucket}{period}P", None
    return "unknown", None
# ╰──────────────────────────────────────────────────────────────╯

class BaseLLM(ABC):
    name: str
    @abstractmethod
    def generate(self, prompt: str, **kw) -> str: ...
    def _log(self, start: float, prompt: str, resp: str):
        logging.info(
            "%s %.2fs | prompt %d chars | resp %d chars",
            self.name,
            time.perf_counter() - start,
            len(prompt),
            len(resp),
        )

# ── OpenAI client ──────────────────────────────────────────────
class OpenAIClient(BaseLLM):
    name = "openai"
    def __init__(self, model: str):
        import openai
        self.model  = model
        self.legacy = version.parse(openai.__version__) < version.parse("1.0.0")
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.cli = openai if self.legacy else openai.OpenAI(api_key=key)
        if self.legacy:
            openai.api_key = key

    def generate(self, prompt: str, temperature: float = 0.2, **kw) -> str:
        max_tries, base = 5, 2.0
        for attempt in range(1, max_tries + 1):
            start = time.perf_counter()
            try:
                if self.legacy:
                    r = self.cli.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        **kw,
                    )
                    txt = r.choices[0].message.content.strip()
                else:
                    r = self.cli.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        **kw,
                    )
                    txt = r.choices[0].message.content.strip()
                self._log(start, prompt, txt)
                return txt

            except (RateLimitError, HTTPStatusError) as e:
                resp = getattr(e, "response", None)
                bucket, reset = _classify_rate_limit(resp)
                hdr_secs = _retry_after_to_seconds(
                    resp.headers.get("retry-after") if resp else None
                )
                wait = (
                    hdr_secs
                    if hdr_secs is not None
                    else reset
                    if reset is not None
                    else base * (2 ** (attempt - 1))
                )
                wait = max(wait, 5.0) + random.uniform(0, 1)
                logging.warning(
                    "Rate-limit (%s) attempt %d/%d → sleep %.2fs",
                    bucket,
                    attempt,
                    max_tries,
                    wait,
                )
                time.sleep(wait)
                continue

            except Exception as e:
                if attempt == max_tries:
                    raise
                wait = base * (2 ** (attempt - 1))
                logging.warning("OpenAI error %s (attempt %d) → %.1fs", e, attempt, wait)
                time.sleep(wait)

        fb = os.getenv("ASTRA_FALLBACK_MODEL")
        if fb and fb.lower() != self.model.lower():
            logging.error("Switching to fallback model %s", fb)
            return get_client(fb).generate(prompt, temperature=temperature, **kw)
        raise RuntimeError("OpenAI retries exhausted")

# --------------------------------------------------------------------------- #
# Hugging-Face transformers / local model
# --------------------------------------------------------------------------- #
class HFClient(BaseLLM):
    name = "hf"

    def __init__(self, model: str):
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            pipeline,
        )

        self.model_name = (
            model or os.getenv("HF_MODEL", "meta-llama/Llama-3-8b-instruct")
        )
        logging.info("Loading HF model %s  (may take a few minutes)", self.model_name)
        tok = AutoTokenizer.from_pretrained(self.model_name)
        mod = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto"
        )
        self.pipe = pipeline(
            "text-generation", model=mod, tokenizer=tok, max_new_tokens=512
        )

    def generate(self, prompt: str, temperature: float = 0.2, **kw) -> str:
        start = time.perf_counter()
        text = self.pipe(prompt, temperature=temperature, **kw)[0]["generated_text"]
        self._log_usage(start, prompt, text)
        return text


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
CLIENTS = {
    "gpt-4o":      OpenAIClient,
    "gpt-4o-mini":    OpenAIClient,
    "local":      HFClient,
    # add more aliases -> class
}

_instances: Dict[str, BaseLLM] = {}


def get_client(alias: str) -> BaseLLM:
    alias = alias.lower()
    if alias not in CLIENTS:
        raise ValueError(f"Unknown LLM alias '{alias}'")
    if alias not in _instances:
        _instances[alias] = CLIENTS[alias](alias)
    return _instances[alias]


