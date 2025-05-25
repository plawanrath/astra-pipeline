"""
ASTRA – slim LLM abstraction
Back-ends:
  • OpenAI (gpt-4o, gpt-3.5, etc.)
  • Hugging-Face local / HF Inference
  • Meta Llama API  (https://llama.developer.meta.com)
Handles RPM/TPM/RPD/TPD buckets, Retry-After ('4.8s', '120ms'), and fallback.
"""

from __future__ import annotations
import os, re, time, random, logging, json, requests
from abc import ABC, abstractmethod
from typing import Dict
from packaging import version
from openai import RateLimitError
from httpx import HTTPStatusError

# ── helper: retry-after parsing & bucket classifier ─────────────────────────
BUCKET_RE = re.compile(r"(requests|tokens) per (minute|day)", re.I)

def _retry_after_to_s(raw: str | None) -> float | None:
    if not raw: return None
    raw = raw.strip().lower()
    try:
        if raw.endswith("ms"): return float(raw[:-2]) / 1000
        if raw.endswith("s"):  return float(raw[:-1])
        return float(raw)
    except ValueError:
        return None

def _bucket(resp) -> tuple[str, float | None]:
    if resp is None: return "unknown", None
    h = {k.lower(): v for k, v in resp.headers.items()}
    now = time.time()
    if "x-ratelimit-reset-requests" in h:
        d = float(h["x-ratelimit-reset-requests"]) - now
        return ("RPM" if d < 120 else "RPD", max(d, 0))
    if "x-ratelimit-reset-tokens" in h:
        d = float(h["x-ratelimit-reset-tokens"]) - now
        return ("TPM" if d < 120 else "TPD", max(d, 0))
    try:
        msg = resp.json().get("error", {}).get("message", "")
    except Exception:
        msg = resp.text
    m = BUCKET_RE.search(msg)
    if m:
        bucket = "R" if m.group(1).lower() == "requests" else "T"
        period = "M" if m.group(2).startswith(("M","m")) else "D"
        return f"{bucket}{period}P", None
    return "unknown", None

# ── abstract base ───────────────────────────────────────────────────────────
class BaseLLM(ABC):
    name: str
    @abstractmethod
    def generate(self, prompt: str, **kw) -> str: ...
    def _log(self, start: float, prompt: str, resp: str):
        logging.info("%s %.2fs | prompt %d | resp %d",
                     self.name, time.perf_counter()-start, len(prompt), len(resp))

# ── OpenAI client ───────────────────────────────────────────────────────────
class OpenAIClient(BaseLLM):
    name = "openai"
    def __init__(self, model: str):
        import openai
        self.model   = model
        self.legacy  = version.parse(openai.__version__) < version.parse("1.0.0")
        key = os.getenv("OPENAI_API_KEY") or ""
        if not key: raise RuntimeError("OPENAI_API_KEY not set")
        if self.legacy:
            openai.api_key = key
            self.cli = openai
        else:
            self.cli = openai.OpenAI(api_key=key, max_retries=0)

    def _call(self, prompt, temperature, **kw):
        if self.legacy:
            r = self.cli.ChatCompletion.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature, **kw)
            return r.choices[0].message.content.strip()
        r = self.cli.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature, **kw)
        return r.choices[0].message.content.strip()

    def generate(self, prompt, temperature=0.2, **kw):
        max_try, base = 5, 2.0
        for attempt in range(1, max_try+1):
            start = time.perf_counter()
            try:
                txt = self._call(prompt, temperature, **kw)
                self._log(start, prompt, txt); return txt
            except (RateLimitError, HTTPStatusError) as e:
                resp = getattr(e, "response", None)
                bucket, reset = _bucket(resp)
                hdr = _retry_after_to_s(resp.headers.get("retry-after") if resp else None)
                wait = max(hdr or reset or base*(2**(attempt-1)), 5) + random.uniform(0,1)
                logging.warning("Rate-limit (%s) %d/%d → %.2fs", bucket, attempt, max_try, wait)
                time.sleep(wait)
            except Exception as e:
                if attempt == max_try: raise
                wait = base*(2**(attempt-1))
                logging.warning("OpenAI err %s (attempt %d) → %.1fs", e, attempt, wait)
                time.sleep(wait)
        fb = os.getenv("ASTRA_FALLBACK_MODEL")
        if fb and fb.lower()!=self.model.lower():
            logging.error("Fallback to %s", fb)
            return get_client(fb).generate(prompt, temperature=temperature, **kw)
        raise RuntimeError("OpenAI retries exhausted")

# ── HF local / hub ──────────────────────────────────────────────────────────
class HFClient(BaseLLM):
    name = "hf"
    def __init__(self, model: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        self.model = model or os.getenv("HF_MODEL","meta-llama/Llama-3-8b-instruct")
        logging.info("Loading HF %s …", self.model)
        tok = AutoTokenizer.from_pretrained(self.model)
        mod = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto")
        self.pipe = pipeline("text-generation", model=mod, tokenizer=tok, max_new_tokens=512)
    def generate(self, prompt, temperature=0.2, **kw):
        start=time.perf_counter()
        txt = self.pipe(prompt,temperature=temperature,**kw)[0]["generated_text"]
        self._log(start,prompt,txt); return txt

# ── Meta official Llama API ─────────────────────────────────────────────────
class LlamaMetaClient(BaseLLM):
    name = "llama-meta"
    ENDPOINT = "https://api.meta.ai/v1/chat/completions"
    def __init__(self, model:str):
        self.model = model or "Llama-3.3-70B-Instruct"
        self.key   = os.getenv("LLAMA_API_KEY") or ""
        if not self.key: raise RuntimeError("LLAMA_API_KEY not set")
    def generate(self, prompt, temperature=0.2, **kw):
        start=time.perf_counter()
        r = requests.post(
            self.ENDPOINT,
            headers={"Authorization":f"Bearer {self.key}",
                     "Content-Type":"application/json"},
            json={
                "model": self.model,
                "messages":[{"role":"user","content":prompt}],
                "temperature":temperature,
                "max_tokens":512,
            }, timeout=60
        ); r.raise_for_status()
        txt=r.json()["choices"][0]["message"]["content"].strip()
        self._log(start,prompt,txt); return txt

# ── factory + alias table ──────────────────────────────────────────────────
CLIENTS = {
    "gpt-4o":OpenAIClient,"gpt-4o-mini":OpenAIClient,"gpt-4":OpenAIClient, "gpt-4.1":OpenAIClient, "o4-mini": OpenAIClient,
    "gpt-3.5":OpenAIClient,"openai":OpenAIClient,
    "local":HFClient,"hf":HFClient,
    "llama-meta":LlamaMetaClient
}
ALIASES = {
    "mini":"gpt-4o-mini","4o":"gpt-4o","4.1":"gpt-4.1", "o4-mini": "o4-mini",
    "llama":"Llama-4-Maverick-17B-128E-Instruct-FP8",
}
_instances: Dict[str,BaseLLM]={}

def get_client(alias:str)->BaseLLM:
    key = ALIASES.get(alias.lower(), alias.lower())
    if key not in CLIENTS: raise ValueError(f"Unknown model alias '{alias}'")
    if key not in _instances: _instances[key]=CLIENTS[key](alias)
    return _instances[key]
