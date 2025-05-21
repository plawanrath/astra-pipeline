"""
Unified LLM interface: OpenAI GPT-4, GPT-3.5, or a local HF/Llama-cpp model.

Add new back-ends by subclassing `BaseLLM` and registering in `CLIENTS`.
"""
from __future__ import annotations
import os, time, logging, json
from abc import ABC, abstractmethod
from typing import Dict

# ---------- abstract layer -------------------------------------------------- #

class BaseLLM(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...

    # Optional helper for usage stats
    def _log_usage(self, start: float, prompt: str, response: str):
        elapsed = time.perf_counter() - start
        logging.info("%s took %.2fs | prompt %d chars | resp %d chars",
                     self.name, elapsed, len(prompt), len(response))

# ---------- OpenAI client --------------------------------------------------- #

class OpenAIClient(BaseLLM):
    name = "openai"

    def __init__(self, model: str):
        import openai                     # lazy-import
        self.model     = model
        self.openai    = openai
        self.openai.api_key  = os.getenv("OPENAI_API_KEY")
        if not self.openai.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY env var")

    def generate(self, prompt: str, temperature: float = 0.2, **kw) -> str:
        start = time.perf_counter()
        resp  = self.openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role":"user","content":prompt}],
                    temperature=temperature,
                    **kw,
                )
        text  = resp.choices[0].message.content.strip()
        self._log_usage(start, prompt, text)
        return text

# ---------- Hugging-Face Transformers -------------------------------------- #

class HFClient(BaseLLM):
    """
    Loads a text-generation pipeline.  
    By default expects the model path in HF_MODEL or passes model name to HF hub.
    """
    name = "hf"

    def __init__(self, model: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        self.model_name = model or os.getenv("HF_MODEL", "meta-llama/Llama-3-8b-instruct")
        logging.info("Loading HF model %s … this may take a while", self.model_name)
        tok   = AutoTokenizer.from_pretrained(self.model_name)
        m     = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.pipe = pipeline("text-generation", model=m, tokenizer=tok, max_new_tokens=512)

    def generate(self, prompt: str, temperature: float = 0.2, **kw) -> str:
        start = time.perf_counter()
        out   = self.pipe(prompt, temperature=temperature)[0]["generated_text"]
        self._log_usage(start, prompt, out)
        return out

# ---------- factory --------------------------------------------------------- #

CLIENTS = {
    "gpt-4o":      OpenAIClient,
    "gpt-o4-mini-high":    OpenAIClient,
    "local":      HFClient,
    # add more aliases -> class
}

_instances: Dict[str, BaseLLM] = {}

def get_client(model_name: str) -> BaseLLM:
    """Return a singleton client for requested model/alias."""
    alias = model_name.lower()
    cls   = CLIENTS.get(alias)
    if not cls:
        raise ValueError(f"Unknown LLM alias '{model_name}'")
    # cache per alias to avoid re-loading heavyweight models
    if alias not in _instances:
        _instances[alias] = cls(model_name)
    return _instances[alias]
