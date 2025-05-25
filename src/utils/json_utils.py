"""
Robust helper for pulling the first JSON object/array out of an LLM reply.
"""

from __future__ import annotations
import json, re

# greedy capture of the first {...} or [...] block, DOTALL for newlines
_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.S)

def safe_extract(text: str):
    """
    Return a Python object from the first JSON blob found in `text`.

    Strategy:
    1. If the trimmed text itself starts with '{' or '[', try json.loads() on
       the *entire* string (fast path).
    2. Otherwise, search for the first JSON object/array using a greedy regex,
       then json.loads() that substring.
    Raises ValueError if parsing fails.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty LLM response")

    if stripped[0] in "{[":
        try:
            return json.loads(stripped)
        except Exception:
            # fall back to regex in case the LLM added trailing notes
            pass

    m = _JSON_RE.search(stripped)
    if not m:
        raise ValueError("No JSON found")
    return json.loads(m.group())
