# -*- coding: utf-8 -*-
"""Unified LLM generator interface for the causal-attribution experiments.

Two backends:
  * OllamaGenerator  -- local models via the Ollama HTTP API (gemma3, qwen3,
    gpt-oss, deepseek-r1, ...). Handles "thinking" models by stripping reasoning.
  * OpenAIGenerator  -- OpenAI chat models, optionally returning token logprobs
    (used for the gray-box confidence signal).

All generators expose .generate(prompt) -> GenResult(text, logprob, meta) so the
RAG pipeline and the counterfactual attribution code are backend-agnostic.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# load OPENAI_API_KEY from .env if present
_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_ANALYSIS_RE = re.compile(r"<\|channel\|>analysis.*?<\|channel\|>final", re.S)


@dataclass
class GenResult:
    text: str
    logprob: float | None = None     # sum token logprob of the generation (if available)
    n_tokens: int | None = None
    meta: dict = field(default_factory=dict)


def _strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _ANALYSIS_RE.sub("", text)
    return text.strip()


class OllamaGenerator:
    def __init__(self, model: str, base_url: str = OLLAMA_URL, think: bool = False,
                 timeout: float = 180.0, num_ctx: int = 4096):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.think = think          # ask thinking-capable models to skip reasoning
        self.timeout = timeout
        self.num_ctx = num_ctx      # cap context window: default 128k KV cache is huge/slow
        self.name = f"ollama:{model}"

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> GenResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": self.think,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": self.num_ctx},
        }
        for attempt in range(3):
            try:
                r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
                r.raise_for_status()
                d = r.json()
                text = _strip_thinking(d.get("response", ""))
                return GenResult(text=text, n_tokens=d.get("eval_count"),
                                 meta={"think": d.get("thinking", "")[:200] if d.get("thinking") else ""})
            except Exception as e:
                if attempt == 2:
                    return GenResult(text="", meta={"error": str(e)[:160]})
                time.sleep(1.5 * (attempt + 1))


class OpenAIGenerator:
    def __init__(self, model: str = "gpt-4.1-mini", logprobs: bool = True, timeout: float = 90.0):
        from openai import OpenAI
        self.client = OpenAI(timeout=timeout)
        self.model = model
        self.want_logprobs = logprobs
        self.name = f"openai:{model}"
        # reasoning models (o-series, gpt-5*) reject temperature/logprobs
        self._reasoning = bool(re.match(r"^(o\d|gpt-5)", model))

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> GenResult:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._reasoning:
            kwargs["max_completion_tokens"] = max(max_tokens, 2048)
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
            if self.want_logprobs:
                kwargs["logprobs"] = True
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(**kwargs)
                ch = r.choices[0]
                text = (ch.message.content or "").strip()
                lp = None
                if ch.logprobs and ch.logprobs.content:
                    lp = sum(t.logprob for t in ch.logprobs.content)
                return GenResult(text=_strip_thinking(text), logprob=lp,
                                 n_tokens=r.usage.completion_tokens if r.usage else None)
            except Exception as e:
                if attempt == 3:
                    return GenResult(text="", meta={"error": str(e)[:160]})
                time.sleep(2.0 * (attempt + 1))


def make_generator(spec: str):
    """spec: 'ollama:gemma3:4b', 'ollama:qwen3:30b-a3b:think', 'openai:gpt-4.1-mini'."""
    if spec.startswith("openai:"):
        return OpenAIGenerator(model=spec.split(":", 1)[1])
    if spec.startswith("ollama:"):
        rest = spec.split(":", 1)[1]
        think = rest.endswith(":think")
        if think:
            rest = rest[: -len(":think")]
        return OllamaGenerator(model=rest, think=think)
    raise ValueError(f"unknown generator spec: {spec}")


if __name__ == "__main__":
    import sys
    g = make_generator(sys.argv[1] if len(sys.argv) > 1 else "ollama:gemma3:4b")
    print(g.name, "->", g.generate("Answer with one word only. Q: capital of Japan?", max_tokens=10))
