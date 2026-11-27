"""Dual-mode LLM client with zero external dependencies.

Supports:
1. Live Mode (`live` or `auto` when API key is set): Calls any OpenAI-compatible
   `/chat/completions` endpoint (OpenAI, DeepSeek, Qwen, Gemini OpenAI-compat, vLLM)
   using Python's standard `urllib.request`—so bioinformatics Conda environments
   do NOT need extra `openai` or `langchain` packages installed.
2. Offline / Deterministic Trace Mode (`offline` or fallback in `auto` when no API
   key is configured): Synthesizes structured, tool-grounded reasoning directly from
   actual computed tool outputs so demos to advisors run with zero failure risk.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class LLMCallRecord:
    """Audit log entry for every Agent LLM invocation."""

    agent_name: str
    stage: str
    mode_used: str  # "live_api" or "offline_grounded"
    model_name: str
    prompt_preview: str
    response_text: str


class DualModeLLMClient:
    """OpenAI-compatible REST client with deterministic tool-grounded fallback."""

    def __init__(
        self,
        mode: str = "auto",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout_sec: int = 60,
    ) -> None:
        self.mode = mode.lower()
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "").strip()
        self.model = model or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.call_history: List[LLMCallRecord] = []

    @property
    def is_live_available(self) -> bool:
        if self.mode == "offline":
            return False
        if self.mode == "live" and not self.api_key:
            raise RuntimeError(
                "LLM mode is set to 'live', but no API key was provided via config or environment variable."
            )
        return bool(self.api_key)

    def complete(
        self,
        agent_name: str,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        offline_fallback_fn: Callable[[], str],
        expect_json: bool = False,
    ) -> str:
        """Execute an LLM completion or fall back cleanly to deterministic tool-grounded synthesis."""
        if self.is_live_available:
            try:
                text = self._call_openai_compatible(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    expect_json=expect_json,
                )
                self.call_history.append(
                    LLMCallRecord(
                        agent_name=agent_name,
                        stage=stage,
                        mode_used="live_api",
                        model_name=self.model,
                        prompt_preview=user_prompt[:300],
                        response_text=text,
                    )
                )
                return text
            except Exception as exc:
                if self.mode == "live":
                    raise
                fallback_note = f"[Auto-Fallback to Offline Grounded Mode due to API error: {exc}]\n"
                text = offline_fallback_fn()
                self.call_history.append(
                    LLMCallRecord(
                        agent_name=agent_name,
                        stage=stage,
                        mode_used="offline_grounded_after_error",
                        model_name="deterministic-synthesizer",
                        prompt_preview=user_prompt[:300],
                        response_text=fallback_note + text,
                    )
                )
                return text

        text = offline_fallback_fn()
        self.call_history.append(
            LLMCallRecord(
                agent_name=agent_name,
                stage=stage,
                mode_used="offline_grounded",
                model_name="deterministic-tool-grounded",
                prompt_preview=user_prompt[:300],
                response_text=text,
            )
        )
        return text

    def _call_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = False,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if expect_json:
            payload["response_format"] = {"type": "json_object"}

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
        parsed = json.loads(raw)
        return parsed["choices"][0]["message"]["content"].strip()
