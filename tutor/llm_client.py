"""Thin Anthropic / OpenAI wrapper implementing the LLMClient protocol (§10).

Active client is selected by LLM_PROVIDER env var.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

from tutor.schemas import CompletionResult


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> CompletionResult: ...


class AnthropicClient:
    def __init__(self, model: str | None = None) -> None:
        import anthropic  # lazy import so the module loads without the key
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> CompletionResult:
        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user or "Respond now."}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = next(
            (block.text for block in response.content if hasattr(block, "text")), ""
        )
        if not text.strip():
            log.warning(
                "AnthropicClient got empty/whitespace response. "
                "stop_reason=%r content_blocks=%d output_tokens=%d",
                response.stop_reason,
                len(response.content),
                response.usage.output_tokens,
            )
        return CompletionResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model_name=self.model,
        )


class OpenAIClient:
    def __init__(self, model: str | None = None) -> None:
        import openai  # lazy import
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> CompletionResult:
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        choice = response.choices[0]
        usage = response.usage
        return CompletionResult(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            model_name=self.model,
        )


def build_llm_client(provider: str | None = None) -> LLMClient:
    """Factory: returns the correct client from LLM_PROVIDER env var."""
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "openai":
        return OpenAIClient()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Choose 'anthropic' or 'openai'.")


def build_judge_client(answerer_provider: str | None = None) -> LLMClient:
    """Return a judge client that is a DIFFERENT model from the answerer (§6.5).

    If answerer is Anthropic → judge is OpenAI, and vice versa.
    Falls back to AnthropicClient when openai package/key is unavailable (logs warning
    about same-model bias).
    """
    import logging
    log = logging.getLogger(__name__)

    answerer_provider = answerer_provider or os.getenv("LLM_PROVIDER", "anthropic")
    if answerer_provider == "anthropic":
        try:
            import openai  # noqa: F401
            if not os.getenv("OPENAI_API_KEY"):
                raise ImportError("OPENAI_API_KEY not set")
            return OpenAIClient()
        except (ImportError, ModuleNotFoundError):
            log.warning(
                "openai package not installed or OPENAI_API_KEY missing — "
                "falling back to AnthropicClient as judge (same-model bias applies)."
            )
            return AnthropicClient()
    return AnthropicClient()
