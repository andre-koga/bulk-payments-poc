"""LLM model factory.

Centralises provider selection so the rest of the agent code only passes a
model string (e.g. "gpt-4o-mini" or "claude-3-5-haiku-20241022") and gets
back a LangChain chat model configured with temperature=0 for deterministic
outputs. This makes LangSmith A/B comparison straightforward — swap the
model string, nothing else changes.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel


def get_chat_model(model: str, temperature: float = 0.0) -> BaseChatModel:
    """Return a LangChain chat model for the given model string.

    Supported prefixes:
    - gpt-*  / o*        → langchain_openai.ChatOpenAI
    - claude-*           → langchain_anthropic.ChatAnthropic

    Raises ImportError if the required provider package is not installed.
    Raises ValueError for unrecognised model strings.
    """
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for OpenAI models. "
                "Install with: pip install 'bulk-payments[agent]'"
            ) from exc
        return ChatOpenAI(model=model, temperature=temperature)

    if model.startswith("claude-"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required for Anthropic models. "
                "Install with: pip install 'bulk-payments[agent]'"
            ) from exc
        return ChatAnthropic(model=model, temperature=temperature)

    raise ValueError(
        f"Unrecognised model '{model}'. Expected a model starting with 'gpt-', "
        "'o1', 'o3', 'o4', or 'claude-'."
    )
