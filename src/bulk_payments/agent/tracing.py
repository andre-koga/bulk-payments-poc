"""LangSmith tracing configuration and helpers."""
from __future__ import annotations

import os
from typing import Any


def configure_tracing(
    project: str = "bulk-payments-poc",
    tags: list[str] | None = None,
) -> None:
    """Idempotently configure LangSmith tracing via environment variables.

    Call once at process startup (CLI or API layer). Tags are appended to
    LANGCHAIN_TAGS so existing values in the environment are preserved.
    """
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", project)

    if tags:
        existing = os.environ.get("LANGCHAIN_TAGS", "")
        merged = ",".join(filter(None, [existing] + tags))
        os.environ["LANGCHAIN_TAGS"] = merged


def run_metadata(
    *,
    tenant_id: str,
    payment_id: str,
    rules_decision: str,
    model_name: str,
) -> dict[str, Any]:
    """Return a consistent metadata dict for LangSmith run tagging.

    Pass the result as `metadata=` to `@traceable` or `RunnableConfig`.
    Having a uniform key schema lets us filter/compare runs in LangSmith.
    """
    return {
        "tenant_id": tenant_id,
        "payment_id": payment_id,
        "rules_decision": rules_decision,
        "model_name": model_name,
    }


def is_tracing_enabled() -> bool:
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes")
