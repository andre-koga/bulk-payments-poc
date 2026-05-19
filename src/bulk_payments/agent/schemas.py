"""Pydantic schemas for AI agent structured output."""
from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class AgentAction(str, Enum):
    PROPOSE_MATCH = "propose_match"
    NO_MATCH = "no_match"
    NEED_MORE_INFO = "need_more_info"


class AgentResolution(BaseModel):
    """Structured output produced by the resolver agent for every payment.

    `reasoning` is the accountant-style narrative that explains *why* the
    agent reached its conclusion. It is stored in SQLite for offline eval and
    surfaced in the UI review screen. The full LLM message/tool trace lives in
    LangSmith; `langsmith_run_id` links the two.
    """

    action: AgentAction
    bill_ids: list[str] = Field(
        default_factory=list,
        description="Proposed bill IDs to allocate. Empty when action is no_match or need_more_info.",
    )
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    reasoning: str = Field(
        description="Concise accountant-style explanation for the decision.",
    )
    model_id: str
    langsmith_run_id: str | None = None

    @model_validator(mode="after")
    def _validate_bill_ids_present_for_match(self) -> "AgentResolution":
        if self.action == AgentAction.PROPOSE_MATCH and not self.bill_ids:
            raise ValueError("bill_ids must be non-empty when action is propose_match")
        return self
