from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SupportResponse(BaseModel):
    """Structured customer-support reply produced by an LLM or extractive backend."""

    answer: str = Field(..., description="Customer-facing answer or refusal message.")
    can_answer: bool = Field(..., description="True when the question is in-scope and grounded.")
    sources: List[str] = Field(default_factory=list, description="Knowledge-base filenames used.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Required when can_answer is False; unused otherwise.",
    )

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_answer(cls, value):
        if value is None:
            return ""
        return str(value)


class GuardrailResult(BaseModel):
    """Outcome of schema, grounding, and out-of-domain checks."""

    passed: bool
    status: str = Field(..., description="pass | refuse | schema_error")
    grounding_ratio: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
    response: Optional[SupportResponse] = None
