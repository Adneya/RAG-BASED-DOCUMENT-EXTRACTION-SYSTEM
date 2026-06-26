from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    tool: str
    action: str
    status: Literal["pending", "running", "completed", "skipped", "failed", "partial"]
    detail: str = ""


class ExtractedContent(BaseModel):
    filename: str
    source_type: Literal["text", "image", "pdf", "audio", "youtube", "unknown"]
    text: str = ""
    ocr_confidence: float | None = Field(
        default=None, description="OCR confidence from 0 to 1 when available."
    )
    duration_seconds: float | None = None
    warnings: list[str] = Field(default_factory=list)


class CostEstimate(BaseModel):
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_usd: float | None = None
    note: str = ""


class AgentResponse(BaseModel):
    status: Literal["completed", "needs_clarification", "partial"]
    follow_up_question: str | None = None
    final_output: str
    extracted_texts: list[ExtractedContent] = Field(default_factory=list)
    plan_trace: list[PlanStep] = Field(default_factory=list)
    cost_estimate: CostEstimate = Field(default_factory=CostEstimate)
