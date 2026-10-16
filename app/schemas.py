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


class RetrievedContext(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    source_type: str
    text: str
    score: float
    chunk_index: int = 0
    page_number: int | None = None


class ResearchPerspective(BaseModel):
    key: Literal["executive", "evidence", "risks", "opportunities", "actions"]
    title: str
    summary: str
    confidence: float = Field(ge=0, le=1)
    sources: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    status: Literal["completed", "needs_clarification", "partial"]
    follow_up_question: str | None = None
    final_output: str
    extracted_texts: list[ExtractedContent] = Field(default_factory=list)
    retrieved_contexts: list[RetrievedContext] = Field(default_factory=list)
    perspectives: list[ResearchPerspective] = Field(default_factory=list)
    plan_trace: list[PlanStep] = Field(default_factory=list)
    cost_estimate: CostEstimate = Field(default_factory=CostEstimate)


class RagDocument(BaseModel):
    document_id: str
    filename: str
    source_type: str
    chunk_count: int = 0
    token_count: int = 0


class RagIngestResponse(BaseModel):
    status: Literal["completed", "partial"]
    documents: list[RagDocument] = Field(default_factory=list)
    chunks_indexed: int = 0
    extracted_texts: list[ExtractedContent] = Field(default_factory=list)
    plan_trace: list[PlanStep] = Field(default_factory=list)


class RagQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=12)


class RagQueryResponse(BaseModel):
    status: Literal["completed", "partial"]
    final_output: str
    retrieved_contexts: list[RetrievedContext] = Field(default_factory=list)
    plan_trace: list[PlanStep] = Field(default_factory=list)
    cost_estimate: CostEstimate = Field(default_factory=CostEstimate)


class AnalyticsColumn(BaseModel):
    name: str
    kind: Literal["date", "number", "category", "text"]
    non_empty: int = 0
    unique_values: int = 0


class AnalyticsPoint(BaseModel):
    label: str
    value: float


class AnalyticsCategory(BaseModel):
    label: str
    value: float


class AnalyticsInsight(BaseModel):
    finding: str
    impact: str
    action: str
    confidence: float
    source: str


class AnalyticsResponse(BaseModel):
    status: Literal["completed", "partial"]
    filename: str
    row_count: int
    columns: list[AnalyticsColumn] = Field(default_factory=list)
    date_column: str | None = None
    measure_column: str | None = None
    category_column: str | None = None
    kpis: dict[str, float] = Field(default_factory=dict)
    trend: list[AnalyticsPoint] = Field(default_factory=list)
    categories: list[AnalyticsCategory] = Field(default_factory=list)
    anomalies: list[AnalyticsPoint] = Field(default_factory=list)
    distribution: list[AnalyticsPoint] = Field(default_factory=list)
    insights: list[AnalyticsInsight] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AuthUser(BaseModel):
    id: str
    name: str
    email: str


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str
