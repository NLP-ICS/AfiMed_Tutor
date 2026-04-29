"""Pydantic models for all request/response contracts (§10)."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrieved guideline chunk with metadata."""
    chunk_id: str
    text: str
    source_doc: str
    section_title: str
    condition: Optional[str] = None
    page_number: Optional[int] = None
    score: float = 0.0


class CompletionResult(BaseModel):
    """Raw output from an LLM completion call."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str


class MCQOption(BaseModel):
    key: str   # e.g. "A", "B", "C", "D"
    text: str


class MCQItem(BaseModel):
    """One multiple-choice question from AfriMed-QA."""
    question_id: str
    question: str
    options: list[MCQOption]
    gold_answer: str         # option key, e.g. "B"
    gold_rationale: str
    specialty: str
    source: str = ""


class SAQItem(BaseModel):
    """One short-answer clinical scenario from AfriMed-QA."""
    case_id: str
    scenario: str
    expert_answer: str
    specialty: str


class RelatedQuestion(BaseModel):
    question_id: str
    question: str
    specialty: str
    retrieval_score: float


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: list[Chunk]
    related_questions: list[RelatedQuestion] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str


class QuizResponse(BaseModel):
    explanation: str
    is_correct: bool
    retrieved_chunks: list[Chunk]
    related_questions: list[RelatedQuestion] = Field(default_factory=list)
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str


class ExplainResponse(BaseModel):
    comparison: str
    retrieved_chunks: list[Chunk]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str


class JudgeScore(BaseModel):
    question_id: str
    groundedness: int          # 0-2
    groundedness_justification: str
    citation_accuracy: int     # 0-2
    citation_justification: str
    consistency: int           # 0-2
    consistency_justification: str
    generated_explanation: str
    retrieved_chunks_text: str
    gold_rationale: str
