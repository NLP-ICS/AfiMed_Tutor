"""Mode router and prompt composer (§5, §10).

Each handle_* function:
  1. Retrieves relevant guideline chunks.
  2. Formats the appropriate system prompt.
  3. Makes exactly one LLM call.
  4. Retrieves related questions.
  5. Returns a typed response schema.
"""

from __future__ import annotations

import os
import time

from tutor.llm_client import LLMClient
from tutor.prompts import ASK_SYSTEM, EXPLAIN_SYSTEM, QUIZ_SYSTEM
from tutor.quiz import QuizLoader, format_options
from tutor.explain import SAQLoader
from tutor.related_questions import build_related_retriever
from tutor.retriever import Retriever
from tutor.schemas import (
    AskResponse,
    Chunk,
    ExplainResponse,
    QuizResponse,
)


def _render_chunks(chunks: list[Chunk]) -> str:
    """Format retrieved chunks for prompt insertion."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Excerpt {i}] {chunk.source_doc} — {chunk.section_title}"
        if chunk.condition:
            header += f" ({chunk.condition})"
        if chunk.page_number:
            header += f", p. {chunk.page_number}"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts) if parts else "(No relevant guideline excerpts found.)"


def handle_ask(
    question: str,
    llm_client: LLMClient,
    retriever: Retriever,
    k: int | None = None,
    related_k: int = 3,
) -> AskResponse:
    k = k or int(os.getenv("RETRIEVER_TOP_K", "5"))
    chunks = retriever.search(question, k=k)
    system = ASK_SYSTEM.format(
        retrieved_chunks=_render_chunks(chunks),
        question=question,
    )
    result = llm_client.complete(system=system, user=question, max_tokens=1024)

    related = []
    try:
        rel_retriever = build_related_retriever()
        related = rel_retriever.get_related(question, k=related_k)
    except Exception:
        pass  # related questions are non-critical

    return AskResponse(
        answer=result.text,
        retrieved_chunks=chunks,
        related_questions=related,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        model_name=result.model_name,
    )


def handle_quiz_submit(
    question_id: str,
    student_choice: str,
    llm_client: LLMClient,
    retriever: Retriever,
    quiz_loader: QuizLoader,
    related_k: int = 3,
) -> QuizResponse:
    item = quiz_loader.get_by_id(question_id)
    k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    chunks = retriever.search(item.question, k=k)

    system = QUIZ_SYSTEM.format(
        question=item.question,
        options=format_options(item.options),
        student_choice=student_choice,
        gold_answer=f"{item.gold_answer}. "
        + next((o.text for o in item.options if o.key == item.gold_answer), ""),
        gold_rationale=item.gold_rationale,
        retrieved_chunks=_render_chunks(chunks),
    )
    result = llm_client.complete(system=system, user="Please explain this question.", max_tokens=1024)

    is_correct = student_choice.strip().upper() == item.gold_answer.strip().upper()

    related = []
    try:
        rel_retriever = build_related_retriever(exclude_id=question_id)
        related = rel_retriever.get_related(item.question, k=related_k)
    except Exception:
        pass

    return QuizResponse(
        explanation=result.text,
        is_correct=is_correct,
        retrieved_chunks=chunks,
        related_questions=related,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        model_name=result.model_name,
    )


def handle_explain(
    case_id: str,
    student_reasoning: str,
    llm_client: LLMClient,
    retriever: Retriever,
    saq_loader: SAQLoader,
) -> ExplainResponse:
    case = saq_loader.get_by_id(case_id)
    k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    chunks = retriever.search(case.scenario, k=k)

    system = EXPLAIN_SYSTEM.format(
        scenario=case.scenario,
        student_reasoning=student_reasoning,
        expert_answer=case.expert_answer,
        retrieved_chunks=_render_chunks(chunks),
    )
    result = llm_client.complete(
        system=system, user="Please compare my reasoning.", max_tokens=1200
    )

    return ExplainResponse(
        comparison=result.text,
        retrieved_chunks=chunks,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        model_name=result.model_name,
    )
