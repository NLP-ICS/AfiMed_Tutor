"""Basic CLI fallback for demo day (§14 risk register).

Usage:
    python -m tutor.cli ask "What is the treatment for malaria?"
    python -m tutor.cli quiz --specialty "Infectious Disease"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def cmd_ask(args) -> None:
    from tutor.llm_client import build_llm_client
    from tutor.orchestrator import handle_ask
    from tutor.retriever import build_retriever

    retriever = build_retriever()
    llm_client = build_llm_client()
    resp = handle_ask(question=args.question, llm_client=llm_client, retriever=retriever)
    print("\n=== AfriMed Tutor Answer ===\n")
    print(resp.answer)
    print(f"\n[Tokens: {resp.input_tokens}in/{resp.output_tokens}out | {resp.latency_ms:.0f}ms]")
    if resp.related_questions:
        print("\nRelated questions:")
        for q in resp.related_questions:
            print(f"  [{q.specialty}] {q.question}")


def cmd_quiz(args) -> None:
    import random
    from tutor.llm_client import build_llm_client
    from tutor.orchestrator import handle_quiz_submit
    from tutor.quiz import QuizLoader, format_options
    from tutor.retriever import build_retriever

    loader = QuizLoader()
    item = loader.sample(specialty=args.specialty or None)

    print(f"\n=== Quiz — {item.specialty} ===\n")
    print(item.question)
    print()
    for opt in item.options:
        print(f"  {opt.key}. {opt.text}")
    print()

    answer = input("Your answer (A/B/C/D): ").strip().upper()
    if answer not in {o.key for o in item.options}:
        print("Invalid answer.")
        return

    retriever = build_retriever()
    llm_client = build_llm_client()
    resp = handle_quiz_submit(
        question_id=item.question_id,
        student_choice=answer,
        llm_client=llm_client,
        retriever=retriever,
        quiz_loader=loader,
    )

    print("\n=== Explanation ===\n")
    print(resp.explanation)
    verdict = "CORRECT" if resp.is_correct else f"INCORRECT (gold: {item.gold_answer})"
    print(f"\n[{verdict} | {resp.latency_ms:.0f}ms]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="tutor", description="AfriMed Tutor CLI")
    sub = parser.add_subparsers(dest="command")

    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("question")

    quiz_parser = sub.add_parser("quiz")
    quiz_parser.add_argument("--specialty", default=None)

    args = parser.parse_args()
    if args.command == "ask":
        cmd_ask(args)
    elif args.command == "quiz":
        cmd_quiz(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
