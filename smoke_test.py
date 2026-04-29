"""Quick smoke test for the full Ask pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from tutor.llm_client import build_llm_client
from tutor.retriever import build_retriever
from tutor.orchestrator import handle_ask

retriever = build_retriever()
llm = build_llm_client()

resp = handle_ask(
    "What is the first-line treatment for uncomplicated malaria in children under 5?",
    llm_client=llm,
    retriever=retriever,
)
print("=== ANSWER ===")
print(resp.answer[:800])
print(f"\nTokens: {resp.input_tokens}in/{resp.output_tokens}out  |  Latency: {resp.latency_ms:.0f}ms")
print(f"Retrieved {len(resp.retrieved_chunks)} chunks, {len(resp.related_questions)} related Qs")
for c in resp.retrieved_chunks:
    print(f"  [{c.score:.3f}] {c.source_doc} — {c.section_title}")
