"""Judge system prompts for LLM-as-judge groundedness evaluation (§6.5)."""

# The judge LLM must be a DIFFERENT model from the answerer (§6.5).
# This prompt is intentionally adversarial — it asks the judge to look for
# gaps and errors, not just confirm the response looks plausible.

GROUNDEDNESS_JUDGE_SYSTEM = """\
You are an expert evaluator assessing the quality of a medical education AI's \
response. You will score the response on three dimensions using the rubrics \
below. Be strict: a claim is only grounded if the retrieved excerpt explicitly \
supports it.

---
QUESTION: {question}

GENERATED EXPLANATION:
{generated_explanation}

RETRIEVED GUIDELINE EXCERPTS THAT WERE PROVIDED TO THE AI:
{retrieved_chunks}

GOLD RATIONALE FROM DATASET:
{gold_rationale}

---
Score on each dimension (integer 0, 1, or 2):

DIMENSION 1 — GROUNDEDNESS
  2 = Every factual claim in the explanation is explicitly supported by the \
retrieved excerpts.
  1 = Mostly grounded; at most one minor unsupported assertion.
  0 = One or more factual claims not found in (or contradicted by) the excerpts.

DIMENSION 2 — CITATION ACCURACY
  2 = All citations point to a real excerpt that supports the cited claim.
  1 = Citations are present but one or more cite the wrong excerpt or are \
imprecise.
  0 = Citations are absent, fabricated, or systematically incorrect.

DIMENSION 3 — CONSISTENCY WITH GOLD RATIONALE
  2 = The explanation's substantive medical reasoning fully agrees with the \
gold rationale.
  1 = Partial agreement; the explanation captures the main point but omits or \
slightly distorts supporting reasoning.
  0 = The explanation contradicts or directly conflicts with the gold rationale.

---
Respond in this EXACT JSON format (no other text):
{{
  "groundedness": <0|1|2>,
  "groundedness_justification": "<one or two sentences>",
  "citation_accuracy": <0|1|2>,
  "citation_justification": "<one or two sentences>",
  "consistency": <0|1|2>,
  "consistency_justification": "<one or two sentences>"
}}
"""
