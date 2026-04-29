# AfriMed Tutor — Additional Experiments and Ablation Studies

This document lists follow-on experiments and ablation studies that could strengthen the evaluation and analysis of AfriMed Tutor beyond the baseline protocol required by the project spec. Each entry includes a summary, detailed description, implementation guidance, and the expected finding or contribution to the report.

---

## E1 — Chunk Size Ablation

**Summary:** Measure how chunk size (100 / 200 / 400 / 800 tokens) affects MCQ accuracy and retrieval recall on the 100-question test set.

**Details:**

The current corpus uses a target of roughly 800 tokens as the upper size limit before splitting, resulting in a mean chunk size of ~145 words (~200 tokens). Smaller chunks may improve precision (the retrieved text is more focused) but reduce recall (a single excerpt may not contain the full clinical reasoning). Larger chunks carry more context but dilute relevance scores with off-topic text.

*Protocol:*
1. Re-run `corpus/build_corpus.py` three additional times with `MAX_CHUNK_TOKENS` set to 100, 200, and 400 (keeping the 800-token variant as the fourth condition). Produce four separate `chunks_{size}.jsonl` and `faiss_{size}.index` files.
2. Re-embed and re-index each corpus variant.
3. Run `eval/run_mcq_eval.py --condition rag` for each retriever variant (dense only for speed) on the 100-question test set.
4. Record per-condition accuracy and a qualitative sample of 5 retrieved chunks per condition to judge coherence.

*Metric:* MCQ accuracy, retrieval score distributions, mean chunk relevance (qualitative, 1–5 scale on a 10-question sample).

*Expected finding:* A chunk size around 200–300 tokens should optimize the precision-recall trade-off for MCQ-style queries; very small chunks (100 tokens) will miss multi-step clinical protocols; very large chunks (800 tokens) will retrieve noisy context.

*Report contribution:* Justifies the chosen chunk size as an empirical decision rather than an arbitrary default. Adds a figure (accuracy vs chunk size) to §3.2.

---

## E2 — Top-k Retrieval Sweep

**Summary:** Vary the number of retrieved chunks (k = 1, 3, 5, 10, 15) to identify the accuracy-cost-latency sweet spot.

**Details:**

The current system retrieves k=5 chunks by default. More chunks provide broader context but increase prompt length (cost and latency), and may introduce off-topic content that distracts the model.

*Protocol:*
1. For each k ∈ {1, 3, 5, 10, 15}, run RAG MCQ eval on the 100-question test set using dense retrieval.
2. Record: accuracy, mean input tokens, mean latency, estimated cost per 100 interactions.
3. Plot accuracy and cost as a function of k.

*Metric:* MCQ accuracy, input token count, latency.

*Expected finding:* Accuracy improves from k=1 to k=5, plateaus or slightly decreases at k=10 and k=15 (noise from irrelevant chunks). Input cost increases linearly with k.

*Report contribution:* Justifies the k=5 default. Adds a figure (accuracy and cost vs k) and informs the latency table in §3.7.

---

## E3 — Cosine Similarity Score Threshold Sweep

**Summary:** Evaluate the effect of the retrieval score cutoff (threshold ∈ {0.0, 0.15, 0.30, 0.45, 0.60}) on accuracy and the "no relevant excerpts" fallback rate.

**Details:**

The current threshold of 0.30 was set by qualitative inspection of 20 queries. A systematic sweep determines whether this is empirically justified.

*Protocol:*
1. For each threshold value, run dense RAG MCQ eval on the 100-question test set.
2. Record: accuracy, fraction of questions where all k chunks fell below threshold (triggering the "(No relevant guideline excerpts found.)" fallback), mean retrieval score.
3. Compute accuracy separately for questions where at least one chunk exceeded the threshold vs. those that fell back to no context.

*Metric:* MCQ accuracy, fallback rate.

*Expected finding:* Below 0.15, irrelevant chunks degrade accuracy. Above 0.45, too many questions fall back to no context, also degrading accuracy. The optimal threshold is in the 0.25–0.35 range.

*Report contribution:* Converts an arbitrary hyperparameter into an empirically chosen value. Adds a supporting figure or table to §3.3.

---

## E4 — Structure-Aware vs. Naive Fixed-Size Chunking

**Summary:** Directly compare structure-aware chunking (current implementation) against a naive fixed-size character-based splitter on MCQ accuracy and a qualitative retrieval coherence assessment.

**Details:**

The spec (§6.1) explicitly requires this comparison but the current evaluation does not surface it because only one chunker was built. This experiment implements the naive baseline and measures the delta.

*Protocol:*
1. Implement a `NaiveChunker` in `corpus/build_corpus.py`: split the raw extracted text into 800-character windows with 200-character overlap, no header detection, no section metadata.
2. Produce `corpus/chunks_naive.jsonl` and `corpus/faiss_naive.index`.
3. Run dense RAG MCQ eval on both corpora.
4. Qualitative assessment: for 10 hand-selected queries, display top-5 retrieved chunks from each corpus and rate coherence (does the chunk read as a complete clinical unit, or does it cut mid-sentence/mid-protocol?).

*Metric:* MCQ accuracy delta, qualitative coherence rating (1–5) averaged over 10 queries.

*Expected finding:* Structure-aware chunking improves accuracy by 3–8 percentage points, especially for Surgery and Pediatrics questions where the guidelines have well-defined protocol blocks. Naive chunking will show more mid-protocol cuts that confuse the LLM.

*Report contribution:* This is the primary NLP-substantive finding specified in the rubric (§6.1 of the spec). It directly justifies the complexity of the structure-aware chunker.

---

## E5 — Embedding Model Comparison

**Summary:** Compare three embedding models — `all-MiniLM-L6-v2` (local, free), `text-embedding-3-small` (OpenAI, $0.02/M tokens), and `voyage-3-lite` (Voyage AI, ~$0.02/M tokens) — on retrieval quality and downstream MCQ accuracy.

**Details:**

The current system uses `all-MiniLM-L6-v2` for cost and offline-reproducibility reasons. The spec (§6.2) suggests OpenAI or Voyage AI embeddings and requests justification. This experiment provides the empirical justification.

*Protocol:*
1. Re-embed the corpus with each model: requires OpenAI and Voyage AI API keys and ~$0.10–0.20 per provider for the 5,633-chunk corpus.
2. Build separate FAISS indexes for each embedding model.
3. Run dense RAG MCQ eval on the 100-question test set for each.
4. Report accuracy, embedding cost, embedding dimensionality (384 / 1536 / 1024), and qualitative retrieval relevance on a 5-question sample.

*Metric:* MCQ accuracy, embedding cost, retrieval score distributions.

*Expected finding:* OpenAI and Voyage embeddings may improve accuracy by 2–5 percentage points due to stronger biomedical semantic understanding; the gap is smaller than the gap from retrieval method (dense vs. sparse), suggesting diminishing returns from embedding model choice at this scale.

*Report contribution:* Directly satisfies the spec's requirement to "justify the embedding model choice" in §3.3. Adds an embedding model comparison table.

---

## E6 — Hybrid Retrieval with Reciprocal Rank Fusion

**Summary:** Combine dense and sparse retrieval rankings using Reciprocal Rank Fusion (RRF) and evaluate whether the hybrid outperforms either single retriever.

**Details:**

Dense retrieval excels at semantic matching; BM25 excels at exact terminology matching. RRF (Cormack et al., 2009) is a simple, parameter-free fusion that assigns each document a score of 1/(k+rank_i) summed across retrievers, where k=60 by default. It consistently outperforms individual retrievers in IR literature.

*Protocol:*
1. Add a `HybridRetriever` class to `tutor/retriever.py` that:
   - Runs both `DenseRetriever` and `SparseRetriever` with k=10 each.
   - Computes RRF scores and returns the top-5 fused results.
2. Run MCQ eval on the 100-question test set.
3. Compare to individual retrievers.

*Implementation sketch:*
```python
class HybridRetriever:
    def __init__(self, k_rrf=60):
        self._dense = DenseRetriever()
        self._sparse = SparseRetriever()
        self._k_rrf = k_rrf

    def search(self, query, k=5):
        dense_results = self._dense.search(query, k=10)
        sparse_results = self._sparse.search(query, k=10)
        scores = {}
        for rank, chunk in enumerate(dense_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (self._k_rrf + rank + 1)
        for rank, chunk in enumerate(sparse_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (self._k_rrf + rank + 1)
        # Sort by fused score, return top-k
        ...
```

*Metric:* MCQ accuracy vs dense-only and sparse-only.

*Expected finding:* Hybrid retrieval matches or slightly outperforms the best individual retriever (+2–4 pp). The improvement is larger for Infectious Disease, where BM25's exact term matching complements dense retrieval's semantic understanding.

*Report contribution:* Demonstrates awareness of the current state of retrieval research beyond naive dense/sparse dichotomy.

---

## E7 — Prompt Ablation: Citation and Refusal Instructions

**Summary:** Measure the effect of removing individual prompt constraints (citation obligation, refusal clause, no-outside-knowledge rule) on output quality.

**Details:**

The Ask-mode and Quiz-mode system prompts contain three explicit behavioral constraints. This ablation tests whether each constraint actually changes model behavior.

*Protocol:*
1. Create three prompt variants:
   - **No citation:** Remove the `[Source: <doc>, Section: <section>]` instruction.
   - **No refusal:** Remove the "if excerpts do not contain enough information, say so explicitly" instruction.
   - **No outside-knowledge ban:** Remove "Do not draw on outside knowledge."
2. For each variant, run LLM-as-judge evaluation on the same 30-item stratified sample as the main eval (§3.5), scoring all three groundedness dimensions.
3. Compare judge scores across prompt variants.

*Metric:* Mean groundedness, citation accuracy, and consistency scores per prompt variant.

*Expected finding:* Removing the citation instruction sharply reduces citation accuracy (from ~1.45 to near 0). Removing the refusal clause may slightly increase groundedness score (model no longer hedges when context is thin). Removing the outside-knowledge ban likely has minimal effect since the model already has a tendency to extend beyond retrieved context.

*Report contribution:* Empirically validates the prompt design decisions. Demonstrates that prompt engineering has measurable behavioral effects — a key NLP contribution.

---

## E8 — Cross-Encoder Re-Ranking

**Summary:** Add a cross-encoder re-ranker on top of the initial dense or sparse retrieval and measure accuracy improvement.

**Details:**

Bi-encoder retrieval (the current dense retriever) is fast but limited by the independent encoding of query and document. A cross-encoder jointly encodes the query-document pair and produces a more accurate relevance score, at the cost of running inference on each candidate chunk.

*Protocol:*
1. Use `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers (runs locally, no API key).
2. Implement a `RerankedRetriever` that:
   - Retrieves 20 candidates from the dense retriever.
   - Re-scores all 20 with the cross-encoder.
   - Returns the top-5 by re-ranked score.
3. Run MCQ eval on the 100-question test set.

*Metric:* MCQ accuracy vs bi-encoder-only dense retrieval.

*Expected finding:* Re-ranking improves accuracy by 3–7 percentage points, with the largest gains in Infectious Disease where initial retrieval quality is lowest. Latency increases by ~500–800 ms per query (cross-encoder inference over 20 candidates).

*Report contribution:* Demonstrates a concrete retrieval improvement pathway and quantifies the latency-accuracy trade-off for potential production deployment.

---

## E9 — HyDE: Hypothetical Document Embeddings for Query Expansion

**Summary:** Use the LLM to generate a hypothetical guideline excerpt that would answer the query, then embed that hypothetical text as the retrieval query instead of the original question.

**Details:**

Gao et al. (2022) introduced HyDE: instead of embedding the user's question (which is short and syntactically unlike the long guideline chunks it targets), generate a plausible answer with the LLM and embed that. The embedding of the hypothetical answer should be more similar to real relevant guideline chunks than the embedding of the question.

*Protocol:*
1. Before retrieval, make a preliminary LLM call with:
   ```
   System: "Write a short clinical guideline excerpt that would answer the following question. 
   Write it in the style of an African clinical guideline. 2–3 sentences only."
   User: {question}
   ```
   Use `max_tokens=150`.
2. Embed the generated hypothetical text as the retrieval query.
3. Retrieve top-5 chunks using the hypothetical embedding.
4. Proceed with the normal RAG pipeline.
5. Run MCQ eval on 100 questions.

*Metric:* MCQ accuracy vs standard dense retrieval; additional LLM call cost (+$0.003/query at current rates).

*Expected finding:* HyDE improves dense retrieval accuracy by 3–6 percentage points, particularly for questions phrased as exam questions ("A 35-year-old woman presents with...") rather than as guideline queries ("Treatment of malaria").

*Report contribution:* Introduces an NLP technique (query reformulation / expansion) that directly addresses the semantic gap between MCQ question phrasing and guideline prose style.

---

## E10 — Specialty-Aware Retrieval Routing

**Summary:** Detect the medical specialty of the query and restrict retrieval to guideline documents most relevant to that specialty, reducing retrieval noise.

**Details:**

The current retriever searches all 5,633 chunks regardless of specialty. A Surgery question may retrieve Infectious Disease chunks (topically similar but clinically irrelevant) because the embedding space does not capture specialty as a dimension. Routing can be implemented as a lightweight classifier or a simple keyword heuristic.

*Protocol:*
1. Implement a specialty classifier using one of:
   - **Keyword heuristic:** maintain a specialty keyword dictionary and tag queries.
   - **Embedding-based:** train a logistic regression classifier on AfriMed-QA question embeddings with specialty labels (available in the dataset).
2. Map each specialty to the most relevant source documents (e.g., Pediatrics → WHO IMCI + SA PHC STG; Surgery → SA Hospital STG + Kenya CG).
3. At query time, restrict the FAISS search to the chunk subset for the detected specialty.
4. Run MCQ eval on 100 questions.

*Metric:* MCQ accuracy per specialty and overall; specialty detection accuracy.

*Expected finding:* Specialty routing improves Pediatrics accuracy (strong IMCI alignment) and Internal Medicine accuracy. May reduce Infectious Disease accuracy if routing misclassifies HIV-related Surgery or Ob/Gyn questions.

*Report contribution:* Demonstrates that domain-aware retrieval design — a key NLP engineering decision — can meaningfully affect downstream task performance.

---

## E11 — Temperature Sensitivity in Explanation Generation

**Summary:** Evaluate how LLM temperature (0.0, 0.3, 0.7, 1.0) affects the groundedness, citation accuracy, and consistency of Quiz-mode explanations.

**Details:**

All evaluations use temperature=0.0 (greedy decoding) for reproducibility. A higher temperature may produce more readable, varied explanations but may also hallucinate more freely. This ablation quantifies the trade-off.

*Protocol:*
1. For each temperature value, run LLM-as-judge evaluation on the same 30-item stratified sample.
2. Use the same judge prompts and scoring rubric.

*Metric:* Mean judge scores (groundedness, citation accuracy, consistency) per temperature.

*Expected finding:* Groundedness decreases with temperature (more varied generation → more extrapolation beyond retrieved context). Citation accuracy also decreases. Consistency with gold rationale may be robust to temperature since the gold rationale is explicitly included in the prompt.

*Report contribution:* Informs operational configuration recommendations for practitioners deploying similar systems.

---

## E12 — Related-Question Method A vs Method B: Quantitative Comparison

**Summary:** Formally compare the two related-question retrieval methods (semantic embedding similarity vs. BM25+specialty) on a user-study-proximate metric: human rating of pedagogical relevance.

**Details:**

The spec's §6.4 requires a qualitative comparison of 10 cases. This experiment extends it to 30 cases with a structured rating task.

*Protocol:*
1. Sample 30 questions from the MCQ pool across 5 specialties.
2. For each question, retrieve 3 related questions with Method A and 3 with Method B (blinded, randomized order).
3. Have each team member independently rate each set of 3 on two dimensions:
   - **Topical relevance** (1–5): how clinically related are these questions to the seed?
   - **Pedagogical diversity** (1–5): do the related questions cover different aspects of the topic, or do they all ask the same thing?
4. Compute inter-rater agreement (Cohen's kappa) and mean ratings per method.

*Metric:* Mean topical relevance, mean pedagogical diversity, inter-rater kappa.

*Expected finding:* Method A (semantic) scores higher on pedagogical diversity; Method B (BM25+specialty) scores higher on topical relevance for same-specialty questions. Neither method is strictly dominant.

*Report contribution:* Provides quantitative backing for the default choice of Method A and demonstrates awareness of multi-dimensional evaluation design.

---

## E13 — LLM-as-Judge Validation: Cohen's Kappa Against Human Scores

**Summary:** Compute inter-rater agreement between the automated LLM judge and human-scored examples on the first 10 items of the groundedness sample, to validate the judge's reliability.

**Details:**

The spec (§6.5) explicitly requires validation of the LLM judge on 10 manually-scored examples before trusting its scores on the remaining 20. This experiment completes that validation and reports the agreement number.

*Protocol:*
1. Each team member independently scores the first 10 items in `results/qualitative_sample.md` on the three dimensions (groundedness, citation accuracy, consistency), using the same 0–2 rubric as the judge prompt.
2. Compute:
   - **Human inter-rater agreement:** Cohen's kappa between any two team member scores per dimension.
   - **Human-judge agreement:** Cohen's kappa between the mean human score and the LLM judge score per dimension.
3. Report kappa values; thresholds: κ < 0.40 = poor, 0.40–0.60 = moderate, 0.60–0.80 = substantial.

*Expected finding:* Human inter-rater agreement is moderate (κ ≈ 0.50–0.65) — medical quality assessment is inherently subjective. Human-judge agreement is lower (κ ≈ 0.30–0.50) due to same-model bias in the judge configuration used in this run.

*Report contribution:* Required by the spec (§6.5). Provides the methodological credibility check that separates a rigorous NLP evaluation from "we counted answers."

---

## Prioritization Guide

| Priority | Experiment | Effort | Report Impact |
|---|---|---|---|
| **Must-do** | E13 — LLM judge validation (kappa) | Low | High (required by spec) |
| **Must-do** | E4 — Structure-aware vs. naive chunking | Medium | Very high (core NLP finding) |
| **High** | E1 — Chunk size ablation | Medium | High |
| **High** | E2 — Top-k sweep | Low | High |
| **High** | E6 — Hybrid retrieval (RRF) | Medium | Medium-High |
| **Medium** | E5 — Embedding model comparison | Medium (needs API keys) | Medium |
| **Medium** | E7 — Prompt ablation | Low | Medium |
| **Medium** | E9 — HyDE query expansion | Low | Medium |
| **Medium** | E3 — Score threshold sweep | Low | Medium |
| **Low** | E8 — Cross-encoder re-ranking | Medium | Medium |
| **Low** | E10 — Specialty routing | High | Medium |
| **Low** | E11 — Temperature sensitivity | Low | Low |
| **Low** | E12 — Related-Q quantitative comparison | Medium | Low |

E13 and E4 should be run before submitting the report. E1, E2, and E6 are high-value low-risk additions if time permits. E8, E10, and E12 are best suited for a follow-on paper or thesis extension.
