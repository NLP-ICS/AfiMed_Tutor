"""Generate report/additional_experiments.pdf using fpdf2."""
from fpdf import FPDF
from fpdf.enums import XPos, YPos

MARGIN = 20
PAGE_W = 210
CONTENT_W = PAGE_W - 2 * MARGIN

# ── Person assignments ─────────────────────────────────────────────────────────
# Person 1: Built the base system (all core infrastructure)
# Person 2: Chunking and corpus quality experiments
# Person 3: Advanced retrieval and embedding experiments
# Person 4: Evaluation, prompt, and query experiments


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "AfriMed Tutor: Additional Experiments and Ablation Studies", align="L",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(180, 180, 180)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def section_heading(self, text, level=1):
        self.ln(4)
        if level == 1:
            self.set_font("Helvetica", "B", 13)
            self.set_fill_color(30, 80, 140)
            self.set_text_color(255, 255, 255)
            self.cell(CONTENT_W, 8, f"  {text}", fill=True,
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(0, 0, 0)
        elif level == 2:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(30, 80, 140)
            self.cell(CONTENT_W, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, text, indent=0):
        self.set_font("Helvetica", "", 10)
        self.set_x(MARGIN + indent)
        self.multi_cell(CONTENT_W - indent, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def label_value(self, label, value, indent=4):
        self.set_x(MARGIN + indent)
        self.set_font("Helvetica", "B", 10)
        self.cell(28, 5.5, label + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(CONTENT_W - indent - 28, 5.5, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(0.5)

    def experiment_box(self, exp_id, title, person, summary, details_blocks):
        """Render one experiment block."""
        # Check page space
        if self.get_y() > 240:
            self.add_page()

        # Title bar
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(240, 245, 255)
        self.set_draw_color(30, 80, 140)
        self.rect(MARGIN, self.get_y(), CONTENT_W, 8, style="FD")
        self.set_x(MARGIN + 2)
        self.cell(CONTENT_W - 2, 8, f"{exp_id}  {title}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200, 200, 200)

        # Assigned-to badge
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 80, 140)
        self.set_text_color(255, 255, 255)
        self.set_x(MARGIN)
        self.cell(40, 5.5, f"  Assigned to: {person}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(2)

        # Summary
        self.set_font("Helvetica", "BI", 10)
        self.set_x(MARGIN + 2)
        self.cell(18, 5, "Summary:", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Helvetica", "I", 10)
        self.multi_cell(CONTENT_W - 20, 5, summary, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

        for label, text in details_blocks:
            self.label_value(label, text, indent=4)

        self.ln(3)
        self.set_draw_color(200, 200, 200)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(3)

    def priority_table(self, rows):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(30, 80, 140)
        self.set_text_color(255, 255, 255)
        widths = [16, 72, 30, 28, 24]
        headers = ["ID", "Experiment", "Assigned to", "Effort", "Priority"]
        self.set_x(MARGIN)
        for w, h in zip(widths, headers):
            self.cell(w, 6, h, border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)
        fill = False
        for row in rows:
            self.set_font("Helvetica", "", 8.5)
            self.set_fill_color(245, 248, 255) if fill else self.set_fill_color(255, 255, 255)
            self.set_x(MARGIN)
            # Row height: compute max lines in description column
            for w, val in zip(widths, row):
                self.cell(w, 6, str(val), border=1, fill=fill)
            self.ln()
            fill = not fill
        self.ln(4)


# ══════════════════════════════════════════════════════════════════════════════
pdf = PDF()
pdf.set_margins(MARGIN, 18, MARGIN)
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

# ── Title page block ──────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(30, 80, 140)
pdf.multi_cell(CONTENT_W, 10, "AfriMed Tutor\nAdditional Experiments and Ablation Studies",
               align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "", 11)
pdf.multi_cell(CONTENT_W, 6, "ICS4554 Natural Language Processing  |  Ashesi University  |  Spring 2026",
               align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(4)
pdf.set_draw_color(30, 80, 140)
pdf.set_line_width(0.8)
pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
pdf.set_line_width(0.2)
pdf.ln(5)

# ── Group roles overview ──────────────────────────────────────────────────────
pdf.section_heading("Group Role Overview", level=1)

pdf.body_text(
    "Person 1 built the complete base system: corpus download and structure-aware chunking, FAISS dense "
    "retrieval and BM25 sparse retrieval, AfriMed-QA data loading and stratified splits, all three "
    "orchestrator modes with prompt engineering, the Streamlit UI, and all quantitative evaluations "
    "(MCQ accuracy, retriever comparison, LLM-as-judge groundedness). The experiments in this document "
    "are assigned to the remaining three group members to extend and strengthen the project's NLP "
    "contribution."
)

role_data = [
    ("Person 1", "Base system (all infrastructure, core evaluation pipeline, Streamlit UI)",
     "Complete"),
    ("Person 2", "Chunking quality and corpus ablations (E1, E2, E3, E4)",
     "Additional experiments"),
    ("Person 3", "Advanced retrieval and embedding experiments (E5, E6, E7, E8)",
     "Additional experiments"),
    ("Person 4", "Evaluation, prompt engineering, and query experiments (E9, E10, E11, E12, E13)",
     "Additional experiments"),
]
pdf.set_font("Helvetica", "B", 9)
pdf.set_fill_color(30, 80, 140)
pdf.set_text_color(255, 255, 255)
pdf.set_x(MARGIN)
for label, w in [("Person", 22), ("Responsibility", 102), ("Role", 46)]:
    pdf.cell(w, 6, label, border=1, fill=True)
pdf.ln()
pdf.set_text_color(0, 0, 0)
fill = False
for person, resp, role in role_data:
    pdf.set_font("Helvetica", "" if fill else "B", 9)
    pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(235, 242, 255)
    pdf.set_x(MARGIN)
    pdf.cell(22, 6, person, border=1, fill=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(102, 6, resp, border=1, fill=fill)
    pdf.cell(46, 6, role, border=1, fill=fill)
    pdf.ln()
    fill = not fill
pdf.ln(6)


# ══════════════════════════════════════════════════════════════════════════════
# PERSON 2 EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
pdf.section_heading("Person 2  -  Chunking Quality and Corpus Ablations", level=1)

pdf.body_text(
    "Person 2 investigates how the design of the document chunking pipeline affects retrieval quality and "
    "downstream MCQ accuracy. The chunking strategy is one of the two primary NLP contributions cited in "
    "the project spec (section 6.1), and these experiments provide the empirical backing for the "
    "structure-aware design choice."
)

pdf.experiment_box(
    "E1", "Structure-Aware vs. Naive Fixed-Size Chunking", "Person 2",
    "Directly compare the structure-aware chunker (current implementation) against a naive fixed-size "
    "character-based splitter on MCQ accuracy and a qualitative retrieval coherence assessment.",
    [
        ("Why it matters",
         "The spec explicitly requires this comparison (section 6.1). It is the primary NLP finding "
         "that justifies the complexity of the structure-aware chunker and distinguishes the project "
         "from a generic RAG exercise."),
        ("Protocol",
         "Implement a NaiveChunker in corpus/build_corpus.py that splits raw extracted text into "
         "800-character windows with 200-character overlap and no header detection or section metadata. "
         "Produce chunks_naive.jsonl and faiss_naive.index. Run dense RAG MCQ eval on both corpora "
         "using the 100-question test set. For 10 hand-selected queries, display top-5 retrieved "
         "chunks from each corpus and rate coherence on a 1 to 5 scale (does the chunk read as a "
         "complete clinical unit, or does it cut mid-sentence or mid-protocol?)."),
        ("Metric",
         "MCQ accuracy delta between structure-aware and naive; mean coherence rating over 10 queries."),
        ("Expected finding",
         "Structure-aware chunking improves accuracy by 3 to 8 percentage points, especially for "
         "Surgery and Pediatrics questions where guidelines have well-defined protocol blocks. Naive "
         "chunking will show more mid-protocol cuts that degrade LLM explanation quality."),
        ("Output", "Accuracy comparison table, 5 example chunk pairs (structure-aware vs naive) for "
         "the report section 3.2."),
    ]
)

pdf.experiment_box(
    "E2", "Chunk Size Ablation", "Person 2",
    "Measure how chunk size (100 / 200 / 400 / 800 tokens) affects MCQ accuracy and retrieval recall "
    "on the 100-question test set.",
    [
        ("Why it matters",
         "The current mean chunk size of 145 words was chosen heuristically. A sweep converts this "
         "into an empirical decision and quantifies the precision-recall trade-off."),
        ("Protocol",
         "Re-run corpus/build_corpus.py three additional times with MAX_CHUNK_TOKENS set to 100, 200, "
         "and 400 (keeping 800 as the fourth condition). Produce separate chunks and FAISS index files "
         "for each. Run dense RAG MCQ eval on the 100-question test set for each variant. Record "
         "accuracy, retrieval score distributions, and mean chunk relevance on a qualitative "
         "5-question sample."),
        ("Metric", "MCQ accuracy, retrieval score distributions, mean chunk size and count per condition."),
        ("Expected finding",
         "A chunk size around 200 to 300 tokens optimizes the precision-recall trade-off. Chunks of "
         "100 tokens will miss multi-step clinical protocols; chunks of 800 tokens will retrieve "
         "noisy context that dilutes relevance."),
        ("Output",
         "Accuracy vs chunk size figure and supporting table for the report. Justifies the chosen "
         "default in section 3.2."),
    ]
)

pdf.experiment_box(
    "E3", "Top-k Retrieval Sweep", "Person 2",
    "Vary the number of retrieved chunks (k = 1, 3, 5, 10, 15) to identify the accuracy-cost-latency "
    "sweet spot.",
    [
        ("Why it matters",
         "More chunks provide broader context but increase prompt length (cost and latency) and may "
         "introduce off-topic content. This sweep justifies the k=5 default empirically."),
        ("Protocol",
         "For each k value, run RAG MCQ eval on the 100-question test set using dense retrieval. "
         "Record accuracy, mean input tokens, mean latency, and estimated cost per 100 interactions."),
        ("Metric", "MCQ accuracy, input token count, latency, estimated cost."),
        ("Expected finding",
         "Accuracy improves from k=1 to k=5, then plateaus or slightly decreases at k=10 and k=15 "
         "due to retrieval noise. Input cost increases approximately linearly with k."),
        ("Output",
         "Accuracy and cost vs k figure. Informs the latency and cost table in report section 3.7."),
    ]
)

pdf.experiment_box(
    "E4", "Cosine Similarity Score Threshold Sweep", "Person 2",
    "Evaluate the effect of the retrieval score cutoff (0.0, 0.15, 0.30, 0.45, 0.60) on accuracy and "
    "the no-relevant-excerpts fallback rate.",
    [
        ("Why it matters",
         "The current threshold of 0.30 was set by qualitative inspection of 20 queries. A systematic "
         "sweep determines whether this value is empirically justified."),
        ("Protocol",
         "For each threshold, run dense RAG MCQ eval on the 100-question test set. Record accuracy, "
         "fraction of questions where all k chunks fell below threshold (triggering the 'No relevant "
         "guideline excerpts found' fallback), and mean retrieval score. Compute accuracy separately "
         "for questions that retrieved at least one chunk above threshold vs those that fell back "
         "to no context."),
        ("Metric", "MCQ accuracy, fallback rate per threshold."),
        ("Expected finding",
         "Below 0.15 irrelevant chunks degrade accuracy. Above 0.45 too many questions fall back to "
         "no context. The optimal threshold lies in the 0.25 to 0.35 range."),
        ("Output",
         "Threshold sweep figure for report section 3.3. Converts an arbitrary hyperparameter into "
         "an empirically justified value."),
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# PERSON 3 EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
pdf.section_heading("Person 3  -  Advanced Retrieval and Embedding Experiments", level=1)

pdf.body_text(
    "Person 3 investigates improvements to the retrieval pipeline beyond the baseline dense and sparse "
    "configurations. These experiments explore embedding model quality, retrieval fusion, and re-ranking, "
    "providing a systematic comparison of retrieval design choices that is central to the NLP contribution."
)

pdf.experiment_box(
    "E5", "Embedding Model Comparison", "Person 3",
    "Compare three embedding models: all-MiniLM-L6-v2 (local, free), text-embedding-3-small (OpenAI, "
    "$0.02 per million tokens), and voyage-3-lite (Voyage AI) on retrieval quality and downstream "
    "MCQ accuracy.",
    [
        ("Why it matters",
         "The spec (section 6.2) requests justification of the embedding model choice. This experiment "
         "provides the empirical justification comparing cost, dimensionality, and accuracy."),
        ("Protocol",
         "Re-embed the corpus with each model (requires OpenAI and Voyage AI API keys; approximately "
         "$0.10 to $0.20 per provider for 5,633 chunks). Build separate FAISS indexes. Run dense RAG "
         "MCQ eval on the 100-question test set for each model. Report accuracy, embedding cost, "
         "embedding dimensionality (384, 1536, 1024), and qualitative retrieval relevance on a "
         "5-question sample."),
        ("Metric",
         "MCQ accuracy, embedding cost per corpus, retrieval score distributions."),
        ("Expected finding",
         "OpenAI and Voyage embeddings may improve accuracy by 2 to 5 percentage points due to "
         "stronger biomedical semantic understanding. The gap is smaller than the dense-vs-sparse "
         "difference, suggesting diminishing returns from embedding model choice at this scale."),
        ("Output",
         "Embedding model comparison table for report section 3.3. Directly satisfies the spec "
         "requirement to justify the embedding model choice."),
    ]
)

pdf.experiment_box(
    "E6", "Hybrid Retrieval with Reciprocal Rank Fusion", "Person 3",
    "Combine dense and sparse retrieval rankings using Reciprocal Rank Fusion (RRF) and evaluate "
    "whether the hybrid outperforms either individual retriever.",
    [
        ("Why it matters",
         "Dense retrieval excels at semantic matching; BM25 excels at exact terminology matching. "
         "RRF (Cormack et al., 2009) is a parameter-free fusion that consistently outperforms "
         "individual retrievers in IR literature. This is a natural next step given that sparse "
         "already outperforms dense in our evaluation."),
        ("Protocol",
         "Add a HybridRetriever class to tutor/retriever.py. Retrieve k=10 candidates from each "
         "of DenseRetriever and SparseRetriever. Compute RRF scores: score = sum(1 / (60 + rank)) "
         "across retrievers. Return top-5 fused results. Run MCQ eval on the 100-question test set."),
        ("Metric", "MCQ accuracy vs dense-only and sparse-only."),
        ("Expected finding",
         "Hybrid retrieval matches or slightly outperforms the best individual retriever by 2 to 4 "
         "percentage points. The improvement is larger for Infectious Disease where initial retrieval "
         "quality is lowest."),
        ("Output",
         "Retriever comparison table updated with hybrid results. New HybridRetriever class that "
         "becomes available in the production app via RETRIEVER_BACKEND=hybrid."),
    ]
)

pdf.experiment_box(
    "E7", "Cross-Encoder Re-Ranking", "Person 3",
    "Add a cross-encoder re-ranker on top of initial dense retrieval and measure the accuracy "
    "improvement and latency cost.",
    [
        ("Why it matters",
         "Bi-encoder retrieval is fast but limited by independent encoding of query and document. "
         "A cross-encoder jointly encodes query-document pairs and produces more accurate relevance "
         "scores, at a latency cost."),
        ("Protocol",
         "Use cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers (runs locally, no "
         "API key needed). Implement a RerankedRetriever that retrieves 20 candidates from the dense "
         "retriever, re-scores all 20 with the cross-encoder, and returns the top-5 by re-ranked "
         "score. Run MCQ eval on the 100-question test set. Record accuracy and end-to-end latency."),
        ("Metric", "MCQ accuracy and mean end-to-end latency vs dense-only."),
        ("Expected finding",
         "Re-ranking improves accuracy by 3 to 7 percentage points with the largest gains in "
         "Infectious Disease. Latency increases by approximately 500 to 800 ms per query due to "
         "cross-encoder inference over 20 candidates."),
        ("Output",
         "Accuracy-latency trade-off figure. New RerankedRetriever class selectable via "
         "RETRIEVER_BACKEND=reranked."),
    ]
)

pdf.experiment_box(
    "E8", "Specialty-Aware Retrieval Routing", "Person 3",
    "Detect the medical specialty of the query and restrict retrieval to guideline documents most "
    "relevant to that specialty, reducing retrieval noise.",
    [
        ("Why it matters",
         "The current retriever searches all 5,633 chunks regardless of specialty. A Surgery question "
         "may retrieve Infectious Disease chunks because the embedding space does not capture specialty "
         "as a dimension. Domain-aware routing directly addresses this."),
        ("Protocol",
         "Implement a specialty classifier using either a keyword heuristic (maintain a specialty "
         "keyword dictionary) or an embedding-based logistic regression trained on AfriMed-QA "
         "question embeddings with specialty labels. Map each specialty to the most relevant source "
         "documents (e.g., Pediatrics maps to WHO IMCI and SA PHC STG; Surgery maps to SA Hospital "
         "STG and Kenya CG). At query time, restrict the FAISS search to the chunk subset for the "
         "detected specialty. Run MCQ eval on the 100-question test set."),
        ("Metric",
         "MCQ accuracy per specialty and overall; specialty detection accuracy."),
        ("Expected finding",
         "Routing improves Pediatrics and Internal Medicine accuracy. It may reduce Infectious "
         "Disease accuracy if HIV-related questions are misclassified into Surgery or Ob/Gyn."),
        ("Output",
         "Per-specialty accuracy heatmap comparing routed vs unrouted retrieval. Specialty "
         "detection accuracy report."),
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# PERSON 4 EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
pdf.section_heading("Person 4  -  Evaluation, Prompt Engineering, and Query Experiments", level=1)

pdf.body_text(
    "Person 4 handles the evaluation methodology and prompt engineering ablations. This includes the "
    "mandatory judge validation (required by the project spec), query expansion experiments, and the "
    "quantitative comparison of the related-question retrieval methods. These tasks are directly linked "
    "to the LLM-as-judge evaluation that the spec identifies as the methodological core of the project."
)

pdf.experiment_box(
    "E9", "LLM-as-Judge Validation: Cohen's Kappa Against Human Scores", "Person 4",
    "Compute inter-rater agreement between the automated LLM judge and human-scored examples on the "
    "first 10 items of the groundedness sample, validating the judge's reliability.",
    [
        ("Why it matters",
         "The spec (section 6.5) explicitly requires validation of the LLM judge on 10 manually-scored "
         "examples before trusting its scores on the full sample. This experiment completes that "
         "required step and reports the kappa agreement number in the report."),
        ("Protocol",
         "Each group member independently scores the first 10 items in results/qualitative_sample.md "
         "on the three dimensions (groundedness, citation accuracy, consistency) using the same 0 to 2 "
         "rubric as the judge prompt. Compute human inter-rater agreement (Cohen's kappa between any "
         "two team member scores per dimension). Compute human-judge agreement (kappa between mean "
         "human score and LLM judge score per dimension). Thresholds: kappa below 0.40 is poor, "
         "0.40 to 0.60 is moderate, 0.60 to 0.80 is substantial."),
        ("Metric", "Cohen's kappa per evaluation dimension, for both human-human and human-judge pairs."),
        ("Expected finding",
         "Human inter-rater agreement is moderate (kappa approximately 0.50 to 0.65) since medical "
         "quality assessment is inherently subjective. Human-judge agreement is lower (approximately "
         "0.30 to 0.50) due to same-model bias from the fallback judge configuration."),
        ("Output",
         "Kappa table for report section 3.5. This is a required deliverable per the spec."),
    ]
)

pdf.experiment_box(
    "E10", "Prompt Ablation: Citation and Refusal Instructions", "Person 4",
    "Measure the effect of removing individual prompt constraints (citation obligation, refusal clause, "
    "no-outside-knowledge rule) on output quality as scored by the LLM judge.",
    [
        ("Why it matters",
         "The Ask-mode and Quiz-mode prompts contain three explicit behavioral constraints. This "
         "ablation tests whether each constraint actually changes model behavior, empirically validating "
         "the prompt engineering decisions."),
        ("Protocol",
         "Create three prompt variants in a new file tutor/prompts_ablation.py: (1) No citation: "
         "remove the [Source: doc, Section: section] instruction. (2) No refusal: remove the 'if "
         "excerpts do not contain enough information, say so explicitly' clause. (3) No outside-knowledge "
         "ban: remove the 'Do not draw on outside knowledge' instruction. For each variant, run "
         "LLM-as-judge evaluation on the same 30-item stratified sample as the main eval, scoring "
         "all three groundedness dimensions. Compare judge scores across prompt variants."),
        ("Metric",
         "Mean groundedness, citation accuracy, and consistency scores per prompt variant."),
        ("Expected finding",
         "Removing the citation instruction sharply reduces citation accuracy toward zero. Removing "
         "the refusal clause may slightly increase groundedness since the model no longer hedges when "
         "context is thin. Removing the outside-knowledge ban likely has minimal measurable effect "
         "since the model already extends beyond retrieved context regardless."),
        ("Output",
         "Prompt ablation table for report section 3.5. Demonstrates that prompt engineering has "
         "measurable behavioral effects."),
    ]
)

pdf.experiment_box(
    "E11", "HyDE: Hypothetical Document Embeddings for Query Expansion", "Person 4",
    "Use the LLM to generate a hypothetical guideline excerpt that would answer the query, then embed "
    "that text as the retrieval query instead of the original question.",
    [
        ("Why it matters",
         "MCQ questions are phrased as exam questions ('A 35-year-old woman presents with...') while "
         "guideline chunks are written as clinical protocols. The semantic gap between question "
         "phrasing and guideline prose style limits dense retrieval. HyDE (Gao et al., 2022) "
         "addresses this by querying with a hypothetical answer instead of the question."),
        ("Protocol",
         "Before retrieval, make a preliminary LLM call asking the model to write a 2 to 3 sentence "
         "hypothetical clinical guideline excerpt that would answer the query, in the style of an "
         "African clinical guideline. Embed the generated text as the retrieval query. Proceed with "
         "the normal RAG pipeline using the hypothetical embedding. Run MCQ eval on the 100-question "
         "test set. Note the additional cost of one preliminary LLM call per query."),
        ("Metric",
         "MCQ accuracy vs standard dense retrieval; additional cost per query."),
        ("Expected finding",
         "HyDE improves dense retrieval accuracy by 3 to 6 percentage points, particularly for "
         "MCQ-style questions phrased as clinical scenarios rather than direct guideline queries."),
        ("Output",
         "Accuracy comparison table and cost analysis. Demonstrates awareness of query reformulation "
         "as an NLP technique."),
    ]
)

pdf.experiment_box(
    "E12", "Temperature Sensitivity in Explanation Generation", "Person 4",
    "Evaluate how LLM temperature (0.0, 0.3, 0.7, 1.0) affects the groundedness, citation accuracy, "
    "and consistency of Quiz-mode explanations.",
    [
        ("Why it matters",
         "All evaluations use temperature 0.0 for reproducibility. A higher temperature may produce "
         "more readable explanations but more hallucination. This ablation quantifies the trade-off "
         "and informs operational configuration recommendations."),
        ("Protocol",
         "For each temperature value, run LLM-as-judge evaluation on the same 30-item stratified "
         "sample using the same judge prompts and scoring rubric as the main eval."),
        ("Metric",
         "Mean judge scores (groundedness, citation accuracy, consistency) per temperature."),
        ("Expected finding",
         "Groundedness and citation accuracy decrease with temperature as generation becomes more "
         "varied and extrapolates beyond retrieved context. Consistency with gold rationale is more "
         "robust to temperature since the gold rationale is explicitly included in the prompt."),
        ("Output",
         "Temperature sensitivity table for report appendix. Informs the choice of temperature "
         "for the production deployment."),
    ]
)

pdf.experiment_box(
    "E13", "Related-Question Method A vs Method B: Quantitative Comparison", "Person 4",
    "Formally compare semantic embedding similarity (Method A) versus BM25 plus specialty filtering "
    "(Method B) for related-question retrieval using a structured team rating task.",
    [
        ("Why it matters",
         "The spec requires a qualitative comparison of 10 cases (section 6.4). This experiment "
         "extends it to 30 cases with a structured rating protocol and inter-rater agreement, "
         "providing quantitative backing for the default choice of Method A."),
        ("Protocol",
         "Sample 30 questions from the MCQ pool across 5 specialties. For each question, retrieve "
         "3 related questions with Method A and 3 with Method B (blinded, randomized order). Have "
         "each team member independently rate each set of 3 on: (1) Topical relevance (1 to 5, how "
         "clinically related are these questions to the seed?) and (2) Pedagogical diversity (1 to 5, "
         "do the related questions cover different aspects or ask the same thing?). Compute Cohen's "
         "kappa for inter-rater agreement and mean ratings per method."),
        ("Metric",
         "Mean topical relevance, mean pedagogical diversity, inter-rater kappa."),
        ("Expected finding",
         "Method A scores higher on pedagogical diversity. Method B scores higher on topical "
         "relevance for same-specialty questions. Neither method strictly dominates."),
        ("Output",
         "Rating comparison table for report section 3.6. Justifies the Method A default with "
         "quantitative evidence."),
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.section_heading("Prioritization Summary", level=1)

pdf.body_text(
    "The table below lists all experiments in priority order. Must-do experiments are required for "
    "full marks on the rubric. High-priority experiments substantially strengthen the NLP contribution. "
    "Medium and low priority experiments are suitable as extension work or for the discussion section."
)

rows = [
    ["E9",  "LLM judge kappa validation",                 "Person 4", "Low",    "Must-do"],
    ["E1",  "Structure-aware vs naive chunking",          "Person 2", "Medium", "Must-do"],
    ["E2",  "Chunk size ablation",                        "Person 2", "Medium", "High"],
    ["E3",  "Top-k retrieval sweep",                      "Person 2", "Low",    "High"],
    ["E6",  "Hybrid retrieval (RRF)",                     "Person 3", "Medium", "High"],
    ["E10", "Prompt ablation",                            "Person 4", "Low",    "High"],
    ["E5",  "Embedding model comparison",                 "Person 3", "Medium", "Medium"],
    ["E11", "HyDE query expansion",                       "Person 4", "Low",    "Medium"],
    ["E4",  "Score threshold sweep",                      "Person 2", "Low",    "Medium"],
    ["E7",  "Cross-encoder re-ranking",                   "Person 3", "Medium", "Medium"],
    ["E8",  "Specialty-aware routing",                    "Person 3", "High",   "Medium"],
    ["E12", "Temperature sensitivity",                    "Person 4", "Low",    "Low"],
    ["E13", "Related-Q quantitative comparison",          "Person 4", "Medium", "Low"],
]

pdf.priority_table(rows)

pdf.body_text(
    "E9 (judge kappa) and E1 (structure-aware vs naive chunking) should be completed before submitting "
    "the report: E9 is explicitly required by the spec and E1 is the core NLP finding that justifies "
    "the chunking design. E2, E3, and E6 are high-value additions with low implementation risk. "
    "E5, E7, E8, and E11 require additional API keys or significant compute and are best treated as "
    "extension work if time permits."
)


# ── Save ──────────────────────────────────────────────────────────────────────
pdf.output("report/additional_experiments.pdf")
print("Saved: report/additional_experiments.pdf")
