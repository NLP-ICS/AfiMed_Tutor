# AfriMed Tutor

AfriMed Tutor is a guideline-grounded study assistant for African medical students. It combines medical guideline retrieval, question answering, quiz generation, and explanation support using a hybrid retrieval-and-generation architecture.

Built for ICS4554 NLP — Ashesi University, Spring 2026.

## Project overview

This project is designed to support medical education by:
- Answering free-text clinical questions with cited guideline evidence
- Delivering multiple-choice quizzes from the AfriMed-QA dataset
- Comparing student reasoning to expert clinical explanations
- Evaluating retrieval quality and answer groundedness using automated metrics

The system uses a retriever over prebuilt guideline chunks, a generative LLM for response synthesis, and a small evaluation suite for model benchmarking.

## Installation

```bash
git clone <repo-url>
cd afrimed-tutor
python -m pip install -r requirements.txt
```

### Environment setup

```bash
copy .env.example .env
```

Edit `.env` and set the API keys and configuration values required by your provider(s):
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for completion
- `OPENAI_API_KEY` (or other embedding key) for embedding generation

### Required files and data

The repository includes scripts to build the retrieval corpus and prepare the AfriMed-QA dataset. These files are not tracked in source control because they are generated during setup.

## Usage

### Build the corpus

```bash
python corpus/build_corpus.py
```

This script:
- downloads guideline PDFs and raw sources defined in `corpus/sources.yaml`
- splits the text into retrieval chunks
- generates embeddings for each chunk
- builds a FAISS index for dense retrieval

### Load the dataset

```bash
python data/load_afrimedqa.py
```

This script loads the AfriMed-QA dataset and creates train/test splits used by the quiz and evaluation pipelines.

### Run the app

```bash
streamlit run app.py
```

The UI supports three main modes:
- `Ask`: free-text clinical questions with citation-backed answers
- `Quiz`: multiple-choice questions with correct answer explanation
- `Explain`: clinical scenarios with comparative reasoning feedback

## File structure

```text
afrimed-tutor/
├── app.py                    Streamlit UI entrypoint
├── corpus/                   Retrieval corpus builder and source metadata
│   ├── build_corpus.py       PDF → chunks → FAISS index pipeline
│   ├── chunks.jsonl          Generated chunk text (not tracked)
│   ├── chunks_meta.jsonl     Generated chunk metadata (not tracked)
│   ├── faiss.index          Index file for dense retrieval (not tracked)
│   ├── raw/                  Downloaded guideline sources
│   └── sources.yaml          Guideline source definitions
├── data/                    AfriMed-QA loader and dataset preparation
│   ├── afrimedqa_mcq_pool.jsonl
│   ├── afrimedqa_mcq_test.jsonl
│   ├── afrimedqa_saq.jsonl
│   ├── load_afrimedqa.py
│   └── question_embeddings.npy
├── eval/                    Evaluation scripts and judge prompts
│   ├── judge_prompts.py
│   ├── run_groundedness_judge.py
│   ├── run_mcq_eval.py
│   └── run_retriever_comparison.py
├── notebooks/                Analysis notebooks for corpus and evaluation
├── results/                 Generated evaluation outputs and summaries
├── tests/                   Unit tests for core modules
└── tutor/                   Core application logic and model orchestration
    ├── cli.py                CLI fallback interface
    ├── explain.py            SAQ/explanation logic
    ├── llm_client.py         Anthropic / OpenAI wrapper
    ├── orchestrator.py       Task routing and prompt orchestration
    ├── prompts.py            System and user prompt templates
    ├── quiz.py               MCQ loader and sampling
    ├── related_questions.py  Related question retrieval
    ├── retriever.py          Dense + sparse retrieval logic
    ├── schemas.py            Pydantic request/response models
```

## Configuration

All tunable values can be configured in `.env`.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `voyage` |
| `RETRIEVER_BACKEND` | `dense` | `dense` (FAISS) or `sparse` (BM25) |
| `RETRIEVER_TOP_K` | `5` | Number of chunks retrieved per query |
| `RETRIEVER_THRESHOLD` | `0.30` | Minimum similarity threshold for dense retrieval |
| `RELATED_Q_METHOD` | `semantic` | `semantic` or `keyword` retrieval for related questions |

## Evaluation

Run the available evaluation scripts after building the corpus and dataset:

```bash
python eval/run_mcq_eval.py
python eval/run_retriever_comparison.py
python eval/run_groundedness_judge.py
```

## Testing

Run the unit test suite with:

```bash
pytest tests/ -v
```

## Notes

- The project is intended for research and education, not clinical use.
- Generated files such as `corpus/faiss.index`, `corpus/chunks.jsonl`, `data/question_embeddings.npy`, and evaluation outputs are excluded from version control.

## License

MIT — see [LICENSE](LICENSE).
