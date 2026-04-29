# Cost and Latency Report

_Populate this file after running evaluations (§7.4)._

## Summary table

| Condition | N | Mean input tokens | Mean output tokens | p50 latency (ms) | p95 latency (ms) | Est. cost / 100 interactions |
|---|---|---|---|---|---|---|
| Baseline (no RAG) | 100 | — | — | — | — | — |
| RAG (dense) | 100 | — | — | — | — | — |
| RAG (sparse) | 100 | — | — | — | — | — |

_Run `notebooks/02_evaluation_summary.ipynb` to fill this table from the CSV results._

## Embedding costs

| Provider | Model | Dimensionality | Price per 1M tokens | Total corpus tokens (est.) | Total cost (est.) |
|---|---|---|---|---|---|
| OpenAI | text-embedding-3-small | 1536 | $0.02 | — | — |

## Notes

- LLM prices based on published API rates at time of evaluation (April 2026).
- Token counts include both prompt (system + user) and completion tokens.
- Latency measured end-to-end from API call to response receipt.
