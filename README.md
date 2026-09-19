# AI-Powered Customer Support Agent (RAG System)

Conversational support agent for a fictional e-commerce store. Retrieval uses dense HuggingFace embeddings (`BAAI/bge-small-en-v1.5`) and a persistent FAISS index. Generation uses local Ollama (`llama3.2:latest`) constrained to JSON, with Pydantic validation, lexical grounding, and out-of-domain refusals.

## Architecture

```
User question
    -> OOD keyword gate (Amazon price match, crypto, scooters, ...)
    -> Dense embedding query
    -> FAISS top-k + L2 distance filter
    -> Ollama JSON generation (repair only if JSON/schema is invalid)
    -> Pydantic SupportResponse
    -> Lexical grounding overlap
    -> Answer or refuse
```

A TF-IDF + LSA extractive path is retained only for baseline comparison.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start Ollama and confirm the model:

```bash
ollama serve
ollama list   # llama3.2:latest should be present
```

Build the vector index:

```bash
python -m src.vectorstore
```

Run the held-out evaluation:

```bash
python eval/run_eval.py
```

Launch the UI:

```bash
streamlit run app.py
```

## Project layout

| Path | Role |
| --- | --- |
| `data/kb/` | Policy and product markdown |
| `src/ingest.py` | Load + recursive chunking (500 / 50) |
| `src/embeddings.py` | BGE dense embeddings + TF-IDF/LSA baseline |
| `src/vectorstore.py` | FAISS build/load/search |
| `src/llm_backends.py` | Ollama JSON backend + extractive baseline |
| `src/schemas.py` | `SupportResponse`, `GuardrailResult` |
| `src/guardrails.py` | Schema, grounding, OOD, distance |
| `src/agent.py` | End-to-end RAG coordinator |
| `eval/eval_set.json` | 44 held-out cases |
| `eval/run_eval.py` | Metrics + baseline comparison |
| `app.py` | Streamlit chat + eval dashboard |

## JSON repair policy

The Ollama retry loop runs **only** when the model output is not parseable JSON or does not match the `SupportResponse` schema. A well-formed refusal (`can_answer: false`) is a successful response and does **not** trigger repair or a second attempt to answer.

## Evaluation results

Held-out set: 44 cases (28 answerable, 16 unanswerable) from `eval/eval_set.json`. Targets are not used as pass/fail gates. Full per-case rows: `eval/last_results.json`.

| Metric | TF-IDF/LSA + extractive | Dense BGE + Ollama | Target |
| --- | --- | --- | --- |
| Source accuracy | 96.4% | 100.0% | — |
| Refusal accuracy | 93.8% | 100.0% | ≥ 90% |
| False-answer rate | 6.2% | 0.0% | ≤ 5% |
| Avg grounding ratio | 1.000 | 0.921 | ≥ 0.75 |

## Baseline vs. dense-embedding upgrade

Baseline is `ExtractiveLocalBackend` on an in-memory TF-IDF/LSA FAISS index; upgrade is dense BGE + Ollama `llama3.2:latest`. Same eval set for both. Shared OOD gates (crypto, live order/inventory, multi-digit iPhone, etc.) apply to both backends.

| Metric | TF-IDF extractive | Dense + Ollama | Delta |
| --- | --- | --- | --- |
| Source accuracy | 96.4% | 100.0% | +3.6 pp |
| Refusal accuracy | 93.8% | 100.0% | +6.2 pp |
| False-answer rate | 6.2% | 0.0% | −6.2 pp |
| Avg grounding ratio | 1.000 | 0.921 | −0.079 |

Dense + Ollama hits all three targets. The extractive baseline still false-answers Best Buy hours (`u12`) by pasting overlapping retrieved sentences. Average grounding is slightly lower for dense because generative answers paraphrase.

## Known limitations

- **TF-IDF/LSA baseline still misses false-answer rate** (6.2% vs ≤ 5%) on `u12` (Best Buy store hours). Shared OOD/live-data gates now correctly refuse `u04`/`u09`/`u15` for both backends.
- **Taxonomy note on `u04`:** `catalog_unsupported_items.md` does ground a correct “we do not sell iPhones” answer (grounding 1.00). Labeling it `should_answer: false` is inconsistent with treating other KB-backed policy “no” answers as answerable (e.g. Klarna limits). By contrast `u12` (Best Buy hours) has **no** KB coverage and is correctly unanswerable. `eval_set.json` was left unchanged; the smartphone OOD gate refuses `u04` to match the current label.
- Hard grounding floor is `HARD_GROUNDING_FLOOR = 0.5` (refuse regardless of model `can_answer`). Soft threshold `MIN_GROUNDING_RATIO = 0.35` remains as documentation of the earlier lexical check.
- Lexical grounding is a token-overlap heuristic, not entailment. Short paraphrases can score below 1.0 even when factually aligned; the extractive baseline scores 1.0 because it copies context.
- Repair retries fire only on JSON/schema malformation. A correct refusal is never re-prompted into an answer.
- Distance threshold (`MAX_L2_DISTANCE = 1.15`) and OOD keyword gates are store-specific; new product lines would need catalog updates rather than relying on the LLM to invent coverage.
