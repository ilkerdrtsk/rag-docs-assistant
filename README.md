# RAG Docs Assistant

[![CI](https://github.com/ilkerdrtsk/rag-docs-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/ilkerdrtsk/rag-docs-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Claude](https://img.shields.io/badge/LLM-Claude-d97757)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Ask questions about your own documents and get short answers **with numbered citations**
back to the exact passages they came from. Built with **FastAPI**, **PostgreSQL + pgvector**
and **Claude**.

Example (shape of a response, shortened):

```text
Q: When will my refund arrive?
A: Refunds go back to your original payment method within 5 business days after the
   warehouse inspects the item [1].

[1] Returns and Refunds (part 3): "Refunds go back to the original payment method
    within 5 business days after the warehouse inspects the item."
```

## Highlights

- **Grounded answers with citations.** Each retrieved chunk is sent to Claude as a citable
  document. The citations Claude returns are turned into `[1]`-style markers plus a source
  list, so every claim can be checked.
- **No context, no guess.** If retrieval finds nothing relevant, the API answers "not found"
  without calling the model. That costs nothing and gives the model no chance to make
  something up.
- **Hybrid retrieval.** PostgreSQL full-text search (exact terms, product names, error codes)
  is combined with pgvector semantic search (paraphrases) using Reciprocal Rank Fusion. Without
  an embeddings key the app runs on keyword search alone.
- **Measured retrieval quality.** `python -m app.evaluate` scores retrieval on a labelled
  question set (hit@k and MRR) and runs in CI, so changes to chunking or search are judged
  by numbers.
- **Heading-aware chunking with overlap.** Each chunk keeps its Markdown section heading, and
  neighbouring chunks overlap, so answers split across a boundary are still found.
- **Production details:** duplicate uploads detected by content hash, optional API key, safe
  upload limits, clear `502` and `429` errors when a provider is down or busy, refusal
  handling with server-side model fallback, Alembic migrations, Docker, and CI against a real
  PostgreSQL with pgvector.

## How it works

```mermaid
flowchart LR
    Q["Question"] --> K["Keyword search<br/>PostgreSQL tsvector"]
    Q --> V["Vector search<br/>pgvector, optional"]
    K --> F["Reciprocal Rank Fusion"]
    V --> F
    F -->|top k chunks| G{"Anything found?"}
    G -->|no| N["Not found<br/>no model call"]
    G -->|yes| C["Claude<br/>chunks as citable documents"]
    C --> A["Answer + [n] markers + sources"]
```

Ingestion: `POST /documents` → split into heading-aware, overlapping chunks → optional
embeddings (Voyage AI) → stored with a generated `tsvector` column (GIN index) and a
`vector(1024)` column (HNSW index).

## Quick start

Requirements: Docker and a [Claude API key](https://platform.claude.com/settings/keys).

```bash
git clone https://github.com/ilkerdrtsk/rag-docs-assistant.git
cd rag-docs-assistant
cp .env.example .env               # put your key in ANTHROPIC_API_KEY
docker compose up --build
```

The app migrates the database, loads three sample documents (returns, shipping and warranty
policies of a fictional outdoor shop) and serves the API on <http://localhost:8000>.
Interactive docs are at <http://localhost:8000/docs>.

```bash
curl -s localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Is a broken zipper covered by the warranty?"}'

# Retrieval only, without calling Claude: see what the model would be given
curl -s localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "express shipping cost", "top_k": 3}'

# Add your own documents
curl -s localhost:8000/documents/upload -F "file=@handbook.md"
```

Without `ANTHROPIC_API_KEY`, documents and `/search` still work, and `/ask` answers
`502 llm_unavailable`. Set `VOYAGE_API_KEY` to turn on semantic search. Documents added
before that can be embedded afterwards with `python -m app.reindex`.

## API

| Method | Path | Description |
| --- | --- | --- |
| POST | `/ask` | Answer a question with citations |
| POST | `/search` | Retrieved chunks with their keyword and vector ranks (no model call) |
| POST | `/documents` | Add a document from JSON (`title`, `text`, `source`) |
| POST | `/documents/upload` | Upload a `.txt` or `.md` file (UTF-8, up to 2 MB) |
| GET | `/documents` | List documents |
| DELETE | `/documents/{id}` | Delete a document and its chunks |
| GET | `/health` | Database check and search mode (`keyword` or `hybrid`) |

A repeated upload of identical content returns `200` with `"created": false` instead of
adding duplicate chunks. When `API_KEY` is set, every endpoint except `/health` requires the
`X-API-Key` header.

## The Claude call

`app/answering.py` is the only place that talks to Claude:

- Model `claude-opus-5-5` with `effort: medium`. Both are configurable.
- One `document` block per retrieved chunk with `citations: {"enabled": true}`, followed by
  the question, in a single user turn. The system prompt tells the model to answer only from
  those documents and to say so when they do not cover the question.
- Server-side fallback (`fallbacks: "default"`): if a safety classifier declines, the API
  retries on a fallback model within the same call. A final refusal is reported as such.
- Errors map to `429 llm_rate_limited` or `502 llm_unavailable`, never to an unhandled 500.

## Evaluating retrieval

```bash
python -m app.seed
python -m app.evaluate --k 3
```

```text
 rank 1  How many days do I have to return something?  ->  Returns and Refunds
 rank 1  How much does express shipping cost?  ->  Shipping
 rank 1  My zipper broke after a year, is that covered?  ->  Warranty
 ...
hit@3: 100%   MRR: 1.00   (9 questions)
```

Add your own questions to `evals/retrieval.jsonl` (`{"question": ..., "expected": "<document title>"}`).

## Development

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
docker compose up -d db            # PostgreSQL 16 with pgvector
alembic upgrade head
python -m app.seed
uvicorn app.main:app --reload
```

Checks that CI runs:

```bash
ruff check . && ruff format --check .
alembic upgrade head && alembic check
pytest --cov
```

The tests use a separate `<database>_test` database. They never call Claude or Voyage: Claude
responses are built with the SDK's own response types, and the Voyage client is tested
against a mocked HTTP transport.

## Project layout

```text
app/
  chunking.py    heading-aware chunking with overlap
  embeddings.py  optional Voyage AI embeddings
  retrieval.py   full-text + vector search, Reciprocal Rank Fusion
  answering.py   the Claude call, citations to [n] markers
  ingest.py      storing documents, duplicate detection, backfilling embeddings
  evaluate.py    retrieval eval (hit@k, MRR)
  api/           FastAPI routes and dependencies
samples/         demo documents
evals/           labelled retrieval questions
migrations/      Alembic migrations (enables the vector extension)
tests/
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+asyncpg://rag:rag@127.0.0.1:5432/rag` | PostgreSQL with pgvector |
| `ANTHROPIC_API_KEY` | | Claude API key |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used for answers |
| `CLAUDE_EFFORT` | `medium` | `low`, `medium`, `high`, `xhigh` or `max` |
| `VOYAGE_API_KEY` | | Turns on semantic search |
| `VOYAGE_MODEL` | `voyage-4` | Embedding model (1024 dimensions) |
| `TOP_K` | `6` | Chunks given to the model per question |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Chunking, in characters |
| `API_KEY` | | When set, required in the `X-API-Key` header |

## License

[MIT](LICENSE)
