# AiInfra

## Enterprise RAG System (local-first)

Modular PDF RAG service using Python 3.11, FastAPI, Qdrant and NVIDIA-hosted models. Services are containerized independently so the same images/configuration can be deployed to EC2. Keep the NVIDIA key in `.env` for local work and in a secret manager in production.

## Project layout

```text
rag-system/
├── frontend/                 # FastAPI-served chat UI
├── backend/
│   ├── ingestion/            # PDF extraction, OCR and fixed chunking
│   ├── retrieval/            # Qdrant storage, similarity search and rerank
│   ├── generation/           # Context-grounded answer generation
│   ├── monitoring/           # Prometheus metrics
│   ├── evaluation/            # JSONL evaluation metrics
│   ├── api/                  # FastAPI endpoints
│   └── core/                 # Settings, logging and NVIDIA client
├── data/                     # Place PDF files here
├── logs/                     # JSONL evaluation and source request database
├── docker/
├── tests/
├── docker-compose.yml
└── requirements.txt
```

## Local setup

1. Install Docker Desktop with Compose. For running Python directly, install Python 3.11, Tesseract OCR, and dependencies with `python -m pip install -r requirements.txt`.
2. Set `NVIDIA_API_KEY` in `.env`. Confirm the configured model IDs are enabled for your NVIDIA NIM account; all model names and the reranking URL are configurable.
3. Copy PDF files into `data/`.
4. Start everything: `docker compose up --build`
5. Open the Study Library at http://localhost:8501. Enter your NVIDIA API key in the UI to enable model-backed features. A key entered there is held in backend memory only and is not saved in the browser or written into `.env`; restarting the backend clears it. API docs are at http://localhost:8000/docs; Qdrant dashboard is at http://localhost:6333/dashboard.
6. Add PDFs through the Study Library upload form (50 MB maximum) or place them in `data/` and call `POST /ingest`. Uploads are indexed immediately. Ingestion is idempotent for stable document/page/chunk identifiers; successfully re-ingested PDFs also have stale chunks removed if the document shrinks.

The backend creates the `enterprise_rag` Qdrant collection on first successful document embedding, using the embedding vector dimension returned by NVIDIA and cosine similarity. Scanned-PDF OCR uses Tesseract; the backend container installs the binary. High-resolution extraction and table/image extraction use Unstructured. OCR fallback uses PyMuPDF rendering plus Tesseract and pdfplumber tables.

## Run without Docker

Start Qdrant separately (for example `docker compose up qdrant -d`), set `QDRANT_URL=http://localhost:6333`, and run:

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
python -m uvicorn frontend.main:app --host 0.0.0.0 --port 8501
```

Use `QDRANT_URL=http://localhost:6333`, `DATA_DIR=./data`, and `LOG_DIR=./logs` in `.env` for direct local Python runs. The checked-in `.env` defaults are Docker-oriented.

## API examples

```powershell
curl.exe -X POST http://localhost:8000/ingest
curl.exe -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the policy?\"}"
curl.exe http://localhost:8000/sources/<request_id>
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/metrics
curl.exe http://localhost:8000/collections
```

`/ask` returns an answer, confidence estimate, and source metadata with page, chunk, retrieval score, rerank score and text. Sources are also retrievable by `request_id`. An optional `expected_source_names` list on `/ask` enables labeled retrieval-hit-rate evaluation. Metrics are exposed in Prometheus text format. Evaluation metrics are JSONL in `logs/evaluation.jsonl`; the default evaluator uses lexical proxies, not semantic judge models.

## Operational notes

- Configure TLS, authentication/authorization, CORS allowlists, rate limiting, and secret management at the deployment boundary before exposing the API publicly.
- The local UI's NVIDIA key-setting endpoint is intentionally for local development and is not authenticated. Do not expose it publicly without adding authentication and access controls.
- The default Qdrant and SQLite volumes are local-development storage. Back up Qdrant and `logs/sources.sqlite3`; use managed storage/database when appropriate on AWS.
- The NVIDIA reranker API has an NVIDIA NIM-specific request schema. If your account exposes another schema/model ID, configure `NVIDIA_RERANK_URL` and adapt the small `NvidiaClient.rerank` adapter.
- Health checks validate Qdrant connectivity. NVIDIA API calls fail explicitly and return logged 502 errors from `/ask`; the system does not return fabricated fallback answers.
