# Build Logs

### 1.4.0 (2026-07-07)
- Added prompt template management in `app/services/prompts.py`
- Added LangSmith tracing configuration for RAG runs
- Added local RAG evaluation dataset and deterministic eval runner
- Added LangSmith dataset sync and experiment runner
- Added RAGAS evaluation with faithfulness, answer relevancy, context precision, and context recall metrics
- Added RAG evaluation strategy documentation and production quality gates
- Improved GitHub ingestion resilience with shared HTTP client, timeout settings, and retries
- Added Cohere rerank fallback to vector search order when rerank is unavailable

### 1.3.2 (2026-05-16)
- Fixed README.md wrong change log dates

### 1.3.1 (2026-05-16)
- Fixed README.md conflict

### 1.3.0 (2026-05-16)
- Added `/api/query/stream` for Server-Sent Events streaming responses
- Documented streaming event types: `metadata`, `token`, and `done`

### 1.2.1 (2026-05-15)
- Added fallback prompt behavior for greetings and simple small talk when no RAG context is found

### 1.2.0 (2026-05-15)
- Added PostgreSQL persistence for documents, chunks, conversations, messages, and retrieval traces
- Added local PostgreSQL Docker Compose setup
- Changed Pinecone metadata to store chunk previews and identifiers instead of full chunk text
- Added conversation-aware query rewriting with `conversation_id`
- Added full chunk and neighboring chunk retrieval from PostgreSQL
- Added `ENABLE_SPARSE_SEARCH` to keep sparse/hybrid search optional per Pinecone index configuration
- Added PostgreSQL conversation cleanup helper for retention jobs

### 1.1.1 (2026-05-13)
- Switched dependency management to `pyproject.toml` and `uv.lock`
- Removed generated `requirements*.txt` files
- Updated Docker, CI, and development docs to use `uv`
- Added a dynamic README version badge sourced from `pyproject.toml`

### 1.1.0 (2026-04-03)
- Added API token authentication
- Added request rate limiting
