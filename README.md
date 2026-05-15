# RAG Server

[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Froger-twan%2Frag-server%2Fmain%2Fpyproject.toml&query=%24.project.version&label=version&color=blue)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.13-pink.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-8.1.0-orange.svg)](https://www.pinecone.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.23-32CD32.svg)](https://python.langchain.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12.29-F9D072.svg)](https://www.llamaindex.ai/)
[![Tests](https://img.shields.io/badge/pytest-9.0.2-green.svg)](https://pytest.org/)

A FastAPI-based Retrieval-Augmented Generation (RAG) server with **PostgreSQL-backed context storage**, **Pinecone vector retrieval**, **conversation-aware query rewriting**, **streaming responses**, **incremental ingestion**, and **multi-source data loaders**.

## Architecture Overview

```mermaid
flowchart LR
    subgraph Sources["Data Sources"]
        GH["GitHub Repos"]
        WEB["Website"]
        NOTES["Notes"]
    end

    subgraph Pipeline["Ingestion Pipeline"]
        LOADERS["Loaders<br/>(GitHub API, Sitemap)"]
        CHUNK["LlamaIndex<br/>Text Chunking"]
        EMBED["OpenAI Embeddings<br/>text-embedding-3-small"]
        BM25["Optional BM25 Sparse<br/>Encoding"]
    end

    subgraph Storage["Storage"]
        PG[("PostgreSQL<br/>Documents, Chunks,<br/>Conversations, Traces")]
        PINE[("Pinecone<br/>Vector Index<br/>chunk_id + preview")]
    end

    subgraph Retrieval["Retrieval & Generation"]
        REWRITE["Query Rewrite<br/>(Conversation Context)"]
        SEARCH["Dense Search<br/>(Optional Hybrid)"]
        EXPAND["Fetch Full + Neighbor Chunks<br/>from PostgreSQL"]
        RERANK["Cohere Rerank<br/>rerank-v3.5"]
        ANSWER["Gemini/DeepSeek/OpenAI<br/>Answer Generation + Streaming"]
    end

    GH --> LOADERS
    WEB --> LOADERS
    NOTES --> LOADERS
    
    LOADERS --> CHUNK
    CHUNK --> EMBED
    CHUNK --> BM25

    CHUNK --> PG
    EMBED --> PINE
    BM25 --> PINE

    REWRITE --> SEARCH
    PINE --> SEARCH
    SEARCH --> EXPAND
    PG --> EXPAND
    EXPAND --> RERANK
    RERANK --> ANSWER
    ANSWER --> PG
```

## Technology Stack

### Core Framework
- **FastAPI** - Modern async web framework
- **LlamaIndex** - Document chunking and processing
- **LangChain** - LLM integration and orchestration

### Vector Database & Search
- **Pinecone** - Managed vector database for retrieval
  - Dense vectors: OpenAI `text-embedding-3-small` (1024d)
- **PostgreSQL** - Persistent context and trace storage
  - Full document chunks
  - Conversation and message history
  - Retrieval traces for debugging
- **Optional sparse search** - BM25 sparse vectors behind `ENABLE_SPARSE_SEARCH`

### LLM & Embeddings
- **OpenAI** - `text-embedding-3-small` for embeddings, `gpt-4o` for generation
- **Cohere** - `rerank-v3.5` for result reranking
- **Google Gemini** - Answer generation
- **DeepSeek** - Alternative LLM provider

### Data Loaders
- **LlamaIndex** - Document chunking and processing
- **GitHub API** - Repository metadata, README, dependency files
- **Website Crawler** - Sitemap-based page ingestion

### Development & CI/CD
- **Docker Compose** - Local PostgreSQL development database
- **Pre-commit** - Black, isort, flake8, pytest for changed files
- **GitHub Actions** - PR checks for format, lint, and tests
- **pytest** - Async test support with mocking

## Key Improvements

### 1. Smart Ingestion (Incremental Updates)

**Problem:** Re-indexing would delete entire namespaces and re-process all documents, wasting API calls and time.

**Solution:** Implemented content hash tracking for incremental updates:

```python
# Deterministic document ID
# Each document gets a unique, deterministic ID based on source + path
doc_id = compute_doc_id("github_repos", "my-repo")

# Content hash tracking
# Only changed documents trigger re-embedding
if old_hash == new_hash:
    skip_unchanged()  # No API call needed
else:
    update_document()  # Only update changed docs
```

**Benefits:**
- 90%+ reduction in embedding API calls for unchanged content
- Granular deletion - remove single documents without clearing namespace
- Traceable metadata: `doc_id`, `doc_hash`, `chunk_hash`, `last_updated`

### 2. PostgreSQL-Backed Context Storage

**Problem:** Pinecone metadata is not a reliable full-text document store. Large chunks can be truncated, conversation state is hard to persist, and retrieval debugging needs more than vector metadata.

**Solution:** Store full documents, chunks, conversations, messages, and retrieval traces in PostgreSQL:

```text
documents
chunks
conversations
messages
retrieval_traces
```

**Benefits:**
- Pinecone stores only retrieval metadata such as `chunk_id`, `text_preview`, `doc_id`, and `chunk_index`
- PostgreSQL stores the full chunk text
- Retrieval can expand a matched chunk with neighboring chunks from the same document
- Conversation history and retrieval traces are persisted for follow-up questions and debugging

### 3. Conversation-Aware Query Rewriting

**Problem:** A single query string cannot resolve follow-up questions like "what database does it use?" without chat history.

**Solution:** `/api/query` accepts an optional `conversation_id`. The service stores messages in PostgreSQL and rewrites the latest user question into a standalone retrieval query before searching.

**Example:**
```text
User: What framework does this project use?
Assistant: ...
User: What database does it use?
Rewritten query: What database does the RAG server project use?
```

### 4. Streaming RAG Responses

**Problem:** Waiting for the full LLM response makes RAG queries feel slow even when the model has already started generating.

**Solution:** `/api/query/stream` returns Server-Sent Events (SSE), so clients can render answer tokens as they arrive.

```text
event: metadata
data: {"conversation_id": "...", "rewritten_query": "...", "sources": [...]}

event: token
data: {"content": "The"}

event: token
data: {"content": " answer"}

event: done
data: {"answer": "The answer..."}
```

**How it works:**
- Retrieval happens before generation: relevant chunks are fetched, expanded, reranked, and passed to the LLM as one formatted context
- Streaming applies to LLM output tokens, not to the retrieved document chunks
- The final `done.answer` is assembled from the same token events sent during the stream
- The completed assistant message and retrieval traces are persisted after generation finishes

### 5. Dense Search with Optional Hybrid Search

**Default:** Dense retrieval is enabled by default and works with standard Pinecone dense indexes.

**Optional:** BM25 sparse vectors can be enabled with:

```text
ENABLE_SPARSE_SEARCH=true
```

Sparse search requires a Pinecone index configuration that supports sparse values. If the index does not support sparse query values, retrieval falls back to dense-only search.

### 6. Improved Document Tracking

**Deterministic IDs:**
```python
# Normal: random UUIDs
"id": "550e8400-e29b-41d4-a716-446655440000"

# Improved: deterministic from source + path
"id": "a3f9b2c1d8e7_0"  # hash("github_repos:my-repo") + chunk_index
```

**Metadata Enrichment:**
```json
{
  "doc_id": "a3f9b2c1d8e7",
  "doc_hash": "sha256:abc123...",
  "chunk_hash": "sha256:def456...",
  "chunk_index": 0,
  "total_chunks": 5,
  "source": "github_repos",
  "repo": "my-project",
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### 7. Granular Document Operations

**Delete by Document ID:**
```python
# Normal: clear entire namespace
clear_namespace("github_repos")  # Deletes everything!

# Improved: delete single document
delete_document("a3f9b2c1d8e7", "github_repos")  # Only that repo
```

**Query Document Chunks:**
```python
# Get all chunks belonging to a specific document
chunks = await get_document_chunks(doc_id, namespace)
```

### 8. Markdown Blog Post Optimization

Special handling for markdown blog posts:

**Frontmatter Parsing:**
- Extracts YAML metadata
- Supports Obsidian-style frontmatter format

**Semantic Header-Based Chunking:**
- Splits by Markdown headers (`#`, `##`, `###`) instead of fixed size
- Preserves document structure and context
- Merges small sections, splits large ones at paragraph boundaries

**Context Enrichment for Embeddings:**
```
Document: My Blog Title
Section: Technical > Python
Subsection: Async Programming

[actual content...]
```

## Project Structure

```
rag-server/
├── app/
│   ├── api/
│   │   ├── routes.py            # REST endpoints
│   │   └── webhooks.py          # GitHub webhook handler
│   ├── core/
│   │   ├── config.py            # Pydantic settings
│   │   └── events.py            # FastAPI lifespan (startup/shutdown)
│   ├── db/
│   │   ├── pinecone.py          # Pinecone client & BM25 encoder
│   │   └── postgres.py          # SQLAlchemy models and persistence helpers
│   ├── loaders/
│   │   ├── github_loader.py     # GitHub repos + dependency parsing
│   │   └── website_loader.py    # Website sitemap crawler
│   ├── indexers/
│   │   └── vector_indexer.py    # Chunking, embedding, upsert logic
│   ├── services/
│   │   ├── answer_generator.py  # LLM answer generation (Google/OpenAI/DeepSeek)
│   │   ├── chunker.py           # Simple text chunking
│   │   ├── embedding.py         # OpenAI embeddings
│   │   ├── frontmatter_parser.py# Markdown frontmatter extraction
│   │   ├── ingestion.py         # Smart upsert/delete
│   │   ├── markdown_chunker.py # Markdown-aware chunking
│   │   └── retriever.py         # Dense search, context expansion, reranking
│   ├── utils/
│   └── main.py                  # FastAPI app factory
├── tests/
│   ├── test_vector_indexer.py   # Hash, doc_id, upsert tests
│   ├── test_ingestion.py        # Ingestion service tests
│   ├── test_github_loader.py    # Dependency parser tests
│   ├── test_sync_jobs.py        # Sync job tests
│   └── test_api.py              # API endpoint tests
├── scripts/
│   └── run_changed_tests.py     # Pre-commit test runner
├── .github/workflows/ci.yml     # PR checks (format, lint, test)
└── .pre-commit-config.yaml      # Local pre-commit hooks
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/query?q={question}` | Query RAG system |
| GET | `/api/query?q={question}&conversation_id={id}` | Continue a conversation-aware query |
| GET | `/api/query/stream?q={question}` | Stream a RAG answer with Server-Sent Events |
| POST | `/api/ingest/website` | Sync website content |
| POST | `/api/ingest/github-all-repos` | Sync all GitHub repos |
| POST | `/api/ingest/notes` | Sync notes repo (blog) manually |
| POST | `/api/webhooks/github` | GitHub push webhook |

Query responses include:

```json
{
  "query": "What database does it use?",
  "result": "...",
  "conversation_id": "8e5c...",
  "rewritten_query": "What database does the RAG server project use?",
  "sources": []
}
```

Streaming query responses use `text/event-stream` and emit:

| Event | Payload | Description |
|-------|---------|-------------|
| `metadata` | `conversation_id`, `rewritten_query`, `sources` | Sent before answer generation starts |
| `token` | `content` | A generated answer fragment from the LLM |
| `done` | `answer` | The complete answer assembled from all token events |

The streaming endpoint uses the same public API token header as `/api/query`.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/roger-twan/rag-server.git
cd rag-server
uv sync --dev
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys:
# - ENVIRONMENT=development
# - DATABASE_URL=postgresql://rag:rag@localhost:5432/rag_server_db
# - PINECONE_API_KEY
# - PINECONE_INDEX_HOST
# - ENABLE_SPARSE_SEARCH=false
# - OPENAI_API_KEY
# - COHERE_API_KEY
# - GOOGLE_API_KEY
# - GITHUB_TOKEN (for repo loading)
# - GITHUB_WEBHOOK_SECRET
# - DEEPSEEK_API_KEY (for DeepSeek LLM)
# - PUBLIC_API_TOKEN (for /query endpoint request)
# - ADMIN_API_TOKEN (for ingest endpoints request)
```

### 3. Start local Postgres

```bash
docker compose up -d postgres
```

The local database uses:

```text
host: localhost
port: 5432
database: rag_server_db
user: rag
password: rag
```

These credentials are for local development only. Production deployments should use secret-managed credentials and a different password.

### 4. Install pre-commit hooks

```bash
pre-commit install
```

### 5. Run server

```bash
# Development
uv run fastapi dev app/main.py
```

### 6. Ingest data

Ingestion requires the admin API token:

```bash
curl -X POST "http://127.0.0.1:8000/api/ingest/notes" \
  -H "X-Api-Token: $ADMIN_API_TOKEN"

curl -X POST "http://127.0.0.1:8000/api/ingest/website" \
  -H "X-Api-Token: $ADMIN_API_TOKEN"

curl -X POST "http://127.0.0.1:8000/api/ingest/github-all-repos" \
  -H "X-Api-Token: $ADMIN_API_TOKEN"
```

Ingestion writes vector metadata to Pinecone and full document chunks to PostgreSQL.

### 7. Query locally

```bash
curl "http://127.0.0.1:8000/api/query?q=What%20framework%20does%20this%20project%20use%3F" \
  -H "X-Api-Token: $PUBLIC_API_TOKEN"
```

For follow-up questions, pass the `conversation_id` returned by the previous response:

```bash
curl "http://127.0.0.1:8000/api/query?q=What%20database%20does%20it%20use%3F&conversation_id=<conversation_id>" \
  -H "X-Api-Token: $PUBLIC_API_TOKEN"
```

To stream an answer with Server-Sent Events:

```bash
curl -N "http://127.0.0.1:8000/api/query/stream?q=What%20framework%20does%20this%20project%20use%3F" \
  -H "X-Api-Token: $PUBLIC_API_TOKEN"
```

For a streaming follow-up question, pass the same `conversation_id`:

```bash
curl -N "http://127.0.0.1:8000/api/query/stream?q=What%20database%20does%20it%20use%3F&conversation_id=<conversation_id>" \
  -H "X-Api-Token: $PUBLIC_API_TOKEN"
```

Example stream:

```text
event: metadata
data: {"conversation_id": "8e5c...", "rewritten_query": "What database does the RAG server project use?", "sources": [...]}

event: token
data: {"content": "It"}

event: token
data: {"content": " uses PostgreSQL"}

event: done
data: {"answer": "It uses PostgreSQL..."}
```

The retrieved RAG chunks are sent to the LLM up front as context. The `token` events are generated answer fragments, and `done.answer` is the concatenation of those fragments.

### Reset local data

Clear PostgreSQL data:

```bash
docker compose exec postgres psql -U rag -d rag_server_db -c \
"TRUNCATE retrieval_traces, messages, conversations, chunks, documents RESTART IDENTITY CASCADE;"
```

Clear Pinecone namespaces:

```bash
uv run python -c "from app.db.pinecone import clear_namespace; [clear_namespace(ns) for ns in ('github_notes', 'github_repos', 'website_roger_ink')]"
```

Then rerun the ingestion endpoints.

Clean up old conversation history and retrieval traces:

```bash
uv run python -c "from app.db.postgres import cleanup_conversations_older_than; print(cleanup_conversations_older_than(30))"
```

## Development

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_vector_indexer.py -v

# Tests for changed files only (via pre-commit)
uv run python scripts/run_changed_tests.py
```

### Code Quality

```bash
# Format code
uv run black .
uv run isort .

# Lint
uv run flake8 app tests scripts

# Run pre-commit manually
uv run pre-commit run --all-files
```

## CI/CD Pipeline

**Pull Request Checks:**
1. **Black** - Code formatting
2. **isort** - Import sorting  
3. **flake8** - Linting
4. **pytest** - Full test suite

All checks must pass before merging to `main`.

## Change Log

### 1.3.0 (2026-10-16)
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

## TODO
- [x] Add request rate limiting and authentication (v1.1.0)
- [ ] Support sync specific GitHub repos
- [ ] Complete sync blog by GitHub Push Webhook
- [x] Add chat memory (v1.2.0)
- [ ] Add evaluation strategy
<<<<<<< Updated upstream
- [x] Support streaming responses
=======
- [x] Support streaming responses (v1.3.0)
>>>>>>> Stashed changes
