# RAG Server

A FastAPI-based Retrieval-Augmented Generation (RAG) server with Qdrant vector database integration.

## Dependencies

- **fastapi[standard]==0.118.0** - Web framework
- **pydantic-settings==2.11.0** - Settings management
- **qdrant-client==1.15.1** - Vector database client

## Project Structure

```
rag-server/
├── app/
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── core/
│   │   └── config.py          # Configuration and settings
│   ├── db/
│   │   └── qdrant_client.py   # Qdrant client and collection setup
│   ├── services/
│   │   ├── chunker.py         # Text chunking logic
│   │   ├── embedding.py       # Embedding generation
│   │   └── retriever.py       # Document retrieval
│   └── main.py                # FastAPI application entry point
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd rag-server
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:
- `ENVIRONMENT` - Application environment (development/production)
- `QDRANT_URL` - Qdrant server URL
- `QDRANT_API_KEY` - Qdrant API key

### 5. Run the server

```bash
fastapi dev app/main.py
```

The server will start at `http://localhost:8000`

## API Endpoints

## Development

### Running in development mode

```bash
fastapi dev app/main.py
```

### Running in production mode

```bash
fastapi run app/main.py
```

## API Endpoints

see `localhost:8000/docs`
