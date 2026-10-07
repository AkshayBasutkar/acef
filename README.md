# ACEF — Agentic Context Engineering Framework

ACEF is a lean, modular context engineering architecture designed for AI coding agents and technical assistants. It combines semantic caching, structured GraphRAG for codebases, and vector-based MemRAG for document and conversational memory.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────┐
│  1. Cache Check │──── hit ────► Return cached response (0ms retrieval)
└────────┬────────┘
         │ miss
         ▼
┌─────────────────┐
│  2. Classify    │────► SIMPLE / CODE / MEMORY
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    ▼         ▼              ▼
 SIMPLE     CODE           MEMORY
 (skip)   GraphRAG         MemRAG
          (Neo4j)       (ChromaDB)
    │        │               │
    └────┬───┴───────────────┘
         ▼
┌─────────────────┐
│   4. Call LLM   │  (Gemini or Groq via OpenAI client)
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. Respond &    │
│    Write-back   │  (Store in cache; append to session memory if MEMORY)
└─────────────────┘
```

---

## Tech Stack

- **API Gateway**: FastAPI + Uvicorn
- **Code Graph (GraphRAG)**: Neo4j (Docker) + Python `ast` parser
- **Vector Store (MemRAG & Cache)**: Embedded ChromaDB (persistent local storage)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Gemini (`gemini-2.0-flash`) or Groq (`llama-3.3-70b-versatile`) via unified OpenAI client
- **Testing**: `pytest`, `pytest-asyncio`, `httpx`

---

## Quickstart

### 1. Prerequisites & Environment
Ensure Python 3.11+ and Docker Desktop are running.

```bash
# Clone and enter directory
cd acef

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and add your GEMINI_API_KEY or GROQ_API_KEY
```

### 2. Start Neo4j
```bash
docker compose up -d
```
Neo4j browser will be available at [http://localhost:7474](http://localhost:7474) (credentials: `neo4j` / `acef_password`).

### 3. Initialize Graph Schema
```bash
python -m scripts.init_neo4j
```
This idempotently creates the `symbolSearch` fulltext index and `file_path_unique` constraint.

### 4. Verify System Health
```bash
python -m scripts.smoke_test
```

### 5. Launch the API Gateway
```bash
uvicorn api.gateway:app --reload
```
Swagger UI documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## API Endpoints & Usage

### 1. Ingest Codebase (GraphRAG)
```bash
curl -X POST http://localhost:8000/ingest/code \
  -H "Content-Type: application/json" \
  -d '{"root_path": "./data/demo_codebase"}'
```

### 2. Ingest Document (MemRAG)
```bash
curl -X POST http://localhost:8000/ingest/document \
  -F "file=@data/demo_documents/pricing.md"
```

### 3. Execute Queries

- **General / Simple query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

- **Code query (triggers GraphRAG):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the login function call in auth.py?"}'
```

- **Document query (triggers MemRAG):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the pricing tiers in the document?"}'
```

- **Semantic Cache Hit (repeat query):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
# -> from_cache: true
```

---

## Running the Test Suite

```bash
# Run unit tests (57 tests)
python -m pytest tests/unit -v

# Run integration tests
python -m pytest tests/integration -v
```
