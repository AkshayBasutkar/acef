# ACEF Project Presentation Content

## Purpose and evidence base

This document is a slide-ready content outline for the **ACEF** project. It was prepared from the current source code, dashboard, configuration, test suite, evaluation data, `README.md`, and `progress.md` in this repository.

**Naming note:** the README expands ACEF as **Agentic Context Engineering Framework**, while the web dashboard calls it **Adaptive Context Engineering Framework**. Use one expansion consistently in the final deck. This outline uses **Agentic Context Engineering Framework**, which matches the README and repository heading.

**Suggested deck length:** 19 slides, plus the optional appendix.

---

## Slide 1. Title

### Agentic Context Engineering Framework (ACEF)

**A hybrid retrieval architecture for technical AI assistants**

- Project presentation
- Prepared by: *[Name(s), USN, department, guide]*
- Institution: *[Institution name]*
- Date: *[Presentation date]*

**Visual:** use a clean screenshot of the ACEF Web Studio dashboard or a simple pipeline graphic.

---

## Slide 2. Introduction

- Technical assistants need the right context before they can give reliable answers.
- A general question does not need local retrieval, but a question about a repository needs code-aware evidence.
- Questions about uploaded documents or previous turns need persistent memory with session isolation.
- ACEF combines semantic caching, code-graph retrieval, document retrieval, and LLM generation behind one API and web interface.

**Key idea:** route each query to the smallest context source that can answer it accurately.

**Visual:** three inputs labelled General question, Codebase question, and Document or conversation question leading to ACEF.

---

## Slide 3. Problem Statement

### Problem statement

Most LLM applications send every prompt directly to the language model or attach large amounts of irrelevant context. This increases latency and cost, can dilute relevant evidence, and does not reliably answer questions about source code, uploaded documents, or prior conversation.

ACEF addresses this problem by building a context-engineering layer that:

- recognizes whether a query needs no retrieval, code retrieval, or memory retrieval
- retrieves repository structure through a knowledge graph for code-specific questions
- retrieves document and session evidence through a vector store for memory-specific questions
- reuses safe answers through a guarded semantic cache
- returns a structured response with source identifiers, route, cache status, and latency

**Visual:** a comparison table: Direct LLM versus ACEF. Suggested rows: context selection, source traceability, session continuity, repeat-query handling.

---

## Slide 4. Objectives and Scope

### Objectives

- Build a modular, locally deployable framework for context-aware AI assistance.
- Support three query routes: `SIMPLE`, `CODE`, and `MEMORY`.
- Build GraphRAG from Python source code using AST parsing and Neo4j relationships.
- Support `.md`, `.txt`, and text-based `.pdf` document ingestion with vector retrieval.
- Preserve conversation continuity with per-session memory.
- Reduce repeated work through semantic caching without reusing an answer for the wrong entity, number, plan, file, or identifier.
- Provide REST endpoints and a visual dashboard for ingestion, querying, graph inspection, health monitoring, and cache management.

### Current scope

- Python codebases for AST ingestion
- Neo4j as the only external service
- ChromaDB embedded locally for vector memory and cache
- Gemini or Groq through OpenAI-compatible APIs

---

## Slide 5. High-Level Design

### Five-step query lifecycle

```text
User query
    |
1. Semantic cache lookup
    |-- valid hit --> cached response
    |
2. Query classification
    |-- SIMPLE --> no retrieval
    |-- CODE   --> GraphRAG on Neo4j
    |-- MEMORY --> MemRAG on ChromaDB
    |
3. Retrieve relevant context when required
    |
4. Generate answer with Gemini or Groq
    |
5. Return structured response and write back state
    |-- cache successful SIMPLE answers
    |-- record successful turns in session memory
```

**Design principle:** retrieval failures do not terminate the request. The pipeline logs the failure and can continue with an LLM call without retrieved context.

**Visual:** recreate the lifecycle as a horizontal pipeline with a three-way branch after classification.

---

## Slide 6. Component Design

| Component | Responsibility | Main technology |
|---|---|---|
| FastAPI gateway | Exposes query, ingestion, health, graph, and cache operations | FastAPI, Uvicorn |
| Pipeline | Coordinates cache, routing, retrieval, generation, and write-back | Python |
| RAGCacheLite | Finds semantically equivalent SIMPLE queries safely | ChromaDB, SentenceTransformers |
| Query classifier | Chooses SIMPLE, CODE, or MEMORY | Heuristic rules by default, optional LLM router |
| GraphRAG | Retrieves related code symbols and relationships | Python AST, Neo4j |
| MemRAG | Retrieves uploaded-document chunks and session turns | ChromaDB |
| LLM client | Generates grounded responses | Gemini or Groq through OpenAI SDK |
| Web Studio | Provides ingestion controls, route telemetry, chat, and graph view | HTML, CSS, JavaScript |

**Speaker note:** all large reusable services are initialized once by the API gateway. The embedding model loads lazily and is shared.

---

## Slide 7. Query Routing Design

### Three-way classification

| Route | When ACEF selects it | Context source |
|---|---|---|
| `SIMPLE` | The current query can be answered without local evidence | No retrieval |
| `CODE` | The answer depends on repository files, symbols, definitions, dependencies, or call relationships | Neo4j GraphRAG |
| `MEMORY` | The answer depends on uploaded documents or earlier turns in the current session | ChromaDB MemRAG |

### Routing signals

- Explicit repository or file references, code-navigation questions, and known indexed symbols support the CODE route.
- Explicit references to uploaded documents or prior discussion support the MEMORY route.
- A document-similarity probe supports implicit document questions such as pricing queries.
- A short follow-up after at least three turns can use session memory, unless it clearly continues a code route.
- Generic programming questions and pasted self-contained code remain SIMPLE. Technical vocabulary alone does not force code retrieval.

### Classifier options

- **Heuristic:** default mode, deterministic, no API cost, no routing latency.
- **LLM classifier:** uses a small provider-specific model at temperature 0 and falls back to the heuristic classifier on any failure.
- **Local classifier:** reserved for a future fine-tuned MiniLM implementation.

---

## Slide 8. GraphRAG for Code Understanding

### Code ingestion and retrieval

- ACEF parses Python files with the standard `ast` module.
- It creates `File`, `Class`, and `Symbol` nodes in Neo4j.
- It records relationships such as `DEFINES`, `CONTAINS`, `IMPORTS`, and `CALLS`.
- Symbols retain qualified names, source locations, docstrings, parent class details, and bounded source-code excerpts.
- Code retrieval extracts query terms, performs safe Lucene full-text matching, and traverses the graph up to the configured maximum of two hops.
- Ingestion replaces prior data for the same codebase root, which avoids stale graph records and supports isolation between multiple repositories.

### Example

```text
main.py: main()
    calls
auth.py: login()
    calls
utils.py: hash_password()
```

**Visual:** show this call chain with file and function labels. Use a live dashboard graph screenshot after ingestion.

---

## Slide 9. MemRAG and Session Memory

### Document and conversation retrieval

- Supports Markdown, plain text, and text-extractable PDF documents.
- Splits documents into approximately 300-word chunks with a 50-word overlap.
- Stores document chunks with stable source metadata in ChromaDB.
- Re-ingesting a document removes its older chunks before upsert, preventing stale content from remaining after an update.
- Keeps document retrieval and conversation retrieval separate, then merges relevant candidates by distance.
- Maintains separate retrieval quotas so conversation history cannot crowd out uploaded-document evidence.
- Filters low-relevance chunks using a configurable similarity floor of 0.30.
- Records successful responses in session memory using a session UUID.

### Important reliability settings

- Document-route similarity threshold: **0.33**
- Memory relevance floor: **0.30**
- Default retrieved memory chunks: **5**
- Document evidence takes priority when it conflicts with prior conversation.

**Visual:** two ChromaDB partitions labelled Uploaded documents and Session turns joining before the LLM.

---

## Slide 10. Semantic Cache and Response Safety

### RAGCacheLite

- Stores only successful SIMPLE-query answers because code and memory answers can depend on changing repository or session state.
- Uses the `all-MiniLM-L6-v2` embedding model and a default semantic-similarity threshold of **0.92**.
- Reuses an answer only when the candidate also passes two lexical safety checks.

### Dual guard

| Guard | Protection provided |
|---|---|
| Salient-token guard | Rejects matches that change important entities or numbers |
| Content-term guard | Rejects matches that change lower-case entities, plan tiers, file identifiers, or function identifiers |

**Example:** a cached answer for a question about France must not answer a semantically similar question about Germany.

**Visual:** query A and query B with the similarity test, then the two guard checks before a cache hit decision.

---

## Slide 11. Implementation

### Technology stack

- **Backend:** Python 3.11+, FastAPI, Uvicorn, Pydantic
- **Graph layer:** Neo4j 5 in Docker, Neo4j Python driver, Python AST
- **Vector and embeddings:** ChromaDB and SentenceTransformers `all-MiniLM-L6-v2`
- **Generation:** OpenAI Python SDK connected to Gemini `gemini-2.0-flash` or Groq `llama-3.3-70b-versatile`
- **Document processing:** `pypdf` for PDF text extraction
- **Testing:** pytest, pytest-asyncio, httpx
- **Frontend:** vanilla HTML, CSS, JavaScript with an interactive force-directed graph canvas

### API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /query` | Executes the five-step pipeline |
| `POST /ingest/code` | Parses and stores a Python codebase in Neo4j |
| `POST /ingest/document` | Uploads one or more supported documents |
| `POST /ingest/documents` | Batch document upload |
| `GET /health` | Reports Neo4j, ChromaDB, LLM, provider, model, and classifier status |
| `GET /graph/data` | Returns nodes and links for the graph visualizer |
| `POST` or `DELETE /graph/clear` | Clears the graph and refreshes related state |
| `POST` or `DELETE /cache/clear` | Clears the semantic cache |

---

## Slide 12. User Interface

### ACEF Web Studio

- Provides a Code Ingestion panel with a configurable codebase path.
- Provides a Document Ingestion panel that accepts single or batch `.md`, `.txt`, and `.pdf` uploads.
- Shows live status for Neo4j, ChromaDB, LLM availability, and classifier mode.
- Visualizes the five-stage pipeline and highlights the route taken by each query.
- Displays response metadata such as route, source references, cache status, and latency.
- Includes a graph explorer with search, zoom, pan, node inspection, and graph clearing controls.
- Allows the user to start a new session, which generates a new session ID and isolates conversation memory.

**Visual:** full-page dashboard screenshot with callouts for code ingestion, document ingestion, telemetry, pipeline map, chat, and graph tabs.

---

## Slide 13. Testing Strategy

### Test coverage

- **Unit tests:** schemas, cache behavior, classifier backends, corpus-aware routing, GraphRAG parsing and retrieval, MemRAG ingestion and retrieval, LLM-client error handling, pipeline behavior, and API endpoints.
- **Integration tests:** end-to-end SIMPLE, CODE, and MEMORY pipeline routes plus asynchronous gateway behavior.
- **Live tests:** optional tests that require a running Neo4j instance and a real LLM API key. Pytest deselects them by default.
- **Evaluation datasets:**
  - 40 query-routing records, including 16 hard negatives
  - 20 MemRAG records covering factual-document queries, paraphrases, conversation follow-ups, near misses, and unrelated queries

### Important test cases

- Reject false semantic-cache hits when a country, number, plan tier, file, or function changes.
- Keep generic programming questions on the SIMPLE route.
- Route repository-specific questions to CODE and document-dependent questions to MEMORY.
- Maintain document and conversation quotas independently.
- Return 429 for provider rate limits and 503 for provider failures.
- Keep health checks responsive while a slow query runs in a worker thread.

---

## Slide 14. Results

### Verified results from the current checkout

- Pytest collection found **194 tests**, with **1 live test deselected by default**.
- The routing and pipeline verification subset completed successfully: **24 passed in 1.96 seconds**.
- The selected tests cover the query-routing boundaries, hard-negative evaluation, cache-hit short circuit, SIMPLE flow, CODE flow, MEMORY flow, graceful retrieval degradation, and LLM failure handling.
- The current test suite includes both mocked tests and environment-dependent integration or live tests. Do not claim that all 194 tests passed unless the complete suite is rerun in the final deployment environment.

### Functional results demonstrated by the implementation

- Repeated valid SIMPLE questions can return from the semantic cache.
- Repository questions can retrieve code symbols and call relationships from the Neo4j graph.
- Uploaded-document questions can retrieve chunks from ChromaDB.
- Session follow-ups can retrieve earlier turns by session ID.
- The health endpoint separates graph connectivity, vector-store readiness, and LLM configuration status.

**Visual:** a compact results table. Use only measured timings from the final live demo. Do not insert fabricated cache or retrieval latency values.

---

## Slide 15. Output Snapshots

### Screenshots to capture for the final PPT

Capture these from a locally running ACEF instance after configuring the API key and starting Neo4j. Redact API keys and full session IDs.

| Snapshot | Steps | What the image should prove |
|---|---|---|
| Dashboard and health | Open `http://localhost:8000/` after startup | Web Studio loads and service status is visible |
| Code ingestion | Ingest `./data/demo_codebase` | AST ingestion summary and successful graph preparation |
| Knowledge graph | Open the Knowledge Graph view after code ingestion | Files, classes or functions, and their relationships are visible |
| CODE query | Ask `What does the login function call?` | Route is CODE, GraphRAG sources are displayed, and the answer is grounded in the code graph |
| Document ingestion | Upload `data/demo_documents/pricing.md` | Document chunks are accepted and stored |
| MEMORY query | Ask `What are the pricing tiers?` | Route is MEMORY and `pricing.md` appears as evidence |
| Session follow-up | Ask a second question with the same session ID | The route retrieves prior session context |
| Semantic cache | Ask the same SIMPLE question twice | The second response shows `from_cache: true` or a Cache Hit indicator |
| Test output | Run `python -m pytest tests/unit/test_query_analyzer.py tests/unit/test_pipeline.py -q` | `24 passed` verification output |

### Sample response shape for an API screenshot

```json
{
  "query_id": "<uuid>",
  "session_id": "<uuid>",
  "response": "<generated answer>",
  "query_type": "code",
  "from_cache": false,
  "sources": ["<retrieved source>"],
  "latency_ms": 0.0
}
```

**Note:** The repository did not contain saved output screenshots. Capture the listed screenshots before finalizing the PPT.

---

## Slide 16. Conclusion

- ACEF delivers one framework for three different context needs: direct questions, repository-aware questions, and document or conversation-aware questions.
- It combines a semantic cache, Neo4j GraphRAG, ChromaDB MemRAG, session memory, configurable routing, and LLM generation in a modular architecture.
- The framework exposes the complete flow through a FastAPI service and an interactive dashboard.
- The current design emphasizes reliable routing, source-aware retrieval, cache safety, graceful degradation, and local development simplicity.

**Closing statement:** ACEF provides a practical foundation for building technical AI assistants that use the right context instead of treating every query the same way.

---

## Slide 17. Future Enhancements

- Add Tree-sitter parsing to support TypeScript, Java, Go, Rust, and other languages.
- Implement and evaluate the reserved local classifier with a fine-tuned MiniLM model.
- Add a confidence score and an LLM fallback only for uncertain heuristic routing decisions.
- Add reranking and semantic context compression before LLM generation.
- Support secure ZIP repository uploads instead of requiring a server-local directory.
- Add user authentication, per-user document isolation, role-based access, audit logs, and safer ingestion-root configuration.
- Move ChromaDB to client-server mode for multi-process or multi-user deployments.
- Add benchmark dashboards for cache hit rate, route accuracy, retrieval quality, token usage, latency, and cost.
- Add answer evaluation with citation quality, groundedness, and regression tests over real repositories.
- Add a response verification or reflection stage when high assurance is more important than latency.

---

## Slide 18. Paper Published

### Publication status

No paper manuscript, paper title, author list, journal or conference name, DOI, acceptance letter, or publication link was found in this repository.

Use one of the following options in the final presentation:

1. **If no paper has been published:**

   `Publication status: This project has not yet been published. A paper is planned as future work.`

2. **If a paper has been published or submitted:** replace this section with verified details only.

```text
Paper title:
Authors:
Journal or conference:
Publication or submission date:
DOI or URL:
Indexing status, if applicable:
```

Do not present a submission as a publication. Distinguish clearly between submitted, accepted, and published.

---

## Slide 19. References

### Project references

1. ACEF repository source code and test suite, accessed locally.
2. `README.md`, *ACEF — Agentic Context Engineering Framework*.
3. `progress.md`, *ACEF — Implementation Progress & System Audit*.
4. `config/settings.py`, `core/pipeline.py`, `core/classifier.py`, `core/query_analyzer.py`, and `core/corpus_index.py`.
5. `modules/cache.py`, `modules/graph_rag.py`, `modules/mem_rag.py`, and `modules/llm_client.py`.
6. `ingestion/code_ingest.py`, `ingestion/document_ingest.py`, `api/gateway.py`, and `static/` dashboard assets.

### Technical references

7. FastAPI. *FastAPI Documentation*. https://fastapi.tiangolo.com/
8. Neo4j. *Neo4j Graph Database Documentation*. https://neo4j.com/docs/
9. Chroma. *Chroma Documentation*. https://docs.trychroma.com/
10. Reimers, N. and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. Proceedings of EMNLP-IJCNLP. https://aclanthology.org/D19-1410/
11. Pydantic. *Pydantic Documentation*. https://docs.pydantic.dev/
12. OpenAI. *OpenAI Python Library*. https://github.com/openai/openai-python
13. Google AI for Developers. *Gemini API Documentation*. https://ai.google.dev/gemini-api/docs
14. Groq. *API Documentation*. https://console.groq.com/docs

---

## Appendix. Demo setup and commands

Use these commands to prepare the screenshots and demonstration.

```powershell
cd acef
pip install -r requirements.txt
docker compose up -d
python -m scripts.init_neo4j
uvicorn api.gateway:app --reload
```

Before launching, create `.env` from `.env.example` and configure either `GEMINI_API_KEY` or `GROQ_API_KEY`.

Suggested live-demo sequence:

1. Open `http://localhost:8000/`.
2. Ingest `./data/demo_codebase`.
3. Open the graph view.
4. Upload `data/demo_documents/pricing.md`.
5. Ask a CODE query, then a MEMORY query, then repeat a SIMPLE query to demonstrate a cache hit.
6. Run the selected test command shown in Slide 15.
