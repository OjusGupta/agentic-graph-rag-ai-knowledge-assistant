# Agentic Graph RAG AI Knowledge Assistant

An AI-powered academic knowledge assistant that answers technical questions by combining semantic vector search, cross-encoder reranking, knowledge graph traversal, and LLM-based answer generation — all grounded in a corpus of 17 real textbooks.

---

## What It Does

You ask a question. The system:
1. Rewrites vague follow-up queries using conversation history
2. Searches 4,362 indexed chunks using hybrid BM25 + vector search
3. Reranks the top results using a cross-encoder for precision
4. Traverses a 16-node knowledge graph to find related documents
5. Sends the best context to `llama-3.3-70b-versatile` via Groq
6. Returns a structured, cited answer with source references

---

## Architecture

```
User Query
    │
    ▼
Query Rewriter  ──── (expands vague follow-ups using history)
    │
    ▼
Hybrid Retriever
    ├── Vector Search  (BGE bge-base-en-v1.5 + Chroma)
    └── BM25 Keyword Boost
    │
    ▼
Cross-Encoder Reranker  (ms-marco-MiniLM-L-6-v2)
    │
    ▼
Knowledge Graph  ──── (16-node concept graph, cross-document links)
    │
    ▼
Groq LLM  (llama-3.3-70b-versatile)
    │
    ▼
Structured Answer + Source Citations
```

---

## Project Structure

```
agentic-graph-rag-ai-knowledge-assistant/
│
├── backend/
│   ├── app.py                  # FastAPI server — /api/query, /api/health
│   └── templates/
│       └── index.html          # Chat UI (dark/light theme, graph canvas)
│
├── ingestion/
│   ├── loader.py               # Recursive PDF reader using pypdf
│   ├── splitter.py             # Sentence-aware chunker (1000 chars, 200 overlap)
│   └── run_ingestion.py        # Delegates to root run_ingestion.py
│
├── retrieval/
│   ├── retriever.py            # build_vector_store() + hybrid query_store()
│   ├── agent_engine.py         # Query rewriting, reranking, Groq LLM call
│   ├── graph_manager.py        # Builds knowledge_graph.json from flat_index
│   └── vector_store.py         # Re-exports from retriever.py
│
├── knowledge_base/             # 17 academic PDFs across 17 domains
├── vector_db/                  # Generated — Chroma DB + flat_index.json + knowledge_graph.json
├── tests/
│   └── test_app.py             # FastAPI endpoint tests
│
├── run.py                      # Start server: uvicorn backend.app:app
├── run_ingestion.py            # Full ingestion pipeline entry point
├── test_ingestion.py           # Ingestion smoke test
├── requirements.txt
└── .env                        # GROQ_API_KEY (never committed)
```

---

## Knowledge Base — 17 Domains

| Domain | PDF |
|--------|-----|
| Artificial Intelligence | (R17A1204) Artificial Intelligence.pdf |
| Neural Networks (ANN) | Neural-Networks.pdf |
| Computer Networks | COMPUTER NETWORKS NOTES.pdf |
| Data Mining & Warehousing | DATA WAREHOUSING AND DATA MINING (R18A0524).pdf |
| DBMS | DBMS.pdf |
| Deep Learning | deep learning.pdf |
| Data Structures (DSA) | DATA STRUCTURES DIGITAL NOTES.pdf |
| Java Programming | Java Programming.pdf |
| LangChain | LangChain_From_0_To_1_public_1_PpuSgEN.pdf |
| Machine Learning | Introduction to Machine Learning with Python.pdf |
| NLP | Natural Language Processing-1.pdf |
| Operating Systems | os.pdf |
| Python Programming | (R18A0513) Python Programming digital notes.pdf |
| RAG | RAG.pdf |
| Reinforcement Learning | RL.pdf |
| SQL | SQL-Manual.pdf |
| Theory of Computation | TOC.pdf |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Embeddings | `BAAI/bge-base-en-v1.5` (HuggingFace) |
| Vector Store | ChromaDB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Knowledge Graph | Custom JSON graph (concept-based) |
| Frontend | Vanilla JS, Canvas API, Lucide icons |
| PDF Parsing | pypdf |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com)

### 3. Add PDFs to knowledge_base/

Place your PDFs inside the appropriate subfolder under `knowledge_base/`.

### 4. Run ingestion

```bash
python run_ingestion.py
```

This will:
- Read all PDFs recursively
- Split into 4,362 sentence-aware chunks
- Embed with BGE and store in Chroma
- Write `flat_index.json`
- Build `knowledge_graph.json` (16 nodes)

### 5. Start the server

```bash
python run.py
```

Open `http://localhost:8000` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Chat UI |
| GET | `/api/health` | Health check |
| POST | `/api/query` | Ask a question |

### POST /api/query

```json
{
  "query": "Explain how transformers work",
  "k": 5,
  "source_filter": "deep learning.pdf",
  "history": [
    { "role": "user", "content": "What is attention?" },
    { "role": "assistant", "content": "Attention is a mechanism..." }
  ]
}
```

**Response:**
```json
{
  "answer": "Transformers use self-attention to...",
  "context_chunks": [
    { "source_file": "deep learning.pdf", "preview": "...", "confidence": 87 }
  ],
  "graph_cross_references": ["Neural-Networks.pdf", "NLP.pdf"]
}
```

---

## Key Design Decisions

- **BGE over MiniLM** — `bge-base-en-v1.5` is specifically trained for retrieval tasks, giving significantly better semantic matching on academic text
- **Hybrid search** — BM25 keyword boost on top of vector similarity catches exact-term matches that pure vector search misses
- **Cross-encoder reranking** — retrieves 8 candidates, reranks to top 3 for precision before sending to LLM
- **Query rewriting** — detects vague follow-ups ("explain its components", "tell me more") and enriches them with the last assistant turn before retrieval
- **Conversation history** — last 6 turns sent to LLM for coherent multi-turn dialogue
- **Groq over OpenAI** — free tier, faster inference, `llama-3.3-70b-versatile` is competitive with GPT-4o on reasoning tasks

---

## Notes

- `vector_db/` is gitignored — run ingestion after cloning
- `.env` is gitignored — never commit your API key
- Re-ingestion required if you change the embedding model
