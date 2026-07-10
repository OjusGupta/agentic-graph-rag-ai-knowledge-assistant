# Agentic Graph RAG AI Knowledge Assistant — Project Documentation

**Status:** Functional MVP — fully working end-to-end  
**Difficulty:** Advanced (5/5)  
**Stack:** Python, FastAPI, ChromaDB, BGE Embeddings, Groq LLM, LangChain

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [System Architecture](#system-architecture)
4. [Component Breakdown](#component-breakdown)
5. [Data Pipeline](#data-pipeline)
6. [Retrieval Pipeline](#retrieval-pipeline)
7. [Agent & LLM Layer](#agent--llm-layer)
8. [Knowledge Graph](#knowledge-graph)
9. [Frontend UI](#frontend-ui)
10. [API Reference](#api-reference)
11. [Tech Stack](#tech-stack)
12. [Project Structure](#project-structure)
13. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Problem Statement

Students and developers studying multiple technical domains (ML, databases, networking, programming) face a fragmented learning experience — answers are scattered across dozens of textbooks and PDFs. Searching manually is slow, and generic LLMs hallucinate or lack domain-specific depth.

**Goal:** Build an AI assistant that retrieves answers directly from a curated academic corpus, reasons over cross-document relationships, and generates structured, cited responses.

---

## Solution Overview

An end-to-end Retrieval-Augmented Generation (RAG) system with:

- **Semantic search** over 4,362 chunks from 17 academic textbooks
- **Hybrid retrieval** combining vector similarity and BM25 keyword scoring
- **Cross-encoder reranking** for precision before LLM generation
- **Knowledge graph** linking documents by shared concepts
- **Query rewriting** for coherent multi-turn conversations
- **Groq LLM** (`llama-3.3-70b-versatile`) for fast, structured answer generation
- **Chat UI** with dark/light theme, live graph visualization, and source filtering

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Chat UI)                       │
│   Dark/light theme · Graph canvas · Source filter · History     │
└────────────────────────────┬────────────────────────────────────┘
                             │  POST /api/query
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                             │
│              backend/app.py — /api/query, /api/health           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT ENGINE                                 │
│              retrieval/agent_engine.py                          │
│                                                                 │
│  1. Query Rewriter  — expands vague follow-ups using history    │
│  2. Retriever       — hybrid BM25 + vector search               │
│  3. Reranker        — cross-encoder scores top-8, keeps top-3   │
│  4. Prompt Builder  — injects context + history into prompt     │
│  5. Groq LLM Call   — llama-3.3-70b-versatile, max 1500 tokens  │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────┐       ┌─────────────────────────────┐
│   VECTOR STORE      │       │     KNOWLEDGE GRAPH         │
│   retriever.py      │       │     graph_manager.py        │
│                     │       │                             │
│  ChromaDB           │       │  16-node concept graph      │
│  BGE embeddings     │       │  JSON adjacency list        │
│  4,362 chunks       │       │  Cross-doc references       │
└─────────────────────┘       └─────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE BASE                              │
│              17 academic PDFs · 17 domains                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### ingestion/loader.py
Recursively scans `knowledge_base/` using `pypdf.PdfReader`. Extracts text page by page, stores `source` (filename) and `page` number in metadata. Skips damaged pages gracefully.

### ingestion/splitter.py
Custom sentence-aware splitter. Splits on paragraph boundaries first (`\n\n`), then sentence boundaries (`[.!?]\s`), falls back to hard character splits only when a single sentence exceeds `chunk_size=1000`. Maintains `chunk_overlap=200` at hard-split seams to preserve context.

**Why custom instead of LangChain's splitter?** LangChain's `RecursiveCharacterTextSplitter` cuts mid-sentence. This splitter respects sentence boundaries, producing cleaner chunks for embedding.

### retrieval/retriever.py
Two responsibilities:

**1. `build_vector_store(chunks, persist_directory)`**
- Cleans surrogate Unicode characters from all text (PDF extraction artifact)
- Validates and filters empty/malformed chunks
- Writes `flat_index.json` (needed by graph_manager)
- Embeds with `BAAI/bge-base-en-v1.5` and stores in Chroma

**2. `query_store(query_text, chunk_count, source_filter)`**
- Loads Chroma, applies optional `$eq` source filter
- Runs vector similarity search for `chunk_count` candidates
- Applies BM25-style keyword boost: counts query term hits in each chunk, adds up to `+0.15` to similarity score
- Sorts by boosted confidence score
- Loads `knowledge_graph.json` and appends cross-references for matched sources

### retrieval/agent_engine.py
Orchestrates the full query-to-answer pipeline:

**`_rewrite_query(query_text, history)`**
Detects vague trigger words (`it`, `its`, `this`, `explain more`, `continue`, etc.). If triggered, appends the first 300 characters of the last assistant message to the query before retrieval. This prevents Chroma from receiving context-free queries like "explain its components".

**`_rerank(query, chunks, top_n=3)`**
Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to score each (query, chunk) pair. Returns top 3 by cross-encoder score. Falls back to top-3 by vector score if reranker unavailable.

**`generate_agent_response(...)`**
Full pipeline: rewrite → retrieve 8 → rerank to 3 → build prompt → call Groq API → return answer + chunks + graph refs.

Conversation history (last 6 turns) is injected between system message and current user message for multi-turn coherence.

### retrieval/graph_manager.py
Scans `flat_index.json` for 4 concept categories (`java`, `database`, `ai_ml`, `os_networks`) using keyword lists. Builds a document-level adjacency graph: two documents are linked if they share at least one concept category. Writes `knowledge_graph.json`.

**Example:** `DBMS.pdf` and `SQL-Manual.pdf` both contain `sql`, `query`, `transaction` → linked in graph → when DBMS is retrieved, SQL-Manual appears as a cross-reference.

### backend/app.py
FastAPI application with 3 endpoints. Loads `.env` on startup. Strips quotes/whitespace from `GROQ_API_KEY` (common copy-paste issue). Serves `index.html` directly from `backend/templates/`.

---

## Data Pipeline

```
knowledge_base/ PDFs
        │
        ▼  loader.py
Page-level Documents (with source + page metadata)
        │
        ▼  splitter.py
4,362 sentence-aware chunks (1000 chars, 200 overlap)
        │
        ├──▶  retriever.build_vector_store()
        │         ├── Surrogate char cleaning
        │         ├── flat_index.json  ──▶  graph_manager.py
        │         └── Chroma (BGE embeddings)
        │
        └──▶  graph_manager.build_knowledge_graph()
                  └── knowledge_graph.json (16 nodes)
```

Run with:
```bash
python run_ingestion.py
```

---

## Retrieval Pipeline

```
User Query: "explain its components"
        │
        ▼  _rewrite_query()
Enriched: "explain its components (context: LangChain is a framework...)"
        │
        ▼  query_store(chunk_count=8)
        ├── Chroma vector search  →  8 candidates
        └── BM25 keyword boost    →  re-scored
        │
        ▼  _rerank(top_n=3)
Cross-encoder scores all 8 pairs → top 3 chunks
        │
        ▼  knowledge_graph.json
Cross-references: ["Neural-Networks.pdf", "deep learning.pdf"]
        │
        ▼  Groq LLM
Structured answer with citations
```

---

## Agent & LLM Layer

**Model:** `llama-3.3-70b-versatile` via Groq API  
**Temperature:** 0.3 (factual, low creativity)  
**Max tokens:** 1500  
**Context window used:** system prompt + last 6 history turns + 3 reranked chunks + current query

**System prompt instructs the model to:**
- Use retrieved context as primary source
- Use general knowledge only to clarify terminology
- Structure answers with bullets/headers
- End with 1-2 follow-up questions
- Cite source documents

**Why Groq over OpenAI?**
- Free tier with generous rate limits
- Sub-second inference on 70B model
- `llama-3.3-70b-versatile` matches GPT-4o on reasoning benchmarks

---

## Knowledge Graph

A lightweight concept-based document graph stored as a JSON adjacency list.

**Concept categories scanned:**
- `java` — class, object, inheritance, polymorphism, jdbc, multithreading
- `database` — sql, dbms, query, transaction, schema, normalization
- `ai_ml` — neural network, regression, cnn, reinforcement learning, dataset
- `os_networks` — process, thread, memory, tcp, ip, routing, packet

**Output:** `vector_db/knowledge_graph.json`
```json
{
  "DBMS.pdf": ["SQL-Manual.pdf", "DATA WAREHOUSING AND DATA MINING.pdf"],
  "deep learning.pdf": ["Neural-Networks.pdf", "Introduction to Machine Learning.pdf"],
  ...
}
```

**Used in retrieval:** After vector search, the graph is queried for each matched source. Related documents are returned as `graph_cross_references` in the API response, shown in the UI as suggested related reads.

---

## Frontend UI

Single-page chat interface served directly by FastAPI.

**Features:**
- Dark / light theme toggle
- Conversation history (last 12 turns, resets on new session)
- Source filter dropdown — all 17 exact PDF filenames
- Top-k stepper (1–6 chunks)
- Live latency display (measured client-side per request)
- Animated knowledge graph canvas (16 nodes, edges pulse on query)
- Retrieved snippets panel with confidence meters
- Markdown rendering — bold, headers, bullets, code blocks, inline code
- Sidebar thread history with search
- Keyboard shortcut: `Ctrl+N` for new session
- Responsive — right panel hides below 900px, sidebar hides below 680px

**No React, no build step** — pure HTML/CSS/JS served from `backend/templates/index.html`. Intentional: zero deployment complexity.

---

## API Reference

### GET /api/health
```json
{ "status": "healthy" }
```

### POST /api/query

**Request:**
```json
{
  "query": "How does backpropagation work?",
  "k": 5,
  "source_filter": "Neural-Networks.pdf",
  "history": [
    { "role": "user", "content": "What is a neural network?" },
    { "role": "assistant", "content": "A neural network is..." }
  ]
}
```

**Response:**
```json
{
  "answer": "Backpropagation works by...\n\n**Sources:** Neural-Networks.pdf",
  "context_chunks": [
    {
      "source_file": "Neural-Networks.pdf",
      "preview": "The backpropagation algorithm computes...",
      "confidence": 91
    }
  ],
  "graph_cross_references": ["deep learning.pdf", "Introduction to Machine Learning.pdf"]
}
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend framework | FastAPI + Uvicorn | Async, fast, auto-docs |
| Embeddings | `BAAI/bge-base-en-v1.5` | Trained for retrieval, outperforms MiniLM on academic text |
| Vector database | ChromaDB | Persistent, local, no infra needed |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision reranking without heavy compute |
| LLM | Groq `llama-3.3-70b-versatile` | Free, fast, GPT-4o competitive |
| PDF parsing | pypdf | Pure Python, no Java/Tesseract dependency |
| LLM orchestration | LangChain (loaders, Chroma wrapper) | Standard RAG tooling |
| Frontend | Vanilla JS + Canvas API | Zero build step, easy deployment |
| Environment | python-dotenv | Secure key management |

---

## Project Structure

```
agentic-graph-rag-ai-knowledge-assistant/
│
├── backend/
│   ├── app.py                   # FastAPI app
│   ├── __init__.py
│   └── templates/
│       └── index.html           # Full chat UI
│
├── ingestion/
│   ├── loader.py                # PDF reader
│   ├── splitter.py              # Sentence-aware chunker
│   ├── run_ingestion.py         # Delegates to root
│   └── __init__.py
│
├── retrieval/
│   ├── retriever.py             # Vector store + hybrid search
│   ├── agent_engine.py          # Rewrite + rerank + LLM
│   ├── graph_manager.py         # Knowledge graph builder
│   ├── vector_store.py          # Re-exports from retriever
│   └── __init__.py
│
├── knowledge_base/              # 17 domain PDFs
│   ├── AI/
│   ├── ANN/
│   ├── Computer_Networks/
│   ├── Data_Mining_Warehousing/
│   ├── DBMS/
│   ├── Deep_Learning/
│   ├── DSA/
│   ├── Java/
│   ├── LangChain/
│   ├── Machine_Learning/
│   ├── NLP/
│   ├── Operating_System/
│   ├── Python/
│   ├── RAG/
│   ├── RL/
│   ├── SQL/
│   └── Theory_of_computation/
│
├── tests/
│   └── test_app.py              # Endpoint tests
│
├── vector_db/                   # Generated (gitignored)
│   ├── chroma/                  # ChromaDB files
│   ├── flat_index.json          # All chunks for graph builder
│   └── knowledge_graph.json    # 16-node concept graph
│
├── run.py                       # Start server
├── run_ingestion.py             # Full ingestion pipeline
├── test_ingestion.py            # Ingestion smoke test
├── requirements.txt
├── .env                         # GROQ_API_KEY (gitignored)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Known Limitations & Future Work

| Limitation | Impact | Planned Fix |
|-----------|--------|-------------|
| Knowledge graph is concept-keyword based | Misses semantic relationships | Replace with entity extraction using spaCy or LLM |
| No persistent chat sessions | History lost on page refresh | Add localStorage or backend session store |
| `vector_db/` not committed | Must re-ingest after every clone/deploy | Add Render persistent disk or pre-built index |
| Single-user, no auth | Not production-safe | Add API key auth or OAuth |
| Frontend is single HTML file | Hard to scale UI features | Migrate to React if UI grows |
| No streaming responses | Answer appears all at once | Add SSE/WebSocket streaming from Groq |
| Graph has 16 nodes only | AWS PDF was missing | Add more domain PDFs to expand coverage |
