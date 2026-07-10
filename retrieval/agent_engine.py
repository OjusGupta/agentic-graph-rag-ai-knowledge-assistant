import os
from typing import Dict, Any, List, Optional

import requests

from retrieval.retriever import query_store

try:
    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False
    print("[WARN] CrossEncoder not available — reranking step skipped.")

_VAGUE_TRIGGERS = {
    "it", "its", "this", "these", "those", "they", "them",
    "yes", "go on", "continue", "elaborate", "explain more", "tell me more",
}

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "llama-3.3-70b-versatile"


def _rerank(query: str, chunks: List[Dict], top_n: int = 3) -> List[Dict]:
    """Re-score retrieved chunks against the query using a cross-encoder."""
    if not RERANKER_AVAILABLE or not chunks:
        return chunks[:top_n]
    scores = _reranker.predict([(query, c["preview"]) for c in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_n]]


def _rewrite_query(query_text: str, history: Optional[List[Dict]]) -> str:
    """Expand vague follow-up queries using the last assistant turn as context."""
    if not history:
        return query_text
    lowered = query_text.lower().strip()
    if any(lowered.startswith(t) or lowered == t for t in _VAGUE_TRIGGERS):
        for msg in reversed(history):
            if msg["role"] == "assistant":
                return f"{query_text} (context: {msg['content'][:300]})"
    return query_text


def generate_agent_response(
    query_text: str,
    chunk_count: int = 5,
    source_filter: str = None,
    history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Retrieve, rerank, and generate a grounded answer via Groq LLM."""
    retrieval_query = _rewrite_query(query_text, history)
    retrieval_data = query_store(retrieval_query, chunk_count=max(chunk_count, 8), source_filter=source_filter)

    if not retrieval_data.get("success"):
        error = retrieval_data.get("error", "Chroma returned no results or vector_db is empty.")
        return {"answer": f"Retrieval failed: {error}", "context_chunks": [], "graph_cross_references": []}

    chunks = _rerank(query_text, retrieval_data.get("context_chunks", []), top_n=3)
    graph_refs = retrieval_data.get("graph_cross_references", [])

    context_str = ""
    sources = set()
    for i, c in enumerate(chunks, 1):
        context_str += f"\n[Chunk {i} — Source: {c['source_file']}]\n{c['preview']}\n"
        sources.add(c["source_file"])

    prompt = (
        f"You are an AI Academic Assistant. Your knowledge is strictly limited to the following 17 textbooks:\n"
        f"Artificial Intelligence, Neural Networks, Computer Networks, Data Mining & Warehousing, DBMS, "
        f"Deep Learning, Data Structures (DSA), Java Programming, LangChain, Machine Learning, NLP, "
        f"Operating Systems, Python Programming, RAG, Reinforcement Learning, SQL, Theory of Computation.\n\n"
        f"Retrieved Context (from: {', '.join(sources) or 'unknown'}):\n{context_str}\n\n"
        f"Guidelines:\n"
        f"1. Answer ONLY using the retrieved context above. This is your sole source of truth.\n"
        f"2. If the retrieved context does not contain enough information to answer, say: "
        f"'I could not find sufficient information on this topic in the indexed textbooks. "
        f"Try selecting a specific textbook filter or rephrasing your question.'\n"
        f"3. Do NOT use outside knowledge, internet data, or general training data to answer.\n"
        f"4. Structure your answer clearly — use bullet points, numbered steps, or bold headers.\n"
        f"5. End with 1-2 follow-up questions that go deeper into the topic.\n"
        f"6. Always cite which textbook source(s) your answer is based on.\n\n"
        f"Student Question: {query_text}\n\nAnswer:"
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"answer": "GROQ_API_KEY is missing. Check your .env file.", "context_chunks": chunks, "graph_cross_references": graph_refs}

    system_msg = {
        "role": "system",
        "content": (
            "You are a strict Academic Knowledge Assistant. You ONLY answer questions based on the "
            "retrieved context from 17 indexed textbooks covering: Artificial Intelligence, Neural Networks, "
            "Computer Networks, Data Mining, DBMS, Deep Learning, DSA, Java, LangChain, Machine Learning, "
            "NLP, Operating Systems, Python, RAG, Reinforcement Learning, SQL, and Theory of Computation. "
            "Never use outside knowledge. If the context does not cover the question, say so clearly "
            "and suggest the student pick a relevant textbook filter."
        ),
    }
    messages = [system_msg] + (history or [])[-6:] + [{"role": "user", "content": prompt}]

    try:
        response = requests.post(
            _GROQ_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
            json={"model": _MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 1500},
            timeout=30,
        )
        data = response.json()

        if "error" in data:
            return {"answer": f"Groq API Error: {data['error'].get('message', 'Unknown error')}", "context_chunks": chunks, "graph_cross_references": graph_refs}

        answer = data["choices"][0]["message"]["content"] if data.get("choices") else "Groq connected but returned no choices."
        return {"answer": answer, "context_chunks": chunks, "graph_cross_references": graph_refs}

    except Exception as e:
        return {"answer": f"Groq connection error: {e}", "context_chunks": chunks, "graph_cross_references": graph_refs}
