# import os
# from typing import Dict, Any, List, Optional

# import requests

# from retrieval.retriever import query_store

# try:
#     from sentence_transformers import CrossEncoder
#     _reranker = None
#     RERANKER_AVAILABLE = True
# except Exception:
#     RERANKER_AVAILABLE = False
#     print("[WARN] CrossEncoder not available — reranking step skipped.")

# def _get_reranker():
#     """Lazy-load reranker on first use."""
#     global _reranker
#     if RERANKER_AVAILABLE and _reranker is None:
#         _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#     return _reranker

# _VAGUE_TRIGGERS = {
#     "it", "its", "this", "these", "those", "they", "them",
#     "yes", "go on", "continue", "elaborate", "explain more", "tell me more",
# }

# _GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# _MODEL = "llama-3.3-70b-versatile"


# def _rerank(query: str, chunks: List[Dict], top_n: int = 3) -> List[Dict]:
#     """Re-score retrieved chunks against the query using a cross-encoder."""
#     reranker = _get_reranker()
#     if not reranker or not chunks:
#         return chunks[:top_n]
#     scores = reranker.predict([(query, c["preview"]) for c in chunks])
#     ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
#     return [c for _, c in ranked[:top_n]]


# def _rewrite_query(query_text: str, history: Optional[List[Dict]]) -> str:
#     """Expand vague follow-up queries using the last assistant turn as context."""
#     if not history:
#         return query_text
#     lowered = query_text.lower().strip()
#     if any(lowered.startswith(t) or lowered == t for t in _VAGUE_TRIGGERS):
#         for msg in reversed(history):
#             if msg["role"] == "assistant":
#                 return f"{query_text} (context: {msg['content'][:300]})"
#     return query_text


# def generate_agent_response(
#     query_text: str,
#     chunk_count: int = 5,
#     source_filter: str = None,
#     history: Optional[List[Dict]] = None,
# ) -> Dict[str, Any]:
#     """Retrieve, rerank, and generate a grounded answer via Groq LLM."""
#     retrieval_query = _rewrite_query(query_text, history)
#     retrieval_data = query_store(retrieval_query, chunk_count=max(chunk_count, 8), source_filter=source_filter)

#     if not retrieval_data.get("success"):
#         error = retrieval_data.get("error", "Chroma returned no results or vector_db is empty.")
#         return {"answer": f"Retrieval failed: {error}", "context_chunks": [], "graph_cross_references": []}

#     chunks = _rerank(query_text, retrieval_data.get("context_chunks", []), top_n=3)
#     graph_refs = retrieval_data.get("graph_cross_references", [])

#     context_str = ""
#     sources = set()
#     for i, c in enumerate(chunks, 1):
#         context_str += f"\n[Chunk {i} — Source: {c['source_file']}]\n{c['preview']}\n"
#         sources.add(c["source_file"])

#     prompt = (
#         f"You are an AI Academic Assistant. Your knowledge is strictly limited to the following 17 textbooks:\n"
#         f"Artificial Intelligence, Neural Networks, Computer Networks, Data Mining & Warehousing, DBMS, "
#         f"Deep Learning, Data Structures (DSA), Java Programming, LangChain, Machine Learning, NLP, "
#         f"Operating Systems, Python Programming, RAG, Reinforcement Learning, SQL, Theory of Computation.\n\n"
#         f"Retrieved Context (from: {', '.join(sources) or 'unknown'}):\n{context_str}\n\n"
#         f"Guidelines:\n"
#         f"1. Answer ONLY using the retrieved context above. This is your sole source of truth.\n"
#         f"2. If the retrieved context does not contain enough information to answer, say: "
#         f"'I could not find sufficient information on this topic in the indexed textbooks. "
#         f"Try selecting a specific textbook filter or rephrasing your question.'\n"
#         f"3. Do NOT use outside knowledge, internet data, or general training data to answer.\n"
#         f"4. Structure your answer clearly — use bullet points, numbered steps, or bold headers.\n"
#         f"5. End with 1-2 follow-up questions that go deeper into the topic.\n"
#         f"6. Always cite which textbook source(s) your answer is based on.\n\n"
#         f"Student Question: {query_text}\n\nAnswer:"
#     )

#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         return {"answer": "GROQ_API_KEY is missing. Check your .env file.", "context_chunks": chunks, "graph_cross_references": graph_refs}

#     system_msg = {
#         "role": "system",
#         "content": (
#             "You are a strict Academic Knowledge Assistant. You ONLY answer questions based on the "
#             "retrieved context from 17 indexed textbooks covering: Artificial Intelligence, Neural Networks, "
#             "Computer Networks, Data Mining, DBMS, Deep Learning, DSA, Java, LangChain, Machine Learning, "
#             "NLP, Operating Systems, Python, RAG, Reinforcement Learning, SQL, and Theory of Computation. "
#             "Never use outside knowledge. If the context does not cover the question, say so clearly "
#             "and suggest the student pick a relevant textbook filter."
#         ),
#     }
#     messages = [system_msg] + (history or [])[-6:] + [{"role": "user", "content": prompt}]

#     try:
#         response = requests.post(
#             _GROQ_URL,
#             headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
#             json={"model": _MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 1500},
#             timeout=30,
#         )
#         data = response.json()

#         if "error" in data:
#             return {"answer": f"Groq API Error: {data['error'].get('message', 'Unknown error')}", "context_chunks": chunks, "graph_cross_references": graph_refs}

#         answer = data["choices"][0]["message"]["content"] if data.get("choices") else "Groq connected but returned no choices."
#         return {"answer": answer, "context_chunks": chunks, "graph_cross_references": graph_refs}

#     except Exception as e:
#         return {"answer": f"Groq connection error: {e}", "context_chunks": chunks, "graph_cross_references": graph_refs}
import os
import re
from typing import Dict, Any, List, Optional
import requests

from retrieval.retriever import query_store

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "llama-3.3-70b-versatile"


def _optimize_and_clean_query(query_text: str, history: Optional[List[Dict]]) -> Dict[str, Any]:
    """
    Uses a fast pre-pass LLM instruction to fix typos, resolve follow-up 
    ambiguities (like 'yes' or 'explain more'), and classify intent.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not history:
        return {"is_conversational": False, "optimized_query": query_text}

    # Format history context for the query cleanup pass
    history_str = ""
    for msg in history[-4:]:
        history_str += f"{msg['role'].upper()}: {msg['content']}\n"

    system_prompt = (
        "You are a text processing utility for an AI search assistant. Analyze the user's latest query alongside the recent conversation history.\n"
        "Tasks:\n"
        "1. Fix any spelling mistakes or typos in the user's query.\n"
        "2. If the query is an implicit follow-up (e.g., 'yes', 'explain more', 'give code for that'), rewrite it into a complete, standalone query containing all necessary context from history.\n"
        "3. Detect if the query is pure casual chit-chat, a greeting, praise, or gratitude (e.g., 'hello', 'thanks', 'you are great') instead of an academic question.\n\n"
        "Output your assessment strictly in the following JSON format without code blocks or extra commentary:\n"
        "{\n"
        '  "is_conversational": true|false,\n'
        '  "optimized_query": "The fully corrected, standalone query text"\n'
        "}"
    )

    try:
        response = requests.post(
            _GROQ_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
            json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"History:\n{history_str}\nLatest Query: {query_text}"}
                ],
                "temperature": 0.1,
                "max_tokens": 150
            },
            timeout=5
        )
        res_data = response.json()
        content = res_data["choices"][0]["message"]["content"].strip()
        
        import json
        clean_json = re.search(r'\{.*\}', content, re.DOTALL)
        if clean_json:
            return json.loads(clean_json.group(0))
    except Exception:
        pass 

    return {"is_conversational": False, "optimized_query": query_text}


def generate_agent_response(
    query_text: str,
    chunk_count: int = 5,
    source_filter: str = None,
    history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Retrieve, filter, and dynamically generate grounded academic or conversational responses."""
    history = history or []

    # Step 1: Run pre-pass to fix typos, handle dialogue context, and check conversational intent
    analysis = _optimize_and_clean_query(query_text, history)
    is_conversational = analysis.get("is_conversational", False)
    search_query = analysis.get("optimized_query", query_text)

    chunks = []
    graph_refs = []
    context_str = ""
    sources = set()

    # Step 2: Only perform database retrieval if the query is an actual academic question
    if not is_conversational:
        retrieval_data = query_store(search_query, chunk_count=max(chunk_count, 15), source_filter=source_filter)
        if retrieval_data.get("success"):
            chunks = retrieval_data.get("context_chunks", [])
            graph_refs = retrieval_data.get("graph_cross_references", [])
            for c in chunks:
                context_str += f"\n[Textbook Source: {c['source_file']}]\n{c['preview']}\n"
                sources.add(c["source_file"])

    # Step 3: Build a highly adaptive system prompt
    system_prompt = (
        "You are an intelligent, realistic AI Academic Mentor matching context across 17 technical textbooks.\n\n"
        "DYNAMIC HANDLING RULES:\n"
        "1. IF the user is greeting you, thanking you, or offering compliments/feedback (e.g., 'thanks', 'you are doing great'), "
        "respond naturally, dynamically, and warmly as an encouraging mentor. DO NOT mention textbooks, chunks, or missing data in this conversational mode.\n"
        "2. IF the user is asking an academic or technical question, strictly ground your answers using the context provided below. "
        "If no context blocks are available for a technical topic, politely state that you couldn't find sufficient information in the 17 indexed textbooks.\n\n"
        "UI CLEANLINESS RULES:\n"
        "- NEVER use terms like 'Chunk X', 'Source file:', or file extensions like '.pdf' within your written dialogue.\n"
        "- Synthesize robust code implementations naturally if requested, integrating complete documentation patterns.\n"
        "- If answering an academic question, close with 1-2 sharp, contextual follow-up questions to push learning further."
    )

    prompt = (
        f"Context Blocks:\n{context_str or 'No textbook context retrieved (Conversational Mode Enabled)'}\n\n"
        f"Student Input: {search_query}\n\n"
        f"Response:"
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"answer": "GROQ_API_KEY configuration is missing.", "context_chunks": chunks, "graph_cross_references": graph_refs}

    messages = [
        {"role": "system", "content": system_prompt}
    ] + history[-6:] + [
        {"role": "user", "content": prompt}
    ]

    try:
        response = requests.post(
            _GROQ_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
            json={"model": _MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 1500},
            timeout=30,
        )
        data = response.json()

        if "error" in data:
            return {"answer": f"Generation Error: {data['error'].get('message')}", "context_chunks": chunks, "graph_cross_references": graph_refs}

        answer = data["choices"][0]["message"]["content"] if data.get("choices") else "Error pulling token generations."
        return {"answer": answer, "context_chunks": chunks, "graph_cross_references": graph_refs}

    except Exception as e:
        return {"answer": f"Connection pipeline error: {e}", "context_chunks": chunks, "graph_cross_references": graph_refs}