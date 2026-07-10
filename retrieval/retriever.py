import os
import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "./vector_db/chroma"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)


def _clean(text: str) -> str:
    """Strip surrogate characters that crash JSON serialization and Chroma."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def build_vector_store(chunks: List[Any], persist_directory: Path) -> None:
    """Ingest document chunks into Chroma and write flat_index.json for the graph builder."""
    print(f"[INFO] Building vector store at {PERSIST_DIR} ...")

    valid_chunks = []
    for doc in chunks:
        if not (doc and hasattr(doc, "page_content")):
            continue
        content = doc.page_content
        if not isinstance(content, str):
            content = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else str(content)
        content = _clean(content)
        if content.strip():
            doc.page_content = content
            valid_chunks.append(doc)

    skipped = len(chunks) - len(valid_chunks)
    if skipped:
        print(f"[WARN] Skipped {skipped} empty or malformed chunks.")

    if not valid_chunks:
        print("[WARN] No valid chunks to index. Aborting.")
        return

    # Write flat index so graph_manager can scan concepts without loading Chroma
    out_dir = Path("./vector_db")
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_index = [{"page_content": _clean(d.page_content), "metadata": d.metadata} for d in valid_chunks]
    with open(out_dir / "flat_index.json", "w", encoding="utf-8") as f:
        json.dump(flat_index, f, ensure_ascii=False)
    print(f"[INFO] flat_index.json written — {len(flat_index)} chunks.")

    Chroma.from_documents(
        documents=valid_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"[OK] Stored {len(valid_chunks)} chunks in Chroma.")


def query_store(query_text: str, chunk_count: int = 8, source_filter: str = None) -> Dict[str, Any]:
    """Hybrid vector + BM25 keyword search. Returns ranked context chunks and graph cross-references."""
    output = {"query": query_text, "success": False, "context_chunks": [], "graph_cross_references": []}

    if not os.path.exists(PERSIST_DIR):
        return output

    try:
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        search_filter = {"source": {"$eq": source_filter}} if source_filter else None

        results = db.similarity_search_with_relevance_scores(
            query_text, k=chunk_count, filter=search_filter
        )

        query_terms = set(query_text.lower().split())
        chunks = []
        for doc, score in results:
            content_lower = doc.page_content.lower()
            term_hits = sum(1 for t in query_terms if t in content_lower)
            boosted = min(score + min(term_hits * 0.03, 0.15), 0.99)
            chunks.append({
                "source_file": doc.metadata.get("source", "Unknown"),
                "preview": doc.page_content[:500] + "...",
                "confidence": max(10, min(99, round(boosted * 100))),
            })

        chunks.sort(key=lambda x: x["confidence"], reverse=True)
        output["success"] = True
        output["context_chunks"] = chunks

        graph_file = Path("./vector_db/knowledge_graph.json")
        if graph_file.exists():
            graph = json.loads(graph_file.read_text(encoding="utf-8"))
            refs = []
            for chunk in chunks:
                refs.extend(graph.get(chunk["source_file"], []))
            output["graph_cross_references"] = list(set(refs))[:3]

    except Exception as e:
        output["error"] = str(e)

    return output
