from pathlib import Path

from ingestion.loader import load_documents
from ingestion.splitter import split_documents
from retrieval.retriever import build_vector_store
from retrieval.graph_manager import build_knowledge_graph


def main() -> None:
    kb_dir = Path("./knowledge_base")
    persist_dir = Path("./vector_db")

    print("[INFO] Step 1: Loading PDF documents ...")
    documents = load_documents(kb_dir)
    print(f"[INFO] Loaded {len(documents)} pages.")

    if not documents:
        print("[ERROR] No documents found. Check knowledge_base/ folder.")
        return

    print("[INFO] Step 2: Splitting into chunks ...")
    chunks = split_documents(documents)
    print(f"[INFO] Created {len(chunks)} chunks.")

    print("[INFO] Step 3: Building Chroma vector store ...")
    build_vector_store(chunks, persist_dir)

    print("[INFO] Step 4: Building knowledge graph ...")
    graph = build_knowledge_graph()
    if "error" in graph:
        print(f"[WARN] Graph: {graph['error']}")

    print("[OK] Ingestion complete.")


if __name__ == "__main__":
    main()
