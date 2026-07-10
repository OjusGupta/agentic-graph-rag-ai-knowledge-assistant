"""
Smoke test for the ingestion pipeline.
Run from project root: python test_ingestion.py
Requires knowledge_base/ PDFs to be present.
"""
from pathlib import Path
from ingestion.loader import load_documents
from ingestion.splitter import split_documents


def main() -> None:
    kb_dir = Path("./knowledge_base")
    documents = load_documents(kb_dir)
    print(f"[INFO] Loaded {len(documents)} pages.")
    assert documents, "No documents loaded — check knowledge_base/ folder."

    chunks = split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    assert chunks, "No chunks produced."

    print("[OK] Ingestion smoke test passed.")


if __name__ == "__main__":
    main()
