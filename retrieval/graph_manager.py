import json
from pathlib import Path
from typing import Dict, List, Set, Any

_CONCEPTS = {
    "java": ["class", "object", "inheritance", "polymorphism", "jdbc", "multithreading"],
    "database": ["sql", "dbms", "query", "transaction", "schema", "normalization"],
    "ai_ml": ["neural network", "regression", "cnn", "reinforcement learning", "dataset"],
    "os_networks": ["process", "thread", "memory", "tcp", "ip", "routing", "packet"],
}

_PERSIST_DIR = Path("./vector_db")


def build_knowledge_graph() -> Dict[str, Any]:
    """
    Scan flat_index.json for concept keywords and build a cross-document
    relationship graph. Writes knowledge_graph.json to vector_db/.
    """
    index_file = _PERSIST_DIR / "flat_index.json"
    graph_file = _PERSIST_DIR / "knowledge_graph.json"

    if not index_file.exists():
        return {"error": "flat_index.json not found. Run ingestion first."}

    chunks = json.loads(index_file.read_text(encoding="utf-8"))

    file_concepts: Dict[str, Set[str]] = {}
    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source", "Unknown")
        content = chunk.get("page_content", "").lower()
        file_concepts.setdefault(source, set())
        for category, keywords in _CONCEPTS.items():
            if any(kw in content for kw in keywords):
                file_concepts[source].add(category)

    graph: Dict[str, List[str]] = {}
    files = list(file_concepts.keys())
    for file_a in files:
        graph[file_a] = [
            file_b for file_b in files
            if file_a != file_b and file_concepts[file_a] & file_concepts[file_b]
        ]

    graph_file.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"[OK] Knowledge graph built — {len(graph)} document nodes.")
    return graph


if __name__ == "__main__":
    build_knowledge_graph()
