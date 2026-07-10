import re
from typing import List

from langchain_core.documents import Document

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into chunks at paragraph/sentence boundaries.
    Falls back to hard character splits only when a single sentence
    exceeds chunk_size, preserving chunk_overlap at the seam.
    """
    chunked = []

    for doc in documents:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.page_content) if p.strip()]
        units = []
        for para in paragraphs:
            units.extend(s for s in _SENTENCE_BOUNDARY.split(para) if s)
        if not units:
            units = [doc.page_content]

        current = ""
        for unit in units:
            candidate = (current + " " + unit).strip() if current else unit
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunked.append(Document(page_content=current, metadata=doc.metadata))
            if len(unit) <= chunk_size:
                current = unit
            else:
                start = 0
                while start < len(unit):
                    chunked.append(Document(page_content=unit[start:start + chunk_size], metadata=doc.metadata))
                    start += chunk_size - chunk_overlap
                current = ""
        if current:
            chunked.append(Document(page_content=current, metadata=doc.metadata))

    return chunked
