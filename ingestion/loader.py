from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader


def load_documents(directory_path: Path) -> List[Document]:
    """Recursively read all PDF files under directory_path into LangChain Documents."""
    documents = []
    for pdf_path in Path(directory_path).rglob("*.pdf"):
        try:
            print(f"[INFO] Reading: {pdf_path.name}")
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_num + 1},
                    ))
        except Exception as e:
            print(f"[WARN] Skipping damaged sections in {pdf_path.name}: {e}")
    return documents
