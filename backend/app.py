import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from retrieval.agent_engine import generate_agent_response

# Load .env from project root — works whether launched from root or backend/
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

if os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY").strip().replace('"', "").replace("'", "")

app = FastAPI(title="Agentic Graph RAG Assistant API")


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    source_filter: str = None
    history: Optional[List[Message]] = []


@app.get("/", response_class=HTMLResponse)
def read_root():
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        return HTMLResponse(
            content="<h3>Frontend Error: index.html not found. Check backend/templates/.</h3>",
            status_code=404,
        )
    return template_path.read_text(encoding="utf-8")


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/query")
def handle_query(payload: QueryRequest):
    print(f"[INFO] Query: '{payload.query}' | k={payload.k} | filter={payload.source_filter}")
    return generate_agent_response(
        query_text=payload.query,
        chunk_count=payload.k,
        source_filter=payload.source_filter,
        history=[{"role": m.role, "content": m.content} for m in (payload.history or [])],
    )
