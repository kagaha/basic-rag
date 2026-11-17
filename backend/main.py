"""FastAPI backend for the Basic RAG web app.

Wraps the same building blocks as the command-line scripts (BGE embeddings,
ChromaDB with cosine similarity, Gemini, history-aware question rewriting)
behind a small REST API.

Design notes
- Each upload creates an in-memory Chroma collection tied to a session id, so
  nothing is written to disk and different visitors never see each other's
  documents.
- The Gemini API key is sent by the browser in the `X-Api-Key` header on each
  request. It is used for that request only: it is never stored, logged or
  returned.

Run from the repository root:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

# Allow `from ingestion_pipeline import ...` when started from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

# Ingestion (normalising, chunking, embedding, storing) is shared with the
# command-line scripts, so any improvement there applies here too.
from ingestion_pipeline import (
    DEFAULT_CHUNK_OVERLAP as CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE as CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    create_vector_store,
    documents_from_texts,
    get_embedding_model,
    split_documents,
)

# ---------------------------------------------------------------- settings --

LLM_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
TOP_K = 6  # chunks retrieved per question

MAX_FILE_BYTES = 5 * 1024 * 1024  # per uploaded file
MAX_TOTAL_CHARS = 2_000_000       # across all documents in one upload
MAX_QUESTION_CHARS = 2000
MAX_SESSIONS = 20                 # oldest sessions are dropped beyond this
HISTORY_MESSAGES = 12             # most recent messages sent back to the model

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

# ------------------------------------------------------------------- state --

_embeddings = None
_embeddings_lock = threading.Lock()


def get_embeddings():
    """Load the embedding model once and reuse it for every request."""
    global _embeddings
    with _embeddings_lock:
        if _embeddings is None:
            _embeddings = get_embedding_model()
        return _embeddings


class Session:
    def __init__(self, db: Chroma):
        self.db = db
        self.history: list = []
        self.created = time.time()
        self.lock = threading.Lock()


SESSIONS: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def _drop_session(session: Session) -> None:
    try:
        session.db.delete_collection()
    except Exception:
        pass


def register_session(session_id: str, session: Session) -> None:
    with _sessions_lock:
        SESSIONS[session_id] = session
        while len(SESSIONS) > MAX_SESSIONS:
            oldest_id = min(SESSIONS, key=lambda k: SESSIONS[k].created)
            _drop_session(SESSIONS.pop(oldest_id))


# ----------------------------------------------------------------- helpers --


def make_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=api_key)


def to_text(response) -> str:
    """Gemini can return a list of content blocks; normalise to plain text."""
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content.strip()


def safe_error(exc: Exception, api_key: str) -> str:
    """Short error message that can never contain the API key."""
    message = str(exc).strip() or exc.__class__.__name__
    if api_key:
        message = message.replace(api_key, "***")
    return message[:400]


def require_key(x_api_key: Optional[str]) -> str:
    key = (x_api_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Please enter your Gemini API key.")
    return key


def decode_text(raw: bytes, name: str) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail=f"{name} is not valid UTF-8 text."
        )


# --------------------------------------------------------------------- app --

app = FastAPI(title="Basic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Basic RAG API is running. Open the web app at http://localhost:5173",
        "docs": "/docs",
    }


@app.get("/api/config")
def config():
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "vector_store": "ChromaDB (cosine similarity)",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
    }


@app.post("/api/documents")
def upload_documents(
    files: list[UploadFile] = File(default=[]),
    texts: list[str] = Form(default=[]),
    text_names: list[str] = Form(default=[]),
    x_api_key: Optional[str] = Header(default=None),
):
    api_key = require_key(x_api_key)

    # 1) Collect (name, text) pairs from uploaded .txt files and pasted text.
    items: list[tuple[str, str]] = []

    for upload in files:
        name = upload.filename or "document.txt"
        if not name.lower().endswith(".txt"):
            raise HTTPException(
                status_code=400, detail=f"{name}: only .txt files are supported."
            )
        raw = upload.file.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"{name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.",
            )
        items.append((name, decode_text(raw, name)))

    for i, content in enumerate(texts):
        name = text_names[i].strip() if i < len(text_names) and text_names[i].strip() else f"Pasted text {i + 1}"
        items.append((name, content))

    # Normalises line endings and skips empty texts (shared with the CLI).
    docs = documents_from_texts(items)

    if not docs:
        raise HTTPException(
            status_code=400, detail="Add at least one non-empty .txt document."
        )

    total_chars = sum(len(d.page_content) for d in docs)
    if total_chars > MAX_TOTAL_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Documents are too large ({total_chars:,} characters). The limit is {MAX_TOTAL_CHARS:,}.",
        )

    # 2) Check the key early, before spending time on embeddings.
    try:
        make_llm(api_key).invoke("Reply with the single word: OK")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Gemini rejected this API key or request: {safe_error(exc, api_key)}",
        )

    # 3) Chunk the documents (same function as the CLI; drops citation lists).
    chunks = split_documents(docs, verbose=False)
    if not chunks:
        raise HTTPException(status_code=400, detail="No usable text found in the documents.")

    # 4) Embed and store in an in-memory Chroma collection for this session.
    session_id = uuid.uuid4().hex
    db = create_vector_store(
        chunks,
        persist_directory=None,
        collection_name=f"s{session_id}",
        embedding_model=get_embeddings(),
        verbose=False,
    )
    register_session(session_id, Session(db))

    return {
        "session_id": session_id,
        "documents": [
            {"name": d.metadata["source"], "characters": len(d.page_content)}
            for d in docs
        ],
        "chunks": len(chunks),
        "characters": total_chars,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
    }


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.post("/api/ask")
def ask(req: AskRequest, x_api_key: Optional[str] = Header(default=None)):
    api_key = require_key(x_api_key)

    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="This session has expired. Please upload your documents again.",
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Questions are limited to {MAX_QUESTION_CHARS} characters.",
        )

    llm = make_llm(api_key)

    with session.lock:
        history = session.history[-HISTORY_MESSAGES:]

        try:
            # Step 1: turn a follow-up into a standalone, searchable question.
            search_question = question
            rewritten = None
            if history:
                rewrite_messages = (
                    [
                        SystemMessage(
                            content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
                        )
                    ]
                    + history
                    + [HumanMessage(content=f"New question: {question}")]
                )
                candidate = to_text(llm.invoke(rewrite_messages))
                if candidate:
                    search_question = candidate
                    if candidate != question:
                        rewritten = candidate

            # Step 2: retrieve the most relevant chunks with their scores.
            results = session.db.similarity_search_with_relevance_scores(
                search_question, k=TOP_K
            )

            if not results:
                answer = "I don't have enough information to answer that question based on the provided documents."
            else:
                context = "\n".join(f"- {doc.page_content}" for doc, _ in results)
                combined_input = f"""Based on the following documents, please answer this question: {question}

Documents:
{context}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
                # Step 3: generate the grounded answer.
                messages = (
                    [
                        SystemMessage(
                            content="You are a helpful assistant that answers questions based on provided documents and conversation history."
                        )
                    ]
                    + history
                    + [HumanMessage(content=combined_input)]
                )
                answer = to_text(llm.invoke(messages))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini request failed: {safe_error(exc, api_key)}",
            )

        # Step 4: remember the exchange for follow-up questions.
        session.history.append(HumanMessage(content=question))
        session.history.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "rewritten_question": rewritten,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "score": round(float(score), 3),
                "content": doc.page_content,
            }
            for doc, score in results
        ],
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
    }


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    with _sessions_lock:
        session = SESSIONS.pop(session_id, None)
    if session is not None:
        _drop_session(session)
    return {"ok": True}