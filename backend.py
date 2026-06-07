"""
MITRA Backend — Single File FastAPI
No routers folder. No disk ChromaDB. No deployment problems.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import json
import time
from typing import List, Optional
import traceback

# ── LLM ──────────────────────────────────────────────────────────────────────
from groq import Groq

# ── Document parsing ──────────────────────────────────────────────────────────
import io

# ── Vector store (in-memory — no disk, no permission errors) ─────────────────
import chromadb

# ── Text splitting ────────────────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── PDF / DOCX ────────────────────────────────────────────────────────────────
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# =============================================================================
# APP SETUP
# =============================================================================
app = FastAPI(title="MITRA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# GLOBAL STATE  (in-memory — resets on restart, fine for free tier)
# =============================================================================
# ChromaDB in-memory client — no disk, no permission errors
chroma_client = chromadb.Client()

# Spaces store: { space_id: { name, docs: [...], created_at } }
SPACES: dict = {}

# =============================================================================
# GROQ CLIENT
# =============================================================================
def get_groq():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment")
    return Groq(api_key=api_key)

# =============================================================================
# HELPERS
# =============================================================================
def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    
    if ext == "pdf" and PDF_SUPPORT:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    
    elif ext in ("docx", "doc") and DOCX_SUPPORT:
        doc = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    
    elif ext == "txt":
        return file_bytes.decode("utf-8", errors="ignore")
    
    else:
        # Try plain text fallback
        return file_bytes.decode("utf-8", errors="ignore")


def get_or_create_collection(space_id: str):
    col_name = f"space_{space_id}".replace("-", "_")
    try:
        return chroma_client.get_collection(col_name)
    except Exception:
        return chroma_client.create_collection(col_name)


def embed_document(space_id: str, doc_id: str, text: str, filename: str):
    """Chunk text and add to ChromaDB collection for this space."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    
    collection = get_or_create_collection(space_id)
    
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": filename, "chunk": i} for i in range(len(chunks))]
    
    # ChromaDB uses its own embeddings by default (no API key needed)
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def search_space(space_id: str, query: str, n_results: int = 5) -> List[str]:
    """Semantic search across all docs in a space."""
    try:
        collection = get_or_create_collection(space_id)
        results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


# =============================================================================
# ROUTES — SPACES
# =============================================================================
@app.get("/")
def root():
    return {"status": "MITRA API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "spaces": len(SPACES)}


@app.post("/spaces/create")
def create_space(name: str = Form(...)):
    space_id = str(uuid.uuid4())[:8]
    SPACES[space_id] = {
        "id": space_id,
        "name": name,
        "docs": [],
        "created_at": time.time(),
        "message_count": 0
    }
    # Pre-create the ChromaDB collection
    get_or_create_collection(space_id)
    return {"space_id": space_id, "name": name}


@app.get("/spaces")
def list_spaces():
    return {"spaces": list(SPACES.values())}


@app.get("/spaces/{space_id}")
def get_space(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return SPACES[space_id]


@app.delete("/spaces/{space_id}")
def delete_space(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    # Delete ChromaDB collection
    col_name = f"space_{space_id}".replace("-", "_")
    try:
        chroma_client.delete_collection(col_name)
    except Exception:
        pass
    del SPACES[space_id]
    return {"deleted": space_id}


# =============================================================================
# ROUTES — DOCUMENTS
# =============================================================================
@app.post("/spaces/{space_id}/upload")
async def upload_document(space_id: str, file: UploadFile = File(...)):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    
    file_bytes = await file.read()
    filename = file.filename or "document.txt"
    
    # Extract text
    try:
        text = extract_text(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="File appears to be empty or unreadable")
    
    doc_id = str(uuid.uuid4())[:8]
    
    # Embed into ChromaDB
    try:
        chunk_count = embed_document(space_id, doc_id, text, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    
    doc_info = {
        "id": doc_id,
        "filename": filename,
        "size": len(file_bytes),
        "chunks": chunk_count,
        "chars": len(text),
        "uploaded_at": time.time()
    }
    SPACES[space_id]["docs"].append(doc_info)
    
    return doc_info


@app.delete("/spaces/{space_id}/docs/{doc_id}")
def delete_document(space_id: str, doc_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    
    # Remove from ChromaDB
    collection = get_or_create_collection(space_id)
    try:
        # Get all chunk IDs for this doc
        results = collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception:
        pass
    
    # Remove from SPACES
    SPACES[space_id]["docs"] = [d for d in SPACES[space_id]["docs"] if d["id"] != doc_id]
    return {"deleted": doc_id}


# =============================================================================
# ROUTES — CHAT
# =============================================================================
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    mode: Optional[str] = "chat"  # chat | summary | mindmap | quiz | explain


@app.post("/spaces/{space_id}/chat")
def chat_with_space(space_id: str, req: ChatRequest):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    
    space = SPACES[space_id]
    if not space["docs"]:
        raise HTTPException(status_code=400, detail="No documents in this space. Upload something first.")
    
    # Retrieve relevant context
    context_chunks = search_space(space_id, req.message, n_results=6)
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."
    
    # Build system prompt based on mode
    doc_names = ", ".join(d["filename"] for d in space["docs"])
    
    system_prompts = {
        "chat": f"""You are MITRA, an intelligent research assistant. 
You have access to documents: {doc_names}

Use the context below to answer the user's question accurately.
If the answer isn't in the context, say so clearly.
Be concise, insightful, and cite which document you're referencing when relevant.

CONTEXT FROM DOCUMENTS:
{context}""",

        "summary": f"""You are MITRA. Provide a comprehensive, structured summary of the documents.
Documents: {doc_names}

Format your response with:
- Key Topics
- Main Arguments  
- Important Facts & Data
- Conclusions

CONTEXT:
{context}""",

        "quiz": f"""You are MITRA. Generate 5 insightful quiz questions based on the documents.
Documents: {doc_names}

For each question:
Q1: [Question]
A: [Answer]

Make questions test deep understanding, not just surface facts.

CONTEXT:
{context}""",

        "explain": f"""You are MITRA. Explain the key concepts from these documents in simple terms.
Documents: {doc_names}

Use analogies and examples. Make complex ideas accessible.

CONTEXT:
{context}""",

        "mindmap": f"""You are MITRA. Create a structured mind map outline of the documents.
Documents: {doc_names}

Format:
CENTRAL TOPIC
├── Branch 1
│   ├── Sub-point
│   └── Sub-point
├── Branch 2
...

CONTEXT:
{context}"""
    }
    
    system = system_prompts.get(req.mode, system_prompts["chat"])
    
    # Build messages
    messages = []
    # Add recent history (last 6 messages)
    for msg in (req.history or [])[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.message})
    
    # Call Groq
    try:
        groq = get_groq()
        response = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=1500,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        SPACES[space_id]["message_count"] = SPACES[space_id].get("message_count", 0) + 1
        
        return {
            "answer": answer,
            "sources": [c["filename"] for c in space["docs"]],
            "chunks_used": len(context_chunks),
            "mode": req.mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


# =============================================================================
# ROUTES — SPECIAL FEATURES
# =============================================================================
@app.post("/spaces/{space_id}/summary")
def get_summary(space_id: str):
    """Generate a full document summary."""
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    req = ChatRequest(message="Give me a complete summary of all documents", mode="summary")
    return chat_with_space(space_id, req)


@app.post("/spaces/{space_id}/quiz")
def get_quiz(space_id: str):
    """Generate quiz questions."""
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    req = ChatRequest(message="Generate quiz questions from these documents", mode="quiz")
    return chat_with_space(space_id, req)


@app.post("/spaces/{space_id}/mindmap")
def get_mindmap(space_id: str):
    """Generate mind map structure."""
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    req = ChatRequest(message="Create a detailed mind map of these documents", mode="mindmap")
    return chat_with_space(space_id, req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
