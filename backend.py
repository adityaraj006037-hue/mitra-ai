"""
MITRA Backend — Single File FastAPI
No pydantic models. No routers. No disk ChromaDB.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, time, io
from typing import List, Optional, Annotated

# ── LLM ──────────────────────────────────────────────────────────────────────
from groq import Groq

# ── Vector store (in-memory) ──────────────────────────────────────────────────
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
# APP
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
# GLOBAL STATE
# =============================================================================
chroma_client = chromadb.Client()
SPACES = {}

# =============================================================================
# HELPERS
# =============================================================================
def get_groq():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    return Groq(api_key=api_key)

def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf" and PDF_SUPPORT:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext in ("docx", "doc") and DOCX_SUPPORT:
        doc = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return file_bytes.decode("utf-8", errors="ignore")

def get_or_create_collection(space_id: str):
    col_name = "space_" + space_id.replace("-", "_")
    try:
        return chroma_client.get_collection(col_name)
    except Exception:
        return chroma_client.create_collection(col_name)

def embed_document(space_id: str, doc_id: str, text: str, filename: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    collection = get_or_create_collection(space_id)
    ids = [doc_id + "_chunk_" + str(i) for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": filename, "chunk": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)

def search_space(space_id: str, query: str, n_results: int = 5):
    try:
        collection = get_or_create_collection(space_id)
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(query_texts=[query], n_results=min(n_results, count))
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []

# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
def root():
    return {"status": "MITRA API running", "version": "1.0.0"}

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
    col_name = "space_" + space_id.replace("-", "_")
    try:
        chroma_client.delete_collection(col_name)
    except Exception:
        pass
    del SPACES[space_id]
    return {"deleted": space_id}

@app.post("/spaces/{space_id}/upload")
async def upload_document(space_id: str, file: UploadFile = File(...)):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    file_bytes = await file.read()
    filename = file.filename or "document.txt"
    try:
        text = extract_text(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read file: " + str(e))
    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty or unreadable")
    doc_id = str(uuid.uuid4())[:8]
    try:
        chunk_count = embed_document(space_id, doc_id, text, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Embedding failed: " + str(e))
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
    collection = get_or_create_collection(space_id)
    try:
        results = collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except Exception:
        pass
    SPACES[space_id]["docs"] = [d for d in SPACES[space_id]["docs"] if d["id"] != doc_id]
    return {"deleted": doc_id}

@app.post("/spaces/{space_id}/chat")
def chat_with_space(
    space_id: str,
    message: str = Body(...),
    history: list = Body(default=[]),
    mode: str = Body(default="chat")
):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    space = SPACES[space_id]
    if not space["docs"]:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")

    context_chunks = search_space(space_id, message, n_results=6)
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."
    doc_names = ", ".join(d["filename"] for d in space["docs"])

    system_prompts = {
        "chat": "You are MITRA, an intelligent research assistant.\nDocuments: " + doc_names + "\nAnswer accurately using the context below. Cite sources when relevant.\n\nCONTEXT:\n" + context,
        "summary": "You are MITRA. Give a structured summary with Key Topics, Main Arguments, Important Facts, and Conclusions.\nDocuments: " + doc_names + "\n\nCONTEXT:\n" + context,
        "quiz": "You are MITRA. Generate 5 insightful quiz questions with answers.\nDocuments: " + doc_names + "\n\nCONTEXT:\n" + context,
        "explain": "You are MITRA. Explain key concepts simply with analogies.\nDocuments: " + doc_names + "\n\nCONTEXT:\n" + context,
        "mindmap": "You are MITRA. Create a structured mind map outline using tree format with branches and sub-points.\nDocuments: " + doc_names + "\n\nCONTEXT:\n" + context,
    }

    system = system_prompts.get(mode, system_prompts["chat"])
    messages = []
    for msg in (history or [])[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

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
        return {"answer": answer, "sources": [d["filename"] for d in space["docs"]], "chunks_used": len(context_chunks), "mode": mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail="LLM error: " + str(e))

@app.post("/spaces/{space_id}/summary")
def get_summary(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return chat_with_space(space_id, message="Summarise all documents", history=[], mode="summary")

@app.post("/spaces/{space_id}/quiz")
def get_quiz(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return chat_with_space(space_id, message="Generate quiz questions", history=[], mode="quiz")

@app.post("/spaces/{space_id}/mindmap")
def get_mindmap(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return chat_with_space(space_id, message="Create a mind map", history=[], mode="mindmap")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
