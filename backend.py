"""
MITRA Backend v5.0 — Full Auth + Supabase
- Supabase JWT verification
- Each user sees only their own spaces
- Persistent storage
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, time, io, httpx, re

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YT_SUPPORT = True
except ImportError:
    YT_SUPPORT = False

try:
    from bs4 import BeautifulSoup
    BS_SUPPORT = True
except ImportError:
    BS_SUPPORT = False

# =============================================================================
# CONFIG
# =============================================================================
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# =============================================================================
# APP
# =============================================================================
app = FastAPI(title="MITRA API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CHROMADB
# =============================================================================
chroma_client = chromadb.Client()

# =============================================================================
# AUTH — get user from token
# =============================================================================
async def get_user(request: Request) -> str:
    """Extract user_id from Supabase JWT token."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = auth.replace("Bearer ", "")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": SUPABASE_KEY
                }
            )
        if res.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        data = res.json()
        user_id = data.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Could not get user ID")
        return user_id
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Auth error: " + str(e))

# =============================================================================
# SUPABASE HELPERS
# =============================================================================
def supa_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

async def supa_get(path: str, params: str = ""):
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/{path}{params}",
            headers=supa_headers()
        )
    return res.json()

async def supa_post(path: str, data):
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            f"{SUPABASE_URL}/rest/v1/{path}",
            headers=supa_headers(),
            json=data
        )
    if res.status_code not in (200, 201):
        raise Exception(f"Supabase error: {res.text}")
    return res.json()

async def supa_delete(path: str, params: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.delete(
            f"{SUPABASE_URL}/rest/v1/{path}{params}",
            headers=supa_headers()
        )
    return res.status_code

# =============================================================================
# STARTUP — rebuild ChromaDB
# =============================================================================
@app.on_event("startup")
async def rebuild_chroma():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        chunks = await supa_get("chunks", "?select=*&limit=10000")
        if not isinstance(chunks, list):
            return
        by_space = {}
        for chunk in chunks:
            sid = chunk.get("space_id")
            if sid not in by_space:
                by_space[sid] = []
            by_space[sid].append(chunk)

        for space_id, space_chunks in by_space.items():
            col_name = "space_" + space_id.replace("-", "_")
            try:
                col = chroma_client.get_or_create_collection(col_name)
                existing = col.get()["ids"]
                new_chunks = [c for c in space_chunks if c["id"] not in existing]
                if new_chunks:
                    col.add(
                        documents=[c["content"] for c in new_chunks],
                        ids=[c["id"] for c in new_chunks],
                        metadatas=[{"doc_id": c["doc_id"], "filename": c["filename"], "chunk": c["chunk_index"]} for c in new_chunks]
                    )
            except Exception as e:
                print(f"Error restoring space {space_id}: {e}")

        print(f"Restored {len(chunks)} chunks from Supabase")
    except Exception as e:
        print(f"Startup restore error: {e}")

# =============================================================================
# GROQ
# =============================================================================
async def call_groq(system: str, messages: list, max_tokens: int = 2000):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    payload = {
        "model": "llama-3.1-8b-instant",
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "messages": [{"role": "system", "content": system}] + messages
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Groq API error: " + response.text)
    return response.json()["choices"][0]["message"]["content"]

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
    else:
        return file_bytes.decode("utf-8", errors="ignore")

def get_or_create_collection(space_id: str):
    col_name = "space_" + space_id.replace("-", "_")
    try:
        return chroma_client.get_collection(col_name)
    except Exception:
        return chroma_client.create_collection(col_name)

async def embed_document(space_id: str, doc_id: str, text: str, filename: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    if not chunks:
        return 0

    collection = get_or_create_collection(space_id)
    chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": filename, "chunk": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=chunk_ids, metadatas=metadatas)

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            chunk_rows = [{"id": chunk_ids[i], "space_id": space_id, "doc_id": doc_id, "filename": filename, "content": chunks[i], "chunk_index": i} for i in range(len(chunks))]
            for i in range(0, len(chunk_rows), 100):
                await supa_post("chunks", chunk_rows[i:i+100])
        except Exception as e:
            print(f"Chunk save error: {e}")

    return len(chunks)

def search_space(space_id: str, query: str, n_results: int = 6):
    try:
        collection = get_or_create_collection(space_id)
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(query_texts=[query], n_results=min(n_results, count))
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []

def extract_youtube_id(url: str) -> str:
    patterns = [r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})']
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""

def build_system(mode: str, doc_names: str, context: str) -> str:
    base = f"Sources: {doc_names}\n\nCONTEXT:\n{context}"
    prompts = {
        "chat":      f"You are MITRA, an intelligent research assistant. Answer accurately using the context.\n\n{base}",
        "summary":   f"You are MITRA. Give a structured summary with Key Topics, Main Arguments, Important Facts, Conclusions.\n\n{base}",
        "quiz":      f"You are MITRA. Generate 5 quiz questions with detailed answers.\n\n{base}",
        "explain":   f"You are MITRA. Explain key concepts simply using analogies.\n\n{base}",
        "mindmap":   f"You are MITRA. Create a mind map:\nCENTRAL TOPIC\n├── Branch 1\n│   ├── Sub-point\n\n{base}",
        "flashcard": f"You are MITRA. Generate 15 flashcards:\nFRONT: [term]\nBACK: [answer]\n---\n\n{base}",
        "studyplan": f"You are MITRA. Create a detailed 30-day study plan. For each day: Day number, Topic, Time estimate, Key tasks.\n\n{base}",
        "debate":    f"You are MITRA. Present BOTH sides:\nSIDE A — FOR:\n[arguments]\n\nSIDE B — AGAINST:\n[arguments]\n\nVERDICT: [conclusion]\n\n{base}",
        "eli5":      f"You are MITRA. Explain everything like the reader is 10 years old. Simple words, fun analogies.\n\n{base}",
        "glossary":  f"You are MITRA. Extract key terms:\nTERM: [word]\nDEFINITION: [definition]\n---\nExtract at least 15 terms.\n\n{base}",
    }
    return prompts.get(mode, prompts["chat"])

# =============================================================================
# ROUTES — HEALTH (no auth needed)
# =============================================================================
@app.get("/")
def root():
    return {"status": "MITRA API running", "version": "5.0.0"}

@app.get("/health")
async def health():
    count = 0
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            spaces = await supa_get("spaces", "?select=id")
            count = len(spaces) if isinstance(spaces, list) else 0
        except:
            pass
    return {"status": "ok", "version": "5.0.0", "spaces": count}

# =============================================================================
# ROUTES — SPACES (auth required)
# =============================================================================
@app.post("/spaces/create")
async def create_space(request: Request, name: str = Form(...)):
    user_id = await get_user(request)
    space_id = str(uuid.uuid4())[:8]
    space_data = {
        "id": space_id,
        "name": name,
        "user_id": user_id,
        "message_count": 0,
        "created_at": time.time()
    }
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            await supa_post("spaces", space_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Could not save space: " + str(e))
    get_or_create_collection(space_id)
    return {"space_id": space_id, "name": name}

@app.get("/spaces")
async def list_spaces(request: Request):
    user_id = await get_user(request)
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            # Only return spaces belonging to this user
            spaces = await supa_get("spaces", f"?user_id=eq.{user_id}&select=*,documents(*)")
            if not isinstance(spaces, list):
                return {"spaces": []}
            result = []
            for s in spaces:
                docs = s.get("documents", []) or []
                result.append({
                    "id": s["id"],
                    "name": s["name"],
                    "message_count": s.get("message_count", 0),
                    "created_at": s.get("created_at", 0),
                    "docs": docs
                })
            return {"spaces": result}
        except Exception as e:
            return {"spaces": []}
    return {"spaces": []}

@app.delete("/spaces/{space_id}")
async def delete_space(request: Request, space_id: str):
    user_id = await get_user(request)
    if SUPABASE_URL and SUPABASE_KEY:
        # Verify ownership before deleting
        spaces = await supa_get("spaces", f"?id=eq.{space_id}&user_id=eq.{user_id}&select=id")
        if not spaces or not isinstance(spaces, list) or len(spaces) == 0:
            raise HTTPException(status_code=403, detail="Not authorized to delete this space")
        await supa_delete("spaces", f"?id=eq.{space_id}")
    try:
        chroma_client.delete_collection("space_" + space_id.replace("-", "_"))
    except:
        pass
    return {"deleted": space_id}

# =============================================================================
# ROUTES — DOCUMENTS
# =============================================================================
@app.post("/spaces/{space_id}/upload")
async def upload_document(request: Request, space_id: str, file: UploadFile = File(...)):
    user_id = await get_user(request)
    file_bytes = await file.read()
    filename = file.filename or "document.txt"

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    try:
        text = extract_text(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read file: " + str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty or unreadable")

    doc_id = str(uuid.uuid4())[:8]
    chunk_count = await embed_document(space_id, doc_id, text, filename)

    doc_info = {
        "id": doc_id,
        "space_id": space_id,
        "filename": filename,
        "type": "file",
        "size": len(file_bytes),
        "chunks": chunk_count,
        "chars": len(text),
        "uploaded_at": time.time()
    }

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            await supa_post("documents", doc_info)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Could not save document: " + str(e))

    return doc_info

@app.post("/spaces/{space_id}/youtube")
async def import_youtube(request: Request, space_id: str, url: str = Body(..., embed=True)):
    user_id = await get_user(request)
    video_id = extract_youtube_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    if not YT_SUPPORT:
        raise HTTPException(status_code=500, detail="YouTube support not available")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi', 'en-IN'])
        text = " ".join([t['text'] for t in transcript_list])
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not get transcript. Video may not have captions.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty")

    doc_id = str(uuid.uuid4())[:8]
    filename = f"YouTube_{video_id}.txt"
    chunk_count = await embed_document(space_id, doc_id, text, filename)

    doc_info = {
        "id": doc_id,
        "space_id": space_id,
        "filename": filename,
        "type": "youtube",
        "url": url,
        "size": len(text),
        "chunks": chunk_count,
        "chars": len(text),
        "uploaded_at": time.time()
    }

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            await supa_post("documents", doc_info)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Could not save document: " + str(e))

    return doc_info

@app.post("/spaces/{space_id}/url")
async def import_url(request: Request, space_id: str, url: str = Body(..., embed=True)):
    user_id = await get_user(request)
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Could not fetch URL: HTTP {response.status_code}")

        if BS_SUPPORT:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            text = soup.get_text(separator='\n', strip=True)
            lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 30]
            text = '\n'.join(lines)
        else:
            text = re.sub(r'<[^>]+>', ' ', response.text)
            text = re.sub(r'\s+', ' ', text).strip()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not fetch URL: " + str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable content found")

    text = text[:50000]
    doc_id = str(uuid.uuid4())[:8]
    domain = url.split('/')[2] if '/' in url else url
    filename = f"Web_{domain}.txt"
    chunk_count = await embed_document(space_id, doc_id, text, filename)

    doc_info = {
        "id": doc_id,
        "space_id": space_id,
        "filename": filename,
        "type": "url",
        "url": url,
        "size": len(text),
        "chunks": chunk_count,
        "chars": len(text),
        "uploaded_at": time.time()
    }

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            await supa_post("documents", doc_info)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Could not save document: " + str(e))

    return doc_info

@app.delete("/spaces/{space_id}/docs/{doc_id}")
async def delete_document(request: Request, space_id: str, doc_id: str):
    user_id = await get_user(request)
    collection = get_or_create_collection(space_id)
    try:
        results = collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            collection.delete(ids=results["ids"])
    except:
        pass

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            await supa_delete("chunks", f"?doc_id=eq.{doc_id}")
            await supa_delete("documents", f"?id=eq.{doc_id}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"deleted": doc_id}

# =============================================================================
# ROUTES — CHAT
# =============================================================================
@app.post("/spaces/{space_id}/chat")
async def chat_with_space(
    request: Request,
    space_id: str,
    message: str = Body(...),
    history: list = Body(default=[]),
    mode: str = Body(default="chat")
):
    user_id = await get_user(request)

    docs = []
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            docs = await supa_get("documents", f"?space_id=eq.{space_id}&select=*")
            if not isinstance(docs, list):
                docs = []
        except:
            pass

    if not docs:
        raise HTTPException(status_code=400, detail="No documents in this space. Upload something first.")

    context_chunks = search_space(space_id, message, n_results=6)
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."
    doc_names = ", ".join(d["filename"] for d in docs)
    system = build_system(mode, doc_names, context)

    messages = []
    for msg in (history or [])[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        answer = await call_groq(system, messages)
        return {"answer": answer, "sources": [d["filename"] for d in docs], "chunks_used": len(context_chunks), "mode": mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail="LLM error: " + str(e))

# =============================================================================
# SHORTCUT ENDPOINTS
# =============================================================================
@app.post("/spaces/{space_id}/summary")
async def get_summary(request: Request, space_id: str):
    return await chat_with_space(request, space_id, message="Give me a complete structured summary", history=[], mode="summary")

@app.post("/spaces/{space_id}/quiz")
async def get_quiz(request: Request, space_id: str):
    return await chat_with_space(request, space_id, message="Generate 5 quiz questions with answers", history=[], mode="quiz")

@app.post("/spaces/{space_id}/mindmap")
async def get_mindmap(request: Request, space_id: str):
    return await chat_with_space(request, space_id, message="Create a detailed mind map", history=[], mode="mindmap")

@app.post("/spaces/{space_id}/flashcards")
async def get_flashcards(request: Request, space_id: str):
    return await chat_with_space(request, space_id, message="Generate 15 flashcards", history=[], mode="flashcard")

@app.post("/spaces/{space_id}/studyplan")
async def get_studyplan(request: Request, space_id: str):
    return await chat_with_space(request, space_id, message="Create a 30-day study plan", history=[], mode="studyplan")

@app.post("/spaces/{space_id}/debate")
async def get_debate(request: Request, space_id: str):
    return await chat_with_space(request, space_id, message="Present both sides", history=[], mode="debate")

@app.post("/spaces/{space_id}/eli5")
async def get_eli5(request: Request, space_id: str):
    return await chat_with_space(request, space_id, message="Explain like I am 10", history=[], mode="eli5")

@app.post("/spaces/{space_id}/glossary")
async def get_glossary(request: Request, space_id: str):
    return await chat_with_space(request, space_id, message="Generate a glossary", history=[], mode="glossary")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
