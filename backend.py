"""
MITRA Backend v3.0 — Full Feature
- YouTube transcript import
- URL/website import  
- All existing features preserved
- Direct HTTP to Groq, no library conflicts
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
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
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# =============================================================================
# APP
# =============================================================================
app = FastAPI(title="MITRA API", version="3.0.0")

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
# GROQ VIA DIRECT HTTP
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
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
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

def embed_document(space_id: str, doc_id: str, text: str, filename: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    if not chunks:
        return 0
    collection = get_or_create_collection(space_id)
    ids = [doc_id + "_chunk_" + str(i) for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": filename, "chunk": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
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
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""

def build_system(mode: str, doc_names: str, context: str) -> str:
    base = f"Documents/Sources: {doc_names}\n\nCONTEXT:\n{context}"
    prompts = {
        "chat":      f"You are MITRA, an intelligent research assistant. Answer accurately using the context. Cite sources when relevant.\n\n{base}",
        "summary":   f"You are MITRA. Give a structured summary with: Key Topics, Main Arguments, Important Facts, Conclusions.\n\n{base}",
        "quiz":      f"You are MITRA. Generate 5 insightful quiz questions with detailed answers. Format: Q1: [question]\nA: [answer]\n\n{base}",
        "explain":   f"You are MITRA. Explain key concepts simply using analogies and examples. Make complex ideas accessible.\n\n{base}",
        "mindmap":   f"You are MITRA. Create a structured mind map using tree format:\nCENTRAL TOPIC\n├── Branch 1\n│   ├── Sub-point\n│   └── Sub-point\n├── Branch 2\n\n{base}",
        "flashcard": f"You are MITRA. Generate 15 flashcards. Format exactly like this:\nFRONT: [term or question]\nBACK: [definition or answer]\n---\nFRONT: [next term]\nBACK: [next answer]\n---\n\n{base}",
        "studyplan": f"You are MITRA. Create a detailed 30-day study plan based on these documents. For each day include: Day number, Topic, Time estimate, Key tasks, and Revision notes. Make it practical and achievable.\n\n{base}",
        "debate":    f"You are MITRA. Present BOTH sides of the main argument in these documents.\nFormat:\nSIDE A — FOR:\n[3-4 strong arguments]\n\nSIDE B — AGAINST:\n[3-4 strong counterarguments]\n\nVERDICT: [balanced conclusion]\n\n{base}",
        "eli5":      f"You are MITRA. Explain everything in these documents like the reader is 10 years old. Use simple words, fun analogies, and short sentences. No jargon.\n\n{base}",
        "glossary":  f"You are MITRA. Extract all important terms and create a glossary. Format:\nTERM: [word or phrase]\nDEFINITION: [clear simple definition]\n---\n\nExtract at least 15 key terms.\n\n{base}",
    }
    return prompts.get(mode, prompts["chat"])

# =============================================================================
# ROUTES — HEALTH
# =============================================================================
@app.get("/")
def root():
    return {"status": "MITRA API running", "version": "3.0.0"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "spaces": len(SPACES),
        "youtube_support": YT_SUPPORT,
        "url_support": BS_SUPPORT
    }

# =============================================================================
# ROUTES — SPACES
# =============================================================================
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
    try:
        chroma_client.delete_collection("space_" + space_id.replace("-", "_"))
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

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Maximum 10MB.")

    try:
        text = extract_text(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read file: " + str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty or unreadable")

    doc_id = str(uuid.uuid4())[:8]
    chunk_count = embed_document(space_id, doc_id, text, filename)

    doc_info = {
        "id": doc_id,
        "filename": filename,
        "type": "file",
        "size": len(file_bytes),
        "chunks": chunk_count,
        "chars": len(text),
        "uploaded_at": time.time()
    }
    SPACES[space_id]["docs"].append(doc_info)
    return doc_info

# =============================================================================
# ROUTES — YOUTUBE IMPORT
# =============================================================================
@app.post("/spaces/{space_id}/youtube")
async def import_youtube(space_id: str, url: str = Body(..., embed=True)):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")

    video_id = extract_youtube_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    if not YT_SUPPORT:
        raise HTTPException(status_code=500, detail="YouTube support not available")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi', 'en-IN'])
        text = " ".join([t['text'] for t in transcript_list])
        timestamps = [{"time": t['start'], "text": t['text']} for t in transcript_list[::10]]
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not get transcript. Video may not have captions. Error: " + str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty")

    doc_id = str(uuid.uuid4())[:8]
    filename = f"YouTube_{video_id}.txt"
    chunk_count = embed_document(space_id, doc_id, text, filename)

    doc_info = {
        "id": doc_id,
        "filename": filename,
        "type": "youtube",
        "url": url,
        "video_id": video_id,
        "size": len(text),
        "chunks": chunk_count,
        "chars": len(text),
        "timestamps": timestamps[:20],
        "uploaded_at": time.time()
    }
    SPACES[space_id]["docs"].append(doc_info)
    return doc_info

# =============================================================================
# ROUTES — URL IMPORT
# =============================================================================
@app.post("/spaces/{space_id}/url")
async def import_url(space_id: str, url: str = Body(..., embed=True)):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Could not fetch URL: HTTP {response.status_code}")

        if BS_SUPPORT:
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            title = soup.find('title')
            title_text = title.get_text().strip() if title else url
            text = soup.get_text(separator='\n', strip=True)
            lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 30]
            text = '\n'.join(lines)
        else:
            text = re.sub(r'<[^>]+>', ' ', response.text)
            text = re.sub(r'\s+', ' ', text).strip()
            title_text = url

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not fetch URL: " + str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable content found at this URL")

    text = text[:50000]

    doc_id = str(uuid.uuid4())[:8]
    domain = url.split('/')[2] if '/' in url else url
    filename = f"Web_{domain}.txt"
    chunk_count = embed_document(space_id, doc_id, text, filename)

    doc_info = {
        "id": doc_id,
        "filename": filename,
        "type": "url",
        "url": url,
        "title": title_text if BS_SUPPORT else url,
        "size": len(text),
        "chunks": chunk_count,
        "chars": len(text),
        "uploaded_at": time.time()
    }
    SPACES[space_id]["docs"].append(doc_info)
    return doc_info

# =============================================================================
# ROUTES — DELETE DOCUMENT
# =============================================================================
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

# =============================================================================
# ROUTES — CHAT (all modes)
# =============================================================================
@app.post("/spaces/{space_id}/chat")
async def chat_with_space(
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
    system = build_system(mode, doc_names, context)

    messages = []
    for msg in (history or [])[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        answer = await call_groq(system, messages)
        SPACES[space_id]["message_count"] = SPACES[space_id].get("message_count", 0) + 1
        return {
            "answer": answer,
            "sources": [d["filename"] for d in space["docs"]],
            "chunks_used": len(context_chunks),
            "mode": mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="LLM error: " + str(e))

# =============================================================================
# ROUTES — SHORTCUT ENDPOINTS
# =============================================================================
@app.post("/spaces/{space_id}/summary")
async def get_summary(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return await chat_with_space(space_id, message="Summarise all documents", history=[], mode="summary")

@app.post("/spaces/{space_id}/quiz")
async def get_quiz(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return await chat_with_space(space_id, message="Generate quiz questions", history=[], mode="quiz")

@app.post("/spaces/{space_id}/mindmap")
async def get_mindmap(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return await chat_with_space(space_id, message="Create a mind map", history=[], mode="mindmap")

@app.post("/spaces/{space_id}/flashcards")
async def get_flashcards(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return await chat_with_space(space_id, message="Generate flashcards", history=[], mode="flashcard")

@app.post("/spaces/{space_id}/studyplan")
async def get_studyplan(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return await chat_with_space(space_id, message="Create a study plan", history=[], mode="studyplan")

@app.post("/spaces/{space_id}/debate")
async def get_debate(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return await chat_with_space(space_id, message="Present both sides of the argument", history=[], mode="debate")

@app.post("/spaces/{space_id}/eli5")
async def get_eli5(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return await chat_with_space(space_id, message="Explain like I am 10", history=[], mode="eli5")

@app.post("/spaces/{space_id}/glossary")
async def get_glossary(space_id: str):
    if space_id not in SPACES:
        raise HTTPException(status_code=404, detail="Space not found")
    return await chat_with_space(space_id, message="Generate a glossary of key terms", history=[], mode="glossary")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
