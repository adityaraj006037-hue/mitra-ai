# MITRA — Research Intelligence Platform

> Upload documents. Ask questions. Generate summaries, quizzes, and mind maps. Powered by Groq AI.

**Built by Aditya Raj (Radhe) · IIT Patna Generative AI Programme · Device: Tablet · Cost: Rs 0**

---

## What is MITRA?

MITRA is an AI-powered research assistant — think NotebookLM but faster, smarter, and free to deploy.

Upload your PDFs, notes, or documents into a **Space**, then:
- 💬 **Chat** with your documents using natural language
- 📋 **Summarise** entire document sets in seconds
- ❓ **Quiz yourself** — AI generates questions from your content
- 🗺️ **Mind Maps** — auto-generate structured knowledge maps
- 💡 **Explain** complex concepts in simple language
- 🔍 **Semantic search** — finds meaning, not just keywords

---

## Live Demo

> Frontend: Open `index.html` in any browser  
> Backend: Deploy to Render (see below)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI / LLM | Groq API (LLaMA 3.1 8B Instant) |
| Backend | FastAPI (Python) |
| Vector DB | ChromaDB (in-memory) |
| Embeddings | ChromaDB default embeddings |
| RAG | LangChain text splitter |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Render (free tier) |

---

## Project Structure

```
mitra/
├── backend.py        ← Entire FastAPI backend (single file)
├── requirements.txt  ← Python dependencies
├── render.yaml       ← Render deployment config
├── index.html        ← Complete frontend UI
└── README.md
```

**No nested folders. No routers directory. No complex imports.**  
Everything is flat and simple — designed to deploy on the first try.

---

## Deploy in 5 Steps

### 1. Clone or upload to GitHub
Create a new repo and upload all files to the root. No subfolders.

### 2. Connect to Render
- Go to [render.com](https://render.com)
- New → Web Service → Connect your GitHub repo

### 3. Set environment variable
In Render dashboard → Environment:
```
GROQ_API_KEY = your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 4. Deploy
Render will run:
```
pip install -r requirements.txt
uvicorn backend:app --host 0.0.0.0 --port $PORT
```

### 5. Open the frontend
- Open `index.html` in your browser
- Set the Backend URL to your Render URL (e.g. `https://mitra-backend.onrender.com`)
- Create a Space, upload documents, start chatting

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/spaces/create` | Create a new space |
| GET | `/spaces` | List all spaces |
| DELETE | `/spaces/{id}` | Delete a space |
| POST | `/spaces/{id}/upload` | Upload a document |
| DELETE | `/spaces/{id}/docs/{doc_id}` | Remove a document |
| POST | `/spaces/{id}/chat` | Chat with documents |
| POST | `/spaces/{id}/summary` | Generate summary |
| POST | `/spaces/{id}/quiz` | Generate quiz |
| POST | `/spaces/{id}/mindmap` | Generate mind map |

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export GROQ_API_KEY=your_key_here

# Run backend
python backend.py

# Open frontend
# Just open index.html in your browser
# Set backend URL to http://localhost:8000
```

---

## Why MITRA is Different from NotebookLM

| Feature | NotebookLM | MITRA |
|---------|-----------|-------|
| Cost | Free (Google account) | Free (self-hosted) |
| Quiz generation | ❌ | ✅ |
| Mind maps | ❌ | ✅ |
| Open source | ❌ | ✅ |
| Self-hostable | ❌ | ✅ |
| Custom AI model | ❌ | ✅ |
| API access | ❌ | ✅ |

---

## Built With Zero

- No laptop — built on a **tablet**
- No money — **Rs 0** total cost
- No prior coding experience — started from scratch
- AI Tutor — **Claude AI (Anthropic)**
- Programme — **IIT Patna Generative AI & Agentic AI**

---

## About the Builder

**Aditya Raj (Radhe)** — AI Engineer  
IIT Patna · Generative AI Programme · May 2026

16 lessons · 4 AI products · 3 live deployments · Rs 0 budget · Tablet only

---

*MITRA means "friend" in Sanskrit. This is your research friend.*
