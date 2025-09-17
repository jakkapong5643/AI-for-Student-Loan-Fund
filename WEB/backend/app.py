import os, json, time
from typing import List, Optional
from collections import defaultdict, deque

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from ollama import Client

DEFAULT_CSV  = r"C:\Final Project\Code\Code\Agentic_AI\data\Input\Rag_know.csv"
INDEX_DIR    = r"C:\Final Project\Code\Code\RAG\faiss_index_bgem3"
EMB_MODEL_ID = "BAAI/bge-m3"
DEVICE       = "cuda" if ("CUDA_VISIBLE_DEVICES" in os.environ) else "cpu"

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "typhoon-1b-merged")

TOP_K        = 2
NPROBE       = 32

app = FastAPI(title="RAG API (FAISS + bge-m3 + Ollama + Memory)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL_ID,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )

embeddings = build_embeddings()

def _set_nprobe_if_ivf(db, nprobe: int = NPROBE):
    try:
        index = db.index
        if isinstance(index, faiss.IndexPreTransform):
            base = faiss.downcast_index(index.index)
        else:
            base = faiss.downcast_index(index)
        if isinstance(base, (faiss.IndexIVFFlat, faiss.IndexIVFPQ, faiss.IndexIVFScalarQuantizer, faiss.IndexIVF)):
            base.nprobe = nprobe
    except Exception:
        pass

print(f"[INFO] Loading FAISS index from: {INDEX_DIR}")
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
_set_nprobe_if_ivf(db, NPROBE)
retriever = db.as_retriever(search_kwargs={"k": TOP_K})

ollama_client = Client(host=OLLAMA_HOST)

CONV_HISTORY = defaultdict(lambda: deque(maxlen=20))
CONV_SUMMARY = defaultdict(str)
SUMMARIZE_EVERY = 5 

def add_to_history(session_id: str, role: str, content: str):
    CONV_HISTORY[session_id].append({"role": role, "content": content})

def maybe_summarize(session_id: str):
    history = list(CONV_HISTORY[session_id])
    if len(history) % SUMMARIZE_EVERY == 0 and history:
        text = "\n".join([f"{h['role']}: {h['content']}" for h in history])
        res = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "สรุปบทสนทนานี้เป็นข้อความสั้น ๆ ภาษาไทย"},
                {"role": "user", "content": text},
            ],
        )
        summary = res["message"]["content"]
        CONV_SUMMARY[session_id] = summary
        CONV_HISTORY[session_id].clear()
        CONV_HISTORY[session_id].append({"role": "system", "content": f"[Summary]: {summary}"})

def format_context(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        lines.append(f"[{i}] (source: {meta.get('source','unknown')} | chunk:{meta.get('chunk_id','?')})\n{d.page_content}")
    return "\n\n".join(lines)

def build_prompt(question: str, docs: List[Document], session_id: str):
    context = format_context(docs)
    summary = CONV_SUMMARY.get(session_id, "")
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in CONV_HISTORY[session_id]])

    system = (
        "คุณคือผู้ช่วย RAG ภาษาไทย สำหรับกองทุนเงินให้กู้ยืมเพื่อการศึกษา (กยศ.) "
        "หน้าที่ของคุณคือ:\n"
        "1. ตอบคำถามด้วยความสุภาพ กระชับ ชัดเจน\n"
        "2. ใช้ข้อมูลอ้างอิงจาก Context ที่ให้มาเท่านั้น (ห้ามมโน)\n"
        "3. หาก Context ไม่มีข้อมูลที่เกี่ยวข้อง ให้ตอบว่า 'ไม่ทราบ' เพียงครั้งเดียว\n"
        "4. อธิบายเหตุผลประกอบว่าทำไมถึงตอบเช่นนั้น\n"
        "5. ระบุแหล่งที่มาที่ใช้ (source/chunk) ในคำตอบ\n"
        "6. คำตอบสุดท้ายมีเพียงครั้งเดียวเท่านั้น\n"
    )
    user = f"""คำถาม:
{question}

Conversation Summary (สรุปที่มีอยู่):
{summary}

Conversation History (ล่าสุด):
{history_text}

Context (ใช้ประกอบคำตอบ):
{context}

คำตอบภาษาไทย (กระชับ อธิบายเหตุผล และระบุ source/chunk ที่ใช้อ้างอิง):"""
    return system, user

class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    num_ctx: Optional[int] = 1024
    num_predict: Optional[int] = 512
    temperature: Optional[float] = 0.2

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    t0 = time.time()
    session_id = payload.session_id or "default"

    docs = retriever.get_relevant_documents(payload.question)
    if not docs:
        return AskResponse(answer="ไม่ทราบ (Context ไม่มีข้อมูลที่เกี่ยวข้อง)", sources=[])

    sys_msg, user_msg = build_prompt(payload.question, docs, session_id)

    options = {
        "temperature": payload.temperature,
        "num_ctx": int(payload.num_ctx or 1024),
        "num_predict": int(payload.num_predict or 512),
    }
    out = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role":"system","content":sys_msg},
                  {"role":"user","content":user_msg}],
        options=options,
    )

    answer = out["message"]["content"]

    add_to_history(session_id, "user", payload.question)
    add_to_history(session_id, "assistant", answer)
    maybe_summarize(session_id)

    srcs = []
    for d in docs:
        m = d.metadata or {}
        srcs.append({"source": m.get("source","unknown"),
                     "chunk_id": m.get("chunk_id","?"),
                     "path": m.get("path", m.get("source","unknown"))})

    print(f"[INFO] answered in {time.time()-t0:.2f}s (session={session_id})")
    return AskResponse(answer=answer, sources=srcs)
