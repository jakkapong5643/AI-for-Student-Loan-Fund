import os, argparse, uuid, json, time, glob
from datetime import datetime
from collections import Counter
from math import sqrt

import pandas as pd
import torch
import numpy as np
import faiss
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from ollama import Client

DEFAULT_CSV  = r"C:\Final Project\Code\Code\Agentic_AI\data\Input\Rag_know.csv"
INDEX_DIR    = r"C:\Final Project\Code\Code\RAG\faiss_index_bgem3"
EMB_MODEL_ID = "BAAI/bge-m3"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
OLLAMA_HOST  = "http://127.0.0.1:11434"
OLLAMA_MODEL = "typhoon-1b-merged"



USE_IVFPQ = True  
IVF_NLIST = 4096
PQ_M      = 64 
NPROBE    = 32

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
TOP_K         = 4

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not isinstance(text, str) or not text.strip():
        return []
    s = text.strip()
    chunks, start, n = [], 0, len(s)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(s[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    required = {"filename", "cleaned_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV : {missing} Missing")
    if "question_text" not in df.columns: df["question_text"] = None
    if "answer_text" not in df.columns:   df["answer_text"] = None
    if "path" not in df.columns:          df["path"] = df["filename"]
    return df

def pick_batch_size():
    if DEVICE != "cuda":
        return 16
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        if vram_gb <= 4:   return 8
        if vram_gb <= 8:   return 32
        if vram_gb <= 16:  return 48
        return 64
    except Exception:
        return 16

def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL_ID,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": pick_batch_size()},
    )

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

def _save_with_meta(db, meta: dict):
    os.makedirs(INDEX_DIR, exist_ok=True)
    t1 = time.time()
    print("[INFO] Save Index")
    db.save_local(INDEX_DIR)
    print(f"save_local {time.time()-t1:.1f}s")

    with open(os.path.join(INDEX_DIR, "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _precompute_embeddings(docs, embeddings):
    texts = [d.page_content for d in docs]
    vecs  = embeddings.embed_documents(texts)
    X = np.asarray(vecs, dtype="float32")
    return X

def _build_faiss_ivfpq(X: np.ndarray, docs, embeddings):
    n, d = X.shape
    metric = faiss.METRIC_INNER_PRODUCT 
    nlist = min(IVF_NLIST, max(256, int(sqrt(max(n, 1))) * 8))
    index_desc = f"IVF{nlist},PQ{PQ_M}"

    index = faiss.index_factory(d, index_desc, metric)
    t_tr = time.time()
    print(f"[INFO] Train IVF+PQ ({index_desc}) vector{n:,}x{d} ...")
    index.train(X)
    print(f"train {time.time()-t_tr:.1f}s")

    t_add = time.time()
    index.add(X)
    print(f"add({n:,}) {time.time()-t_add:.1f}s")

    index.nprobe = NPROBE

    ids = [d.metadata.get("id", f"{i}") for i, d in enumerate(docs)]
    docstore = InMemoryDocstore(dict(zip(ids, docs)))
    index_to_docstore_id = {i: ids[i] for i in range(len(ids))}
    vs = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    return vs, index_desc

def build_or_load_index(csv_path=DEFAULT_CSV, rebuild=False, include_faq_as_docs=False):
    embeddings = build_embeddings()

    if (not rebuild) and os.path.exists(INDEX_DIR):
        print(f"[INFO] Load Index {INDEX_DIR}")
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        _set_nprobe_if_ivf(db, NPROBE)
        return db, embeddings

    print(f"[INFO] Creat Index{csv_path}")
    df = load_dataframe(csv_path)
    docs = []
    by_source = Counter()
    now_iso = datetime.now().isoformat()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing files", unit="file"):
        filename = str(row["filename"])
        path = str(row.get("path", filename))
        raw_text = row["cleaned_text"]
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        chunks = chunk_text(raw_text)
        by_source[filename] += len(chunks)

        for j, ch in enumerate(chunks):
            meta = {
                "id": f"{uuid.uuid4().hex}",
                "source": filename,
                "chunk_id": j,
                "path": path,
                "datetime": now_iso,
            }
            docs.append(Document(page_content=ch, metadata=meta))

        if include_faq_as_docs and isinstance(row.get("question_text"), str) and isinstance(row.get("answer_text"), str):
            qa_text = f"คำถาม: {row['question_text']}\nคำตอบ: {row['answer_text']}"
            meta = {
                "id": f"{uuid.uuid4().hex}",
                "source": filename,
                "chunk_id": -1,
                "path": path,
                "datetime": now_iso,
            }
            docs.append(Document(page_content=qa_text, metadata=meta))
            by_source[filename] += 1

    if not docs:
        raise ValueError("Missing Document Index")

    print(f"[INFO] {len(docs):,} chunks -> Index")
    t0 = time.time()

    if USE_IVFPQ:
        X = _precompute_embeddings(docs, embeddings) 
        db, index_desc = _build_faiss_ivfpq(X, docs, embeddings)
        index_type = index_desc
    else:
        db = FAISS.from_documents(docs, embeddings)
        index_type = "Flat (IP)"

    print(f"Build index {time.time()-t0:.1f}s")

    _set_nprobe_if_ivf(db, NPROBE)
    meta = {
        "created_at": now_iso,
        "emb_model": EMB_MODEL_ID,
        "device": DEVICE,
        "index_type": index_type,
        "nprobe": NPROBE if USE_IVFPQ else None,
        "files": len(by_source),
        "total_vectors": int(getattr(db.index, "ntotal", len(docs))),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "include_faq_as_docs": bool(include_faq_as_docs),
        "top_sources_by_chunks": by_source.most_common(10),
    }
    _save_with_meta(db, meta)

    print(f"Save: {INDEX_DIR}")
    print(f"total_vectors ~ {meta['total_vectors']:,} | index_type={index_type}")
    return db, embeddings

def print_index_status():
    if not os.path.exists(INDEX_DIR):
        print(f"{INDEX_DIR}")
        return

    meta_path = os.path.join(INDEX_DIR, "index_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print("[STATUS] index_meta.json")
            print(json.dumps(meta, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"{e}")

    try:
        embeddings = build_embeddings()
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        _set_nprobe_if_ivf(db, NPROBE)
        ntotal = getattr(db.index, "ntotal", None)
        print(f"[STATUS] Creat Index (ntotal): {ntotal:,}" if ntotal is not None else "[STATUS] ntotal")
    except Exception as e:
        print(f"{e}")

    files = glob.glob(os.path.join(INDEX_DIR, "*"))
    print(f"[STATUS] {INDEX_DIR}: {len(files)} ")
    for p in files:
        print("  -", os.path.basename(p))

def format_context(docs):
    lines = []
    for k, d in enumerate(docs, 1):
        m = d.metadata or {}
        src = m.get("source", "unknown")
        cid = m.get("chunk_id", "?")
        lines.append(f"[{k}] (source: {src} | chunk:{cid})\n{d.page_content}")
    return "\n\n".join(lines)

def build_prompt(question, docs):
    context = format_context(docs)
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

Context (ใช้ประกอบคำตอบ):
{context}

คำตอบภาษาไทย (กระชับ อธิบายเหตุผล และระบุ source/chunk ที่ใช้อ้างอิง):"""
    return system, user

def answer_with_ollama(question, retriever, client: Client, num_ctx=1024, num_predict=512):
    docs = retriever.invoke(question)  
    if not docs:
        print("\n--- Answer ---\n")
        msg = "ไม่มีข้อมูลที่เกี่ยวข้อง"
        print(msg + "\n")
        return msg, []

    sys_msg, user_msg = build_prompt(question, docs)

    options = {
        "temperature": 0.2,
        "num_ctx": int(num_ctx),
        "num_predict": int(num_predict),
    }

    stream = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_msg}],
        stream=True,
        options=options,
    )

    print("\n--- Answer ---\n")
    full = []
    for chunk in stream:
        token = chunk.get("message", {}).get("content", "")
        print(token, end="", flush=True)
        full.append(token)
    print("\n")
    return "".join(full), docs

def cli_build(args):
    build_or_load_index(
        csv_path=args.csv,
        rebuild=True,
        include_faq_as_docs=args.include_faq
    )
def cli_chat(args):
    db, _ = build_or_load_index(
        csv_path=args.csv,
        rebuild=False,
        include_faq_as_docs=args.include_faq
    )
    _set_nprobe_if_ivf(db, NPROBE)

    retriever = db.as_retriever(search_kwargs={"k": args.k})
    client = Client(host=OLLAMA_HOST)

    print(f"[INFO] Use Ollama: {OLLAMA_MODEL}")
    print("[TIP] 'exit') ")
    while True:
        q = input("\n คำถาม: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        try:
            _ = answer_with_ollama(q, retriever, client, num_ctx=args.ctx, num_predict=args.predict)
        except Exception as e:
            print(f"\n[ERR] {e}\n")

def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build")
    p1.add_argument("--csv", type=str, default=DEFAULT_CSV)
    p1.add_argument("--include-faq", action="store_true")
    p1.set_defaults(func=cli_build)
    
    p2 = sub.add_parser("chat", help="Chat")
    p2.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Build Index ")
    p2.add_argument("--include-faq", action="store_true")
    p2.add_argument("--ctx", type=int, default=1024, help="total context (num_ctx)")
    p2.add_argument("--predict", type=int, default=512, help="Tokens (num_predict)")
    p2.set_defaults(func=cli_chat)

    p3 = sub.add_parser("status", help="check status")
    p3.set_defaults(func=lambda _: print_index_status())

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    os.environ.setdefault("FAISS_NUM_THREADS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()

