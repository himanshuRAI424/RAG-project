"""
College Knowledge Assistant - RAG engine.

Pipeline: load documents -> chunk -> embed -> store -> retrieve -> generate answer.
Every answer carries its supporting sources; if nothing relevant is found,
the assistant refuses instead of guessing.
"""
import os
import re
import pickle
import requests
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------- Configuration ----------------
DOCS_DIR = "docs"
INDEX_PATH = "index/kb.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "gemma:2b"

CHUNK_WORDS = 120      # words per chunk
CHUNK_OVERLAP = 30     # words shared between neighbouring chunks
TOP_K = 4              # chunks passed to the LLM
MIN_SCORE = 0.20       # below this similarity we say "not found"

NOT_FOUND_MSG = "I could not find this information in the college documents."

SYSTEM_PROMPT = f"""You are a college assistant. Answer ONLY using the CONTEXT below.
Rules:
1. If the context does not contain the answer, reply exactly: "{NOT_FOUND_MSG}"
2. Never use outside knowledge and never guess.
3. If two sources give different values for the same thing, do NOT pick one silently.
   Report both values, name each source file with its 'last updated' date,
   and advise the student to confirm with the college office.
4. Keep the answer short and cite sources as [1], [2].
"""


# ---------------- Loading and chunking ----------------
def load_document(path):
    """Return a list of (page_number, text) for a PDF or text file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if ext in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return [(1, f.read())]
    return []


def extract_doc_date(text):
    """Find a line like 'Last updated: 2024-07-01' (used for conflict reporting)."""
    m = re.search(r"last updated:\s*([^\n]+)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else "unknown"


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


# ---------------- Knowledge base ----------------
class KnowledgeBase:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.chunks = []        # list of dicts: source, page, date, text
        self.embeddings = None  # (n_chunks, dim), L2-normalised

    def build(self, docs_dir=DOCS_DIR):
        self.chunks = []
        for name in sorted(os.listdir(docs_dir)):
            path = os.path.join(docs_dir, name)
            pages = load_document(path)
            if not pages:
                continue
            doc_date = extract_doc_date(" ".join(t for _, t in pages))
            for page_no, text in pages:
                for piece in chunk_text(text):
                    self.chunks.append(
                        {"source": name, "page": page_no, "date": doc_date, "text": piece}
                    )
        if not self.chunks:
            raise ValueError("No readable documents found in " + docs_dir)
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "emb": self.embeddings}, f)

    def load(self):
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.chunks, self.embeddings = data["chunks"], data["emb"]

    def search(self, query, k=TOP_K):
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q  # cosine similarity (vectors are normalised)
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


# ---------------- Answer generation ----------------
def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{i}] (file={h['source']}, page={h['page']}, last updated={h['date']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    return f"{SYSTEM_PROMPT}\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"


def call_llm(prompt):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0}},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def answer(question, kb, k=TOP_K, min_score=MIN_SCORE, use_llm=True):
    """Return dict: answer, found, sources, llm_used."""
    hits = [h for h in kb.search(question, k) if h["score"] >= min_score]

    # Guardrail 1: nothing relevant retrieved -> refuse without calling the LLM.
    if not hits:
        return {"answer": NOT_FOUND_MSG, "found": False, "sources": [], "llm_used": False}

    if not use_llm:
        return {"answer": "(retrieval-only mode) Top passage: " + hits[0]["text"],
                "found": True, "sources": hits, "llm_used": False}

    try:
        text = call_llm(build_prompt(question, hits))
    except requests.RequestException:
        # Fallback: show the best passage if Ollama is not running.
        text = ("The language model is not reachable, so here is the most relevant "
                "passage:\n\n" + hits[0]["text"])
        return {"answer": text, "found": True, "sources": hits, "llm_used": False}

    # Guardrail 2: the LLM itself said it does not know.
    if NOT_FOUND_MSG.lower() in text.lower():
        return {"answer": NOT_FOUND_MSG, "found": False, "sources": [], "llm_used": True}

    return {"answer": text, "found": True, "sources": hits, "llm_used": True}
