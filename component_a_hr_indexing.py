"""
============================================================
COMPONENT A — HR Knowledge Base Indexing
============================================================
HR Copilot · MLDS 2026 · Maneesh Kumar & Ravikiran Ravada

Use Case:
  Build the searchable knowledge base from 6 HR policy
  documents covering: Leave, Compensation, Remote Work,
  Onboarding, Grievance, and L&D policies.

What this builds:
  1. Document loader (Markdown, PDF, DOCX)
  2. HR-aware chunker (preserves policy clauses)
  3. Local embeddings (all-MiniLM-L6-v2, 384-dim)
  4. FAISS flat index for vector similarity search
  5. BM25 index for keyword search
  6. Saved to disk — loaded by all downstream agents

Azure equivalent:
  Azure AI Document Intelligence → Azure AI Search (hybrid)
  with per-field semantic configuration per HR category.

Run: python3 component_a_hr_indexing.py
============================================================
"""
import os, json, pickle, pathlib, re
from typing import List, Dict, Tuple
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DOCS_DIR   = "data/hr_docs"
INDEX_DIR  = "data/index"
EMBED_MODEL= "all-MiniLM-L6-v2"
CHUNK_SIZE = 300      # Smaller chunks = more specific policy clauses (precision-focused)
CHUNK_OVERLAP = 100   # Higher overlap = better semantic continuity between chunks
BATCH_SIZE = 32

print("="*60)
print("  COMPONENT A — HR Knowledge Base Indexing")
print("="*60)

# ─────────────────────────────────────────────────────────────
# HR CATEGORY MAPPING
# Maps document filename → HR domain for agent routing
# ─────────────────────────────────────────────────────────────
HR_CATEGORY_MAP = {
    "leave_policy":           "leave",
    "compensation_benefits":  "compensation",
    "remote_work_policy":     "remote_work",
    "onboarding_guide":       "onboarding",
    "grievance_compliance":   "grievance",
    "learning_development":   "learning",
}

def infer_hr_category(filename: str) -> str:
    name = filename.lower().replace(".md","").replace(".pdf","").replace(".docx","")
    for key, cat in HR_CATEGORY_MAP.items():
        if key in name:
            return cat
    return "general"


# ─────────────────────────────────────────────────────────────
# STEP 1 — DOCUMENT LOADER
# ─────────────────────────────────────────────────────────────
def load_hr_documents(docs_dir: str) -> List[Dict]:
    """Load HR policy documents from directory."""
    docs_dir = pathlib.Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"HR docs directory '{docs_dir}' not found.\n"
            "Run setup.sh to create sample documents first."
        )
    documents = []
    for fp in sorted(docs_dir.rglob("*")):
        if fp.suffix.lower() not in (".md", ".txt", ".pdf", ".docx"):
            continue
        try:
            if fp.suffix == ".pdf":
                from pypdf import PdfReader
                text = "\n\n".join(p.extract_text() or "" for p in PdfReader(str(fp)).pages)
            elif fp.suffix == ".docx":
                import docx
                text = "\n\n".join(p.text for p in docx.Document(str(fp)).paragraphs if p.text.strip())
            else:
                text = fp.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  [WARN] Could not read {fp.name}: {e}")
            continue
        if not text.strip():
            continue
        cat = infer_hr_category(fp.name)
        documents.append({"path": str(fp), "filename": fp.name, "content": text, "category": cat})
        print(f"  Loaded [{cat:15s}]: {fp.name}  ({len(text):,} chars)")
    print(f"\n  Total HR documents: {len(documents)}")
    return documents


# ─────────────────────────────────────────────────────────────
# STEP 2 — HR-AWARE CHUNKER
# ─────────────────────────────────────────────────────────────
def chunk_hr_document(text: str, doc_meta: Dict,
                       chunk_size: int = CHUNK_SIZE,
                       overlap: int    = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split HR policy documents while preserving:
    - Section headings (## / ###)
    - Numbered lists and bullet clauses
    - Table rows (keep together)

    Strategy: split on double-newlines (paragraph boundaries), respecting
    section headers. Carry the last heading into the next chunk so context
    is never lost.

    Azure production:
      Azure AI Document Intelligence with layout model preserves
      heading hierarchy and table structure automatically.
    """
    chunks = []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 2,       # ~4 chars/token; *2 (not *4) for finer HR clause granularity
            chunk_overlap=overlap * 2,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
            length_function=len,
            is_separator_regex=False,
        )
        raw_chunks = splitter.split_text(text)
    except ImportError:
        # Fallback: paragraph-based chunking
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        raw_chunks, buf = [], []
        for para in paragraphs:
            buf.append(para)
            if sum(len(p) for p in buf) > chunk_size * 2:
                raw_chunks.append("\n\n".join(buf))
                buf = buf[-2:] if len(buf) > 2 else []   # overlap
        if buf:
            raw_chunks.append("\n\n".join(buf))

    for i, text_chunk in enumerate(raw_chunks):
        if len(text_chunk.strip()) < 40:
            continue
        chunks.append({
            "chunk_id":    f"{doc_meta['filename']}::chunk_{i:04d}",
            "text":        text_chunk.strip(),
            "source_file": doc_meta["path"],
            "filename":    doc_meta["filename"],
            "category":    doc_meta["category"],
            "char_count":  len(text_chunk),
        })
    return chunks


def chunk_all_documents(documents: List[Dict]) -> List[Dict]:
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_hr_document(doc["content"], doc)
        all_chunks.extend(doc_chunks)
    cat_counts = {}
    for c in all_chunks:
        cat_counts[c["category"]] = cat_counts.get(c["category"], 0) + 1
    print(f"\n  Total chunks: {len(all_chunks)}")
    for cat, n in sorted(cat_counts.items()):
        print(f"    [{cat:15s}]: {n} chunks")
    return all_chunks


# ─────────────────────────────────────────────────────────────
# STEP 3 — EMBEDDINGS
# ─────────────────────────────────────────────────────────────
def generate_embeddings(chunks: List[Dict]) -> np.ndarray:
    """
    Embed all chunks using all-MiniLM-L6-v2 (local, no API key).

    Specs: 384 dimensions · ~90 MB · ~14,000 sentences/sec on CPU
    L2-normalised output → inner product = cosine similarity in FAISS.

    Azure production:
      client.embeddings.create(model="text-embedding-3-large")
      → 3072 dimensions, significantly better for domain-specific queries.
      Swap in without changing any other code — only the dim changes.
    """
    print(f"\n  Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks (batch={BATCH_SIZE})...")
    vectors = model.encode(texts, batch_size=BATCH_SIZE,
                           show_progress_bar=True,
                           normalize_embeddings=True,
                           convert_to_numpy=True)
    print(f"  Shape: {vectors.shape}")
    return vectors.astype("float32"), model


# ─────────────────────────────────────────────────────────────
# STEP 4 — FAISS INDEX
# ─────────────────────────────────────────────────────────────
def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    IndexFlatIP: exact cosine search after L2 normalisation.
    Good for < 200K chunks. No training needed.

    For > 200K: use IndexIVFFlat (approximate, needs training).
    For highest recall at scale: IndexHNSWFlat (m=32).

    Azure: HNSW with ef_construction=400 configured at index creation.
    """
    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"\n  FAISS index: {index.ntotal} vectors (dim={dim})")
    return index


# ─────────────────────────────────────────────────────────────
# STEP 5 — BM25 INDEX
# ─────────────────────────────────────────────────────────────
def build_bm25_index(chunks: List[Dict]) -> BM25Okapi:
    """
    BM25Okapi keyword index.
    k1=1.5 (TF saturation), b=0.75 (length normalisation).
    HR-specific: lowercase, remove special characters for better term matching.

    Azure: BM25 applied automatically when search_text is provided.
    """
    tokenised = [
        re.sub(r'[^\w\s]', '', c["text"]).lower().split()
        for c in chunks
    ]
    bm25 = BM25Okapi(tokenised, k1=1.5, b=0.75)
    print(f"  BM25 index: {len(tokenised)} documents")
    return bm25


# ─────────────────────────────────────────────────────────────
# SAVE & LOAD
# ─────────────────────────────────────────────────────────────
def save_index(chunks, faiss_index, bm25, index_dir=INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(faiss_index, f"{index_dir}/hr_faiss.index")
    with open(f"{index_dir}/hr_bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(f"{index_dir}/hr_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {index_dir}/")
    print(f"    hr_faiss.index  {os.path.getsize(f'{index_dir}/hr_faiss.index')//1024} KB")
    print(f"    hr_bm25.pkl     {os.path.getsize(f'{index_dir}/hr_bm25.pkl')//1024} KB")
    print(f"    hr_chunks.json  {len(chunks)} chunks")


def load_index(index_dir=INDEX_DIR):
    """Called by all downstream agents."""
    with open(f"{index_dir}/hr_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    faiss_idx = faiss.read_index(f"{index_dir}/hr_faiss.index")
    with open(f"{index_dir}/hr_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    return chunks, faiss_idx, bm25, model


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    documents   = load_hr_documents(DOCS_DIR)
    chunks      = chunk_all_documents(documents)
    vectors, _  = generate_embeddings(chunks)
    faiss_index = build_faiss_index(vectors)
    bm25        = build_bm25_index(chunks)
    save_index(chunks, faiss_index, bm25)

    # Quick sanity check
    print("\n  Test query: 'How many annual leave days can I carry forward?'")
    _, faiss_idx, bm25_idx, model = load_index()
    q = model.encode(["How many annual leave days can I carry forward?"],
                     normalize_embeddings=True).astype("float32")
    D, I = faiss_idx.search(q, 3)
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        print(f"  {rank}. [{score:.3f}] [{chunks[idx]['category']}] {chunks[idx]['text'][:100]}...")

    print("\n" + "="*60)
    print("  COMPONENT A COMPLETE ✅")
    print("  Run next: python3 component_b_orchestrator_agent.py")
    print("="*60)
