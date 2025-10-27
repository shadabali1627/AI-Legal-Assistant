from typing import List, Tuple, Dict, Any
import os
import json
import numpy as np
import faiss

# Providers
class EmbeddingBackend:
    SENTENCE = "sentence-transformers"
    GOOGLE = "google"  # Gemini-based embeddings


# ---- helpers ---------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length (for cosine similarity with IndexFlatIP)."""
    if x.ndim != 2 or x.size == 0:
        return x.astype("float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / norms).astype("float32")


def _batched(iterable: List[str], batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


# ---- embedders -------------------------------------------------------------

def get_embedder(provider: str):
    """
    Returns:
        (provider_name, embed_fn)
    where embed_fn: List[str] -> np.ndarray[float32] of shape (N, D)
    """
    if provider == EmbeddingBackend.GOOGLE:
        # Google Gemini embeddings via google-generativeai
        import google.generativeai as genai
        from config import settings

        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        embed_model = getattr(settings, "GEMINI_EMBED_MODEL", "text-embedding-004")

        # text-embedding-004 outputs 768-d vectors.
        def embed(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.zeros((0, 768), dtype="float32")
            vectors: List[List[float]] = []
            for batch in _batched(texts, batch_size=32):
                try:
                    try:
                        r = genai.embed_content(model=embed_model, content=batch)
                        if isinstance(r, dict) and "embeddings" in r:
                            for ent in r["embeddings"]:
                                vectors.append(ent["embedding"])
                        else:
                            for item in batch:
                                r1 = genai.embed_content(model=embed_model, content=item)
                                vectors.append(r1["embedding"])
                    except Exception:
                        for item in batch:
                            r1 = genai.embed_content(model=embed_model, content=item)
                            vectors.append(r1["embedding"])
                except Exception as e:
                    print("⚠️ Gemini batch embedding error:", e)
                    vectors.extend([[0.0] * 768 for _ in batch])
            arr = np.array(vectors, dtype="float32")
            return _l2_normalize(arr)

        return EmbeddingBackend.GOOGLE, embed

    # Default: sentence-transformers (local CPU fallback)
    from sentence_transformers import SentenceTransformer
    from backend.config import settings as _settings

    _st_model = SentenceTransformer(_settings.SENTENCE_MODEL)

    def embed(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype="float32")  # 384D default for MiniLM
        emb = _st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    return EmbeddingBackend.SENTENCE, embed


# ---- FAISS index mgmt ------------------------------------------------------

def ensure_faiss_index(
    vector_dir: str,
    faiss_subdir: str,
    texts: List[str],
    metas: List[Dict[str, Any]],
    embedder,
    embedding_backend: str,
    data_dir: str = "backend/data",
    force_rebuild: bool = False,
):
    """
    Build or load a FAISS index for vector retrieval.

    Automatically rebuilds if:
      - The dataset files are newer than the FAISS index, or
      - The index/meta files are missing, or
      - force_rebuild=True

    Uses inner-product on L2-normalized vectors (cosine similarity).
    """
    # Ensure directories
    if os.path.exists(vector_dir) and not os.path.isdir(vector_dir):
        os.remove(vector_dir)
    os.makedirs(vector_dir, exist_ok=True)

    subpath = os.path.join(vector_dir, faiss_subdir)
    if os.path.exists(subpath) and not os.path.isdir(subpath):
        os.remove(subpath)
    os.makedirs(subpath, exist_ok=True)

    index_path = os.path.join(subpath, "index.faiss")
    meta_path = os.path.join(subpath, "meta.npy")
    info_path = os.path.join(subpath, "build_info.json")

    # --- Detect dataset modification time ---
    latest_data_time = 0
    if os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            for f in files:
                try:
                    latest_data_time = max(latest_data_time, os.path.getmtime(os.path.join(root, f)))
                except FileNotFoundError:
                    pass

    index_time = os.path.getmtime(index_path) if os.path.exists(index_path) else 0
    must_rebuild = force_rebuild or (latest_data_time > index_time)

    # --- Try to load existing index ---
    if not must_rebuild and os.path.isfile(index_path) and os.path.isfile(meta_path):
        try:
            index = faiss.read_index(index_path)
            id2meta_list = np.load(meta_path, allow_pickle=True).tolist()
            id2meta = {i: m for i, m in enumerate(id2meta_list)}
            print("✅ Loaded existing FAISS index.")
            return index, id2meta
        except Exception as e:
            print("⚠️ Could not load FAISS index/meta, rebuilding. Reason:", e)

    # --- Build fresh index ---
    print("🔄 Rebuilding FAISS index (new or updated data detected)...")

    if not texts:
        raise ValueError("No texts provided to build FAISS index.")

    vectors = embedder(texts)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(f"Embedding shape mismatch: got {vectors.shape} for {len(texts)} texts.")

    d = int(vectors.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(vectors)

    # Save index + meta
    faiss.write_index(index, index_path)
    id2meta_list = []
    for t, m in zip(texts, metas):
        m2 = dict(m)
        m2["chunk"] = t
        id2meta_list.append(m2)
    np.save(meta_path, np.array(id2meta_list, dtype=object))

    # Save build info
    build_info = {
        "embedding_backend": embedding_backend,
        "dimension": d,
        "dataset_modified": latest_data_time,
        "count": len(texts),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=2)

    id2meta = {i: m for i, m in enumerate(id2meta_list)}
    print("✅ FAISS index built and saved successfully.")
    return index, id2meta


# ---- Search ----------------------------------------------------------------

def search_faiss(
    index,
    embedder,
    embedding_backend: str,
    query: str,
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Search FAISS index for top-k similar chunks to the query."""
    if not query:
        return []
    qv = embedder([query])
    if qv.ndim != 2 or qv.shape[0] == 0:
        return []
    sims, idxs = index.search(qv, top_k)
    sims = sims[0].tolist()
    idxs = idxs[0].tolist()
    return [(float(s), {"__idx__": ix}) for s, ix in zip(sims, idxs)]
