from typing import List, Dict, Any, Tuple
import os
from backend.utils.file_loader import load_cases_as_docs
from backend.utils.text_processing import chunk_texts
from backend.utils.embedding_utils import (
    EmbeddingBackend,
    get_embedder,
    ensure_faiss_index,
    search_faiss,
)
from backend.config import settings


class RAGPipeline:
    def __init__(self):
        self.embedder = None
        self.embedding_backend: EmbeddingBackend = None
        self.index = None
        self.id2meta: Dict[int, Dict[str, Any]] = {}
        self.generator_mode = None  # "GEMINI" | "HF" | "RULES"
        self._gemini_model = None
        self._hf_generator = None

    def initialize(self):
        """Load data, create embeddings, and select generator backend."""
        cases_path = os.path.join(settings.DATA_DIR, "cases.json")
        docs = load_cases_as_docs(cases_path)

        # 1) Chunk texts
        texts, metas = [], []
        for d in docs:
            for chunk in chunk_texts(
                d["text"],
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP,
            ):
                texts.append(chunk)
                metas.append(d["meta"])

        # 2) Build / load FAISS vector index
        self.embedding_backend, self.embedder = get_embedder(settings.EMBEDDING_PROVIDER)
        self.index, self.id2meta = ensure_faiss_index(
            vector_dir=settings.VECTOR_DIR,
            faiss_subdir="faiss_index",
            texts=texts,
            metas=metas,
            embedder=self.embedder,
            embedding_backend=self.embedding_backend,
            data_dir=settings.DATA_DIR,  
        )

        # 3) Choose generation backend (Gemini → HF → Rules)
        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai

                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)
                self.generator_mode = "GEMINI"
                print(f"✅ Using Gemini model: {settings.GEMINI_MODEL}")
                return
            except Exception as e:
                print("⚠️ Gemini initialization failed:", e)

        # Fallback: local Hugging Face model
        try:
            from transformers import pipeline

            self._hf_generator = pipeline(
                "text-generation", model=settings.HF_MODEL, trust_remote_code=True
            )
            self.generator_mode = "HF"
            print(f"✅ Using Hugging Face model: {settings.HF_MODEL}")
        except Exception:
            self.generator_mode = "RULES"
            print("⚠️ Falling back to rules-based summarization.")

    # ---------------- RETRIEVAL ----------------

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        raw = search_faiss(
            index=self.index,
            embedder=self.embedder,
            embedding_backend=self.embedding_backend,
            query=query,
            top_k=k,
        )
        hits = []
        for score, m in raw:
            meta = self.id2meta.get(m["__idx__"], {})
            hits.append((score, meta))
        return hits

    # ---------------- PROMPT CONSTRUCTION ----------------

    def _format_context(self, hits: List[Tuple[float, Dict[str, Any]]]) -> str:
        lines = []
        for i, (score, meta) in enumerate(hits, 1):
            snippet = meta.get("chunk", "")[:800]
            lines.append(
                f"[{i}] {meta.get('case_name')} ({meta.get('year')}), "
                f"{meta.get('court')} | score={score:.3f}\n"
                f"Areas: {', '.join(meta.get('area_of_law', []))}\n"
                f"Summary: {meta.get('summary')}\n"
                f"Text: {snippet}\n"
            )
        return "\n".join(lines)

    def _build_prompt(self, query: str, hits: List[Tuple[float, Dict[str, Any]]]) -> str:
        ctx = self._format_context(hits)
        return (
            "You are 'AI Legal Assistant', a precise Pakistani legal RAG chatbot. "
            "Answer the user's question using ONLY the provided context snippets. "
            "Cite sources using bracket numbers like [1], [2]. If unsure, say you cannot find it in the context.\n\n"
            f"User question: {query}\n\n"
            f"Context snippets:\n{ctx}\n\n"
            "Now provide a concise, accurate answer with citations."
        )

    # ---------------- GENERATORS ----------------

    def _generate_gemini(self, prompt: str) -> str:
        """Generate an answer using Gemini 2.5 Flash."""
        try:
            resp = self._gemini_model.generate_content(prompt)
            return (getattr(resp, "text", None) or "").strip() or "I could not generate an answer."
        except Exception as e:
            print("⚠️ Gemini generation error:", e)
            return "I could not generate an answer using Gemini."

    def _generate_hf(self, prompt: str) -> str:
        txt = self._hf_generator(prompt, max_length=800, do_sample=False)[0]["generated_text"]
        return txt[len(prompt):].strip()[:2000] or "I could not generate an answer."

    def _generate_rules(self, hits: List[Tuple[float, Dict[str, Any]]]) -> str:
        if not hits:
            return "I could not find an answer in the current knowledge base."
        summaries = [f"[{i}] {m['case_name']} ({m['year']}): {m['summary']}" for i, (_, m) in enumerate(hits, 1)]
        return " • ".join(summaries)

    # ---------------- MAIN ANSWER METHOD ----------------

    def answer(self, query: str, k: int = 5) -> Dict[str, Any]:
        hits = self.retrieve(query, k=k)
        prompt = self._build_prompt(query, hits)

        if self.generator_mode == "GEMINI":
            text = self._generate_gemini(prompt)
        elif self.generator_mode == "HF":
            text = self._generate_hf(prompt)
        else:
            text = self._generate_rules(hits)

        # Format sources for frontend
        sources = []
        for i, (score, meta) in enumerate(hits, 1):
            sources.append(
                {
                    "rank": i,
                    "score": float(score),
                    "case_name": meta.get("case_name"),
                    "year": meta.get("year"),
                    "court": meta.get("court"),
                    "citation": meta.get("citation"),
                    "area_of_law": meta.get("area_of_law"),
                    "summary": meta.get("summary"),
                }
            )

        return {"answer": text, "mode": self.generator_mode, "sources": sources}
