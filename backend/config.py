import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    VECTOR_DIR: str = Field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore"))

    # Embeddings
    EMBEDDING_PROVIDER: str = "google"  # "google" | "sentence-transformers" | "openai"
    SENTENCE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # OpenAI (optional fallback)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Gemini (Google)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.5-flash"          # ✅ main LLM
    GEMINI_EMBED_MODEL: str = "text-embedding-004"  # ✅ embedding model

    # HF local generator fallback
    HF_MODEL: str = "distilgpt2"

    # CORS
    CORS_ALLOW_ORIGINS: list = ["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:3000", "*"]

    class Config:
        env_file = ".env"

settings = Settings()
