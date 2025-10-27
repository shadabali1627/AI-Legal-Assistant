from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from rag_pipeline import RAGPipeline
from config import settings
from routes.chat import router as chat_router

rag = RAGPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    rag.initialize()
    app.state.rag = rag
    print("✅ RAG pipeline initialized.")
    yield
    # Shutdown (optional)
    print("🧹 Shutting down...")

app = FastAPI(title="AI Legal Assistant (RAG)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ok", "service": "AI Legal Assistant (RAG)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
