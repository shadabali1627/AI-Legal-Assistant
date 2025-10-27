from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()

class ChatRequest(BaseModel):
    query: str = Field(..., description="User query")
    top_k: int = 5

@router.post("/chat")
async def chat(req: Request, payload: ChatRequest):
    rag = req.app.state.rag
    result = rag.answer(payload.query, k=payload.top_k)
    return result
