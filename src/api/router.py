from fastapi import APIRouter
from pydantic import BaseModel

from indexer import indexer
from indexer.runner import IndexingStatus

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    indexing_status: IndexingStatus


@router.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question
    # TODO: embed question, query Qdrant, pass context to LLM
    answer = f"[RAG not yet implemented] You asked: {question}"
    return ChatResponse(answer=answer, indexing_status=indexer.get_status())


@router.get("/status")
def status():
    return {"indexing_status": indexer.get_status()}
