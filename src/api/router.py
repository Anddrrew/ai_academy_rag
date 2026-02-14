from fastapi import APIRouter
from pydantic import BaseModel

from embedder import embedder
from indexer import indexer
from indexer.runner import IndexingStatus
from llm import llm
from store import store

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    indexing_status: IndexingStatus


@router.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question
    vector = embedder.embed_query(question)
    results = store.search(vector, k=5)
    context_chunks = [r.payload["text"] for r in results]
    answer = llm.chat(question, context_chunks)
    return ChatResponse(answer=answer, indexing_status=indexer.get_status())


@router.get("/status")
def status():
    return {"indexing_status": indexer.get_status()}
