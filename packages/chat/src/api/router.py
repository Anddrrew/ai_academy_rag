from fastapi import APIRouter
from pydantic import BaseModel

from knowledge_storage import knowledge_storage
from shared.services.embedder import embedder
from shared.services.file_manager import file_manager
from llm import llm

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question
    vector = embedder.embed_query(question)
    results = knowledge_storage.search(vector)
    context_chunks = [
        f"[Source: [{r.payload['source']}]({file_manager.get_public_url(r.payload['source'])})]\n{r.payload['text']}"
        for r in results
    ]
    answer = llm.chat(question, context_chunks)
    return ChatResponse(answer=answer)


@router.get("/status")
def status():
    return {"status": "ok"}
