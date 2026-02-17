import logging

from fastapi import APIRouter
from pydantic import BaseModel

from context import context
from shared.services.embedder import embedder
from shared.services.file_manager import file_manager
from llm import llm

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question
    logger.debug("Question: %s", question)

    context_chunks = context.get_chunks(question)
    answer = llm.chat(question, context_chunks)
    logger.debug("Answer length: %d chars", len(answer))
    return ChatResponse(answer=answer)


@router.get("/status")
def status():
    return {"status": "ok"}
