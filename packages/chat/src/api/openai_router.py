import logging
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from context import context
from shared.config import config
from shared.services.embedder import embedder
from shared.services.file_manager import file_manager
from llm import llm

logger = logging.getLogger(__name__)
router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False


@router.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": config.server.display_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rag",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    last_message = request.messages[-1].content if request.messages else ""
    context_chunks = context.get_chunks(last_message)
    messages = [m.model_dump() for m in request.messages]

    if request.stream:
        return StreamingResponse(
            llm.stream(messages, context_chunks),
            media_type="text/event-stream",
        )

    answer = llm.chat_messages(messages, context_chunks)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
    }
