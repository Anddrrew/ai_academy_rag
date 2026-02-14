import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import config
from embedder import embedder
from llm import llm
from store import store

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False


def _get_context_chunks(messages: list[Message]) -> list[str]:
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), None)
    if not last_user:
        return []
    vector = embedder.embed_query(last_user)
    results = store.search(vector, k=5)
    return [r.payload["text"] for r in results]


@router.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": config.llm.display_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rag",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    context_chunks = _get_context_chunks(request.messages)
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
