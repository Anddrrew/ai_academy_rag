import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from rich.logging import RichHandler

from runner import indexer, IndexingStatus
from shared.config import config

logging.basicConfig(
    level="INFO",
    handlers=[RichHandler(rich_tracebacks=True)],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.indexer.start_on_startup:
        indexer.start()
    yield
    indexer.stop()


app = FastAPI(title="Indexer", description="Knowledge base indexing service", lifespan=lifespan)


class StatusResponse(BaseModel):
    status: IndexingStatus


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(status=indexer.get_status())


@app.post("/index", response_model=StatusResponse)
def index() -> StatusResponse:
    indexer.start()
    return StatusResponse(status=indexer.get_status())
