import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from rich.logging import RichHandler

from config import config

logging.basicConfig(
    level=config.server.log_level,
    handlers=[RichHandler(rich_tracebacks=True)],
)

from api import router
from indexer import indexer


@asynccontextmanager
async def lifespan(app: FastAPI):
    indexer.start()
    yield
    indexer.stop()


app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.server.host, port=config.server.port, reload=config.server.reload)
