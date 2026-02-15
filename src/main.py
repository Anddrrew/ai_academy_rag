from indexer import indexer
from api.openai_router import router as openai_router
from api.router import router
from services.file_manager import file_manager
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rich.logging import RichHandler

from config import config

logging.basicConfig(
    level=config.server.log_level,
    handlers=[RichHandler(rich_tracebacks=True)],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    indexer.start()
    yield
    indexer.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(openai_router)
app.mount("/files", StaticFiles(directory=file_manager.knowledge_base_dir), name="files")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.server.host,
                port=config.server.port, reload=config.server.reload)
