import logging
import threading
from enum import Enum
from pathlib import Path

from indexer.loaders import audio_loader, pdf_loader

logger = logging.getLogger("File Indexer")

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent.parent / "knowledge_base"

LOADERS = {
    ".pdf": pdf_loader.load,
    ".mp3": audio_loader.load,
}


class IndexingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    STOPPED = "stopped"


class IndexerRunner:
    _instance: "IndexerRunner | None" = None

    def __new__(cls) -> "IndexerRunner":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._status = IndexingStatus.IDLE
            cls._instance._thread = None
            cls._instance._stop_event = threading.Event()
        return cls._instance

    def start(self) -> None:
        if self._status == IndexingStatus.RUNNING:
            logger.warning("Indexer is already running.")
            return
        self._stop_event.clear()
        self._status = IndexingStatus.RUNNING
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._status = IndexingStatus.STOPPED

    def get_status(self) -> IndexingStatus:
        return self._status

    def _run(self) -> None:
        files = [f for f in KNOWLEDGE_BASE_DIR.iterdir() if f.is_file()]

        if not files:
            logger.warning("No files found in knowledge_base/")
            self._status = IndexingStatus.DONE
            return

        for file_path in files:
            if self._stop_event.is_set():
                logger.info("Indexing interrupted.")
                return

            ext = file_path.suffix.lower()
            loader = LOADERS.get(ext)

            if loader is None:
                logger.warning("Unsupported file type: %s", file_path.name)
                continue

            logger.info("Loading %s", file_path.name)
            text = loader(file_path)
            logger.info("Extracted %d characters from %s", len(text), file_path.name)

            # TODO: split into chunks
            # chunks = chunker.split(text, source=file_path.name)

            # TODO: embed and store in Qdrant
            # embedder.embed_and_store(chunks)

        self._status = IndexingStatus.DONE
        logger.info("Indexing complete.")


indexer = IndexerRunner()
