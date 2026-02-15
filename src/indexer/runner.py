import logging
import threading
from enum import Enum
from pathlib import Path

from chunker import chunker
from embedder import embedder
from file_manager import file_manager
from indexer.loaders import audio_loader, pdf_loader
from store import store


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
            cls._instance.logger = logging.getLogger(cls.__name__)
        return cls._instance

    def start(self) -> None:
        if self._status == IndexingStatus.RUNNING:
            self.logger.warning("Indexer is already running.")
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
    
    def get_loader(self, file_path: Path):
        ext = file_path.suffix.lower()
        loader = LOADERS.get(ext)
        return loader
    
    
    def _run(self) -> None:
        files_processed = 0
        
        for file_path in file_manager.iter_files():
            if self._stop_event.is_set():
                self.logger.info("Indexing interrupted.")
                return
            

            self.logger.info("Processing %s", file_path.name)

            loader = self.get_loader(file_path)
            if loader is None:
                self.logger.warning("Unsupported file type: %s", file_path.name)
                continue

            text = loader(file_path)
            self.logger.info("Extracted %d characters from %s",
                        len(text), file_path.name)

            chunks = chunker.split(text, source=file_path.name)
            vectors = embedder.embed_chunks(chunks)
            store.add_chunks(chunks, vectors)
            files_processed += 1

        if files_processed == 0:
            self.logger.warning("No files found in knowledge base directory.")
        
        self._status = IndexingStatus.DONE
        self.logger.info("Indexing complete.")


indexer = IndexerRunner()
