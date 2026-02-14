import logging
from pathlib import Path

import whisper

from config import config

logger = logging.getLogger(__name__)
model = whisper.load_model(config.whisper.model_size)


def load(file_path: Path) -> str:
    result = model.transcribe(str(file_path))
    logger.debug("Transcribed %s", file_path.name)
    return result["text"]
