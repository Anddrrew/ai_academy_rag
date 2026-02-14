import logging
from pathlib import Path

import whisper

from config import config


model = whisper.load_model(config.whisper.model_size)


def load(file_path: Path) -> str:
    result = model.transcribe(str(file_path))
    return result["text"]
