from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    source: str
    index: int
