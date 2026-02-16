from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_CONFIG_FILE = Path(__file__).parent.parent.parent / ".env"


class QdrantConfig(BaseModel):
    search_k: int = 5
    host: str
    port: int
    collection: str


class EmbeddingConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    vector_size: int = 1024
    public_url: str


class ChunkingConfig(BaseModel):
    size: int = 500
    overlap: int = 50


class OpenAIConfig(BaseModel):
    chat_model: str = "gpt-4o-mini"
    whisper_model: str = "base"
    api_key: str


class ServerConfig(BaseModel):
    display_name: str = "RAG Assistant"
    log_level: str = "INFO"
    public_url: str


class Config(BaseSettings):
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig = ChunkingConfig()
    openai: OpenAIConfig
    server: ServerConfig

    model_config = SettingsConfigDict(
        env_file=ENV_CONFIG_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__"
    )


config = Config()
