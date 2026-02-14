from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
from pydantic_settings.main import PydanticBaseSettingsSource

YAML_CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"
ENV_CONFIG_FILE = Path(__file__).parent.parent / ".env"


class QdrantConfig(BaseModel):
    host: str
    port: int
    collection: str
    force_clear: bool


class EmbeddingConfig(BaseModel):
    model: str
    vector_size: int


class ChunkingConfig(BaseModel):
    size: int
    overlap: int


class WhisperConfig(BaseModel):
    model_size: str


class LLMConfig(BaseModel):
    model: str
    api_key: str


class ServerConfig(BaseModel):
    host: str
    port: int
    reload: bool
    log_level: str


class Config(BaseSettings):
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    whisper: WhisperConfig
    llm: LLMConfig
    server: ServerConfig

    model_config = SettingsConfigDict(
        yaml_file=YAML_CONFIG_FILE,
        env_file=ENV_CONFIG_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
        )


config = Config()
