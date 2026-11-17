"""Application settings, loaded from environment variables or a .env file."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DOCRAG_", extra="ignore")

    # OpenAI (the key uses the standard name, without the DOCRAG_ prefix)
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Vector store
    chroma_path: str = "chroma_db"
    collection: str = "documents"

    # Ingestion
    ocr_languages: str = "eng+deu"
    ocr_dpi: int = 300
    min_text_chars: int = 50  # pages with less embedded text than this get OCR'd
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
