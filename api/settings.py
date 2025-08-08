import os
from functools import lru_cache
from pydantic import BaseModel

class Settings(BaseModel):
    database_url: str = os.getenv("DATABASE_URL", "postgresql+psycopg://dev:dev@localhost:5432/dev_assistant")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    gen_model: str = os.getenv("GEN_MODEL", "llama3")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

@lru_cache
def get_settings():
    return Settings()
