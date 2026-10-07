"""
ACEF Configuration — single Settings class loaded from .env.

All configuration values are centralized here. Modules receive a Settings
instance (or individual values) at construction time — they never read .env
or environment variables directly.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "acef_password"

    # ── Chroma ───────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── LLM provider ─────────────────────────────────────────────────────────
    llm_provider: str = "gemini"                # "gemini" | "groq"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # ── Cache ────────────────────────────────────────────────────────────────
    cache_similarity_threshold: float = 0.92

    # ── Memory retrieval ─────────────────────────────────────────────────────
    memory_top_k: int = 5
    memory_long_conversation_turns: int = 3     # session turn count that counts as "long"

    # ── Context budget ───────────────────────────────────────────────────────
    max_context_chars: int = 6000               # simple character-based truncation

    # ── Demo ─────────────────────────────────────────────────────────────────
    demo_codebase_path: str = "./data/demo_codebase"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance — parsed once, reused everywhere."""
    return Settings()
