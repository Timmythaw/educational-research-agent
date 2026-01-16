"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    
    # API Keys
    google_api_key: str
    
    # Model Configuration
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/text-embedding-004"
    temperature: float = 0.1
    
    # RAG Configuration
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_retrieval: int = 5
    
    # Agent Configuration
    max_iterations: int = 3
    
    # Paths
    data_dir: Path = Path("data")
    papers_dir: Path = data_dir / "papers"
    vector_store_dir: Path = data_dir / "vector_store"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
