"""Configuration management using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    
    # API Keys
    google_api_key: str
    
    # Model Configuration
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    temperature: float = 0.1
    
    # RAG Configuration
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_retrieval: int = 5
    
    # Agent Configuration
    max_iterations: int = 3
    
    # Paths (relative to project root)
    data_dir: Path = PROJECT_ROOT / "data"
    papers_dir: Path = PROJECT_ROOT / "data" / "papers"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings() #type: ignore
