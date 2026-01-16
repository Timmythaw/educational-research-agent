"""Knowledge base management for educational research papers."""

from .loader import DocumentLoader
from .vector_store import VectorStoreManager

__all__ = ["DocumentLoader", "VectorStoreManager"]
