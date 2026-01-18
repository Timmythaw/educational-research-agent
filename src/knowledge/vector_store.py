"""Vector store with thread-safe FAISS operations."""

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)

# Thread lock for FAISS (CRITICAL for memory safety)
_faiss_lock = threading.Lock()


class VectorStoreManager:
    """Manages the FAISS vector store with thread-safe operations."""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
        )
        self.vector_store = None
        self.index_path = settings.vector_store_dir
    
    def load_or_create(self, documents: Optional[List[Document]] = None):
        """Load existing index or create new one with thread safety."""
        with _faiss_lock:  # Lock FAISS operations
            if self.index_path.exists():
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                try:
                    self.vector_store = FAISS.load_local(
                        str(self.index_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info("FAISS index loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
                    if documents:
                        logger.info("Creating new index from provided documents")
                        self.vector_store = FAISS.from_documents(documents, self.embeddings)
                        self.save()
            else:
                if documents:
                    logger.info("Creating new FAISS index from documents")
                    self.vector_store = FAISS.from_documents(documents, self.embeddings)
                    self.save()
                else:
                    logger.warning("No existing index and no documents provided")
    
    def save(self):
        """Save the vector store with thread safety."""
        if self.vector_store:
            with _faiss_lock:  # Lock save operations
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(self.index_path))
                logger.info(f"FAISS index saved to {self.index_path}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search with thread safety."""
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        with _faiss_lock:  # Lock search operations
            try:
                results = self.vector_store.similarity_search(query, k=k)
                logger.info(f"Retrieved {len(results)} documents for query: '{query[:50]}...'")
                return results
            except Exception as e:
                logger.error(f"Similarity search failed: {e}")
                return []
