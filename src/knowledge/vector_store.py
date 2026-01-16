"""Vector store management using FAISS and Google Gemini embeddings."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import faiss
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage FAISS vector store for educational research papers."""
    
    def __init__(self, vector_store_dir: Optional[Path] = None):
        """
        Initialize vector store manager.
        
        Args:
            vector_store_dir: Directory to save/load vector store
        """
        self.vector_store_dir = vector_store_dir or settings.vector_store_dir
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Gemini embeddings (API key read from GOOGLE_API_KEY env var)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
        )
        
        self.vectorstore: Optional[FAISS] = None
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")
        
        logger.info(f"Creating vector store from {len(documents)} documents...")
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        
        logger.info("Vector store created successfully")
        return self.vectorstore
    
    def save_vectorstore(self, vectorstore: Optional[FAISS] = None) -> None:
        """
        Save vector store to disk.
        
        Args:
            vectorstore: FAISS vector store to save (uses self.vectorstore if None)
        """
        vectorstore = vectorstore or self.vectorstore
        
        if vectorstore is None:
            raise ValueError("No vector store to save")
        
        save_path = str(self.vector_store_dir / "faiss_index")
        vectorstore.save_local(save_path)
        
        logger.info(f"Vector store saved to {save_path}")
        
        # Save metadata
        metadata = {
            "num_documents": vectorstore.index.ntotal,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
        
        metadata_path = self.vector_store_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_vectorstore(self) -> FAISS:
        """
        Load vector store from disk.
        
        Returns:
            Loaded FAISS vector store
        """
        load_path = str(self.vector_store_dir / "faiss_index")
        
        if not Path(load_path).exists():
            raise FileNotFoundError(
                f"Vector store not found at {load_path}. "
                "Run build_knowledge_base.py first."
            )
        
        logger.info(f"Loading vector store from {load_path}...")
        
        self.vectorstore = FAISS.load_local(
            load_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,  # We trust our own data
        )
        
        logger.info("Vector store loaded successfully")
        
        # Load metadata
        metadata_path = self.vector_store_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Vector store contains {metadata['num_documents']} documents")
        
        return self.vectorstore
    
    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        Search vector store for relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            score_threshold: Minimum relevance score (0-1)
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not loaded. Call load_vectorstore() first.")
        
        if score_threshold:
            # Search with score threshold
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query, k=k
            )
            # Filter by threshold (lower score = more similar)
            docs = [doc for doc, score in docs_and_scores if score < (1 - score_threshold)]
        else:
            # Regular search
            docs = self.vectorstore.similarity_search(query, k=k)
        
        logger.info(f"Retrieved {len(docs)} documents for query: '{query[:50]}...'")
        return docs
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        if self.vectorstore is None:
            return {"status": "not_loaded"}
        
        num_docs = self.vectorstore.index.ntotal
        
        # Get unique sources from docstore
        sources = set()
        try:
            # Access docstore documents using getattr for type safety
            docstore_dict = getattr(self.vectorstore.docstore, '_dict', {})
            for doc in docstore_dict.values():
                if hasattr(doc, 'metadata') and "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
        except (AttributeError, TypeError):
            # If _dict is not available, we can't get sources
            pass
        
        return {
            "status": "loaded",
            "num_documents": num_docs,
            "num_sources": len(sources),
            "sources": sorted(list(sources)),
        }
