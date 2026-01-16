"""Retrieval tool for the agent."""

import logging
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.documents import Document

from src.knowledge.vector_store import VectorStoreManager
from src.config import settings

logger = logging.getLogger(__name__)

# Global manager instance to avoid reloading FAISS every time
_vector_manager = None

def get_vector_manager() -> VectorStoreManager:
    """Get or initialize the global vector manager."""
    global _vector_manager
    if _vector_manager is None:
        _vector_manager = VectorStoreManager()
        try:
            _vector_manager.load_vectorstore()
        except FileNotFoundError:
            logger.warning("Vector store not found. Search will return empty results.")
    return _vector_manager


class SearchTool:
    """Tool for searching the educational knowledge base."""
    
    @staticmethod
    def search(query: str, k: int = 5) -> Dict[str, Any]:
        """
        Search for educational research papers.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with 'documents' (list of text) and 'sources' (metadata)
        """
        manager = get_vector_manager()
        
        if manager.vectorstore is None:
            return {"error": "Knowledge base not loaded", "documents": []}
            
        try:
            results = manager.search(query, k=k)
            
            # Format for the LLM
            formatted_docs = []
            for doc in results:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", 0)
                content = doc.page_content.replace("\n", " ")
                formatted_docs.append(f"[Source: {source}, Page: {page}] {content}")
            
            return {
                "context_str": "\n\n".join(formatted_docs),
                "raw_docs": results
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e), "documents": []}

# LangChain Tool Definition
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the educational research knowledge base for relevant papers.
    Use this tool to find evidence, facts, and studies to answer student questions.
    """
    result = SearchTool.search(query)
    if "error" in result:
        return f"Error searching knowledge base: {result['error']}"
    return result["context_str"]
