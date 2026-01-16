"""Web search tool using Google Custom Search."""

import logging
from typing import Dict, Any, Optional

from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import tool

from src.config import settings

logger = logging.getLogger(__name__)

# Initialize Google Search Wrapper
search_wrapper: Optional[GoogleSearchAPIWrapper] = None

# Initialize search wrapper only if config is present
if settings.google_cse_id and settings.google_search_api_key:
    try:
        search_wrapper = GoogleSearchAPIWrapper(
            google_cse_id=settings.google_cse_id,
            google_api_key=settings.google_search_api_key
        )
    except Exception:
        search_wrapper = None

class WebSearchTool:
    """Tool for searching the internet using Google."""
    
    @staticmethod
    def search(query: str) -> Dict[str, Any]:
        if search_wrapper is None:
            return {"context_str": "", "error": "Google Search not configured"}
            
        try:
            logger.info(f"Googling: {query}")
            
            # Use .results() instead of .run() to get metadata (links!)
            raw_results = search_wrapper.results(query, num_results=5)
            
            formatted_results = []
            for item in raw_results:
                title = item.get("title", "No Title")
                link = item.get("link", "#")
                snippet = item.get("snippet", "")
                
                # Format: [Title](Link): Snippet
                formatted_results.append(f"Source: [{title}]({link})\nContent: {snippet}\n")
            
            context_str = "\n---\n".join(formatted_results)
            
            return {
                "context_str": f"--- WEB SEARCH RESULTS ---\n{context_str}",
                "source": "google_search"
            }
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return {"context_str": "", "error": str(e)}

@tool
def search_web(query: str) -> str:
    """Search the web using Google Custom Search."""
    result = WebSearchTool.search(query)
    return result["context_str"]
