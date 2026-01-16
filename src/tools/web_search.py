"""Web search tool using Google Custom Search."""

import logging
from typing import Dict, Any, Optional

from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import tool

from src.config import settings

logger = logging.getLogger(__name__)

# Initialize Google Search Wrapper with settings
search_wrapper: Optional[GoogleSearchAPIWrapper] = None

if settings.google_cse_id and settings.google_search_api_key:
    try:
        search_wrapper = GoogleSearchAPIWrapper(
            google_cse_id=settings.google_cse_id,
            google_api_key=settings.google_search_api_key
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Google Search: {e}")
        search_wrapper = None
else:
    logger.warning("Google Search not configured (missing CSE_ID or SEARCH_API_KEY)")

class WebSearchTool:
    """Tool for searching the internet using Google."""
    
    @staticmethod
    def search(query: str) -> Dict[str, Any]:
        if search_wrapper is None:
            logger.warning("Google Search not available - credentials not configured")
            return {"context_str": "", "error": "Google Search not configured"}
            
        try:
            logger.info(f"Googling: {query}")
            results = search_wrapper.run(query)
            return {
                "context_str": f"[Source: Google Search] {results}",
                "source": "google_search"
            }
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return {"context_str": "", "error": str(e)}

@tool
def search_web(query: str) -> str:
    """Search the web using Google Custom Search for the given query."""
    result = WebSearchTool.search(query)
    return result["context_str"]
