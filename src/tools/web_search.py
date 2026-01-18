# src/tools/web_search.py

import logging
import ssl
import certifi
from typing import Dict, Any, Optional

from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import tool

from src.config import settings

logger = logging.getLogger(__name__)

# Create a custom SSL context that's more permissive
def create_ssl_context():
    """Create SSL context with proper certificate verification."""
    try:
        # Use certifi's certificate bundle
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        return ssl_context
    except Exception as e:
        logger.warning(f"Failed to create SSL context: {e}")
        # Fallback to less strict verification
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context

# Initialize Google Search Wrapper with error handling
search_wrapper: Optional[GoogleSearchAPIWrapper] = None

if settings.google_cse_id and settings.google_search_api_key:
    try:
        search_wrapper = GoogleSearchAPIWrapper(
            google_cse_id=settings.google_cse_id,
            google_api_key=settings.google_search_api_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize Google Search: {e}")
        search_wrapper = None

class WebSearchTool:
    """Tool for searching the internet using Google."""
    
    @staticmethod
    def search(query: str, max_retries: int = 3) -> Dict[str, Any]:
        if search_wrapper is None:
            return {"context_str": "", "error": "Google Search not configured"}
            
        # Retry logic for transient SSL errors
        for attempt in range(max_retries):
            try:
                logger.info(f"Googling (attempt {attempt + 1}/{max_retries}): {query}")
                
                # Use .results() to get metadata
                raw_results = search_wrapper.results(query, num_results=5)
                
                formatted_results = []
                for item in raw_results:
                    title = item.get("title", "No Title")
                    link = item.get("link", "#")
                    snippet = item.get("snippet", "")
                    
                    formatted_results.append(
                        f"Source: [{title}]({link})\nContent: {snippet}\n"
                    )
                
                context_str = "\n---\n".join(formatted_results)
                
                return {
                    "context_str": f"--- WEB SEARCH RESULTS ---\n{context_str}",
                    "source": "google_search"
                }
                
            except ssl.SSLError as e:
                logger.warning(f"SSL error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Google search failed after {max_retries} attempts: {e}")
                    return {
                        "context_str": "⚠️ Web search temporarily unavailable due to connection issues.",
                        "error": f"SSL Error: {str(e)}"
                    }
                # Wait before retry
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Google search failed: {e}")
                return {
                    "context_str": "⚠️ Web search failed. Using available knowledge.",
                    "error": str(e)
                }
        
        # Fallback if loop completes without returning
        return {
            "context_str": "⚠️ Web search failed. Using available knowledge.",
            "error": "Max retries exceeded"
        }

@tool
def search_web(query: str) -> str:
    """Search the web using Google Custom Search."""
    result = WebSearchTool.search(query)
    return result.get("context_str", "")
