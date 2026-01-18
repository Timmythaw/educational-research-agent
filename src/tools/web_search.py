"""Web search tool with thread-safe execution and SSL retry logic."""

import logging
import time
import threading
from typing import Dict, Any, Optional
import urllib3

from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import tool

from src.config import settings

logger = logging.getLogger(__name__)

# Disable SSL warnings (only if you're okay with it)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Thread lock for Google Search (prevent concurrent API calls)
_search_lock = threading.Lock()

# Initialize Google Search Wrapper
search_wrapper: Optional[GoogleSearchAPIWrapper] = None

if settings.google_cse_id and settings.google_search_api_key:
    try:
        search_wrapper = GoogleSearchAPIWrapper(
            google_cse_id=settings.google_cse_id,
            google_api_key=settings.google_search_api_key,
            k=5  # Limit results
        )
    except Exception as e:
        logger.error(f"Failed to initialize Google Search: {e}")
        search_wrapper = None


class WebSearchTool:
    """Tool for searching the internet using Google with thread-safety."""
    
    @staticmethod
    def search(query: str, max_retries: int = 3) -> Dict[str, Any]:
        if search_wrapper is None:
            return {
                "context_str": "⚠️ Google Search not configured.",
                "error": "Missing API keys"
            }
        
        # Use thread lock to prevent concurrent searches
        with _search_lock:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Googling (attempt {attempt + 1}/{max_retries}): {query}")
                    
                    # Use .results() to get metadata
                    raw_results = search_wrapper.results(query, num_results=5)
                    
                    if not raw_results:
                        logger.warning(f"No results found for: {query}")
                        return {
                            "context_str": "⚠️ No web results found for this query.",
                            "source": "google_search"
                        }
                    
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
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check if it's an SSL error
                    if "ssl" in error_msg or "certificate" in error_msg:
                        logger.warning(f"SSL error on attempt {attempt + 1}: {e}")
                        
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Wait before retry
                            continue
                        else:
                            return {
                                "context_str": "⚠️ Web search unavailable due to network issues. Using other sources.",
                                "error": f"SSL Error: {str(e)}"
                            }
                    
                    # Check if it's a timeout
                    elif "timeout" in error_msg or "timed out" in error_msg:
                        logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                        
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        else:
                            return {
                                "context_str": "⚠️ Web search timed out. Using other sources.",
                                "error": "Timeout"
                            }
                    
                    # Other errors
                    else:
                        logger.error(f"Google search failed: {e}")
                        return {
                            "context_str": "⚠️ Web search failed. Using knowledge base.",
                            "error": str(e)
                        }
            
            # Fallback if all retries fail
            return {
                "context_str": "⚠️ Web search unavailable after multiple attempts.",
                "error": "Max retries exceeded"
            }


@tool
def search_web(query: str) -> str:
    """
    Search the web using Google Custom Search.
    Thread-safe with automatic retry on SSL/timeout errors.
    """
    result = WebSearchTool.search(query)
    return result.get("context_str", "")
