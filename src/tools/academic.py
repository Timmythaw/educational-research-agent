"""Academic search with thread-safe ArXiv access."""

import logging
import threading
import arxiv
from typing import Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Thread lock for ArXiv API
_arxiv_lock = threading.Lock()


class AcademicSearchTool:
    """Tool for searching academic papers on ArXiv with thread safety."""
    
    @staticmethod
    def search(query: str, max_results: int = 5) -> Dict[str, Any]:
        try:
            logger.info(f"Searching ArXiv for: {query}")
            
            # Lock ArXiv searches
            with _arxiv_lock:
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                results = []
                for paper in search.results():
                    results.append({
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "summary": paper.summary[:300],
                        "url": paper.entry_id,
                        "published": str(paper.published.date())
                    })
            
            if not results:
                return {
                    "context_str": "⚠️ No academic papers found for this query.",
                    "source": "arxiv"
                }
            
            formatted = []
            for paper in results:
                authors_str = ", ".join(paper["authors"][:3])
                formatted.append(
                    f"**{paper['title']}**\n"
                    f"Authors: {authors_str}\n"
                    f"Published: {paper['published']}\n"
                    f"Link: {paper['url']}\n"
                    f"Summary: {paper['summary']}\n"
                )
            
            context_str = "\n---\n".join(formatted)
            
            return {
                "context_str": f"--- ACADEMIC PAPERS (ArXiv) ---\n{context_str}",
                "source": "arxiv"
            }
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return {
                "context_str": "⚠️ Academic search failed.",
                "error": str(e)
            }


@tool
def search_academic(query: str) -> str:
    """
    Search for academic papers on ArXiv.
    Thread-safe implementation.
    """
    result = AcademicSearchTool.search(query)
    return result.get("context_str", "")
