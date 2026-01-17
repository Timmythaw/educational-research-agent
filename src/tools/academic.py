"""Academic search tool using ArXiv."""

import logging
from typing import Dict, Any
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool
import arxiv

logger = logging.getLogger(__name__)

# Initialize ArXiv
# doc_content_chars_max limits the text size so we don't blow up context
arxiv_wrapper = ArxivAPIWrapper(
    arxiv_search=arxiv.Search,
    arxiv_exceptions=arxiv.ArxivError,
    top_k_results=5,
    doc_content_chars_max=3000
)

class AcademicSearchTool:
    @staticmethod
    def search(query: str) -> Dict[str, Any]:
        try:
            logger.info(f"Searching ArXiv for: {query}")
            results = arxiv_wrapper.run(query)
            
            # If ArXiv returns "No good ArXiv Result Found", handle it
            if "No good ArXiv Result" in results:
                return {"context_str": "", "source": "arxiv"}
                
            return {
                "context_str": f"--- ACADEMIC PAPERS (ARXIV) ---\n{results}",
                "source": "arxiv"
            }
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return {"context_str": "", "error": str(e)}

@tool
def search_academic(query: str) -> str:
    """
    Search ArXiv for actual academic research papers, abstracts, and authors.
    Use this when you need deep scientific or technical verification.
    """
    result = AcademicSearchTool.search(query)
    return result["context_str"]
