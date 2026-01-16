"""Nodes for the LangGraph workflow."""

import logging
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.agents.state import AgentState
from src.prompts import MAKER_PROMPT, CHECKER_PROMPT
from src.tools.retriever import SearchTool
from src.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.google_api_key,
    temperature=settings.temperature,
)

def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    Hybrid Retrieval: Search BOTH Knowledge Base and Web.
    """
    query = state['query']
    logger.info(f"Hybrid Retrieval for: {query}")
    
    final_docs = []
    
    # 1. Search Knowledge Base (Academic/Deep Info)
    try:
        kb_result = SearchTool.search(query, k=settings.top_k_retrieval)
        if kb_result.get("raw_docs"):
            logger.info(f"   Found {len(kb_result['raw_docs'])} academic docs")
            # Label them clearly so the LLM knows these are from the papers
            final_docs.append(f"--- ACADEMIC KNOWLEDGE BASE ---\n{kb_result['context_str']}")
    except Exception as e:
        logger.error(f"   KB Search failed: {e}")

    # 2. Search Web (Current/General Info)
    # We ALWAYS search the web for completeness, or you could add a keyword check
    try:
        web_result = WebSearchTool.search(query)
        if web_result.get("context_str"):
            logger.info(f"   Found web results")
            final_docs.append(f"--- WEB SEARCH RESULTS ---\n{web_result['context_str']}")
    except Exception as e:
        logger.error(f"   Web Search failed: {e}")
    
    # 3. Handle case with no results
    if not final_docs:
        final_docs = ["No relevant information found in Knowledge Base or Web."]
    
    return {"retrieved_docs": final_docs}

# ... (Keep maker_node and checker_node exactly the same as before) ...
def maker_node(state: AgentState) -> Dict[str, Any]:
    """Generate a draft answer using retrieved docs."""
    logger.info("Maker: Generating draft answer...")
    
    context = "\n\n".join(state["retrieved_docs"])
    iteration = state.get("iteration", 0)
    
    if state.get("critique"):
        prompt_input = (
            f"Previous Draft: {state['draft_answer']}\n"
            f"Checker's Critique: {state['critique']}\n\n"
            f"Please write a NEW, improved answer that addresses the critique."
        )
    else:
        prompt_input = state["query"]
        
    chain = MAKER_PROMPT | llm
    response = chain.invoke({
        "query": prompt_input,
        "context": context
    })
    
    return {
        "draft_answer": response.content,
        "iteration": iteration + 1
    }

def checker_node(state: AgentState) -> Dict[str, Any]:
    """Validate the draft answer."""
    logger.info("Checker: Validating answer...")
    
    context = "\n\n".join(state["retrieved_docs"])
    draft = state["draft_answer"]
    
    chain = CHECKER_PROMPT | llm
    response = chain.invoke({
        "context": context,
        "draft_answer": draft
    })
    
     # Handle response content which may be a string or have content attribute
    content = response.content if isinstance(response.content, str) else str(response.content)
    critique = content.strip()
    
    if "VALID" in critique.upper() and len(critique) < 50:
        status = "VALID"
    else:
        status = "INVALID"
        
    return {
        "critique": critique,
        "validation_status": status
    }
