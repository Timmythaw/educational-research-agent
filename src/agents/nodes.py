"""Nodes for the LangGraph workflow."""

import logging
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.agents.state import AgentState
from src.prompts import MAKER_PROMPT, CHECKER_PROMPT
from src.tools.retriever import SearchTool
from src.tools.web_search import WebSearchTool  # <--- Import this

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.google_api_key,
    temperature=settings.temperature,
)

def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve documents. Fallback to web search if needed."""
    query = state['query']
    logger.info(f"Retrieving info for: {query}")
    
    # 1. Try Knowledge Base first
    kb_result = SearchTool.search(query, k=settings.top_k_retrieval)
    kb_docs = kb_result.get("context_str", "")
    kb_doc_list = kb_result.get("raw_docs", [])
    
    final_docs = []
    
    # 2. Check if Knowledge Base results are sufficient
    # Heuristic: If we found fewer than 1 doc or text is very short, try Web
    if len(kb_doc_list) == 0 or len(kb_docs) < 200:
        logger.warning("Knowledge Base yielded low results. Falling back to Web Search.")
        
        web_result = WebSearchTool.search(query)
        web_docs = web_result.get("context_str", "")
        
        if web_docs:
            final_docs.append(web_docs)
            # We can also mix them if we found partial info in KB
            if kb_docs:
                final_docs.append(kb_docs)
        else:
            # If web fails too, just return whatever KB had
            final_docs.append(kb_docs)
            
    else:
        # KB was good enough
        final_docs.append(kb_docs)
    
    # Remove empty strings
    final_docs = [d for d in final_docs if d]
    
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
