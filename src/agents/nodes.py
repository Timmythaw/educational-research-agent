"""Nodes for the LangGraph workflow."""

import logging
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.agents.state import AgentState
from src.prompts import MAKER_PROMPT, CHECKER_PROMPT
from src.tools.retriever import SearchTool
from src.tools.validator import validator

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.google_api_key,
    temperature=settings.temperature,
)

def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve documents based on the query."""
    logger.info(f"Retrieving docs for: {state['query']}")
    
    search_result = SearchTool.search(state['query'], k=settings.top_k_retrieval)
    
    # If retrieval fails, return empty list (Maker will handle it)
    docs = search_result.get("context_str", "")
    
    return {"retrieved_docs": [docs]}

def maker_node(state: AgentState) -> Dict[str, Any]:
    """Generate a draft answer using retrieved docs."""
    logger.info("Maker: Generating draft answer...")
    
    context = "\n\n".join(state["retrieved_docs"])
    iteration = state.get("iteration", 0)
    
    # If we have a critique from the Checker, add it to the context
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
    
    # 1. LLM-based Quality Check
    chain = CHECKER_PROMPT | llm
    response = chain.invoke({
        "context": context,
        "draft_answer": draft
    })
    
    # Handle response content which may be a string or have content attribute
    content = response.content if isinstance(response.content, str) else str(response.content)
    critique = content.strip()
    
    # 2. Heuristic Citation Check
    # (We assume context is formatted as a list of strings, but validator needs Document objects
    # For MVP simplicity, we'll skip strict validator.validate_citations here and rely on LLM)
    
    # Determine status based on LLM output
    if "VALID" in critique.upper() and len(critique) < 50:
        status = "VALID"
    else:
        status = "INVALID"
        
    return {
        "critique": critique,
        "validation_status": status
    }
