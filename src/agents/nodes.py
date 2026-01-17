"""Nodes for the LangGraph workflow."""

"""Nodes for the LangGraph workflow."""

import logging
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from src.config import settings
from src.agents.state import AgentState
from src.prompts import (
    META_SYSTEM_PROMPT,
    PLANNER_PROMPT_WITH_HISTORY
)
from src.tools.retriever import search_knowledge_base
from src.tools.web_search import search_web
from src.tools.academic import search_academic

logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.google_api_key,
    temperature=settings.temperature,
)

def planner_node(state: AgentState) -> Dict[str, Any]:
    """Break down complex queries or handle chat history."""
    logger.info("Planner: Analyzing query...")
    
    query = state["query"]
    messages = state.get("messages", [])
    
    # 1. Format History (Safe handling)
    # We take previous messages (excluding the current query which is the last one)
    previous_messages = messages[:-1] if len(messages) > 0 else []
    
    if previous_messages:
        # Take last 5 messages for context
        history_str = "\n".join([f"{m.type.upper()}: {m.content}" for m in previous_messages[-5:]])
    else:
        history_str = "No previous conversation history."
    
    # 3. Invoke LLM
    chain = PLANNER_PROMPT_WITH_HISTORY | llm
    response = chain.invoke({
        "query": query,
        "history": history_str
    })
    
    plan = response.content
    return {"plan": plan}



def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Autonomous ReAct agent that decides which tools to use.
    """
    logger.info("Researcher Agent: Working...")
    
    tools = [search_knowledge_base, search_web]
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=META_SYSTEM_PROMPT
    )
    
    query = state["query"]
    plan = state.get("plan", "")
    critique = state.get("critique", "")
    
    if critique:
        # Don't use f-string with variables that might contain curly braces
        user_message = (
            "Original Query: " + query + "\n\n"
            "Your previous answer was rejected with this critique:\n" + critique + "\n\n"
            "Please research again using your tools and provide an improved answer."
        )
    else:
        user_message = (
            "User Query: " + query + "\n\n"
            "Research Plan:\n" + plan + "\n\n"
            "Execute this plan using your tools (search_knowledge_base, search_web, search_academic). "
            "Follow the response structure defined in your system prompt."
        )
    
    # Pass as dict (not HumanMessage)
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_message}]
    })
    
    final_message = result["messages"][-1]
    
    # Handle different content formats
    if isinstance(final_message.content, list):
        # Content is a list of content blocks (like your case)
        draft_text = ""
        for block in final_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                draft_text += block.get("text", "")
    elif isinstance(final_message.content, str):
        # Content is a plain string
        draft_text = final_message.content
    else:
        # Fallback
        draft_text = str(final_message.content)
    
    return {
        "draft_answer": draft_text,
        "iteration": state.get("iteration", 0) + 1
    }

def checker_node(state: AgentState) -> Dict[str, Any]:
    """Validate the draft answer."""
    logger.info("Checker: Validating answer...")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    draft = state["draft_answer"]
    iteration = state.get("iteration", 0)
    
    # Ensure draft is a string (it might be a list or other type)
    if isinstance(draft, list):
        draft = "\n".join(str(item) for item in draft)
    elif not isinstance(draft, str):
        draft = str(draft)
    
    if iteration >= 2:
        logger.info("Iteration limit approaching. Using lenient validation.")
        # Just check if there's substantial content
        if len(draft) > 200 and ("References" in draft or "Source" in draft):
            return {
                "critique": "VALID (lenient mode due to iteration limit)",
                "validation_status": "VALID"
            }
    # Since the agent handles context internally, we validate based on:
    # 1. Does it answer the query?
    # 2. Does it cite sources?
    # 3. Does it follow the response structure?
    
    # Build the full user message first to avoid template variable issues
    user_message = (
        "Draft Answer:\n" + draft + "\n\n"
        "Task:\n"
        "1. Does this answer the user's query reasonably well?\n"
        "2. Does it cite sources properly?\n"
        "3. Does it sound plausible and avoid hallucinations?\n\n"
        "Output 'VALID' if the answer is acceptable, or provide a specific Critique listing issues to fix."
    )
    
    CHECKER_PROMPT_AGENT = ChatPromptTemplate.from_messages([
        ("system", "You are an academic editor. Be reasonably strict but not overly critical."),
        ("user", "{user_input}")
    ])
    
    chain = CHECKER_PROMPT_AGENT | llm
    response = chain.invoke({"user_input": user_message})
    
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
    