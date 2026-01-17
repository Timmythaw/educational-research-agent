"""Nodes for the LangGraph workflow."""

"""Nodes for the LangGraph workflow."""

import logging
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

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
    if state.get("retrieved_docs"):
        logger.info("Sources already gathered, skipping research.")
        return {}

    tools = [search_knowledge_base, search_web]
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""You are a research assistant. Your job is to:
        1. Use your tools to gather relevant information
        2. Return the raw research findings WITHOUT writing a final answer
        3. Be thorough - use multiple tools and queries

        Do NOT write a final answer. Just gather comprehensive context.""",
    )
    
    query = state["query"]
    plan = state.get("plan", "")
    user_message = (
        "User Query: " + query + "\n\n"
        "Research Plan:\n" + plan + "\n\n"
        "Execute this plan using your tools (search_knowledge_base, search_web, search_academic). "
        "Return all relevant findings."
    )

    # Streaming to capture intermediate steps
    agent_steps = []
    try:
        for event in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
            for key, value in event.items():
                if key == "agent":
                    message = value["messages"][-1]
                    logger.info(f"Research reasoning: {message}")
                    agent_steps.append({"type": "reasoning", "content": str(message)})
                    
                elif key == "tools":
                    tool_calls = value.get("messages", [])
                    for tool_msg in tool_calls:
                        tool_name = getattr(tool_msg, 'name', 'unknown')
                        logger.info(f"Tool called: {tool_name}")
                        agent_steps.append({
                            "type": "tool_call",
                            "tool": tool_name,
                            "result": str(tool_msg.content)[:200]
                        })
    except Exception as e:
        logger.warning(f"Research streaming failed: {e}")
    
    # Pass as dict (not HumanMessage)
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_message}]
    })
    
    final_message = result["messages"][-1]
    
    # Handle different content formats
    if isinstance(final_message.content, list):
        research_context = ""
        for block in final_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                research_context += block.get("text", "")
    elif isinstance(final_message.content, str):
        research_context = final_message.content
    else:
        research_context = str(final_message.content)
    
    # Fallback if no steps captured
    if not agent_steps:
        agent_steps.append({
            "type": "tool_call",
            "tool": "Research Agent",
            "result": "Completed source gathering"
        })
    
    return {
        "retrieved_docs": [research_context],  # Store all gathered context
        "agent_steps": agent_steps
    }

def writer_node(state: AgentState) -> Dict[str, Any]:
    """
    Writer node - Writes answer using pre-gathered sources.
    This node CAN loop if checker rejects.
    """
    logger.info("Writer: Drafting answer...")
    
    query = state["query"]
    plan = state.get("plan", "")
    retrieved_docs = state.get("retrieved_docs", [])
    critique = state.get("critique", "")
    iteration = state.get("iteration", 0)
    
    # Combine all research context
    research_context = "\n\n".join(retrieved_docs)
    
    if critique and iteration > 0:
        # Refinement mode
        old_draft = state.get("draft_answer", "")
        
        WRITER_PROMPT = ChatPromptTemplate.from_messages([
            ("system", META_SYSTEM_PROMPT),
            ("user", """Previous Draft:
{old_draft}

Checker's Critique:
{critique}

Research Context:
{context}

Task: Revise the draft to address the critique. Use the research context to add missing citations or fix errors. Maintain the good parts of the original.""")
        ])
        
        chain = WRITER_PROMPT | llm
        response = chain.invoke({
            "old_draft": old_draft,
            "critique": critique,
            "context": research_context
        })
        
    else:
        # First draft mode
        WRITER_PROMPT = ChatPromptTemplate.from_messages([
            ("system", META_SYSTEM_PROMPT),
            ("user", """User Query: {query}

Research Plan:
{plan}

Research Context (from tools):
{context}

Task: Write a comprehensive answer following the response structure in your system prompt. Use ONLY information from the research context. Cite sources properly.""")
        ])
        
        chain = WRITER_PROMPT | llm
        response = chain.invoke({
            "query": query,
            "plan": plan,
            "context": research_context
        })
    
    draft_text = response.content if isinstance(response.content, str) else str(response.content)
    
    return {
        "draft_answer": draft_text,
        "iteration": iteration + 1
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
    