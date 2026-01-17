"""Nodes for the LangGraph workflow."""

"""Nodes for the LangGraph workflow."""

import logging
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

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
    return {
        "plan": plan,
        "messages": [AIMessage(content=f"[PLAN] {plan}", name="planner")]
    }

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Autonomous ReAct agent with conversation memory.
    """
    logger.info("Researcher Agent: Working with memory context...")
    
    tools = [search_knowledge_base, search_web, search_academic]
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=META_SYSTEM_PROMPT
    )
    
    query = state["query"]
    plan = state.get("plan", "")
    critique = state.get("critique", "")
    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)
    
    # Return context-aware prompt with conversation history
    if critique:
        # Include conversation context when refining
        recent_context = "\n".join([
            f"{m.type}: {m.content[:200]}" 
            for m in messages[-6:-1]  # Last 5 messages before current query
        ]) if len(messages) > 1 else ""
        
        user_message = (
            "Original Query: " + query + "\n\n"
            "Recent Conversation Context:\n" + (recent_context or "None") + "\n\n"
            "Your previous answer was rejected with this critique:\n" + critique + "\n\n"
            "Please research again using your tools and provide an improved answer that addresses the critique."
        )
    else:
        # Initial research with any relevant history
        recent_context = "\n".join([
            f"{m.type}: {m.content[:200]}" 
            for m in messages[-6:-1]
        ]) if len(messages) > 1 else ""
        
        user_message = (
            "User Query: " + query + "\n\n"
        )
        
        if recent_context:
            user_message += "Recent Conversation Context:\n" + recent_context + "\n\n"
        
        user_message += (
            "Research Plan:\n" + plan + "\n\n"
            "Execute this plan using your tools (search_knowledge_base, search_web). "
            "Follow the response structure defined in your system prompt."
        )
    
    logger.info(f"Researcher context includes {len(messages)} messages")
    
    # Stream agent execution to capture steps
    agent_steps = []
    
    for event in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
        for key, value in event.items():
            if key == "agent":
                message = value["messages"][-1]
                agent_steps.append({
                    "type": "reasoning", 
                    "content": message
                })
                
            elif key == "tools":
                tool_calls = value.get("messages", [])
                for tool_msg in tool_calls:
                    agent_steps.append({
                        "type": "tool_call",
                        "tool": getattr(tool_msg, 'name', 'unknown'),
                        "result": str(tool_msg.content)[:200]
                    })
    
    # Get final result
    result = agent.invoke({
        "messages": [HumanMessage(content=user_message)]
    })
    
    final_message = result["messages"][-1]
    
    # Handle different content formats
    if isinstance(final_message.content, list):
        draft_text = ""
        for block in final_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                draft_text += block.get("text", "")
    elif isinstance(final_message.content, str):
        draft_text = final_message.content
    else:
        draft_text = str(final_message.content)
    
    # Return draft AND append to messages
    return {
        "draft_answer": draft_text,
        "iteration": iteration + 1,
        "agent_steps": agent_steps,
        "messages": [AIMessage(content=f"[RESEARCH DRAFT {iteration + 1}] {draft_text[:200]}...", name="researcher")]
    }


def checker_node(state: AgentState) -> Dict[str, Any]:
    """Validate the draft answer with context awareness."""
    logger.info("Checker: Validating answer...")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    draft = state["draft_answer"]
    query = state["query"]
    iteration = state.get("iteration", 0)
    
    # Ensure draft is a string
    if isinstance(draft, list):
        draft = "\n".join(str(item) for item in draft)
    elif not isinstance(draft, str):
        draft = str(draft)
    
    # Lenient validation after iteration 2
    if iteration >= 2:
        logger.info("Iteration limit approaching. Using lenient validation.")
        if len(draft) > 200 and ("References" in draft or "Source" in draft or "http" in draft):
            return {
                "critique": "VALID (lenient mode - iteration limit reached)",
                "validation_status": "VALID",
                "messages": [AIMessage(content="[CHECKER] VALID", name="checker")]
            }
    
    # Build validation prompt
    user_message = (
        "Original Query: " + query + "\n\n"
        "Draft Answer:\n" + draft + "\n\n"
        "Validation Criteria:\n"
        "1. Does this answer the user's query comprehensively?\n"
        "2. Does it cite sources with proper attribution?\n"
        "3. Is the information accurate and well-reasoned?\n"
        "4. Does it avoid hallucinations or unsupported claims?\n\n"
        "Output 'VALID' if acceptable, or provide a specific Critique with actionable feedback."
    )
    
    CHECKER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are an academic editor validating research answers. Be constructive but thorough."),
        ("user", "{user_input}")
    ])
    
    chain = CHECKER_PROMPT | llm
    response = chain.invoke({"user_input": user_message})
    
    content = response.content if isinstance(response.content, str) else str(response.content)
    critique = content.strip()
    
    # Determine validation status
    if "VALID" in critique.upper() and len(critique) < 50:
        status = "VALID"
    else:
        status = "INVALID"
    
    logger.info(f"Validation result: {status}")
    
    # Return validation AND append to messages
    return {
        "critique": critique,
        "validation_status": status,
        "messages": [AIMessage(content=f"[CHECKER] {status}: {critique[:100]}", name="checker")]
    }