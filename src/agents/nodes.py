"""Nodes for the LangGraph workflow."""

"""Nodes for the LangGraph workflow."""

import logging
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

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
    Replaces the fixed retrieval_node + maker_node.
    """
    logger.info("Researcher Agent: Working...")
    
    # 1. Define Tools (KB, Web, ArXiv)
    tools = [search_knowledge_base, search_web, search_academic]
    
    # 2. Create Agent with YOUR META_SYSTEM_PROMPT
    # This ensures the agent follows your citation format and response structure
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=META_SYSTEM_PROMPT
    )
    
    # 3. Construct Input
    query = state["query"]
    plan = state.get("plan", "")
    critique = state.get("critique", "")
    
    if critique:
        # Loop mode: Agent needs to refine based on critique
        user_message = (
            f"Original Query: {query}\n\n"
            f"Your previous answer was rejected with this critique:\n{critique}\n\n"
            f"Please research again using your tools and provide an improved answer."
        )
    else:
        # First mode: Agent executes the plan
        user_message = (
            f"User Query: {query}\n\n"
            f"Research Plan:\n{plan}\n\n"
            f"Execute this plan using your tools (search_knowledge_base, search_web, search_academic). "
            f"Follow the response structure defined in your system prompt."
        )
    
    # 4. Invoke the Agent
    result = agent.invoke({
        "messages": [HumanMessage(content=user_message)]
    })
    
    # 5. Extract Final Answer
    final_message = result["messages"][-1]
    
    return {
        "draft_answer": final_message.content,
        "iteration": state.get("iteration", 0) + 1
    }

def checker_node(state: AgentState) -> Dict[str, Any]:
    """Validate the draft answer."""
    logger.info("Checker: Validating answer...")
    
    from langchain_core.prompts import ChatPromptTemplate
    
    draft = state["draft_answer"]
    
    # Since the agent handles context internally, we validate based on:
    # 1. Does it answer the query?
    # 2. Does it cite sources?
    # 3. Does it follow the response structure?
    
    CHECKER_PROMPT_AGENT = ChatPromptTemplate.from_messages([
        ("system", "You are a strict academic editor."),
        ("user", f"Draft Answer:\n{draft}\n\n"
                 "Task:\n"
                 "1. Does this answer follow the required response structure (Direct Answer, Detailed Synthesis, Key Takeaways, References)?\n"
                 "2. Does it cite sources properly?\n"
                 "3. Does it sound plausible and avoid hallucinations?\n\n"
                 "Output 'VALID' if the answer is acceptable, or provide a specific Critique listing issues to fix.")
    ])
    
    chain = CHECKER_PROMPT_AGENT | llm
    response = chain.invoke({})
    
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
    