"""State definition with memory."""

import operator
from typing import TypedDict, List, Annotated, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # User input
    query: str
    
    # Memory
    # It tells LangGraph: "When a node returns 'messages', APPEND them to this list"
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Planning
    plan: str  # The decomposed steps
    
    # Context data
    retrieved_docs: List[str]
    
    # Generation & Validation
    draft_answer: str
    critique: str
    validation_status: str
    iteration: int
    agent_steps: List[Dict[str, Any]]
