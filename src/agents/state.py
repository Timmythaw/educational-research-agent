"""State definition for the educational research agent."""

import operator
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """The state of the agent during the research process."""
    
    # User input
    query: str
    
    # Conversation history
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Context data
    retrieved_docs: List[str]  # The raw text chunks
    
    # Generation & Validation
    draft_answer: str       # The Maker's latest attempt
    critique: str           # The Checker's feedback
    validation_status: str  # "VALID" or "INVALID"
    
    # Loop control
    iteration: int          # Tracks how many refinement loops we've done
