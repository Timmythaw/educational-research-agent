"""LangGraph workflow definition."""

from asyncio.log import logger
from langgraph.graph import StateGraph, END

from src.config import settings
from src.agents.state import AgentState
from src.agents.nodes import checker_node, planner_node, researcher_node, writer_node

def should_continue(state: AgentState) -> str:
    """Decide whether to continue refinement or end."""
    
    # 1. Check iteration limit FIRST
    max_iterations = 3
    current_iteration = state.get("iteration", 0)
    
    if current_iteration >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached. Ending loop.")
        return "end"
    
    # 2. Check validation status
    validation_status = state.get("validation_status", "INVALID")
    
    if validation_status == "VALID":
        return "end"
    else:
        return "loop"

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("research", researcher_node)   
    workflow.add_node("writer", writer_node)    
    workflow.add_node("checker", checker_node)
    
    # Define Edges
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "research")
    workflow.add_edge("research", "writer")
    workflow.add_edge("writer", "checker")
    
    workflow.add_conditional_edges(
        "checker",
        should_continue,
        {
            "loop": "writer", # Loop back to ReAct Agent
            "end": END
        }
    )
    
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)

