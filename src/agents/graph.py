"""LangGraph workflow definition."""

from langgraph.graph import StateGraph, END

from src.config import settings
from src.agents.state import AgentState
from src.agents.nodes import checker_node, planner_node, researcher_node

def should_continue(state: AgentState) -> str:
    """Decide whether to loop back or finish."""
    status = state["validation_status"]
    iteration = state["iteration"]
    
    if status == "VALID":
        return "end"
    
    if iteration >= settings.max_iterations:
        return "end"  # Force finish to prevent infinite loops
        
    return "loop"

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("checker", checker_node)
    
    # Define Edges
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "checker")
    
    workflow.add_conditional_edges(
        "checker",
        should_continue,
        {
            "loop": "researcher", # Loop back to ReAct Agent
            "end": END
        }
    )
    
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)

