"""LangGraph workflow definition."""

from langgraph.graph import StateGraph, END

from src.config import settings
from src.agents.state import AgentState
from src.agents.nodes import retrieval_node, maker_node, checker_node, planner_node

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
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("maker", maker_node)
    workflow.add_node("checker", checker_node)
    
    # Define Edges
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "retrieve")
    workflow.add_edge("retrieve", "maker")
    workflow.add_edge("maker", "checker")
    
    workflow.add_conditional_edges(
        "checker",
        should_continue,
        {
            "loop": "maker",
            "end": END
        }
    )
    
    # Memory Checkpointer (Required for chat history!)
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)

