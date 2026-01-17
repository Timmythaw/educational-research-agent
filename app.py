"""
Educational Research Agent - Streamlit UI with Async Live Streaming (FIXED)
"""

import logging
import sys
import uuid
import asyncio
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from src.agents.graph import build_graph
from src.agents.state import AgentState
from src.tools.validator import validator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="EduResearch Agent",
    page_icon="🎓",
    layout="wide"
)

# Title & Description
st.title("🎓 Educational Research Assistant")
st.markdown("""
This AI agent helps you research academic topics by combining:
- 🧠 **Planner & Memory** (Context-aware reasoning)
- 📚 **Internal Knowledge Base** (PDFs)
- 🌐 **Web Search** (Google)
- 🛡️ **Safety Guardrails** (Maker-Checker Loop)
""")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display Chat History
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# ✅ Async Live Streaming Function with Proper State Tracking
async def stream_agent_live(initial_state, config, containers):
    """Stream agent execution with real-time UI updates."""
    final_state = None
    all_states = {}  # ✅ Track states from all nodes
    
    status_container = containers["status"]
    plan_container = containers["plan"]
    research_container = containers["research"]
    checker_container = containers["checker"]
    
    try:
        async for event in st.session_state.agent.astream(initial_state, config=config):
            for key, value in event.items():
                # ✅ Store state from each node
                all_states[key] = value
                final_state = value
                
                if key == "planner":
                    plan = value.get("plan", "")
                    
                    with status_container:
                        st.write("**📋 Status:** Planning research strategy...")
                    
                    with plan_container:
                        st.success("✅ **Planning Complete**")
                        with st.expander("📋 Research Strategy", expanded=False):
                            st.info(plan)
                        
                elif key == "researcher":
                    iteration = value.get("iteration", 0)
                    
                    with status_container:
                        if iteration == 1:
                            st.write("**🔬 Status:** Researching sources...")
                        else:
                            st.write(f"**🔬 Status:** Re-researching (iteration {iteration})...")
                    
                    agent_steps = value.get("agent_steps", [])
                    
                    with research_container:
                        if iteration == 1:
                            st.success("✅ **Research Complete**")
                        else:
                            st.info(f"🔄 **Research Iteration {iteration}**")
                        
                        if agent_steps:
                            with st.expander(f"🧠 Agent Reasoning ({len(agent_steps)} steps)", expanded=True):
                                for i, step in enumerate(agent_steps, 1):
                                    if step["type"] == "tool_call":
                                        st.markdown(f"**🔧 Tool {i}:** `{step['tool']}`")
                                        st.caption(step['result'][:400] + "..." if len(step['result']) > 400 else step['result'])
                                        st.divider()
                                    elif step["type"] == "reasoning":
                                        st.markdown(f"**💭 Reasoning {i}:**")
                                        st.caption(str(step['content'])[:300])
                                        st.divider()
                    
                elif key == "checker":
                    valid_status = value.get("validation_status")
                    critique = value.get("critique")
                    iteration = value.get("iteration", 0)
                    
                    with status_container:
                        st.write("**🛡️ Status:** Validating answer...")
                    
                    with checker_container:
                        if valid_status == "VALID":
                            st.success("✅ **Validation Passed** - Answer approved!")
                        else:
                            st.warning(f"⚠️ **Validation Failed (Iteration {iteration})** - Requesting improvements...")
                            with st.expander("📝 Checker Feedback", expanded=False):
                                st.info(critique)
        
        # ✅ Return all states to extract draft_answer
        return all_states, final_state
    
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        raise


# Chat Input
query = st.chat_input("Ask a research question...")

if query:
    # 1. Display User Message
    st.session_state.messages.append(HumanMessage(content=query))
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Safety Check
    with st.spinner("⚡ Running safety checks..."):
        safety = validator.check_safety(query)
        if not safety["is_safe"]:
            with st.chat_message("assistant"):
                error_msg = f"🚫 **Request Blocked:** {safety['reason']}"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=f"Request Blocked: {safety['reason']}"))
            st.stop()

    # 3. Run Agent with Live Streaming
    with st.chat_message("assistant"):
        
        # Create containers for live updates
        status_container = st.empty()
        
        with st.container():
            st.markdown("### 🤖 Agent Workflow")
            plan_container = st.container()
            research_container = st.container()
            checker_container = st.container()
        
        st.divider()
        
        response_container = st.empty()
        
        # Prepare state with memory
        initial_state: AgentState = {
            "query": query,
            "messages": st.session_state.messages.copy(),
            "plan": "",
            "retrieved_docs": [],
            "draft_answer": "",
            "critique": "",
            "validation_status": "",
            "iteration": 0,
            "agent_steps": []
        }
        
        config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        try:
            # Prepare containers
            containers = {
                "status": status_container,
                "plan": plan_container,
                "research": research_container,
                "checker": checker_container
            }
            
            # ✅ Run async live streaming
            with st.spinner("🤖 Agent working..."):
                all_states, final_state = asyncio.run(
                    stream_agent_live(initial_state, config, containers)
                )
            
            # Update final status
            with status_container:
                st.success("**✅ Complete!** Answer generated successfully.")
            
            # ✅ Extract draft_answer from the latest researcher state
            final_answer = ""
            
            # Try to get from final_state first
            if final_state and final_state.get("draft_answer"):
                final_answer = final_state.get("draft_answer")
            # If not in final state, look for it in researcher state
            elif "researcher" in all_states:
                final_answer = all_states["researcher"].get("draft_answer", "")
            
            # Ensure we have content
            if not final_answer or final_answer.strip() == "":
                final_answer = "⚠️ No answer generated. The agent completed but no draft was produced. Please try again."
                logger.warning("No draft_answer found in any state")
                logger.debug(f"All states keys: {all_states.keys()}")
                logger.debug(f"Final state: {final_state}")
            
            with response_container:
                st.markdown("### 📝 Final Answer")
                st.markdown(final_answer)
            
            # ✅ Add assistant response to memory
            st.session_state.messages.append(AIMessage(content=final_answer))
            
        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}")
            logger.error(f"Agent execution error: {e}", exc_info=True)
            
            with response_container:
                st.error("""
                **Something went wrong during research.**
                
                Please try:
                - Rephrasing your question
                - Being more specific
                - Checking your internet connection
                """)
