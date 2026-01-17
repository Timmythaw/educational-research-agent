"""
Educational Research Agent - Streamlit UI
"""

import logging
import sys
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Add project root to path so we can import src
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

# Initialize Graph (Lazy load)
if "agent" not in st.session_state:
    st.session_state.agent = build_graph()

# Initialize Thread ID for Memory (Persists across reruns)
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

# Chat Input
query = st.chat_input("Ask a research question...")

if query:
    # 1. Display User Message
    st.session_state.messages.append(HumanMessage(content=query))
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Safety Check
    with st.status("Running Safety Checks...", expanded=True) as status:
        safety = validator.check_safety(query)
        if not safety["is_safe"]:
            st.error(f"Blocked: {safety['reason']}")
            st.session_state.messages.append(AIMessage(content=f"Request Blocked: {safety['reason']}"))
            status.update(label="Safety Check Failed", state="error")
            st.stop()
        
        status.update(label="Safety Check Passed", state="complete")

    # 3. Run Agent Loop
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Prepare state and config for memory
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
        
        # Config acts as the session key for LangGraph memory
        config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}  # type: ignore
        
        final_answer = ""
        
        try:
            with st.status("Agent is working...", expanded=True) as status:
                
                for event in st.session_state.agent.stream(initial_state, config=config):
                    for key, value in event.items():
                        
                        if key == "planner":
                            plan = value.get("plan", "")
                            st.write("**Planner:** Generated search strategy")
                            with st.expander("View Plan"):
                                st.info(plan)
                                
                        elif key == "researcher":
                            st.write("**Researcher:** Working with tools...")
                            
                            # Display agent steps
                            agent_steps = value.get("agent_steps", [])
                            if agent_steps:
                                with st.expander(f"Agent Reasoning ({len(agent_steps)} steps)", expanded=True):
                                    for i, step in enumerate(agent_steps):
                                        if step["type"] == "tool_call":
                                            st.write(f"Tool {i+1}: `{step['tool']}`")
                                            with st.container():
                                                st.caption(step['result'])
                                        elif step["type"] == "reasoning":
                                            st.write(f"Thought {i+1}:")
                                            st.caption(str(step['content'])[:300])
                                
                        elif key == "checker":
                            valid_status = value.get("validation_status")
                            critique = value.get("critique")
                            
                            if valid_status == "VALID":
                                st.write("Checker: Answer approved!")
                            else:
                                st.warning("Checker: Found issues, refining...")
                                with st.expander("View Critique"):
                                    st.info(critique)
                
                status.update(label="Research Complete!", state="complete")
            # 4. Display Final Answer
            final_state = st.session_state.agent.invoke(initial_state, config=config)
            final_answer = final_state.get("draft_answer", "Error generating answer.")
            
            response_placeholder.markdown(final_answer)
            st.session_state.messages.append(AIMessage(content=final_answer))
            
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Streamlit Error: {e}")
