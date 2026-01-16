"""
Educational Research Agent - Main Entry Point
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.config import settings
from src.agents.graph import build_graph
from src.agents.state import AgentState
from src.tools.validator import validator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_agent():
    """Run the interactive agent loop."""
    load_dotenv()
    
    print("\nEDUCATIONAL RESEARCH AGENT")
    print("=================================")
    print("Ask a question about education (or type 'quit' to exit)")
    print(f"Using Model: {settings.gemini_model}")
    print("=================================\n")
    
    # Initialize Graph
    try:
        agent = build_graph()
    except Exception as e:
        logger.error(f"Failed to build agent: {e}")
        return

    while True:
        try:
            query = input("\nUser: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query:
                continue
                
            # 1. Safety Check (Input Guardrail)
            print("\nRunning Safety Check...")
            safety_result = validator.check_safety(query)
            
            if not safety_result["is_safe"]:
                print(f"BLOCKED: {safety_result['reason']}")
                continue
                
            print("Query Safe. Starting Research...\n")
            
            # 2. Run Agent Workflow
            initial_state: AgentState = {
                "query": query,
                "messages": [HumanMessage(content=query)],
                "iteration": 0,
                "retrieved_docs": [],
                "draft_answer": "",
                "critique": "",
                "validation_status": ""
            }
            
            # Stream events to show progress
            for event in agent.stream(initial_state):
                for key, value in event.items():
                    if key == "retrieve":
                        print(f"Retrieved {len(value.get('retrieved_docs', []))} document contexts.")
                    elif key == "maker":
                        print("Maker generated a draft.")
                    elif key == "checker":
                        status = value.get("validation_status", "UNKNOWN")
                        print(f"Checker validation: {status}")
                        if status == "INVALID":
                            print(f"   Critique: {value.get('critique')[:100]}...")
            
            # 3. Get Final Result (need to fetch final state)
            # Since stream returns intermediate steps, we can just run invoke to get final state
            final_state = agent.invoke(initial_state)
            
            print("\n" + "="*40)
            print("FINAL ANSWER")
            print("="*40)
            print(final_state["draft_answer"])
            print("="*40 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    run_agent()
