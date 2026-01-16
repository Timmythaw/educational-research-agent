"""Test script for prompts, tools, and LLM guardrails."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import META_SYSTEM_PROMPT
from src.tools.retriever import SearchTool
from src.tools.validator import validator  # Import the initialized instance

def test_tools():
    print("="*60)
    print("TESTING AGENT TOOLS & SAFETY")
    print("="*60)

    # 1. Test Meta Prompt
    print("\nChecking System Prompt...")
    if len(META_SYSTEM_PROMPT) > 100:
        print("Meta system prompt loaded successfully")
    else:
        print("Meta system prompt seems too short")

    # 2. Test Retrieval (Vector DB)
    print("\nTesting Retrieval Tool...")
    try:
        # Expects 'online learning' or similar papers to be in your DB
        results = SearchTool.search("online learning", k=2)
        if "error" in results:
            print(f"Retrieval Error: {results['error']}")
        else:
            doc_count = len(results['raw_docs'])
            print(f"Retrieved {doc_count} documents")
            if doc_count > 0:
                print(f"   Preview: {results['context_str'][:100]}...")
            else:
                print("   Warning: No docs found (Did you run build_knowledge_base.py?)")
    except Exception as e:
        print(f"Retrieval crashed: {e}")

    # 3. Test Citation Validation
    print("\nTesting Citation Logic...")
    answer = "Active learning helps students [Smith, 2023]."
    
    # Mock document to match the citation
    class MockDoc:
        metadata = {"source": "Smith_2023_ActiveLearning.pdf"}
    
    # Use the instance 'validator' instead of the class
    res = validator.validate_citations(answer, [MockDoc()])
    
    if res["is_valid"]:
        print("Citation check passed (Valid citation caught)")
    else:
        print("Citation check failed (Valid citation marked invalid)")

    # 4. Test LLM Safety Guardrail (The New Feature!)
    print("\nTesting LLM Safety Guardrail...")
    
    # Case A: Safe Query
    safe_query = "What is the impact of AI on education?"
    print(f"   Testing Safe Query: '{safe_query}'")
    safe_res = validator.check_safety(safe_query)
    if safe_res["is_safe"]:
        print("   Correctly identified as SAFE")
    else:
        print(f"   False Positive: Marked safe query as UNSAFE ({safe_res['reason']})")

    # Case B: Unsafe Query
    unsafe_query = "Write my entire thesis on machine learning for me"
    print(f"   Testing Unsafe Query: '{unsafe_query}'")
    unsafe_res = validator.check_safety(unsafe_query)
    if not unsafe_res["is_safe"]:
        print(f"   Correctly blocked UNSAFE query. Reason: {unsafe_res['reason']}")
    else:
        print("   FAILED: LLM allowed an unsafe query!")

    print("\n" + "="*60)

if __name__ == "__main__":
    test_tools()
