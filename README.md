# Educational Research Agent (Agentic RAG)

Deployed demo: https://timmy-educational-research-agent.streamlit.app/

## Overview
An agentic RAG system that answers educational research questions using:
- Local PDF knowledge base (FAISS vector store)
- Google Web Search fallback (CSE)
- Academic Paper Search from ArXiv (ArXiv API)
- Planner + short-term memory (LangGraph checkpointer)
- Maker-Checker validation loop + LLM safety guardrails

## System architecture
![System Architecture](docs/Education_Researcher_agent.png)

### Safety mechanisms
- LLM-based input guardrail blocks unsafe requests (e.g., academic dishonesty, prompt injection).
- Maker-Checker loop validates groundedness and forces refinement.
- No fabricated citations: sources must come from retrieved context (PDF chunks or web results).

### Example queries & outputs
[See example queries and outputs](docs/Example_Query.md)