"""
Build knowledge base from educational research papers.

Run with: uv run python scripts/build_knowledge_base.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.knowledge.loader import DocumentLoader
from src.knowledge.vector_store import VectorStoreManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_knowledge_base():
    """Build vector store from PDF papers."""
    
    logger.info("=" * 60)
    logger.info("BUILDING EDUCATIONAL RESEARCH KNOWLEDGE BASE")
    logger.info("=" * 60)
    
    # Check if papers directory exists
    if not settings.papers_dir.exists():
        logger.error(f"Papers directory not found: {settings.papers_dir}")
        logger.error("Run 'python scripts/download_papers.py' first")
        return
    
    # Count PDFs
    pdf_files = list(settings.papers_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {settings.papers_dir}")
        logger.error("Run 'python scripts/download_papers.py' first")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    logger.info(f"Chunk size: {settings.chunk_size}")
    logger.info(f"Chunk overlap: {settings.chunk_overlap}")
    logger.info("")
    
    # Step 1: Load and chunk documents
    logger.info("Step 1/3: Loading and chunking documents...")
    loader = DocumentLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    documents = loader.load_directory(settings.papers_dir)
    
    if not documents:
        logger.error("No documents loaded. Check PDF files.")
        return
    
    logger.info(f"Loaded and chunked {len(documents)} document segments")
    logger.info("")
    
    # Step 2: Create vector store
    logger.info("Step 2/3: Creating vector store with Gemini embeddings...")
    logger.info("(This may take a few minutes...)")
    
    vector_manager = VectorStoreManager()
    
    try:
        vector_manager.load_or_create(documents)
        logger.info("Vector store created")
        logger.info("")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        logger.error("Check your GOOGLE_API_KEY in .env file")
        return
    
    # Step 3: Save vector store
    logger.info("Step 3/3: Saving vector store to disk...")
    vector_manager.save()
    logger.info("Vector store saved")
    logger.info("")
    
    # Display statistics
    logger.info("=" * 60)
    logger.info("KNOWLEDGE BASE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total documents: {len(documents)}")
    
    # Get unique sources
    sources = set(doc.metadata.get('source', 'Unknown') for doc in documents)
    logger.info(f"Unique papers: {len(sources)}")
    logger.info(f"Storage location: {settings.vector_store_dir}")
    logger.info("")
    logger.info("Papers included:")
    for source in sorted(sources)[:10]:  # Show first 10
        source_name = Path(source).name if source != 'Unknown' else source
        logger.info(f"  - {source_name}")
    if len(sources) > 10:
        logger.info(f"  ... and {len(sources) - 10} more")
    logger.info("")
    logger.info("=" * 60)
    logger.info("KNOWLEDGE BASE READY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    build_knowledge_base()
