"""Document loading and chunking for educational research papers."""

from pathlib import Path
from typing import List
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load and chunk academic papers intelligently."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize document loader.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Separators that respect academic paper structure
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence breaks
                " ",       # Word breaks
                "",
            ],
            length_function=len,
        )
    
    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load a single PDF file and extract text with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects with text and metadata
        """
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # Add source metadata
            for page in pages:
                page.metadata["source"] = pdf_path.name
                page.metadata["source_path"] = str(pdf_path)
            
            logger.info(f"Loaded {len(pages)} pages from {pdf_path.name}")
            return pages
            
        except Exception as e:
            logger.error(f"Error loading {pdf_path.name}: {e}")
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks while preserving context.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def load_directory(self, papers_dir: Path) -> List[Document]:
        """
        Load all PDFs from a directory and chunk them.
        
        Args:
            papers_dir: Directory containing PDF files
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        papers_dir = Path(papers_dir)
        pdf_files = list(papers_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {papers_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files in {papers_dir}")
        
        all_documents = []
        for pdf_path in pdf_files:
            pages = self.load_pdf(pdf_path)
            all_documents.extend(pages)
        
        # Chunk all documents
        chunked_docs = self.chunk_documents(all_documents)
        
        logger.info(
            f"Loaded {len(all_documents)} pages from {len(pdf_files)} papers, "
            f"created {len(chunked_docs)} chunks"
        )
        
        return chunked_docs
    
    def extract_metadata_from_filename(self, filename: str) -> dict:
        """
        Extract metadata from standardized filename format.
        
        Expected format: author_year_title.pdf or descriptive_title.pdf
        
        Args:
            filename: PDF filename
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {"title": filename.replace("_", " ").replace(".pdf", "")}
        
        # Try to parse author_year_title format
        parts = filename.replace(".pdf", "").split("_")
        if len(parts) >= 2:
            if parts[1].isdigit() and len(parts[1]) == 4:  # Year format
                metadata["author"] = parts[0]
                metadata["year"] = parts[1]
                metadata["title"] = " ".join(parts[2:]) if len(parts) > 2 else parts[0]
        
        return metadata
