"""
Main pipeline for processing Fair Work Act and modern awards documents.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing.pdf_parser import PDFParser
from data_preprocessing.chunking import LegalChunker
from data_preprocessing.metadata_extractor import MetadataExtractor
from embedding import EmbeddingModel, EmbeddingManager
from retrieval import HybridRetriever
from llm import GemmaLLM, LegalChatBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FairWorkRAGPipeline:
    """
    Main pipeline for the Fair Work Act RAG system.
    Orchestrates all components from document processing to chat.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "data/processed",
        embedding_model: str = "text-embedding-3-large",
        retrieval_alpha: float = 0.5,
        retrieval_beta: float = 0.5
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Directory containing raw PDFs
            output_dir: Directory for processed outputs
            embedding_model: OpenAI embedding model name
            retrieval_alpha: Weight for vector search
            retrieval_beta: Weight for BM25
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = PDFParser()
        self.chunker = LegalChunker()
        self.metadata_extractor = MetadataExtractor()
        
        # Embedding
        self.embedding_model = EmbeddingModel(model_name=embedding_model)
        self.embedding_manager = EmbeddingManager(self.embedding_model)
        
        # Retrieval
        self.retriever = None  # Will be set after embedding
        
        logger.info("Initialized FairWorkRAGPipeline")
    
    def process_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Process PDF documents through the pipeline.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of processed chunks
        """
        all_chunks = []
        
        for pdf_path in pdf_paths:
            logger.info(f"Processing: {pdf_path}")
            
            # Parse PDF
            doc = self.parser.parse_pdf(pdf_path)
            logger.info(f"  Parsed {len(doc.pages)} pages")
            
            # Chunk document
            chunks = self.chunker.chunk_document(doc)
            logger.info(f"  Created {len(chunks)} chunks")
            
            # Extract metadata
            for chunk in chunks:
                cross_refs = self.metadata_extractor.extract_cross_references(
                    chunk.text,
                    chunk.section_number
                )
                definitions = self.metadata_extractor.extract_definitions(
                    chunk.text,
                    chunk.section_number
                )
                
                chunk.metadata['cross_references'] = [r.reference_text for r in cross_refs]
                chunk.metadata['definitions'] = [d.term for d in definitions]
            
            # Convert to dicts for embedding
            chunk_dicts = []
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    'chunk_id': f"{Path(pdf_path).stem}_chunk_{i}",
                    'text': chunk.text,
                    'section_number': chunk.section_number,
                    'document_name': Path(pdf_path).stem,
                    'chunk_type': chunk.chunk_type.value,
                    'metadata': {
                        'page_numbers': chunk.page_numbers,
                        'heading': chunk.metadata.get('heading', ''),
                        'cross_references': chunk.metadata.get('cross_references', []),
                        'definitions': chunk.metadata.get('definitions', [])
                    }
                }
                chunk_dicts.append(chunk_dict)
            
            all_chunks.extend(chunk_dicts)
        
        return all_chunks
    
    def build_index(self, chunks: List[Dict]):
        """
        Build vector and BM25 index from chunks.
        
        Args:
            chunks: List of chunk dictionaries
        """
        logger.info("Building embedding index...")
        records = self.embedding_manager.add_chunks_batch(chunks)
        logger.info(f"Added {len(records)} chunks to embedding index")
        
        # Save index
        self.embedding_manager.save_index("fairwork_index")
        
        # Initialize retriever
        self.retriever = HybridRetriever(
            self.embedding_manager,
            alpha=0.5,
            beta=0.5
        )
        
        logger.info("Index built successfully")
    
    def load_index(self, index_name: str = "fairwork_index"):
        """Load pre-built index from disk."""
        logger.info(f"Loading index: {index_name}")
        count = self.embedding_manager.load_index(index_name)
        
        self.retriever = HybridRetriever(
            self.embedding_manager,
            alpha=0.5,
            beta=0.5
        )
        
        logger.info(f"Loaded {count} chunks from index")
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """
        Search the index for relevant documents.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of retrieved chunks with scores
        """
        if not self.retriever:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        results = self.retriever.retrieve(query, k=k)
        
        return [
            {
                'chunk_id': r.chunk_id,
                'text': r.text,
                'section': r.section_number,
                'document': r.document_name,
                'type': r.chunk_type,
                'score': r.combined_score,
                'vector_score': r.vector_score,
                'bm25_score': r.bm25_score
            }
            for r in results
        ]
    
    def chat(
        self,
        query: str,
        llm: GemmaLLM = None,
        use_rag: bool = True
    ) -> Dict:
        """
        Chat with the RAG system.
        
        Args:
            query: User query
            llm: LLM instance (optional, creates new if not provided)
            use_rag: Whether to use RAG
            
        Returns:
            Response dict with answer and sources
        """
        if not self.retriever:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        if llm is None:
            llm = GemmaLLM()
        
        chatbot = LegalChatBot(self.retriever, llm)
        result = chatbot.answer(query, use_rag=use_rag)
        
        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fair Work Act RAG Pipeline')
    parser.add_argument('--pdfs', nargs='+', help='PDF files to process')
    parser.add_argument('--mode', choices=['process', 'query', 'chat'], default='process',
                       help='Mode of operation')
    parser.add_argument('--query', type=str, help='Query for search/chat mode')
    parser.add_argument('--k', type=int, default=10, help='Number of results')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='data/processed', help='Output directory')
    parser.add_argument('--load-index', action='store_true', help='Load existing index')
    parser.add_argument('--index-name', default='fairwork_index', help='Index name')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FairWorkRAGPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    if args.mode == 'process':
        if not args.pdfs:
            print("Error: --pdfs required for processing mode")
            return
        
        # Process PDFs
        chunks = pipeline.process_pdfs(args.pdfs)
        
        # Save processed chunks
        chunks_file = Path(args.output_dir) / "chunks.json"
        with open(chunks_file, 'w') as f:
            json.dump(chunks, f, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")
        
        # Build index
        pipeline.build_index(chunks)
        
        print(f"Processed {len(chunks)} chunks and built index")
    
    elif args.mode == 'query':
        # Load index
        pipeline.load_index(args.index_name)
        
        # Search
        results = pipeline.search(args.query, k=args.k)
        
        print(f"\nSearch results for: '{args.query}'")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['document']} - Section {result['section']}")
            print(f"    Type: {result['type']}")
            print(f"    Score: {result['score']:.3f}")
            print(f"    Text: {result['text'][:200]}...")
    
    elif args.mode == 'chat':
        # Load index
        pipeline.load_index(args.index_name)
        
        # Initialize LLM
        llm = GemmaLLM()
        
        # Chat
        result = pipeline.chat(args.query, llm=llm)
        
        print(f"\nQuestion: {args.query}")
        print("=" * 60)
        print(f"\nAnswer:\n{result['response']}")
        
        if result['sources']:
            print(f"\nSources:")
            for source in result['sources'][:5]:
                print(f"  - {source['document']} - Section {source['section']}")


if __name__ == "__main__":
    main()
