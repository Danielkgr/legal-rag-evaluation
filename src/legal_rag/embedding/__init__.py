"""
Embedding generation module for legal document chunks.
Uses OpenAI text-embedding-3-large for high-quality embeddings.
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

import openai
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRecord:
    """Represents an embedded chunk."""
    chunk_id: str
    text: str
    embedding: List[float]
    section_number: Optional[str]
    document_name: str
    chunk_type: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingModel:
    """
    Wrapper around OpenAI text-embedding-3-large model.
    Handles batch processing and caching.
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: OpenAI embedding model to use
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            batch_size: Number of texts to embed in each batch
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be provided or set as environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI embedding model: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding texts"):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                # Try embedding individually
                for text in batch:
                    try:
                        emb = self.embed_text(text)
                        embeddings.append(emb)
                    except Exception as inner_e:
                        logger.error(f"Failed to embed: {text[:100]}... - {inner_e}")
                        embeddings.append([0.0] * 3072)  # Zero vector for failed embeddings
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        # text-embedding-3-large is 3072 dimensions
        return 3072


class EmbeddingManager:
    """
    Manages embedding storage, retrieval, and indexing.
    Supports persistent storage and efficient retrieval.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        storage_path: str = "data/processed/embeddings"
    ):
        """
        Initialize the embedding manager.
        
        Args:
            embedding_model: EmbeddingModel instance
            storage_path: Path to store embeddings
        """
        self.model = embedding_model
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings: List[EmbeddingRecord] = []
        self._embedding_cache: Dict[str, List[float]] = {}
        
        logger.info(f"Initialized EmbeddingManager at {storage_path}")
    
    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        section_number: Optional[str],
        document_name: str,
        chunk_type: str,
        metadata: Dict = None
    ) -> EmbeddingRecord:
        """
        Embed and store a single chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            text: Text content to embed
            section_number: Section number (if applicable)
            document_name: Name of source document
            chunk_type: Type of chunk (definition, section, etc.)
            metadata: Additional metadata
            
        Returns:
            EmbeddingRecord with embedding
        """
        # Check cache first
        if text in self._embedding_cache:
            embedding = self._embedding_cache[text]
        else:
            embedding = self.model.embed_text(text)
            self._embedding_cache[text] = embedding
        
        record = EmbeddingRecord(
            chunk_id=chunk_id,
            text=text,
            embedding=embedding,
            section_number=section_number,
            document_name=document_name,
            chunk_type=chunk_type,
            metadata=metadata or {}
        )
        
        self.embeddings.append(record)
        return record
    
    def add_chunks_batch(
        self,
        chunks: List[Dict]
    ) -> List[EmbeddingRecord]:
        """
        Embed and store multiple chunks in batch.
        
        Args:
            chunks: List of dicts with 'text', 'chunk_id', and metadata
            
        Returns:
            List of EmbeddingRecord objects
        """
        texts = [c['text'] for c in chunks]
        embeddings = self.model.embed_texts(texts)
        
        records = []
        for i, chunk in enumerate(chunks):
            record = EmbeddingRecord(
                chunk_id=chunk.get('chunk_id', f"chunk_{i}"),
                text=chunk['text'],
                embedding=embeddings[i],
                section_number=chunk.get('section_number'),
                document_name=chunk.get('document_name', 'unknown'),
                chunk_type=chunk.get('chunk_type', 'paragraph'),
                metadata=chunk.get('metadata', {})
            )
            records.append(record)
            self.embeddings.append(record)
        
        return records
    
    def save_index(self, name: str = "default"):
        """
        Save the embedding index to disk.
        
        Args:
            name: Index name for the save
        """
        index_path = self.storage_path / f"{name}_index.json"
        
        data = {
            'embeddings': [
                {
                    'chunk_id': r.chunk_id,
                    'text': r.text,
                    'embedding': r.embedding,
                    'section_number': r.section_number,
                    'document_name': r.document_name,
                    'chunk_type': r.chunk_type,
                    'metadata': r.metadata
                }
                for r in self.embeddings
            ]
        }
        
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved embedding index to {index_path} ({len(self.embeddings)} records)")
    
    def load_index(self, name: str = "default") -> int:
        """
        Load embedding index from disk.
        
        Args:
            name: Index name to load
            
        Returns:
            Number of records loaded
        """
        index_path = self.storage_path / f"{name}_index.json"
        
        if not index_path.exists():
            logger.warning(f"Index not found: {index_path}")
            return 0
        
        with open(index_path, 'r') as f:
            data = json.load(f)
        
        self.embeddings = [
            EmbeddingRecord(
                chunk_id=r['chunk_id'],
                text=r['text'],
                embedding=r['embedding'],
                section_number=r.get('section_number'),
                document_name=r.get('document_name', 'unknown'),
                chunk_type=r.get('chunk_type', 'paragraph'),
                metadata=r.get('metadata', {})
            )
            for r in data['embeddings']
        ]
        
        # Build cache
        for r in self.embeddings:
            self._embedding_cache[r.text] = r.embedding
        
        logger.info(f"Loaded embedding index: {len(self.embeddings)} records")
        return len(self.embeddings)
    
    def get_embeddings_array(self) -> np.ndarray:
        """Get all embeddings as a numpy array."""
        if not self.embeddings:
            return np.array([])
        
        return np.array([r.embedding for r in self.embeddings])
    
    def search_by_text(
        self,
        query: str,
        k: int = 5,
        filter_document: str = None,
        filter_chunk_type: str = None
    ) -> List[Dict]:
        """
        Search for similar chunks by text query.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_document: Filter by document name
            filter_chunk_type: Filter by chunk type
            
        Returns:
            List of matching chunks with scores
        """
        # Embed the query
        query_embedding = self.model.embed_text(query)
        
        # Compute similarity scores
        scores = []
        for r in self.embeddings:
            # Apply filters
            if filter_document and r.document_name != filter_document:
                continue
            if filter_chunk_type and r.chunk_type != filter_chunk_type:
                continue
            
            # Cosine similarity
            score = np.dot(query_embedding, r.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(r.embedding)
            )
            scores.append({
                'chunk': r,
                'score': float(score)
            })
        
        # Sort by score and return top k
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:k]


def main():
    """Example usage."""
    import sys
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize embedding model
    model = EmbeddingModel()
    
    # Create embedding manager
    manager = EmbeddingManager(model)
    
    # Sample chunks (in practice, these would come from the chunker)
    sample_chunks = [
        {
            'chunk_id': 'fw_act_s5_1',
            'text': '5 Definitions in this Act In this Act: ordinary hours means the number of hours in a standard working day for an employee',
            'section_number': '5',
            'document_name': 'fair_work_act_2009',
            'chunk_type': 'definition'
        },
        {
            'chunk_id': 'fw_act_s38_1',
            'text': '38 National employment standards The national employment standards are set out in this Part',
            'section_number': '38',
            'document_name': 'fair_work_act_2009',
            'chunk_type': 'section'
        },
        {
            'chunk_id': 'fw_act_s177_1',
            'text': '177 Modern awards A modern award may make provision concerning any matter relating to employment',
            'section_number': '177',
            'document_name': 'fair_work_act_2009',
            'chunk_type': 'section'
        }
    ]
    
    # Add chunks
    records = manager.add_chunks_batch(sample_chunks)
    print(f"Added {len(records)} chunks")
    
    # Save index
    manager.save_index("sample")
    
    # Test search
    query = "What are ordinary hours of work?"
    results = manager.search_by_text(query, k=2)
    
    print(f"\nSearch results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Section: {result['chunk'].section_number}")
        print(f"   Text: {result['chunk'].text[:100]}...")


if __name__ == "__main__":
    main()
