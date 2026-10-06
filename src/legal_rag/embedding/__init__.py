"""
Embedding generation module for legal document chunks.
Uses OpenAI text-embedding-3-large for high-quality embeddings.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import openai
from tqdm import tqdm

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
    metadata: Optional[Dict] = None

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
        batch_size: int = 100,
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
            raise ValueError(
                "OPENAI_API_KEY must be provided or set as environment variable"
            )

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
            response = self.client.embeddings.create(model=self.model_name, input=text)
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
            batch = texts[i : i + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model_name, input=batch
                )
                embeddings.extend([data.embedding for data in response.data])
            except Exception as error:
                logger.error("Error embedding batch, retrying one at a time: %s", error)
                # A text that still fails raises, rather than going into the
                # index as a zero vector that can never be retrieved.
                embeddings.extend(self.embed_text(text) for text in batch)

        return embeddings


class EmbeddingManager:
    """
    Manages embedding storage, retrieval, and indexing.
    Supports persistent storage and efficient retrieval.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        storage_path: str = "data/processed/embeddings",
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
        metadata: Optional[Dict] = None,
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
            metadata=metadata or {},
        )

        self.embeddings.append(record)
        return record

    def add_chunks_batch(self, chunks: List[Dict]) -> List[EmbeddingRecord]:
        """
        Embed and store multiple chunks in batch.

        Args:
            chunks: List of dicts with 'text', 'chunk_id', and metadata

        Returns:
            List of EmbeddingRecord objects
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.model.embed_texts(texts)

        records = []
        for i, chunk in enumerate(chunks):
            record = EmbeddingRecord(
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                text=chunk["text"],
                embedding=embeddings[i],
                section_number=chunk.get("section_number"),
                document_name=chunk.get("document_name", "unknown"),
                chunk_type=chunk.get("chunk_type", "paragraph"),
                metadata=chunk.get("metadata", {}),
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
            "embeddings": [
                {
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "embedding": r.embedding,
                    "section_number": r.section_number,
                    "document_name": r.document_name,
                    "chunk_type": r.chunk_type,
                    "metadata": r.metadata,
                }
                for r in self.embeddings
            ]
        }

        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"Saved embedding index to {index_path} ({len(self.embeddings)} records)"
        )

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

        with open(index_path, "r") as f:
            data = json.load(f)

        self.embeddings = [
            EmbeddingRecord(
                chunk_id=r["chunk_id"],
                text=r["text"],
                embedding=r["embedding"],
                section_number=r.get("section_number"),
                document_name=r.get("document_name", "unknown"),
                chunk_type=r.get("chunk_type", "paragraph"),
                metadata=r.get("metadata", {}),
            )
            for r in data["embeddings"]
        ]

        # Build cache
        for r in self.embeddings:
            self._embedding_cache[r.text] = r.embedding

        logger.info(f"Loaded embedding index: {len(self.embeddings)} records")
        return len(self.embeddings)

    def search_by_text(
        self,
        query: str,
        k: int = 5,
        filter_document: Optional[str] = None,
        filter_chunk_type: Optional[str] = None,
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
        candidates = [
            r
            for r in self.embeddings
            if (not filter_document or r.document_name == filter_document)
            and (not filter_chunk_type or r.chunk_type == filter_chunk_type)
        ]
        if not candidates or k <= 0:
            return []

        query_vector = np.asarray(self.model.embed_text(query), dtype=float)
        matrix = np.asarray([r.embedding for r in candidates], dtype=float)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
        # A zero vector has no direction, so it scores 0 rather than NaN.
        scores = np.divide(
            matrix @ query_vector, norms, out=np.zeros(len(candidates)), where=norms > 0
        )
        top = np.argsort(-scores, kind="stable")[:k]
        return [{"chunk": candidates[i], "score": float(scores[i])} for i in top]
