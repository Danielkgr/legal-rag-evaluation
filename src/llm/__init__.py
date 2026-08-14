"""
LLM inference module using Gemma 4 12B for local chat interface.
"""

import os
import json
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # 'system', 'user', 'assistant'
    content: str


class GemmaLLM:
    """
    Local LLM inference using Gemma 4 12B (quantized).
    Optimized for legal document reasoning and cross-referencing.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-4-12b-it-qat-q4_0-gguf",
        device: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ):
        """
        Initialize the Gemma LLM.
        
        Args:
            model_name: Model name or path (GGUF format)
            device: Device to run on ('cpu', 'cuda', 'mps')
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.device = device or self._get_device()
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load model and tokenizer
        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # For GGUF models, we'd use llama.cpp or gguf library
        # For now, using HuggingFace format
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32
            )
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info("Trying alternative model path...")
            # Try local path or different format
            self.model = None
        
        logger.info(f"Initialized Gemma LLM on {self.device}")
    
    def _get_device(self) -> str:
        """Get available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if self.model is None:
            return "Error: Model not loaded. Please check model path and try again."
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        # Format prompt
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        
        # Extract assistant response
        # Remove the input prompt from output
        full_text = response
        if "assistant" in full_text:
            parts = full_text.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        return response
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None
    ) -> str:
        """
        Chat with the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            
        Returns:
            Model response
        """
        if self.model is None:
            return "Error: Model not loaded."
        
        max_tokens = max_tokens or self.max_tokens
        
        # Tokenize
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        
        return response
    
    def _get_system_prompt(self) -> str:
        """System prompt for legal document Q&A."""
        return """You are a legal assistant specialized in Australian workplace law.
Your task is to answer questions based on the provided legal documents (Fair Work Act 2009 and modern awards).

 Guidelines:
1. Answer ONLY using information from the provided documents
2. Cite specific section numbers when referencing provisions
3. If information is not in the documents, say "I cannot find this information in the provided documents"
4. For complex questions, break down your reasoning
5. Be precise and accurate - legal advice requires high precision
6. If multiple provisions apply, reference all relevant ones
7. Explain defined terms using the definitions provided in the documents"""


class LegalChatBot:
    """
    Chatbot for interacting with the RAG system.
    Combines retrieval with LLM generation for legal Q&A.
    """
    
    def __init__(
        self,
        retriever,
        llm: GemmaLLM,
        max_retrieved: int = 10,
        max_context_tokens: int = 3000
    ):
        """
        Initialize the legal chatbot.
        
        Args:
            retriever: HybridRetriever instance
            llm: GemmaLLM instance
            max_retrieved: Maximum chunks to retrieve
            max_context_tokens: Maximum tokens for context
        """
        self.retriever = retriever
        self.llm = llm
        self.max_retrieved = max_retrieved
        self.max_context_tokens = max_context_tokens
        
        self.conversation_history: List[Dict[str, str]] = []
        logger.info("Initialized LegalChatBot")
    
    def _build_context(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Build context from retrieved documents.
        
        Args:
            query: User query
            
        Returns:
            Context string and list of source metadata
        """
        # Retrieve relevant chunks
        results = self.retriever.retrieve(query, k=self.max_retrieved)
        
        # Build context
        context_parts = []
        sources = []
        
        for i, result in enumerate(results, 1):
            section_info = f"Section {result.section_number}" if result.section_number else "Unknown section"
            
            chunk_text = f"[{i}] {result.document_name} - {section_info}\nType: {result.chunk_type}\nText: {result.text}"
            context_parts.append(chunk_text)
            
            sources.append({
                'chunk_id': result.chunk_id,
                'section': result.section_number,
                'document': result.document_name,
                'score': result.combined_score
            })
        
        context = "\n\n".join(context_parts)
        return context, sources
    
    def answer(
        self,
        query: str,
        use_rag: bool = True
    ) -> Dict[str, any]:
        """
        Answer a query using RAG or direct LLM.
        
        Args:
            query: User query
            use_rag: Whether to use retrieval-augmented generation
            
        Returns:
            Dict with 'response', 'sources', and 'metadata'
        """
        if use_rag:
            # Retrieve context
            context, sources = self._build_context(query)
            
            # Build prompt
            prompt = f"""Answer the following question using ONLY the provided legal documents.

Question: {query}

Legal Documents:
{context}

Instructions:
1. Answer based ONLY on the provided documents
2. Cite specific section numbers
3. If information is not available, state that clearly
4. Be precise and accurate

Answer:"""
        else:
            # Direct LLM (no RAG)
            prompt = f"""Answer the following question about Australian workplace law:

{query}

Answer:"""
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Update conversation history
        self.conversation_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])
        
        return {
            'response': response,
            'sources': sources if use_rag else [],
            'metadata': {
                'query': query,
                'use_rag': use_rag,
                'conversation_length': len(self.conversation_history)
            }
        }
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


def main():
    """Example usage."""
    print("Gemma LLM Chatbot")
    print("=" * 50)
    print("Note: This requires the Gemma 4 12B model to be available")
    print("Usage:")
    print("  llm = GemmaLLM()")
    print("  chatbot = LegalChatBot(retriever, llm)")
    print("  result = chatbot.answer('What are ordinary hours?')")
    print("  print(result['response'])")


if __name__ == "__main__":
    main()
