"""
Interactive chat interface for the Fair Work RAG system.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import FairWorkRAGPipeline
from llm import GemmaLLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatInterface:
    """Interactive command-line chat interface."""
    
    def __init__(self, pipeline: FairWorkRAGPipeline, llm: GemmaLLM = None):
        """
        Initialize the chat interface.
        
        Args:
            pipeline: FairWorkRAGPipeline instance
            llm: LLM instance (optional)
        """
        self.pipeline = pipeline
        self.llm = llm or GemmaLLM()
        self.conversation_history: List[Dict] = []
        
        logger.info("Initialized ChatInterface")
    
    def start(self):
        """Start the interactive chat loop."""
        print("\n" + "=" * 60)
        print("FAIR WORK ACT & AWARDS RAG CHAT")
        print("=" * 60)
        print("Ask questions about Australian workplace law.")
        print("Commands: /help, /clear, /history, /rag <on|off>, /quit\n")
        
        rag_enabled = True
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input, rag_enabled)
                    continue
                
                # Chat with RAG
                result = self.pipeline.chat(user_input, llm=self.llm, use_rag=rag_enabled)
                
                # Display response
                print(f"\nAssistant: {result['response']}")
                
                # Display sources if available
                if result['sources']:
                    print("\nSources:")
                    for i, source in enumerate(result['sources'][:5], 1):
                        print(f"  [{i}] {source['document']}")
                        if source['section']:
                            print(f"      Section: {source['section']}")
                        print(f"      Score: {source['score']:.3f}")
                
                # Update history
                self.conversation_history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": result['response']}
                ])
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")
    
    def _handle_command(self, command: str, rag_enabled: bool):
        """Handle special commands."""
        cmd_parts = command.split()
        cmd = cmd_parts[0].lower()
        
        if cmd == '/help':
            print("\nCommands:")
            print("  /help    - Show this help message")
            print("  /clear   - Clear conversation history")
            print("  /history - Show conversation history")
            print("  /rag <on|off> - Toggle RAG mode")
            print("  /quit    - Exit the chat")
        
        elif cmd == '/clear':
            self.conversation_history = []
            print("Conversation history cleared.")
        
        elif cmd == '/history':
            if not self.conversation_history:
                print("No conversation history.")
            else:
                print("\nConversation History:")
                for msg in self.conversation_history[-10:]:
                    role = msg['role'].capitalize()
                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    print(f"  {role}: {content}")
        
        elif cmd == '/rag':
            if len(cmd_parts) < 2:
                print(f"RAG mode: {'enabled' if rag_enabled else 'disabled'}")
            elif cmd_parts[1].lower() == 'on':
                rag_enabled = True
                print("RAG mode enabled.")
            elif cmd_parts[1].lower() == 'off':
                rag_enabled = False
                print("RAG mode disabled.")
            else:
                print("Usage: /rag <on|off>")
        
        elif cmd == '/quit':
            print("Goodbye!")
            sys.exit(0)
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fair Work RAG Chat Interface')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='data/processed', help='Output directory')
    parser.add_argument('--index-name', default='fairwork_index', help='Index name')
    parser.add_argument('--load-index', action='store_true', help='Load existing index')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return
    
    # Initialize pipeline
    pipeline = FairWorkRAGPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    if args.load_index:
        pipeline.load_index(args.index_name)
    else:
        # Try to load existing index
        try:
            pipeline.load_index(args.index_name)
            print("Loaded existing index.")
        except Exception as e:
            print(f"Index not found or error loading: {e}")
            print("Please run with --mode process first to build the index.")
            return
    
    # Initialize LLM
    llm = GemmaLLM()
    
    # Start chat interface
    chat = ChatInterface(pipeline, llm)
    chat.start()


if __name__ == "__main__":
    main()
