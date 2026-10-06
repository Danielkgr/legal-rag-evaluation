"""
Interactive chat over the indexed legislation.
"""

import argparse
import logging
import os
import sys

from legal_rag.llm import get_llm
from legal_rag.pipeline import FairWorkRAGPipeline

logger = logging.getLogger(__name__)

COMMANDS = """Commands:
  /help          Show this help
  /clear         Forget earlier turns
  /history       Show the last turns
  /rag on|off    Answer with or without retrieval
  /quit          Leave the chat"""


class ChatInterface:
    """Interactive command-line chat that keeps one conversation."""

    def __init__(self, pipeline: FairWorkRAGPipeline, llm=None, rag_enabled=True):
        self.pipeline = pipeline
        self.chatbot = pipeline.chatbot(llm if llm is not None else get_llm())
        self.rag_enabled = rag_enabled

    def start(self):
        """Run the chat loop until /quit, Ctrl-C, or end of input."""
        print("\n" + "=" * 60)
        print("LEGISLATION CHAT")
        print("=" * 60)
        print(f"Ask questions about {self.chatbot.corpus_description}.")
        print("Commands: /help, /clear, /history, /rag <on|off>, /quit\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye.")
                return
            if not user_input:
                continue
            if user_input.startswith("/"):
                if self.handle_command(user_input) == "quit":
                    return
                continue
            try:
                result = self.chatbot.answer(user_input, use_rag=self.rag_enabled)
            except Exception as error:  # keep the session alive on one bad turn
                logger.error("Error: %s", error)
                print(f"Error: {error}")
                continue
            self.show(result)

    def show(self, result):
        """Print an answer and the sources it was given."""
        print(f"\nAssistant: {result['response']}")
        if result["sources"]:
            print("\nSources:")
            for i, source in enumerate(result["sources"][:5], 1):
                added = (
                    f" (added: {source['expansion']})"
                    if source.get("expansion")
                    else ""
                )
                section = f", section {source['section']}" if source["section"] else ""
                print(f"  [{i}] {source['document']}{section}{added}")

    def handle_command(self, command: str):
        """Act on a slash command.  Returns "quit" when the chat should end."""
        parts = command.split()
        name = parts[0].lower()
        if name == "/help":
            print(COMMANDS)
        elif name == "/clear":
            self.chatbot.clear_history()
            print("Conversation history cleared.")
        elif name == "/history":
            history = self.chatbot.conversation_history[-10:]
            if not history:
                print("No conversation history.")
            for message in history:
                content = message["content"]
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"  {message['role'].capitalize()}: {content}")
        elif name == "/rag":
            if len(parts) < 2:
                print(f"RAG mode: {'on' if self.rag_enabled else 'off'}")
            elif parts[1].lower() in ("on", "off"):
                self.rag_enabled = parts[1].lower() == "on"
                print(f"RAG mode {'on' if self.rag_enabled else 'off'}.")
            else:
                print("Usage: /rag <on|off>")
        elif name == "/quit":
            print("Goodbye.")
            return "quit"
        else:
            print(f"Unknown command: {name}.  Type /help for the commands.")
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    """The command-line options, shared with the menu's tests."""
    parser = argparse.ArgumentParser(description="Chat over the indexed legislation")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--output-dir", default="data/processed", help="Directory holding the index"
    )
    parser.add_argument("--index-name", default="fairwork_index", help="Index name")
    parser.add_argument(
        "--load-index",
        action="store_true",
        help="Load the saved index (the chat always does; kept for compatibility)",
    )
    parser.add_argument(
        "--no-rag", action="store_true", help="Start with retrieval switched off"
    )
    parser.add_argument(
        "--backend",
        choices=["claude", "openai", "gemma"],
        help="Answer backend.  Defaults to CHAT_BACKEND, then to the local servers.",
    )
    return parser


def main():
    """Entry point for python -m legal_rag.chat."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = build_arg_parser().parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.  Questions are embedded through")
        print("the OpenAI client, so set it, or set any placeholder value when")
        print("OPENAI_BASE_URL points at a local embedding server.")
        sys.exit(1)

    pipeline = FairWorkRAGPipeline(data_dir=args.data_dir, output_dir=args.output_dir)
    pipeline.load_index(args.index_name)
    if not pipeline.embedding_manager.embeddings:
        print(f"No index named {args.index_name} in {args.output_dir}/embeddings.")
        print("Build one first with: python -m legal_rag.pipeline --mode process")
        sys.exit(1)

    ChatInterface(pipeline, get_llm(args.backend), rag_enabled=not args.no_rag).start()


if __name__ == "__main__":
    main()
