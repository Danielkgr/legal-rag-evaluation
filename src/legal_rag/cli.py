"""
Interactive menu for processing, searching, chatting, and evaluating.

Run it with `legal-rag` after installing, or `./fairwork` from a clone.  The
default folders, data/raw and data/processed, are relative to the folder you
run it from, so run it from the project root.  Each workflow runs one of the
package's command-line modules in a subprocess, built by the *_command
functions below so that tests can check every command against that module's
own argument parser.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

COLOURS = {
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "white": "97",
}
RESET = "\033[0m"

DEFAULT_PDFS_DIR = Path("data") / "raw"
DEFAULT_OUTPUT_DIR = Path("data") / "processed"
DEFAULT_EVAL_DIR = Path("evaluation_set")

EMBEDDING_MODELS = [
    ("  text-embedding-3-large (default, best quality)", "text-embedding-3-large"),
    ("  text-embedding-3-small (faster)", "text-embedding-3-small"),
    ("  text-embedding-ada-002 (legacy)", "text-embedding-ada-002"),
    ("  Other (type a name)", "custom"),
]


def _c(text: str, colour: str = "white") -> str:
    """Wrap text in ANSI colour codes."""
    code = COLOURS.get(colour)
    return f"\033[{code}m{text}{RESET}" if code else text


# Commands -------------------------------------------------------------------


def process_command(pdf_paths: Sequence[str], out_dir: str, model: str) -> List[str]:
    return [
        sys.executable, "-m", "legal_rag.pipeline", "--mode", "process",
        "--pdfs", *pdf_paths, "--output-dir", out_dir, "--embedding-model", model,
    ]  # fmt: skip


def query_command(query: str, k: int, out_dir: str) -> List[str]:
    return [
        sys.executable, "-m", "legal_rag.pipeline", "--mode", "query",
        "--query", query, "--k", str(k), "--output-dir", out_dir, "--load-index",
    ]  # fmt: skip


def chat_command(rag: bool, out_dir: str) -> List[str]:
    command = [
        sys.executable, "-m", "legal_rag.chat", "--load-index", "--output-dir", out_dir,
    ]  # fmt: skip
    return command if rag else command + ["--no-rag"]


def evaluate_command(
    pdf_paths: Sequence[str], num_queries: int, out_dir: str, eval_dir: str
) -> List[str]:
    return [
        sys.executable, "-m", "legal_rag.evaluate", "--pdfs", *pdf_paths,
        "--num-queries", str(num_queries), "--output-dir", out_dir,
        "--eval-dir", eval_dir,
    ]  # fmt: skip


def _run(command: List[str]) -> None:
    subprocess.run(command, env={**os.environ})


# Prompts --------------------------------------------------------------------


def _print_box(title: str, lines: List[str], width: int = 60) -> None:
    """Render a title and lines inside a ruled box."""
    print(f"\n{f' {title} '.center(width, '-')}")
    for line in lines:
        print(f" {line:<{width - 2}} ")
    print("-" * width)


def _menu(prompt_lines: List[str], choices: List[tuple]) -> Optional[str]:
    """
    Show a numbered choice menu and return the value of the selected item.

    choices is a list of (label, value) pairs.  Typing the first letter of a
    value, its number, or "0" for the last item selects it.
    """
    while True:
        print()
        for line in prompt_lines:
            print(f"  {line}")
        print()
        for label, _ in choices:
            print(f"    {label}")
        print()
        raw = input("  > ").strip().lower()

        for _, value in choices:
            if value and value[0].lower() == raw:
                return value
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(choices):
                return choices[index][1]
            if raw == "0" and choices:
                return choices[-1][1]
        for _, value in choices:
            if value and value.lower() == raw:
                return value
        print(f"  {_c('Unrecognised input.', 'yellow')}")


def _prompt_value(question: str, default=None) -> Optional[str]:
    """Ask a free-text question with an optional default."""
    hint = f" [{default}]" if default is not None else ""
    raw = input(f"\n{question}{hint}: ").strip()
    return raw or (str(default) if default is not None else None)


def _choose_pdf_dir(default: Path) -> Path:
    """Use the default PDF folder or a typed path."""
    choice = _menu(
        ["Where are your PDF documents?"],
        [(f"  {default} (default)", "default"), ("  Type a path", "type")],
    )
    if choice == "type":
        typed = Path(input("  Folder: ").strip())
        if typed.is_dir():
            return typed
        print(f"  {_c('Not a folder, using the default.', 'yellow')}")
    return default


def parse_selection(raw: str, count: int) -> List[int]:
    """Turn "1,3" or "a" into zero-based indexes, ignoring anything invalid."""
    raw = raw.strip().lower()
    if raw in ("a", "all"):
        return list(range(count))
    indexes = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= count and int(part) - 1 not in indexes:
            indexes.append(int(part) - 1)
    return indexes


def _choose_pdfs(directory: Path) -> List[str]:
    """List the PDFs under a folder and return the chosen paths."""
    pdfs = (
        sorted(directory.rglob("*.pdf"), key=lambda p: p.name)
        if directory.is_dir()
        else []
    )
    if not pdfs:
        print(f"\n  {_c(f'No PDF files found in {directory}.', 'yellow')}")
        return []
    print(f"\n  Found {len(pdfs)} PDF file(s):")
    for i, pdf in enumerate(pdfs, 1):
        print(f"    {i}. {pdf.name} ({pdf.stat().st_size / 1024:.0f} KB)")
    while True:
        raw = input("\n  Choose PDFs (numbers separated by commas, or a for all): ")
        indexes = parse_selection(raw, len(pdfs))
        if indexes:
            return [str(pdfs[i]) for i in indexes]
        print(f"  {_c('Choose at least one listed number.', 'yellow')}")


# Workflows ------------------------------------------------------------------


def workflow_process_pdf() -> None:
    """Parse PDFs and build the retrieval index."""
    pdf_paths = _choose_pdfs(_choose_pdf_dir(DEFAULT_PDFS_DIR))
    if not pdf_paths:
        print("  No PDFs selected, cancelling.")
        return
    model = _menu(["Which embedding model?"], EMBEDDING_MODELS)
    if model == "custom":
        model = input("  Model name: ").strip() or "text-embedding-3-large"
    out_dir = _prompt_value("Output directory", default=DEFAULT_OUTPUT_DIR)
    print(f"\n  Processing {len(pdf_paths)} PDF(s)\n")
    _run(process_command(pdf_paths, out_dir, model))


def workflow_query() -> None:
    """Search the index, building it first if asked."""
    option = _menu(
        ["How should the retrieval index be found?"],
        [
            ("  Load the existing index (fairwork_index)", "load"),
            ("  Build a fresh index from PDFs", "build"),
        ],
    )
    out_dir = _prompt_value("Index directory", default=DEFAULT_OUTPUT_DIR)
    if option == "build":
        pdf_paths = _choose_pdfs(_choose_pdf_dir(DEFAULT_PDFS_DIR))
        if not pdf_paths:
            print("  No PDFs selected, cancelling.")
            return
        print(f"\n  Building the index from {len(pdf_paths)} PDF(s)\n")
        _run(process_command(pdf_paths, out_dir, "text-embedding-3-large"))

    query = input("\n  Search query: ").strip()
    if not query:
        print("  Empty query, cancelling.")
        return
    k = _prompt_value("Number of results (k)", default=10)
    _run(query_command(query, int(k) if k and k.isdigit() else 10, out_dir))


def workflow_chat() -> None:
    """Start the chat, with or without retrieval."""
    mode = _menu(
        ["Choose a chat mode:"],
        [
            ("  RAG chat: answers sourced from the indexed legislation", "rag"),
            ("  Direct: ask the model with no retrieved text", "direct"),
        ],
    )
    out_dir = _prompt_value("Index directory", default=DEFAULT_OUTPUT_DIR)
    _run(chat_command(rag=mode == "rag", out_dir=out_dir))


def workflow_evaluate() -> None:
    """Generate a test set and score retrieval against it."""
    count = _prompt_value("Number of test queries to generate", default=100)
    num_queries = int(count) if count and count.isdigit() else 100
    out_dir = _prompt_value(
        "Output directory for chunks and index", default=DEFAULT_OUTPUT_DIR
    )
    eval_dir = _prompt_value("Evaluation output directory", default=DEFAULT_EVAL_DIR)
    pdf_paths = _choose_pdfs(_choose_pdf_dir(DEFAULT_PDFS_DIR))
    if not pdf_paths:
        print("  No PDFs selected, cancelling.")
        return
    print(f"\n  Running the evaluation with {num_queries} queries\n")
    _run(evaluate_command(pdf_paths, num_queries, out_dir, eval_dir))


MENU_ITEMS = [
    ("1. Process PDFs     ", "process", "Parse PDFs and build the retrieval index"),
    ("2. Query            ", "query", "Search the index for relevant passages"),
    ("3. Chat             ", "chat", "Ask questions and get sourced answers"),
    ("4. Evaluate         ", "evaluate", "Generate a test set and score retrieval"),
]


def main() -> None:
    """Show the menu and dispatch to the workflows until the user quits."""
    _print_box(
        "LEGISLATION RAG",
        [
            _c("Retrieval and sourced answers over Australian Acts.", "cyan"),
            "",
            "Select an action below to get started.",
        ],
    )
    if not os.getenv("OPENAI_API_KEY"):
        _print_box(
            "OPENAI_API_KEY is not set",
            [
                _c("Documents and questions are embedded through the", "yellow"),
                _c("OpenAI client, so processing and search need it.", "yellow"),
                "",
                "For OpenAI:        export OPENAI_API_KEY='sk-...'",
                "For a local server: export OPENAI_API_KEY=local",
                "                    export OPENAI_BASE_URL=http://localhost:10001/v1",
            ],
        )

    workflows = {
        "process": workflow_process_pdf,
        "query": workflow_query,
        "chat": workflow_chat,
        "evaluate": workflow_evaluate,
    }
    while True:
        for _, key, description in MENU_ITEMS:
            print(f"  {_c(key.upper().ljust(14))} {description}")
        choices = [(label, key) for label, key, _ in MENU_ITEMS]
        choices.append(("  Quit", "q"))
        selected = _menu(["What would you like to do?"], choices)
        if selected == "q":
            print(f"\n  {_c('Goodbye.', 'green')}")
            break
        try:
            workflows[selected]()
        except KeyboardInterrupt:
            print(f"\n  {_c('Interrupted.', 'yellow')}")
        except Exception as error:  # report and return to the menu
            print(f"\n  {_c(f'Error: {error}', 'red')}")


if __name__ == "__main__":
    main()
