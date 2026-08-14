#!/usr/bin/env python3
"""
Interactive CLI for the Fair Work Act & Modern Awards RAG System.

Guides users through document processing, querying, chat, and evaluation
with a styled menu-driven interface — no need to memorise flags.
"""

from __future__ import annotations

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# ── colour helpers ──────────────────────────────────────────────────────────

colours = {}
for code, fg in [
    ("red", "31"), ("green", "32"), ("yellow", "33"),
    ("blue", "34"), ("magenta", "35"), ("cyan", "36"), ("white", "97"),
]:
    colours[code] = f"\033[{fg}m"
RESET = "\033[0m"

def _c(text: str, colour: str = "white") -> str:
    """Wrap *text* in ANSI colour codes."""
    code = colours.get(colour, "")
    return f"{code}{text}{RESET}" if code else text


# ── constants ───────────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent  # one level up (legal-rag-evaluation root)

DEFAULT_PDFS_DIR = SRC_DIR / "data" / "raw"
DEFAULT_OUTPUT_DIR = SRC_DIR / "data" / "processed"
DEFAULT_EVAL_DIR = PROJECT_DIR / "evaluation_set"


# ── data helpers ────────────────────────────────────────────────────────────

@dataclass
class PDFFile:
    path: Path
    name: str  # display name (basename)

    @classmethod
    def from_path(cls, p: Path) -> "PDFFile":
        return cls(path=p, name=p.name)


def list_pdfs(directory: Path) -> List[PDFFile]:
    """Return PDF files in *directory*, sorted alphabetically."""
    if not directory.is_dir():
        return []
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".pdf")
    return [PDFFile.from_path(f) for f in files]


def find_pdfs(directory: Path) -> List[PDFFile]:
    """Recursively collect PDFs under *directory*."""
    if not directory.is_dir():
        return []
    return sorted(
        [PDFFile.from_path(f) for f in directory.rglob("*.pdf")],
        key=lambda x: x.name,
    )


# ── TUI helpers ─────────────────────────────────────────────────────────────

def _print_box(title: str, lines: List[str], width: int = 60) -> None:
    """Render *title* + *lines* inside a decorative box."""
    pad = " "
    header = f" {title} ".center(width, "─")
    print(f"\n{header}")
    for line in lines:
        print(f"{pad}{line:<{width - 2}}{pad}")
    print("─" * width + "\n", end="")


def _menu(prompt_lines: List[str], choices: List[tuple]) -> Optional[str]:
    """
    Show a numbered choice menu and return the key of the selected item.

    *choices* is a list of (label, value) tuples.
    Returns None if nothing was selected.
    """
    print()
    for line in prompt_lines:
        print(f"  {line}")
    print()
    for label, _ in choices:
        print(f"    {label}")
    print()
    raw = input("  › ").strip().lower()

    # If user typed a letter/number match against the first char of a label
    for _, val in choices:
        if val and val[0].lower() == raw:
            return val

    # Numeric fallback (1-based index, or "0" for the last item)
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx][1]
        if raw == "0" and choices:
            return choices[-1][1]
    except ValueError:
        pass

    # Exact match against values
    for _, val in choices:
        if val and val.lower() == raw:
            return val

    print(f"  {_c('Unrecognised input.', 'yellow')}")
    return _menu(prompt_lines, choices)


def _confirm(question: str) -> bool:
    """Ask a yes/no question. Returns True for y/yes."""
    resp = input(f"\n{question} [y/N]: ").strip().lower()
    return resp in ("y", "yes")


# ── sub‑menus ───────────────────────────────────────────────────────────────

def _choose_pdf_dir(default: Path) -> Path:
    """Let the user pick a directory containing PDFs."""
    options = [
        (f"  {default} (default)", str(default)),
        ("  Browse for folder …", "browse"),
        ("  Type custom path …", "custom"),
    ]
    selected = _menu(["Where should we find your PDF documents?"], options)
    if selected == "browse":
        # Use readline-based simple picker
        sub_options = [
            (f"  {p}", str(p)) for p in sorted(Path("/").iterdir())[:10]
        ]
        if not sub_options:
            sub_options = [(f"  {default}", str(default))]
        selected_dir = _menu(["Pick a parent directory:"], sub_options)
        if selected_dir and Path(selected_dir).is_dir():
            pdfs = list_pdfs(Path(selected_dir))
            if pdfs:
                return Path(selected_dir)
    elif selected == "custom":
        p = input("  Enter path: ").strip()
        if Path(p).is_dir():
            return Path(p)

    return default


def _choose_pdfs(directory: Path, multiple: bool = True) -> List[str]:
    """Show a numbered list of PDFs in *directory* and return paths."""
    pdfs = find_pdfs(directory)
    if not pdfs:
        print(f"\n  {_c('No PDF files found.', 'yellow')}")
        return []

    print(f"\n  Found {len(pdfs)} PDF file(s):")
    if multiple:
        print("  ────────────────────────────────────────")
        for i, pdf in enumerate(pdfs, 1):
            size = f" ({pdf.path.stat().st_size / 1024:.0f}KB)" if pdf.path.is_file() else ""
            print(f"    {i}. {pdf.name}{size}")
        print("  ────────────────────────────────────────")

    while True:
        if multiple and len(pdfs) > 9:
            # Provide a quick "select all" option
            sel = _menu(
                ["Choose PDFs (comma-separated numbers, e.g. 1,3 or 'a' for all):"],
                [("  All files", "all")] + [(f"  {i}. {p.name}", str(i)) for i, p in enumerate(pdfs, 1)],
            )
        else:
            sel = _menu(
                ["Choose a PDF:"],
                [(f"  {i}. {p.name}", str(i)) for i, p in enumerate(pdfs, 1)],
            )

        if sel == "all":
            return [str(p.path) for p in pdfs]

        try:
            indices = [int(x.strip()) - 1 for x in sel.split(",")]
            selected = [pdfs[i] for i in indices if 0 <= i < len(pdfs)]
            if selected:
                return [str(p.path) for p in selected]
        except (ValueError, IndexError):
            pass

    return []


def _prompt_value(question: str, default=None, validator=None) -> Optional[str]:
    """Ask a free-text question with an optional default."""
    hint = f" [{default}]" if default is not None else ""
    raw = input(f"\n{question}{hint}: ").strip()
    value = raw or (str(default) if default is not None else None)
    if validator and value:
        if not validator(value):
            print(f"  {_c('Invalid value.', 'yellow')}")
            return _prompt_value(question, default, validator)
    return value


# ── workflow: Process PDFs ─────────────────────────────────────────────────

def workflow_process_pdf() -> None:
    """Interactive flow to process one or more PDFs through the pipeline."""
    # 1. Choose directory
    pdf_dir = _choose_pdf_dir(DEFAULT_PDFS_DIR)
    pdf_paths = _choose_pdfs(pdf_dir, multiple=True)
    if not pdf_paths:
        print("  No PDFs selected — cancelling.")
        return

    # 2. Embedding model
    models = [
        ("  text-embedding-3-large (default, best quality)", "text-embedding-3-large"),
        ("  text-embedding-3-small (faster)", "text-embedding-3-small"),
        ("  text-embedding-ada-002 (legacy)", "text-embedding-ada-002"),
        ("  Other (type custom)", "custom"),
    ]
    model = _menu(["Which OpenAI embedding model?", models], models)

    if model == "custom":
        model = input("  Enter model name: ").strip() or "text-embedding-3-large"

    # 3. Output directory
    out_dir = _prompt_value(
        "Output directory",
        default=str(DEFAULT_OUTPUT_DIR),
        validator=lambda p: Path(p).is_dir(),
    )
    if not out_dir:
        return

    print(f"\n  Processing {len(pdf_paths)} PDF(s) …\n")
    cmd = [
        sys.executable, str(SRC_DIR / "pipeline.py"),
        "--mode", "process",
        "--pdfs"
    ] + pdf_paths + [
        "--output-dir", out_dir,
        "--embedding-model", model,
    ]
    subprocess.run(cmd, env={**os.environ})


# ── workflow: Query / Search ───────────────────────────────────────────────

def workflow_query() -> None:
    """Interactive flow to run a search query against the index."""
    # Load existing or build new index?
    load_option = _menu(
        ["How should we access the retrieval index?"],
        [
            ("  Load existing index (fairwork_index)", "load"),
            ("  Build a fresh index from PDFs", "build"),
        ],
    )

    if load_option == "build":
        pdf_dir = _choose_pdf_dir(DEFAULT_PDFS_DIR)
        pdf_paths = _choose_pdfs(pdf_dir, multiple=True)
        if not pdf_paths:
            print("  No PDFs selected — cancelling.")
            return
        out_dir = _prompt_value(
            "Output directory", default=str(DEFAULT_OUTPUT_DIR),
            validator=lambda p: Path(p).is_dir(),
        )
        if not out_dir:
            return
        # Build index first
        print(f"\n  Building index from {len(pdf_paths)} PDF(s) …\n")
        cmd = [
            sys.executable, str(SRC_DIR / "pipeline.py"),
            "--mode", "process", "--pdfs"
        ] + pdf_paths + ["--output-dir", out_dir]
        subprocess.run(cmd, env={**os.environ})

    # Now query
    query = input("\n  Enter your search query: ").strip()
    if not query:
        print("  Empty query — cancelling.")
        return

    k_str = _prompt_value("Number of results (k)", default=10)
    k = int(k_str) if k_str else 10

    out_dir = DEFAULT_OUTPUT_DIR
    cmd = [
        sys.executable, str(SRC_DIR / "pipeline.py"),
        "--mode", "query",
        "--query", query,
        "--k", str(k),
        "--output-dir", str(out_dir),
        "--load-index",
    ]
    subprocess.run(cmd, env={**os.environ})


# ── workflow: Chat ─────────────────────────────────────────────────────────

def workflow_chat() -> None:
    """Launch the interactive chat interface."""
    # Check for RAG-enabled or direct chat?
    chat_option = _menu(
        ["Choose chat mode:"],
        [
            ("  RAG chat – ask questions & get sourced answers from the legal corpus", "rag"),
            ("  Direct LLM – ask the model directly (no retrieval context)", "direct"),
        ],
    )

    cmd = [sys.executable, str(SRC_DIR / "chat.py")]

    # Ensure index is loaded
    if chat_option == "rag":
        cmd.append("--load-index")

    subprocess.run(cmd, env={**os.environ})


# ── workflow: Evaluate ─────────────────────────────────────────────────────

def workflow_evaluate() -> None:
    """Interactive flow to run a full evaluation."""
    # Number of queries
    n_str = _prompt_value("Number of test queries to generate", default=100)
    num_queries = int(n_str) if n_str else 100

    # Output dirs
    out_dir = _prompt_value(
        "Output directory (processed chunks & index)",
        default=str(DEFAULT_OUTPUT_DIR),
        validator=lambda p: Path(p).is_dir(),
    )
    if not out_dir:
        return

    eval_dir = _prompt_value(
        "Evaluation output directory",
        default=str(DEFAULT_EVAL_DIR),
        validator=lambda p: Path(p).is_dir(),
    )
    if not eval_dir:
        return

    # Run evaluation
    pdf_dir = _choose_pdf_dir(DEFAULT_PDFS_DIR)
    pdf_paths = _choose_pdfs(pdf_dir, multiple=True)
    if not pdf_paths:
        print("  No PDFs selected — cancelling.")
        return

    print(f"\n  Running evaluation with {num_queries} queries …\n")
    cmd = [
        sys.executable, str(SRC_DIR / "evaluate.py"),
        "--pdfs"
    ] + pdf_paths + [
        "--num-queries", str(num_queries),
        "--output-dir", out_dir,
        "--eval-dir", eval_dir,
    ]
    subprocess.run(cmd, env={**os.environ})


# ── main menu ───────────────────────────────────────────────────────────────

MENU_ITEMS = [
    ("1. Process PDFs     ", "process",   "Ingest PDF docs → build retrieval index"),
    ("2. Query            ", "query",     "Search the index for relevant passages"),
    ("3. Chat             ", "chat",      "Interactive Q&A with sourced answers"),
    ("4. Evaluate         ", "evaluate",  "Run full evaluation (generate + score)"),
]


def main() -> None:
    """Entry point – shows a styled menu and dispatches to workflows."""

    # ── header / branding ───────────────────────────────────────────────

    _print_box(
        "FAIR WORK ACT & AWARDS — RAG SYSTEM",
        [
            f"  {_c('Legal document retrieval, made easy.', 'cyan')}",
            "",
            "  Select an action below to get started.",
        ],
    )

    # ── quick‑checks ────────────────────────────────────────────────────

    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_api_key:
        _print_box(
            "⚠ API Key Not Detected",
            [
                f"  {_c('OPENAI_API_KEY is not set in your environment.', 'yellow')}",
                "",
                "  Many features require an OpenAI API key for embeddings.",
                "",
                f"  You can still browse menus here — you'll be prompted when",
                f"  a key is needed. To set it permanently:",
                "",
                f"    {_c('export OPENAI_API_KEY=\'sk-…\'', 'white')}",
            ],
        )

    # ── menu loop ───────────────────────────────────────────────────────

    workflows = {
        "process": workflow_process_pdf,
        "query": workflow_query,
        "chat": workflow_chat,
        "evaluate": workflow_evaluate,
    }

    while True:
        for _, key, desc in MENU_ITEMS:
            print(f"  {_c(key.upper().ljust(14))} — {desc}")

        choices = [(label, key) for label, key, _ in MENU_ITEMS]
        choices.append(("  Quit", "q"))
        selected = _menu(["What would you like to do?"], choices)

        if selected == "q":
            print(f"\n  {_c('Goodbye!', 'green')}")
            break

        handler = workflows.get(selected)
        if handler:
            try:
                handler()
            except KeyboardInterrupt:
                print(f"\n  {_c('Interrupted.', 'yellow')}")
                continue
            except Exception as exc:
                print(f"\n  {_c(f'Error: {exc}', 'red')}")


if __name__ == "__main__":
    main()
