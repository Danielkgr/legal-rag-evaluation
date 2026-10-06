"""Send the demo questions through the full chat path and record the answers.

Writes the format of results/chat_demo.json: one entry per question with the
answer, the document and section of the first five sources, and the seconds
taken.  This script was written after that file was committed and has not been
run against it; see results/PROVENANCE.md.

Build the index first, then run from the project root:

    python -m legal_rag.pipeline --mode process \\
        --pdfs data/raw/spam_act_2003.pdf data/raw/dnr_act_2006.pdf
    python scripts/chat_demo.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.llm import get_llm  # noqa: E402
from legal_rag.pipeline import FairWorkRAGPipeline  # noqa: E402

QUESTIONS = [
    "Under the Spam Act 2003, when may a sender send an actionable message to a "
    "person? Cite the relevant section.",
    "Under the Do Not Call Register Act 2006, may a telemarketer call a number "
    "listed on the register? Cite the relevant section.",
]


def run(rag: FairWorkRAGPipeline, llm, questions):
    """Answer each question in a fresh conversation and time it."""
    rows = []
    for question in questions:
        started = time.perf_counter()
        result = rag.chat(question, llm=llm)
        rows.append(
            {
                "question": question,
                "answer": result["response"],
                "sources": [
                    {"document": s["document"], "section": s["section"]}
                    for s in result["sources"][:5]
                ],
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
    return rows


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--index-name", default="fairwork_index")
    parser.add_argument("--out", default="results/chat_demo.json")
    args = parser.parse_args()

    rag = FairWorkRAGPipeline(output_dir=args.output_dir)
    rag.load_index(args.index_name)
    rows = run(rag, get_llm(), QUESTIONS)
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
