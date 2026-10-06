"""
Section-level retrieval evaluation against hand-labelled questions.

A gold question names the Act and the sections that answer it.  For each
question the harness retrieves chunks once, at the largest k, and scores the
ranking at each k:

| Measure | Meaning |
|---|---|
| precision@k | Share of the top k chunks that belong to a gold section |
| recall@k | Share of the gold sections with at least one chunk in the top k |
| MRR | Mean of 1 / rank of the first chunk from a gold section, 0 when none is retrieved |
| document hit@k | Share of questions with a chunk from a gold Act in the top k, the routing probe's measure |

A chunk belongs to every section whose heading falls inside it, so a short
section folded into its neighbour still counts.  A schedule clause never
matches a section label.  A document matches an Act label when its name equals
the label or starts with the label and an underscore, so the volumes of a long
Act, such as fair_work_act_2009_vol1.pdf, all match "fair_work_act_2009".
Retrieval is scored without cross-reference or definition expansion.

    python -m legal_rag.evaluation.section_eval --gold gold/fair_work_act_2009.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

KS = (1, 3, 5, 10)


@dataclass
class GoldLabel:
    act: str
    section: Optional[str] = None


@dataclass
class GoldQuestion:
    id: str
    question: str
    gold: List[GoldLabel]
    sources: List[Dict] = field(default_factory=list)

    @property
    def gold_sections(self) -> Set[Tuple[str, str]]:
        return {(g.act, g.section.upper()) for g in self.gold if g.section}

    @property
    def gold_acts(self) -> Set[str]:
        return {g.act for g in self.gold}


def load_gold_set(path: str) -> Tuple[Dict, List[GoldQuestion]]:
    """Read a gold set file.  Returns its header fields and its questions."""
    data = json.loads(Path(path).read_text())
    questions = [
        GoldQuestion(
            id=item["id"],
            question=item["question"],
            gold=[GoldLabel(**label) for label in item["gold"]],
            sources=item.get("sources", []),
        )
        for item in data["questions"]
    ]
    header = {key: value for key, value in data.items() if key != "questions"}
    return header, questions


def document_matches(document_name: str, act: str) -> bool:
    return document_name == act or document_name.startswith(act + "_")


def chunk_sections(result) -> Set[str]:
    """The body sections a retrieved chunk holds, upper-cased."""
    metadata = getattr(result, "metadata", None) or {}
    if metadata.get("schedule"):
        return set()
    sections = metadata.get("sections") or [result.section_number]
    return {str(s).upper() for s in sections if s}


def score_question(results: Sequence, question: GoldQuestion, ks=KS) -> Dict:
    """Score one ranked list of retrieved chunks against one gold question."""
    gold_sections = question.gold_sections

    def hits(result) -> Set[Tuple[str, str]]:
        held = chunk_sections(result)
        return {
            (act, section)
            for act, section in gold_sections
            if section in held and document_matches(result.document_name, act)
        }

    row: Dict = {
        "id": question.id,
        "question": question.question,
        "gold": sorted(f"{act} s {section}" for act, section in gold_sections),
        "retrieved": [
            f"{r.document_name} s {r.section_number}" for r in results[: max(ks)]
        ],
    }
    for k in ks:
        top = list(results[:k])
        row[f"doc_hit@{k}"] = any(
            document_matches(r.document_name, act)
            for r in top
            for act in question.gold_acts
        )
        if gold_sections:
            row[f"precision@{k}"] = sum(bool(hits(r)) for r in top) / k
            covered = set().union(*(hits(r) for r in top)) if top else set()
            row[f"recall@{k}"] = len(covered) / len(gold_sections)
    if gold_sections:
        first = next(
            (rank for rank, r in enumerate(results[: max(ks)], 1) if hits(r)), None
        )
        row["reciprocal_rank"] = 1.0 / first if first else 0.0
    return row


def summarise(rows: Sequence[Dict], ks=KS) -> Dict:
    """Average the per-question scores."""

    def mean(values):
        values = list(values)
        return round(sum(values) / len(values), 4) if values else None

    labelled = [row for row in rows if "reciprocal_rank" in row]
    summary: Dict = {"questions": len(rows), "questions_with_sections": len(labelled)}
    for k in ks:
        summary[f"precision@{k}"] = mean(row[f"precision@{k}"] for row in labelled)
        summary[f"recall@{k}"] = mean(row[f"recall@{k}"] for row in labelled)
    summary["mrr"] = mean(row["reciprocal_rank"] for row in labelled)
    for k in ks:
        summary[f"doc_hit@{k}"] = mean(float(row[f"doc_hit@{k}"]) for row in rows)
    return summary


def evaluate(
    search: Callable[[str, int], Sequence], questions: Sequence[GoldQuestion], ks=KS
) -> Dict:
    """Run every question through search(question, k) and score it."""
    rows = [score_question(search(q.question, max(ks)), q, ks) for q in questions]
    return {"summary": summarise(rows, ks), "results": rows}


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Section-level retrieval evaluation")
    parser.add_argument("--gold", default="gold/fair_work_act_2009.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--index-name", default="fairwork_index")
    parser.add_argument("--out", default="results/section_eval_fair_work_act.json")
    args = parser.parse_args()

    from legal_rag.pipeline import FairWorkRAGPipeline

    header, questions = load_gold_set(args.gold)
    rag = FairWorkRAGPipeline(output_dir=args.output_dir)
    rag.load_index(args.index_name)
    if not rag.embedding_manager.embeddings:
        raise SystemExit(f"No index named {args.index_name} in {args.output_dir}.")

    report = evaluate(lambda q, k: rag.retriever.retrieve(q, k=k), questions)
    report["gold_set"] = {"path": args.gold, **header}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for name, value in report["summary"].items():
        print(f"{name:>24}  {value}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
