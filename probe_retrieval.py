"""Honest, interpretable retrieval probe (no auto-metrics).

The built-in evaluator's auto-annotation marks every chunk relevant to every
query (247 chunks x 100 queries = 24,700 annotations), so its precision/recall
numbers are not a meaningful benchmark. This probe instead measures a claim we
can actually defend: given a question, does the hybrid retriever surface a
chunk from the CORRECT Act in the top-k? Queries and their expected Act were
fixed in advance; the accuracy is whatever the run produces.
"""

import os, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legal_rag.pipeline import FairWorkRAGPipeline

# (question, expected document stem). Written before looking at any results.
PROBE = [
    (
        "When may a sender of a commercial electronic message rely on the recipient's consent?",
        "spam_act_2003",
    ),
    (
        "What rules prohibit using a false or misleading sender identity in an email?",
        "spam_act_2003",
    ),
    ("What is an actionable message under this Act?", "spam_act_2003"),
    ("What is the Do Not Call Register and who keeps it?", "dnr_act_2006"),
    (
        "When may a telemarketer call a number listed on the Do Not Call Register?",
        "dnr_act_2006",
    ),
    (
        "How is a telephone number registered on, or removed from, the Do Not Call Register?",
        "dnr_act_2006",
    ),
    (
        "What civil penalty applies to a body corporate that breaches the sending rules?",
        "spam_act_2003",
    ),
    (
        "Which body administers the register of prohibited telemarketing numbers?",
        "dnr_act_2006",
    ),
]


def main():
    p = FairWorkRAGPipeline(output_dir="data/processed")
    chunks = p.process_pdfs(["data/raw/spam_act_2003.pdf", "data/raw/dnr_act_2006.pdf"])
    p.build_index(chunks)

    rows = []
    hit1 = hit3 = 0
    for q, expect in PROBE:
        res = p.search(q, k=5)
        top = res[0]["document"] if res else None
        docs = [r["document"] for r in res]
        top3 = docs[:3]
        ok1 = top == expect
        ok3 = expect in top3
        hit1 += ok1
        hit3 += ok3
        rows.append(
            {
                "query": q,
                "expected": expect,
                "top1": top,
                "top3_docs": top3,
                "hit@1": bool(ok1),
                "hit@3": bool(ok3),
                "top1_score": round(res[0]["score"], 4) if res else None,
            }
        )
        print(
            f"[{'HIT' if ok1 else 'MISS'}@1] expect={expect:<14} top1={top:<14} | {q[:60]}"
        )

    n = len(PROBE)
    summary = {
        "n_queries": n,
        "documents_in_index": sorted(
            {
                Path(x).name
                for x in ["data/raw/spam_act_2003.pdf", "data/raw/dnr_act_2006.pdf"]
            }
        ),
        "embedding_model_served": os.getenv("OPENAI_BASE_URL", "") + " (local server)",
        "hit@1_document_accuracy": round(hit1 / n, 3),
        "hit@3_document_accuracy": round(hit3 / n, 3),
        "note": "Document-level routing on 8 hand-picked cross-Act queries; not a general benchmark.",
    }
    out = {"summary": summary, "results": rows}
    Path("results").mkdir(exist_ok=True)
    Path("results/retrieval_probe.json").write_text(json.dumps(out, indent=2))
    print("\n", json.dumps(summary, indent=2))
    print("wrote results/retrieval_probe.json")


if __name__ == "__main__":
    main()
