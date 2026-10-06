"""Section-level scoring, and the integrity of the provisional gold set."""

import re
import types
from pathlib import Path

import pytest

from legal_rag.evaluation.section_eval import (
    GoldLabel,
    GoldQuestion,
    evaluate,
    load_gold_set,
    score_question,
)

ROOT = Path(__file__).resolve().parents[1]
ACT = "fair_work_act_2009"


def _chunk(doc, section, sections=None, schedule=None):
    return types.SimpleNamespace(
        document_name=doc,
        section_number=section,
        metadata={"sections": sections or [section], "schedule": schedule},
    )


QUESTION = GoldQuestion(
    id="q",
    question="How long must an employee have worked for a small business?",
    gold=[GoldLabel(ACT, "383"), GoldLabel(ACT, "23")],
)


def test_precision_recall_and_reciprocal_rank_at_section_level():
    ranked = [
        _chunk(f"{ACT}_vol2", "382"),
        _chunk(f"{ACT}_vol2", "383"),
        _chunk(f"{ACT}_vol2", "383"),  # a second piece of the same section
        _chunk(f"{ACT}_vol1", "22", sections=["22", "23"]),  # s 23 folded into s 22
    ]
    row = score_question(ranked, QUESTION, ks=(1, 3, 4))
    assert row["precision@1"] == 0.0 and row["recall@1"] == 0.0
    assert row["precision@3"] == pytest.approx(2 / 3)
    assert row["recall@3"] == 0.5
    assert row["precision@4"] == 0.75 and row["recall@4"] == 1.0
    assert row["reciprocal_rank"] == 0.5
    assert row["doc_hit@1"] is True


def test_other_acts_and_schedule_clauses_do_not_count():
    ranked = [
        _chunk("spam_act_2003", "383"),
        _chunk(ACT, "383", schedule="1"),
        _chunk("fair_work_act_2009x", "383"),
    ]
    row = score_question(ranked, QUESTION, ks=(3,))
    assert row["precision@3"] == 0.0 and row["reciprocal_rank"] == 0.0
    assert row["doc_hit@3"] is True  # the schedule chunk is still from the Act


def test_evaluate_averages_over_questions():
    other = GoldQuestion(
        id="r", question="Unfair dismissal time limit?", gold=[GoldLabel(ACT, "394")]
    )
    report = evaluate(lambda q, k: [_chunk(ACT, "394")], [QUESTION, other], ks=(1,))
    assert report["summary"]["questions"] == 2
    assert report["summary"]["mrr"] == 0.5
    assert report["summary"]["recall@1"] == 0.5
    assert report["summary"]["doc_hit@1"] == 1.0


def test_the_gold_set_is_provisional_and_every_label_has_a_matching_source():
    header, questions = load_gold_set(ROOT / "gold" / "fair_work_act_2009.json")
    assert header["status"] == "provisional"
    assert len(questions) >= 30
    assert len({q.id for q in questions}) == len(questions)
    for question in questions:
        assert question.gold, question.id
        sources = {s["section"]: s for s in question.sources}
        for label in question.gold:
            assert label.act == ACT
            source = sources[label.section]
            # The source is the AustLII page for that very section.
            assert re.search(
                rf"/consol_act/fwa2009114/s{label.section.lower()}\.html$",
                source["url"],
            ), (question.id, label.section)
            assert source["heading"]
