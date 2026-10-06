"""What the built-in test-set generator actually labels as relevant.

These tests pin the behaviour the README and results/PROVENANCE.md describe:
one relevance row for every query and chunk, templated queries that mark every
chunk of their source Act relevant, and Fair Work Act questions written into
the code that have no relevant chunk when no Fair Work Act is indexed.
"""

import random

from legal_rag.evaluation.eval_generator import EvaluationSetGenerator


def _chunk(doc, section, n):
    return {
        "chunk_id": f"{doc}_chunk_{n}",
        "document_name": doc,
        "section_number": section,
        "text": f"commercial message rules for section {section} number {n}",
    }


CHUNKS = [
    _chunk("spam_act_2003", "16", 0),
    _chunk("spam_act_2003", "17", 1),
    _chunk("spam_act_2003", "18", 2),
    _chunk("dnr_act_2006", "11", 3),
    _chunk("dnr_act_2006", "12", 4),
]


def _generate(tmp_path, num_queries=10):
    random.seed(0)
    generator = EvaluationSetGenerator(output_dir=str(tmp_path))
    queries = generator.generate_queries(CHUNKS, num_queries=num_queries)
    annotations = generator.generate_annotations(CHUNKS)
    relevant = {}
    for annotation in annotations:
        if annotation.relevance_score >= 1:
            relevant.setdefault(annotation.query_id, set()).add(annotation.chunk_id)
    return queries, annotations, relevant


def test_there_is_one_relevance_row_for_every_query_and_chunk(tmp_path):
    queries, annotations, _ = _generate(tmp_path)
    assert len(annotations) == len(queries) * len(CHUNKS)


def test_a_templated_query_marks_every_chunk_of_its_source_act_relevant(tmp_path):
    queries, _, relevant = _generate(tmp_path)
    facts = [q for q in queries if q.query_type == "fact"]
    assert facts
    for query in facts:
        (doc,) = query.expected_documents
        same_act = {c["chunk_id"] for c in CHUNKS if c["document_name"] == doc}
        assert same_act <= relevant[query.query_id]


def test_fair_work_act_questions_have_no_relevant_chunk_without_that_act(tmp_path):
    queries, _, relevant = _generate(tmp_path)
    fixed = [q for q in queries if q.query_type != "fact"]
    assert len(fixed) == 6  # 35% plus 25% of 10 queries
    for query in fixed:
        assert query.expected_documents == ["fair_work_act_2009"]
        assert query.query_id not in relevant
