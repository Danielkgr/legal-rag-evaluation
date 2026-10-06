"""Retrieval pulls in the sections a result cites and the definitions it uses.

A bag-of-words embedder stands in for the embedding server, so the hybrid
retriever runs end to end with no network.
"""

import re
import zlib

import pytest

from legal_rag.data_preprocessing.metadata_extractor import MetadataExtractor
from legal_rag.embedding import EmbeddingManager
from legal_rag.retrieval import HybridRetriever


class BagOfWords:
    dims = 256

    def embed_text(self, text):
        vector = [0.0] * self.dims
        vector[0] = 1.0
        for word in re.findall(r"[a-z]+", text.lower()):
            vector[1 + zlib.crc32(word.encode()) % (self.dims - 1)] += 1.0
        return vector

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


def _chunk(doc, section, text, schedule=None):
    return {
        "chunk_id": f"{doc}_s{section}" + (f"_sch{schedule}" if schedule else ""),
        "text": text,
        "section_number": section,
        "document_name": doc,
        "chunk_type": "section",
        "metadata": {"sections": [section], "schedule": schedule},
    }


CHUNKS = [
    _chunk(
        "toy_act",
        "4",
        "4 Definitions\nIn this Act:\n"
        "commercial electronic message has the meaning given by section 6.\n"
        "Register means the Do Not Call Register kept under section 13.",
    ),
    _chunk(
        "toy_act",
        "6",
        "6 Meaning of commercial electronic message\n"
        "A commercial electronic message is an electronic message that offers goods.",
    ),
    _chunk(
        "toy_act",
        "11",
        "11 Calls to registered numbers\n"
        "(1) A person must not make a telemarketing call to a number on the "
        "Register.\n(2) Subsection (1) does not apply to a call covered by "
        "section 12, or to a call under section 7 of the Telecommunications Act 1997.",
    ),
    _chunk(
        "toy_act",
        "12",
        "12 Consent\nA call is covered by this section if the account-holder consented.",
    ),
    _chunk(
        "toy_act",
        "13",
        "13 Keeping the Register\nThe ACMA must keep the Register.",
    ),
    _chunk(
        "toy_act",
        "7",
        "7 Unrelated provision about fees\nFees are set by regulation.",
    ),
    _chunk(
        "toy_act",
        "1",
        "1 Consent of the account-holder\nConsent may be express or inferred.",
        schedule="2",
    ),
]


@pytest.fixture
def retriever(tmp_path):
    manager = EmbeddingManager(BagOfWords(), storage_path=str(tmp_path))
    manager.add_chunks_batch(CHUNKS)
    return HybridRetriever(manager)


QUERY = "May a person make a telemarketing call to a number?"


def test_retrieval_without_expansion_adds_nothing(retriever):
    results = retriever.retrieve(QUERY, k=1)
    assert [r.chunk_id for r in results] == ["toy_act_s11"]
    assert results[0].expansion is None


def test_cited_sections_and_used_definitions_are_added_after_the_results(retriever):
    results = retriever.retrieve(QUERY, k=1, expand=True)
    assert results[0].chunk_id == "toy_act_s11"
    added = {r.chunk_id: r for r in results[1:]}
    # s 11 cites s 12 of this Act; the reference to section 7 of another Act
    # is not followed.
    assert added["toy_act_s12"].expansion == "cross_reference"
    assert "toy_act_s7" not in added
    # s 11 uses "Register", which s 4 defines.
    assert added["toy_act_s4"].expansion == "definition"
    assert "Register" in added["toy_act_s4"].expansion_reason
    assert all(r.combined_score < results[0].combined_score for r in results[1:])


def test_a_definition_given_by_another_section_pulls_in_that_section(retriever):
    question = "Is a commercial electronic message covered by consent?"
    consent = [
        r for r in retriever.retrieve(question, k=7) if r.chunk_id == "toy_act_s12"
    ]
    added = {r.chunk_id: r for r in retriever.expand(consent, question)}
    # s 4 says the term "has the meaning given by section 6", so the
    # definition step adds s 6 rather than the dictionary entry.
    assert added["toy_act_s6"].expansion == "definition"
    assert "toy_act_s4" not in added


def test_expansion_is_bounded(retriever):
    results = retriever.retrieve(QUERY, k=1)
    assert retriever.expand(results, QUERY, 0, 0) == []
    one_each = retriever.expand(results, QUERY, 1, 1)
    kinds = [r.expansion for r in one_each]
    assert kinds.count("cross_reference") == 1 and kinds.count("definition") == 1


def test_a_schedule_reference_pulls_in_the_schedule(retriever):
    results = retriever.retrieve("Who keeps the Register?", k=1)
    record = results[0]
    record.text += "\nSee also Schedule 2."
    added = retriever.expand([record], "Who keeps the Register?")
    assert "toy_act_s1_sch2" in [r.chunk_id for r in added]


def test_extractor_reads_line_anchored_definitions_and_this_act_references():
    extractor = MetadataExtractor()
    definitions = extractor.extract_definitions(CHUNKS[0]["text"])
    assert [(d.term, d.target_section) for d in definitions] == [
        ("commercial electronic message", "6"),
        ("Register", None),
    ]
    assert extractor.extract_cross_references(CHUNKS[2]["text"], ["11"]) == ["12"]
