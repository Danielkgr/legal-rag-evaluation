"""Section references in answers, and the check against what was retrieved."""

import types

import pytest

from legal_rag import llm
from legal_rag.chat import describe_result
from legal_rag.citations import parse_references, verify_citations


def _refs(text):
    return [(r.section, r.schedule, r.act) for r in parse_references(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Under s 11(1), a call", [("11", None, None)]),
        ("section 11 applies", [("11", None, None)]),
        ("Section 11 and s. 12A", [("11", None, None), ("12A", None, None)]),
        ("ss 16-17 apply", [("16", None, None), ("17", None, None)]),
        (
            "sections 16 to 18",
            [("16", None, None), ("17", None, None), ("18", None, None)],
        ),
        (
            "sections 16, 17 and 20",
            [("16", None, None), ("17", None, None), ("20", None, None)],
        ),
        ("subsection 11(1) or (2)", [("11", None, None)]),
        ("paragraph 7(a)", [("7", None, None)]),
        ("see Schedule 2", [(None, "2", None)]),
        (
            "s 7 of the Telecommunications Act 1997",
            [("7", None, "Telecommunications Act 1997")],
        ),
        ("has 30 days, and the penalty is 2 units", []),
        ("the employer's 15 staff", []),
    ],
)
def test_parser_reads_the_usual_ways_of_citing_a_section(text, expected):
    assert _refs(text) == expected


def _chunk(doc, title, section, schedule=None):
    return types.SimpleNamespace(
        document_name=doc,
        section_number=section,
        metadata={"document_title": title, "sections": [section], "schedule": schedule},
    )


DNR = "Do Not Call Register Act 2006"
RETRIEVED = [_chunk("dnr_act_2006", DNR, "11"), _chunk("dnr_act_2006", DNR, "10")]
CORPUS = RETRIEVED + [
    _chunk("dnr_act_2006", DNR, "12"),
    _chunk("dnr_act_2006", DNR, "2", schedule="2"),
    _chunk("spam_act_2003", "Spam Act 2003", "16"),
]


def test_each_reference_is_supported_outside_the_context_or_not_in_the_corpus():
    answer = (
        "Under s 11(1) a person must not call, and s 10 outlines the scheme.  "
        "Section 12 adds detail, and s 99 does not exist.  Section 16 of the "
        "Spam Act 2003 is about email, Schedule 2 covers consent, and s 11 of "
        "the Privacy Act 1988 is not indexed."
    )
    check = verify_citations(answer, RETRIEVED, CORPUS)
    assert check.supported == ["s 11", "s 10"]
    assert check.outside_retrieved == [
        "s 12",
        "s 16 of the Spam Act 2003",
        "Schedule 2",
    ]
    assert check.not_in_corpus == ["s 99", "s 11 of the Privacy Act 1988"]


def test_an_act_named_without_its_year_still_matches():
    check = verify_citations(
        "s 11 of the Do Not Call Register Act forbids it", RETRIEVED, CORPUS
    )
    assert check.supported == ["s 11 of the Do Not Call Register Act"]


def test_a_section_number_alone_does_not_match_a_schedule_clause():
    check = verify_citations("s 2 says so", RETRIEVED, CORPUS)
    assert check.not_in_corpus == ["s 2"]


def test_the_chatbot_attaches_the_check_and_the_chat_prints_it():
    class Retriever:
        embedding_manager = types.SimpleNamespace(embeddings=CORPUS)

        def retrieve(self, query, k=10, expand=False):
            return [
                types.SimpleNamespace(
                    chunk_id=f"c{i}",
                    text="text",
                    chunk_type="section",
                    combined_score=0.1,
                    expansion=None,
                    expansion_reason=None,
                    **vars(chunk),
                )
                for i, chunk in enumerate(RETRIEVED)
            ]

    class Backend:
        def generate(self, prompt, history=None, system=None):
            return "Section 11 forbids it, and s 12 and s 99 are relevant."

    result = llm.LegalChatBot(Retriever(), Backend()).answer("May I call?")
    assert result["citation_check"] == {
        "supported": ["s 11"],
        "outside_retrieved": ["s 12"],
        "not_in_corpus": ["s 99"],
    }
    printed = "\n".join(describe_result(result))
    assert "[1] dnr_act_2006, section 11" in printed
    assert "Not in the indexed Acts: s 99" in printed
    assert "Not in the retrieved text, though in the indexed Acts: s 12" in printed
