"""The Claude backend, against a mocked HTTP transport.  No request leaves the machine.

anthropic 1.x is built on httpx2, so the mock transport comes from httpx2.
"""

import json
import types

import anthropic
import httpx2
import pytest

from legal_rag import llm
from legal_rag.llm.claude import ClaudeError, ClaudeLLM, source_title


@pytest.fixture(autouse=True)
def _no_backend_settings(monkeypatch):
    """Keep the caller's backend settings out of these tests."""
    for name in ("CHAT_BACKEND", "CLAUDE_MODEL", "CLAUDE_EFFORT", "CHAT_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def _message(content, stop_reason="end_turn", **extra):
    body = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 40},
    }
    body.update(extra)
    return body


def _client(responses, requests, status=200):
    """A real SDK client whose transport records requests and replays responses."""
    replies = iter(responses)

    def handler(request):
        requests.append(request)
        return httpx2.Response(status, json=next(replies))

    return anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def _source(section, text, schedule=None, reason=None):
    return types.SimpleNamespace(
        text=text,
        document_name="dnr_act_2006",
        section_number=section,
        expansion_reason=reason,
        metadata={
            "document_title": "Do Not Call Register Act 2006",
            "schedule": schedule,
        },
    )


SOURCES = [
    _source("11", "11 Making telemarketing calls\n(1) A person must not make a call."),
    _source(
        "2",
        "2 Consent\nConsent may be express.",
        schedule="2",
        reason="defines 'consent'",
    ),
]

CITED = _message(
    [
        {"type": "text", "text": "Under s 11(1) "},
        {
            "type": "text",
            "text": "a person must not make the call",
            "citations": [
                {
                    "type": "char_location",
                    "cited_text": "(1) A person must not make a call.",
                    "document_index": 0,
                    "document_title": "Do Not Call Register Act 2006, s 11",
                    "start_char_index": 30,
                    "end_char_index": 64,
                }
            ],
        },
        {"type": "text", "text": "."},
    ]
)


def test_each_chunk_is_a_titled_document_with_citations_and_no_sampling_params():
    requests = []
    claude = ClaudeLLM(client=_client([CITED], requests))
    claude.answer_with_documents("May a telemarketer call?", SOURCES, system="SYS")

    (request,) = requests
    assert request.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "high"}
    assert body["system"] == "SYS"
    for absent in ("temperature", "top_p", "top_k", "thinking"):
        assert absent not in body
    content = body["messages"][-1]["content"]
    documents, question = content[:-1], content[-1]
    assert [d["title"] for d in documents] == [
        "Do Not Call Register Act 2006, s 11",
        "Do Not Call Register Act 2006, Schedule 2, clause 2",
    ]
    assert all(d["citations"] == {"enabled": True} for d in documents)
    assert documents[0]["source"] == {
        "type": "text",
        "media_type": "text/plain",
        "data": SOURCES[0].text,
    }
    assert (
        documents[1]["context"] == "Added after retrieval because it defines 'consent'."
    )
    assert question == {"type": "text", "text": "May a telemarketer call?"}


def test_citations_point_at_the_exact_retrieved_text():
    answer = ClaudeLLM(client=_client([CITED], [])).answer_with_documents("q", SOURCES)
    assert answer.text == "Under s 11(1) a person must not make the call[1]."
    (citation,) = answer.citations
    assert citation.source_index == 0
    assert (
        citation.cited_text == SOURCES[0].text[citation.start_char : citation.end_char]
    )
    assert not answer.refused and not answer.truncated and not answer.served_by_fallback


def test_a_refusal_is_reported_not_read_as_an_answer():
    refusal = _message(
        [],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    answer = ClaudeLLM(client=_client([refusal], [])).answer_with_documents(
        "q", SOURCES
    )
    assert answer.refused and answer.refusal_category == "cyber"
    assert (
        answer.text
        == "Claude declined to answer this question (refusal category: cyber)."
    )
    assert answer.citations == []


def test_a_truncated_answer_says_so():
    cut = _message(
        [{"type": "text", "text": "Section 11 says"}], stop_reason="max_tokens"
    )
    answer = ClaudeLLM(client=_client([cut], [])).answer_with_documents("q", SOURCES)
    assert answer.truncated
    assert answer.text.endswith("[The answer was cut off at the max_tokens limit.]")


def test_an_answer_served_by_the_fallback_model_is_flagged():
    served = _message(
        [
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-5"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {"type": "text", "text": "Section 11 applies."},
        ],
        model="claude-opus-5",
        usage={
            "input_tokens": 120,
            "output_tokens": 40,
            "iterations": [
                {
                    "type": "fallback_message",
                    "model": "claude-opus-5",
                    "input_tokens": 120,
                    "output_tokens": 40,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
            ],
        },
    )
    answer = ClaudeLLM(client=_client([served], [])).answer_with_documents("q", SOURCES)
    assert answer.served_by_fallback and answer.model == "claude-opus-5"
    assert answer.text == "Section 11 applies."


def test_api_errors_become_readable_claude_errors():
    error = {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "slow down"},
    }
    claude = ClaudeLLM(client=_client([error], [], status=429))
    with pytest.raises(ClaudeError, match="rate limiting"):
        claude.answer_with_documents("q", SOURCES)


def test_retrieval_off_sends_plain_text_with_history():
    requests = []
    plain = _message([{"type": "text", "text": "General answer."}])
    claude = ClaudeLLM(client=_client([plain], requests), effort="low")
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert claude.generate("q2", history=history) == "General answer."
    body = json.loads(requests[0].content)
    assert body["messages"][-1] == {"role": "user", "content": "q2"}
    assert body["messages"][:2] == history
    assert body["output_config"] == {"effort": "low"}


def test_get_llm_selects_claude_from_the_environment(monkeypatch):
    monkeypatch.setenv("CHAT_BACKEND", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("CLAUDE_EFFORT", "medium")
    backend = llm.get_llm()
    assert isinstance(backend, ClaudeLLM)
    assert (backend.model, backend.effort) == ("claude-opus-5-5", "medium")
    with pytest.raises(ValueError):
        llm.get_llm("gpt")


def test_the_chatbot_hands_claude_the_retrieved_chunks_and_keeps_its_citations():
    class Retriever:
        def retrieve(self, query, k=10, expand=False):
            return [
                types.SimpleNamespace(
                    chunk_id="dnr_act_2006_chunk_9",
                    combined_score=0.03,
                    expansion=None,
                    **vars(SOURCES[0]),
                )
            ]

    claude = ClaudeLLM(client=_client([CITED], []))
    bot = llm.LegalChatBot(Retriever(), claude, corpus_description="the DNR Act")
    result = bot.answer("May a telemarketer call?")
    assert result["response"].endswith("[1].")
    assert result["citations"][0]["source_index"] == 0
    assert result["sources"][0]["section"] == "11"
    assert result["metadata"]["stop_reason"] == "end_turn"


def test_source_title_names_the_act_and_provision():
    assert source_title(SOURCES[0]) == "Do Not Call Register Act 2006, s 11"
    plain = types.SimpleNamespace(
        text="x", document_name="spam_act_2003", section_number=None, metadata={}
    )
    assert source_title(plain) == "spam_act_2003, text outside any section"
