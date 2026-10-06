"""Chat-side tests that run without a model or a network.

The generation layer historically offered a single backend (a local
``transformers`` Gemma checkpoint), which forced the heavy ML install and could
not point at a plain OpenAI-compatible server. These tests lock in the newer
``OpenAIChatLLM`` client and the ``get_llm`` selection rule, using a stubbed
client so no request leaves the machine.
"""

import types

from legal_rag import llm
from legal_rag.llm import OpenAIChatLLM, get_llm


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self._content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


def _stub_client(content):
    completions = _FakeCompletions(content)
    chat = types.SimpleNamespace(completions=completions)
    return types.SimpleNamespace(chat=chat, _record=completions)


def test_generate_wraps_prompt_with_the_legal_system_message():
    client = OpenAIChatLLM(base_url="http://localhost:9/v1")
    client._client = _stub_client("  A grounded answer.  ")
    out = client.generate("When may a call be made?")
    # .strip() is applied to the assistant turn.
    assert out == "A grounded answer."
    sent = client._client._record.calls[0]["messages"]
    assert sent[0] == {"role": "system", "content": llm.build_system_prompt()}
    assert sent[1] == {"role": "user", "content": "When may a call be made?"}


def test_chat_returns_empty_string_when_content_is_missing():
    client = OpenAIChatLLM(base_url="http://localhost:9/v1")
    client._client = _stub_client(None)
    assert client.chat([{"role": "user", "content": "hi"}]) == ""


def test_get_llm_selects_openai_client_when_an_endpoint_is_configured(monkeypatch):
    monkeypatch.setenv("CHAT_BASE_URL", "http://localhost:9/v1")
    monkeypatch.setenv("CHAT_MODEL", "some-model")
    chosen = get_llm()
    assert isinstance(chosen, OpenAIChatLLM)
    assert chosen.base_url == "http://localhost:9/v1"
    assert chosen.model == "some-model"


def test_get_llm_falls_back_to_gemma_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("CHAT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_CHAT_BASE_URL", raising=False)
    sentinel = object()
    monkeypatch.setattr(llm, "GemmaLLM", lambda **kwargs: sentinel)
    assert get_llm() is sentinel
