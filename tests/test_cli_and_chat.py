"""The menu, the chat loop, and the answer backends, without a model or network."""

import contextlib
import sys
import types
from pathlib import Path

import pytest

from legal_rag import chat, cli, evaluate, llm, pipeline
from legal_rag.data_preprocessing.pdf_parser import PDFParser

# The menu runs each workflow as `python -m <module> <args>`.  Every command
# it builds must parse with that module's own argument parser.


def test_process_command_parses_and_keeps_the_chosen_embedding_model():
    command = cli.process_command(["a.pdf", "b.pdf"], "out", "text-embedding-3-small")
    assert command[1:3] == ["-m", "legal_rag.pipeline"]
    args = pipeline.build_arg_parser().parse_args(command[3:])
    assert args.mode == "process" and args.pdfs == ["a.pdf", "b.pdf"]
    assert args.output_dir == "out"
    assert args.embedding_model == "text-embedding-3-small"


def test_query_chat_and_evaluate_commands_parse():
    args = pipeline.build_arg_parser().parse_args(cli.query_command("q", 5, "out")[3:])
    assert (args.mode, args.query, args.k, args.load_index) == ("query", "q", 5, True)
    assert chat.build_arg_parser().parse_args(cli.chat_command(False, "o")[3:]).no_rag
    assert (
        not chat.build_arg_parser().parse_args(cli.chat_command(True, "o")[3:]).no_rag
    )
    command = cli.evaluate_command(["a.pdf"], 20, "out", "evals")
    args = evaluate.build_arg_parser().parse_args(command[3:])
    assert (args.num_queries, args.eval_dir) == (20, "evals")


def test_menu_defaults_match_the_readme():
    assert cli.DEFAULT_PDFS_DIR == Path("data/raw")
    assert cli.DEFAULT_OUTPUT_DIR == Path("data/processed")


def test_pdf_selection_accepts_lists_and_all():
    assert cli.parse_selection("1, 3", 3) == [0, 2]
    assert cli.parse_selection("a", 2) == [0, 1]
    assert cli.parse_selection("9,x", 3) == []


def test_index_is_saved_under_the_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    rag = pipeline.FairWorkRAGPipeline(output_dir=str(tmp_path / "out"))
    assert rag.embedding_manager.storage_path == tmp_path / "out" / "embeddings"


# Chat loop ------------------------------------------------------------------


class _FakeChatbot:
    corpus_description = "the Toy Act 2024"

    def __init__(self):
        self.calls = []
        self.conversation_history = []

    def answer(self, question, use_rag=True):
        self.calls.append((question, use_rag))
        return {"response": "ok", "sources": [], "metadata": {}}

    def clear_history(self):
        self.conversation_history = []


class _FakePipeline:
    def __init__(self):
        self.bot = _FakeChatbot()
        self.llm = None

    def chatbot(self, llm=None):
        self.llm = llm
        return self.bot


def test_rag_off_changes_how_the_next_question_is_answered(monkeypatch):
    rag = _FakePipeline()
    interface = chat.ChatInterface(rag, llm=object())
    inputs = iter(["first", "/rag off", "second", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    interface.start()
    assert rag.bot.calls == [("first", True), ("second", False)]


def test_chat_main_builds_the_backend_with_get_llm(monkeypatch):
    rag = _FakePipeline()
    rag.embedding_manager = types.SimpleNamespace(embeddings=[object()])
    rag.load_index = lambda name: None
    sentinel = object()
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(chat, "FairWorkRAGPipeline", lambda **kwargs: rag)
    monkeypatch.setattr(chat, "get_llm", lambda: sentinel)
    monkeypatch.setattr(chat.ChatInterface, "start", lambda self: None)
    monkeypatch.setattr(sys, "argv", ["chat", "--no-rag"])
    chat.main()
    assert rag.llm is sentinel


# Chatbot and backends ---------------------------------------------------------


class _RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, history=None, system=None):
        self.calls.append(
            {"prompt": prompt, "history": list(history), "system": system}
        )
        return f"answer {len(self.calls)}"


class _NoResults:
    def retrieve(self, query, k=10, expand=False):
        return []


def test_follow_up_questions_carry_the_earlier_turns():
    backend = _RecordingLLM()
    bot = llm.LegalChatBot(
        _NoResults(), backend, corpus_description="the Spam Act 2003", history_turns=1
    )
    bot.answer("What is a commercial electronic message?")
    bot.answer("Does it include SMS?")
    bot.answer("And email?")
    assert backend.calls[0]["history"] == []
    assert backend.calls[1]["history"] == [
        {"role": "user", "content": "What is a commercial electronic message?"},
        {"role": "assistant", "content": "answer 1"},
    ]
    # history_turns=1 keeps only the last exchange.
    assert backend.calls[2]["history"][0]["content"] == "Does it include SMS?"
    assert "the Spam Act 2003" in backend.calls[0]["system"]


def test_system_prompt_names_the_indexed_documents_not_the_fair_work_act():
    corpus = llm.describe_corpus(
        ["Spam Act 2003", "Do Not Call Register Act 2006", "Spam Act 2003", None]
    )
    assert corpus == "the Spam Act 2003 and the Do Not Call Register Act 2006"
    prompt = llm.build_system_prompt(corpus)
    assert corpus in prompt and "Fair Work" not in prompt


def test_openai_backend_sends_system_history_and_question_in_order():
    backend = llm.OpenAIChatLLM(base_url="http://localhost:9/v1")
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        message = types.SimpleNamespace(content="fine")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    backend._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    backend.generate("q2", history=history, system="SYS")
    assert [m["content"] for m in sent["messages"]] == ["SYS", "q1", "a1", "q2"]


def test_gguf_model_names_fail_with_a_clear_error():
    with pytest.raises(ValueError, match="GGUF"):
        llm.GemmaLLM(model_name="google/gemma-4-12b-it-qat-q4_0-gguf")


class _FakeTensor:
    shape = (1, 3)

    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        self.messages = messages
        return _FakeTensor()

    def decode(self, tokens, skip_special_tokens):
        return " generated " if list(tokens) == [9] else "prompt echoed"


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, 3, 9]]


def _fake_ml_stack(monkeypatch, model=None, error=None):
    def load_model(name, **kwargs):
        if error:
            raise error
        return model

    tokenizer = _FakeTokenizer()
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
        bfloat16="bfloat16",
        float32="float32",
        no_grad=contextlib.nullcontext,
    )
    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name: tokenizer),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load_model),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return tokenizer


def test_a_model_that_fails_to_load_raises_instead_of_answering(monkeypatch):
    _fake_ml_stack(monkeypatch, error=OSError("no such repository"))
    with pytest.raises(RuntimeError, match="no such repository"):
        llm.GemmaLLM(model_name="some/checkpoint")


def test_temperature_zero_means_greedy_decoding_not_the_default(monkeypatch):
    model = _FakeModel()
    tokenizer = _fake_ml_stack(monkeypatch, model=model)
    gemma = llm.GemmaLLM(model_name="some/checkpoint", temperature=0.7)
    assert gemma.generate("q", temperature=0, system="SYS") == "generated"
    assert model.kwargs["do_sample"] is False and "temperature" not in model.kwargs
    gemma.generate("q")
    assert model.kwargs["do_sample"] is True and model.kwargs["temperature"] == 0.7
    # The system prompt is folded into the first user turn.
    assert tokenizer.messages[0]["role"] == "user"


# Document titles ----------------------------------------------------------------


def test_title_comes_from_the_first_page_not_from_any_mention():
    parser = PDFParser()
    pdf = types.SimpleNamespace(metadata={})
    first_page = "Spam Act 2003\nNo. 129, 2003\nSee also the Fair Work Act 2009."
    metadata = parser._extract_metadata(pdf, [first_page])
    assert metadata["title"] == "Spam Act 2003"
    assert metadata["document_type"] == "Act"
