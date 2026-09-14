"""The evaluation pipeline must import without the heavy ML stack.

Historically `llm/__init__.py` imported `torch` and `transformers` at module
load, so importing `pipeline` (and therefore `evaluate`) forced a multi-gigabyte
install even for the retrieval/metrics path, which never touches the chat model.
These tests lock in that importing the core no longer needs torch.
"""

import importlib.util


def test_torch_is_not_installed_in_the_minimal_env():
    # If this starts failing the guarantee below is no longer meaningful.
    assert importlib.util.find_spec("torch") is None


def test_evaluate_imports_without_torch():
    import evaluate  # noqa: F401


def test_pipeline_imports_without_torch():
    import pipeline  # noqa: F401
