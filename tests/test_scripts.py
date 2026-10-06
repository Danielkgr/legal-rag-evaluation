"""scripts/chat_demo.py writes the same shape as results/chat_demo.json."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakePipeline:
    def chat(self, question, llm=None):
        sources = [
            {"document": "dnr_act_2006", "section": str(n), "score": 0.1}
            for n in range(7)
        ]
        return {"response": f"answer to {question}", "sources": sources}


def test_chat_demo_rows_match_the_committed_format():
    rows = _load("chat_demo").run(_FakePipeline(), llm=None, questions=["q1", "q2"])
    committed = json.loads((ROOT / "results" / "chat_demo.json").read_text())
    assert [set(row) for row in rows] == [set(row) for row in committed]
    assert [set(s) for s in rows[0]["sources"]] == [
        set(s) for s in committed[0]["sources"]
    ]
    assert len(rows[0]["sources"]) == len(committed[0]["sources"]) == 5
