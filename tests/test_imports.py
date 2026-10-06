"""The pipeline must import without loading the heavy ML stack.

`llm/__init__.py` once imported `torch` and `transformers` at module load, so
importing the pipeline, and therefore the evaluation runner, forced a
multi-gigabyte install even for the retrieval and metrics path, which never
touches a local model.  This test imports the light modules in a fresh
interpreter and checks that neither package was loaded.  That holds whether or
not torch is installed, so the suite passes in both the test environment and
the full install.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
LIGHT_MODULES = [
    "legal_rag.pipeline",
    "legal_rag.evaluate",
    "legal_rag.chat",
    "legal_rag.cli",
]


def test_light_modules_import_without_loading_torch_or_transformers():
    code = (
        "import json, sys\n"
        + "".join(f"import {module}\n" for module in LIGHT_MODULES)
        + "print(json.dumps(sorted(m for m in ('torch', 'transformers') "
        "if m in sys.modules)))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
