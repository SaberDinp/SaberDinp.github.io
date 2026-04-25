from __future__ import annotations

import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def tmp_path() -> Path:
    """
    Sandbox-friendly replacement for pytest's built-in tmp_path fixture.

    The default Windows temp root can be unreadable in this environment, which
    breaks tests that only need a writable scratch directory. Keep the scratch
    area under the repo's output tree instead.
    """
    base = Path(tempfile.gettempdir()) / "codex_pytest_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
