"""Pytest configuration ensuring the src/ layout is importable."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if "mms_boundary_analysis" not in sys.modules:
    init_py = _SRC / "__init__.py"
    spec = spec_from_file_location("mms_boundary_analysis", init_py)
    if spec and spec.loader:
        module = module_from_spec(spec)
        module.__path__ = [str(_SRC)]
        sys.modules["mms_boundary_analysis"] = module
        spec.loader.exec_module(module)
