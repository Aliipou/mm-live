"""
Shared pytest configuration for mm_live test suite.

Sets up the package import path so tests can import mm_live from the src/
layout without requiring an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path when running tests directly (e.g. without pip install -e .)
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
