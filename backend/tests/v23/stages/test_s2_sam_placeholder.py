"""Ring 4 compatibility smoke — S2 SAM stage public symbol."""

from __future__ import annotations

import importlib


def test_s2_sam_module_present() -> None:
    mod = importlib.import_module("backend.services.v23.stages.s2_sam")
    assert hasattr(mod, "run_s2_sam")
