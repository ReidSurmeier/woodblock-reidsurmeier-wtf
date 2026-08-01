"""Ring 4 compatibility smoke — S1 ingest stage public symbol."""

from __future__ import annotations

import importlib


def test_s1_ingest_module_present() -> None:
    mod = importlib.import_module("backend.services.v23.stages.s1_ingest")
    assert hasattr(mod, "ingest_reference_image")
