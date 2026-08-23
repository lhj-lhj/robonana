from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_robotwin_instructions.py"
SPEC = importlib.util.spec_from_file_location("audit_robotwin_instructions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_instruction_audit_classifies_placeholder_templates() -> None:
    seen = [audit.template_pattern("Pick {A} with {a} and place it.")]
    unseen = [audit.template_pattern("Move {A}.")]

    assert audit.prompt_class("Pick the red block with the left arm and place it.", seen, unseen) == "seen_only"
    assert audit.prompt_class("Move the red block.", seen, unseen) == "unseen_only"
    assert audit.prompt_class("Ignore the block.", seen, unseen) == "unmatched"
