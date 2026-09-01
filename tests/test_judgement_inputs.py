"""The judging engine must never read the customer-facing floor plan.

The floor plan is a drawing a person tidies by hand -- chairs get snapped
around tables, overlaps get pushed apart, walls get traced.  If the engine
read it, tidying the picture would silently change how many seats the app
reports as free.  The one link that is allowed runs the other way: the
editor writes corrected chair ownership back into the layout file
(docs/superpowers/specs/2026-09-01-map-as-output-and-supabase-design.md).

This is a string check on imports rather than a behavioural test because
what it guards against is someone wiring it up without noticing, not
someone deliberately reversing the decision.
"""

import ast
import unittest
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
FORBIDDEN = {"install", "install.floorplan", "install.floor_projection"}


def _imported_modules(path: Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


class JudgementInputsTest(unittest.TestCase):
    def test_engine_never_imports_the_floor_plan(self):
        for source in sorted(ENGINE_DIR.glob("*.py")):
            with self.subTest(module=source.name):
                imported = _imported_modules(source)
                offenders = {
                    name
                    for name in imported
                    if name in FORBIDDEN or name.split(".")[0] == "install"
                }
                self.assertEqual(
                    offenders,
                    set(),
                    f"{source.name} 이 평면도를 읽고 있다: {sorted(offenders)}. "
                    "판정 입력은 calibrate 가 만든 레이아웃 하나여야 한다.",
                )

    def test_the_guard_can_actually_see_imports(self):
        # A guard that reads nothing passes forever.  Pin it to a real import
        # the engine does make, so an ast change that breaks parsing is caught.
        imported = _imported_modules(ENGINE_DIR / "seatnow_core.py")
        self.assertIn("engine.seatnow_layout", imported)


if __name__ == "__main__":
    unittest.main()
