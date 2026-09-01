"""Unit tests for the Codex counting pass.

Codex is never invoked here.  What is checked is the command and the prompt,
because those are where the blinding rule lives: if the grader can reach the
annotated stills or our own log, every number this harness produces is void.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from judge_frames import SCHEMA_PATH, build_codex_command, judge_prompt


CLEAN = Path("frames/angle1/clean/t0015.0s.jpg")


class PromptTests(unittest.TestCase):
    def test_prompt_names_the_image(self):
        self.assertIn(str(CLEAN), judge_prompt(CLEAN))

    def test_prompt_forbids_opening_other_files(self):
        # Blinding rule, spec section 5-1.  Without this line Codex can read
        # marked/ or the JSONL and grade us against our own answer.
        prompt = judge_prompt(CLEAN)
        self.assertIn("다른", prompt)
        self.assertIn("열지 마라", prompt)

    def test_prompt_asks_for_the_uncertain_flag(self):
        self.assertIn("uncertain", judge_prompt(CLEAN))

    def test_prompt_does_not_mention_the_marked_directory(self):
        self.assertNotIn("marked", judge_prompt(CLEAN))


class CommandTests(unittest.TestCase):
    def _command(self):
        return build_codex_command(CLEAN, SCHEMA_PATH, Path("judge/t0015.0s.json"))

    def test_runs_codex_exec(self):
        command = self._command()
        self.assertEqual(command[0], "codex")
        self.assertEqual(command[1], "exec")

    def test_session_is_ephemeral(self):
        # A remembered session anchors the next count to the previous one.
        self.assertIn("--ephemeral", self._command())

    def test_sandbox_is_read_only(self):
        command = self._command()
        self.assertIn("--sandbox", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")

    def test_schema_is_enforced(self):
        command = self._command()
        self.assertIn("--output-schema", command)
        self.assertEqual(
            command[command.index("--output-schema") + 1], str(SCHEMA_PATH)
        )

    def test_answer_is_written_to_a_file(self):
        command = self._command()
        self.assertIn("-o", command)
        self.assertEqual(
            command[command.index("-o") + 1], str(Path("judge/t0015.0s.json"))
        )

    def test_git_repo_check_is_skipped(self):
        self.assertIn("--skip-git-repo-check", self._command())

    def test_prompt_is_the_last_argument(self):
        command = self._command()
        self.assertEqual(command[-1], judge_prompt(CLEAN))

    def test_codex_binary_is_overridable(self):
        command = build_codex_command(
            CLEAN, SCHEMA_PATH, Path("out.json"), codex="/usr/bin/codex"
        )
        self.assertEqual(command[0], "/usr/bin/codex")


class SchemaTests(unittest.TestCase):
    def test_schema_file_exists_and_parses(self):
        self.assertTrue(SCHEMA_PATH.exists())
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_schema_requires_every_field(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            sorted(schema["required"]),
            [
                "note",
                "people_seated",
                "people_standing",
                "people_total",
                "tables_belongings_only",
                "tables_in_use",
                "tables_visible",
                "uncertain",
            ],
        )

    def test_schema_forbids_extra_fields(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])


import tempfile

from frame_dump import CLEAN_DIR
from judge_frames import Judgement, clean_frames, judge_directory, parse_judgement


GOOD = json.dumps(
    {
        "people_total": 3,
        "people_seated": 2,
        "people_standing": 1,
        "tables_visible": 6,
        "tables_in_use": 4,
        "tables_belongings_only": 2,
        "uncertain": False,
        "note": "",
    }
)


class ParseJudgementTests(unittest.TestCase):
    def test_valid_answer_parses(self):
        result = parse_judgement("t0015.0s", GOOD)
        self.assertIsNone(result.error)
        self.assertEqual(result.people_total, 3)
        self.assertEqual(result.people_seated, 2)
        self.assertFalse(result.uncertain)

    def test_answer_wrapped_in_prose_still_parses(self):
        # Codex sometimes frames the JSON with a sentence even under a schema.
        prose = "여기 결과입니다:" + chr(10)
        result = parse_judgement("t0015.0s", prose + GOOD + chr(10))
        self.assertIsNone(result.error)
        self.assertEqual(result.people_total, 3)

    def test_broken_json_becomes_an_error(self):
        result = parse_judgement("t0015.0s", "{not json")
        self.assertIsNotNone(result.error)

    def test_missing_field_becomes_an_error(self):
        result = parse_judgement("t0015.0s", json.dumps({"people_total": 2}))
        self.assertIsNotNone(result.error)

    def test_parts_not_summing_to_total_becomes_an_error(self):
        # A schema cannot express "seated + standing == total", so it is
        # checked here.  An answer that fails it is not a usable ground truth.
        bad = json.dumps(
            {
                "people_total": 3,
                "people_seated": 1,
                "people_standing": 1,
                "uncertain": False,
                "note": "",
            }
        )
        result = parse_judgement("t0015.0s", bad)
        self.assertIsNotNone(result.error)

    def test_error_keeps_the_stem(self):
        self.assertEqual(parse_judgement("t0030.0s", "junk").stem, "t0030.0s")


class CleanFramesTests(unittest.TestCase):
    def test_only_clean_directory_is_listed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / CLEAN_DIR).mkdir()
            (root / "marked").mkdir()
            (root / CLEAN_DIR / "t0015.0s.jpg").write_bytes(b"x")
            (root / "marked" / "t0015.0s.jpg").write_bytes(b"x")
            found = clean_frames(root)
            self.assertEqual([p.name for p in found], ["t0015.0s.jpg"])
            self.assertNotIn("marked", str(found[0].parent))

    def test_results_are_in_time_order(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / CLEAN_DIR).mkdir()
            for name in ("t0105.0s.jpg", "t0015.0s.jpg", "t0000.0s.jpg"):
                (root / CLEAN_DIR / name).write_bytes(b"x")
            self.assertEqual(
                [p.name for p in clean_frames(root)],
                ["t0000.0s.jpg", "t0015.0s.jpg", "t0105.0s.jpg"],
            )

    def test_missing_clean_directory_raises(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(FileNotFoundError):
                clean_frames(Path(raw))


def runner_returning(*answers):
    """A fake Codex: returns the given answers in order, or raises them."""
    remaining = list(answers)
    calls = []

    def run(command, output_path, timeout):
        calls.append(command)
        answer = remaining.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    run.calls = calls  # type: ignore[attr-defined]
    return run


class JudgeDirectoryTests(unittest.TestCase):
    def _frames(self, root: Path, count: int):
        (root / CLEAN_DIR).mkdir(parents=True)
        for index in range(count):
            (root / CLEAN_DIR / f"t{index * 15:04d}.0s.jpg").write_bytes(b"x")

    def test_every_frame_gets_one_call(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "frames"
            self._frames(root, 3)
            runner = runner_returning(GOOD, GOOD, GOOD)
            results = judge_directory(root, Path(raw) / "judge", runner=runner)
            self.assertEqual(len(results), 3)
            self.assertEqual(len(runner.calls), 3)

    def test_one_failure_does_not_stop_the_rest(self):
        # 22 stills must not be thrown away because one call timed out.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "frames"
            self._frames(root, 3)
            runner = runner_returning(GOOD, RuntimeError("timeout"), GOOD)
            results = judge_directory(root, Path(raw) / "judge", runner=runner)
            self.assertEqual(len(results), 3)
            self.assertIsNone(results[0].error)
            self.assertIsNotNone(results[1].error)
            self.assertIsNone(results[2].error)

    def test_results_are_written_next_to_each_other(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "frames"
            out = Path(raw) / "judge"
            self._frames(root, 1)
            judge_directory(root, out, runner=runner_returning(GOOD))
            self.assertTrue((out / "t0000.0s.json").exists())

    def test_error_is_recorded_in_the_written_file(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "frames"
            out = Path(raw) / "judge"
            self._frames(root, 1)
            judge_directory(
                root, out, runner=runner_returning(RuntimeError("boom"))
            )
            written = json.loads((out / "t0000.0s.json").read_text(encoding="utf-8"))
            self.assertIn("boom", written["error"])


class ResolveCodexTests(unittest.TestCase):
    """A missing CLI must fail once, before any still is attempted.

    subprocess does not apply PATHEXT, so a bare "codex" never finds the
    npm shim (codex.CMD) on Windows -- and without an up-front check that
    surfaced as thirty identical FileNotFoundErrors, one per still, which
    reads like thirty judging failures instead of one missing tool.
    """

    def test_existing_binary_resolves_to_a_full_path(self):
        from judge_frames import resolve_codex

        resolved = resolve_codex("python")
        self.assertTrue(Path(resolved).is_absolute())

    def test_missing_binary_raises_before_any_call(self):
        from judge_frames import resolve_codex

        with self.assertRaises(FileNotFoundError) as caught:
            resolve_codex("definitely-not-a-real-binary-xyz")
        self.assertIn("definitely-not-a-real-binary-xyz", str(caught.exception))

    def test_explicit_existing_path_is_kept(self):
        from judge_frames import resolve_codex

        with tempfile.TemporaryDirectory() as raw:
            binary = Path(raw) / "codex-fake"
            binary.write_text("", encoding="utf-8")
            self.assertEqual(resolve_codex(str(binary)), str(binary))


class BelongingsPromptTests(unittest.TestCase):
    """Belongings are the main occupancy path, so they must be in the truth.

    ``occupancy_state_from_evidence`` (seatnow_core.py:890-906) calls a table
    occupied on objects OR a seated person OR a chair holding luggage.  The
    first run found 47% of verdicts riding on belongings, and nothing in the
    harness could say whether those were right.
    """

    def test_prompt_asks_about_belongings(self):
        prompt = judge_prompt(CLEAN)
        self.assertIn("tables_in_use", prompt)
        self.assertIn("tables_belongings_only", prompt)
        self.assertIn("tables_visible", prompt)

    def test_prompt_names_what_counts_as_belongings(self):
        prompt = judge_prompt(CLEAN)
        for example in ("가방", "노트북"):
            self.assertIn(example, prompt)


class BelongingsParseTests(unittest.TestCase):
    def _payload(self, **overrides):
        payload = {
            "people_total": 3,
            "people_seated": 2,
            "people_standing": 1,
            "tables_visible": 6,
            "tables_in_use": 4,
            "tables_belongings_only": 2,
            "uncertain": False,
            "note": "",
        }
        payload.update(overrides)
        return json.dumps(payload)

    def test_table_counts_are_kept(self):
        result = parse_judgement("t0015.0s", self._payload())
        self.assertIsNone(result.error)
        self.assertEqual(result.tables_visible, 6)
        self.assertEqual(result.tables_in_use, 4)
        self.assertEqual(result.tables_belongings_only, 2)

    def test_in_use_above_visible_is_an_error(self):
        result = parse_judgement("t0015.0s", self._payload(tables_in_use=9))
        self.assertIsNotNone(result.error)

    def test_belongings_only_above_in_use_is_an_error(self):
        result = parse_judgement(
            "t0015.0s", self._payload(tables_in_use=1, tables_belongings_only=3)
        )
        self.assertIsNotNone(result.error)

    def test_missing_table_field_is_an_error(self):
        payload = json.loads(self._payload())
        del payload["tables_in_use"]
        result = parse_judgement("t0015.0s", json.dumps(payload))
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
