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
            ["note", "people_seated", "people_standing", "people_total", "uncertain"],
        )

    def test_schema_forbids_extra_fields(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
