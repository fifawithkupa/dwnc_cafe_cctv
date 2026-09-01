# 검출 검사 하네스 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 실제 카페 영상에서 판정 파이프라인을 끝까지 돌리고, 검출·포즈·좌석 판정 세 층 중 어디서 무너지는지 숫자로 낸다.

**Architecture:** `seatnow.py`에 tick마다 사진 두 장(깨끗한 것/주석 달린 것)을 저장하는 옵션을 붙인다. `judge_frames.py`가 깨끗한 사진만 Codex에 하나씩 보여주고 사람 수를 세게 한다(세션마다 새로 열어 기억 오염을 막는다). `inspect_run.py`가 우리 JSONL과 Codex 답을 합쳐 세 층을 한 줄에 놓은 판독표를 낸다.

**Tech Stack:** Python 3.11 / OpenCV / `unittest` / Codex CLI (`codex exec`, `view_image` 툴, `--output-schema`)

**설계 문서:** `docs/superpowers/specs/2026-09-01-detection-inspection-harness-design.md`

## Global Constraints

- 테스트는 `unittest`다. 실행은 `./venv/Scripts/python.exe -m unittest discover tests` (Windows) / `./venv/bin/python -m unittest discover tests` (Linux). 커밋 전 전체 통과 필수 (`README.md:161`)
- 테스트는 **모델을 부르지 않는다.** 순수 함수와 명령 조립을 검사한다 (`tests/test_hwaccel.py` 방식)
- 모든 새 모듈은 `from __future__ import annotations`로 시작하고, **"왜 이게 있는가"를 설명하는 모듈 docstring**과 사용 예시를 단다 (`bench_decode.py:1-17` 방식)
- `PROJECT_DIR = Path(__file__).resolve().parent` 관례를 따른다
- **주석 MP4를 새로 만들지 않는다** (`CLAUDE.md` 개인정보)
- **`seatnow_core.py`의 판정 로직을 건드리지 않는다.** 이 작업은 관찰 도구만 추가한다. `--median-frames` 기본값 2를 그대로 둔다 (설계 §7)
- 검출 재현율의 정의는 **두 개**다. `검출 재현율` = 검출기 person 수 ÷ 실제, `포즈 재현율` = 포즈모델이 낸 사람 수 ÷ 실제. 한 층씩 대응한다
- 커밋 메시지는 한국어, `feat:` / `docs:` / `test:` 접두어

---

### Task 1: `frame_dump.py` — 사진 두 장 저장

**Files:**
- Create: `frame_dump.py`
- Test: `tests/test_frame_dump.py`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces:
  - `frame_stem(timestamp: float) -> str` — `15.0` → `"t0015.0s"`
  - `frame_paths(frame_dir: Path, timestamp: float) -> tuple[Path, Path]` — `(clean, marked)` 순서
  - `save_frame_pair(frame_dir: Path, timestamp: float, clean: np.ndarray, marked: np.ndarray) -> tuple[Path, Path]`
  - 상수 `CLEAN_DIR = "clean"`, `MARKED_DIR = "marked"`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_frame_dump.py`:

```python
"""Unit tests for the per-tick still writer.

No video and no model: the naming rules are pure, and the writing is
checked against a 2x2 array in a temporary directory.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from frame_dump import CLEAN_DIR, MARKED_DIR, frame_paths, frame_stem, save_frame_pair


class FrameStemTests(unittest.TestCase):
    def test_zero_is_padded(self):
        self.assertEqual(frame_stem(0.0), "t0000.0s")

    def test_whole_second_keeps_one_decimal(self):
        self.assertEqual(frame_stem(15.0), "t0015.0s")

    def test_fractional_second_is_kept(self):
        self.assertEqual(frame_stem(1234.5), "t1234.5s")

    def test_stems_sort_in_time_order(self):
        stems = [frame_stem(t) for t in (5.0, 15.0, 105.0, 1005.0)]
        self.assertEqual(stems, sorted(stems))

    def test_negative_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            frame_stem(-0.1)


class FramePathsTests(unittest.TestCase):
    def test_clean_comes_first_and_lives_in_its_own_directory(self):
        clean, marked = frame_paths(Path("frames/angle1"), 15.0)
        self.assertEqual(clean, Path("frames/angle1") / CLEAN_DIR / "t0015.0s.jpg")
        self.assertEqual(marked, Path("frames/angle1") / MARKED_DIR / "t0015.0s.jpg")

    def test_the_two_directories_are_different(self):
        # The blinding rule depends on a grader being able to be pointed at
        # clean/ alone; if they shared a directory that would be impossible.
        self.assertNotEqual(CLEAN_DIR, MARKED_DIR)


class SaveFramePairTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.zeros((2, 2, 3), dtype=np.uint8)
        self.marked = np.full((2, 2, 3), 255, dtype=np.uint8)

    def test_both_files_are_written(self):
        with tempfile.TemporaryDirectory() as raw:
            clean_path, marked_path = save_frame_pair(
                Path(raw), 15.0, self.clean, self.marked
            )
            self.assertTrue(clean_path.exists())
            self.assertTrue(marked_path.exists())

    def test_missing_directories_are_created(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "does" / "not" / "exist"
            clean_path, _ = save_frame_pair(target, 0.0, self.clean, self.marked)
            self.assertTrue(clean_path.exists())

    def test_unwritable_destination_raises(self):
        # Silently skipping a still would be indistinguishable from "that tick
        # was never judged", which is the one thing the harness must not blur.
        with tempfile.TemporaryDirectory() as raw:
            blocker = Path(raw) / CLEAN_DIR
            blocker.write_text("not a directory", encoding="utf-8")
            with self.assertRaises((RuntimeError, OSError, NotADirectoryError, FileExistsError)):
                save_frame_pair(Path(raw), 0.0, self.clean, self.marked)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_frame_dump -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'frame_dump'`

- [ ] **Step 3: 최소 구현을 쓴다**

`frame_dump.py`:

```python
"""Save one clean and one annotated still per judged tick.

The harness that grades detection needs two pictures of the same instant.
The clean one is what a grader counts people in; the marked one is what
explains a disagreement afterwards.  They must never be swapped: showing a
grader the boxes first anchors the count to whatever SeatNow already drew,
which is how a scoring harness quietly starts grading itself.

Stills rather than an MP4 is also what keeps this usable under the
deployment rule that forbids annotated video on disk (CLAUDE.md).

    frames/angle1/clean/t0015.0s.jpg    # counted from
    frames/angle1/marked/t0015.0s.jpg   # diagnosed from
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


CLEAN_DIR = "clean"
MARKED_DIR = "marked"


def frame_stem(timestamp: float) -> str:
    """Media timestamp as a filename stem that sorts in time order.

    Zero-padded because plain ``t105.0s`` would sort before ``t15.0s`` and
    the folder is meant to be paged through in order by a human.
    """
    if timestamp < 0:
        raise ValueError(f"timestamp cannot be negative: {timestamp}")
    return f"t{timestamp:06.1f}s"


def frame_paths(frame_dir: Path, timestamp: float) -> Tuple[Path, Path]:
    """Return (clean, marked) paths for one tick.  Clean is always first."""
    stem = frame_stem(timestamp)
    root = Path(frame_dir)
    return root / CLEAN_DIR / f"{stem}.jpg", root / MARKED_DIR / f"{stem}.jpg"


def save_frame_pair(
    frame_dir: Path,
    timestamp: float,
    clean: np.ndarray,
    marked: np.ndarray,
) -> Tuple[Path, Path]:
    """Write both stills, raising rather than skipping on failure."""
    clean_path, marked_path = frame_paths(frame_dir, timestamp)
    for path, image in ((clean_path, clean), (marked_path, marked)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write frame: {path}")
    return clean_path, marked_path
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_frame_dump -v`
Expected: PASS — 10개 테스트

- [ ] **Step 5: 커밋**

```bash
git add frame_dump.py tests/test_frame_dump.py
git commit -m "feat: frame_dump.py - tick마다 깨끗한 사진과 주석 사진을 따로 저장

채점자에게 박스가 그려진 사진을 먼저 보여주면 판독이 그 박스에 끌려간다.
clean/ 과 marked/ 를 폴더로 갈라 채점자를 clean/ 에만 붙일 수 있게 한다."
```

---

### Task 2: `seatnow.py --frame-dir` 연결

**Files:**
- Modify: `seatnow.py:107` (인자 추가), `seatnow.py:24-40` (import), `seatnow.py:336` (run_context), `seatnow.py:576-587` (저장 호출)
- Test: `tests/test_frame_dump.py` (테스트 클래스 추가)

**Interfaces:**
- Consumes: Task 1의 `save_frame_pair`, `frame_paths`
- Produces: `seatnow.py`의 `--frame-dir` 인자. `args.frame_dir`는 `Optional[Path]`

**배경:** 지금 루프는 `writer.write(render_frame(...))` 한 줄로만 그림을 내보낸다 (`seatnow.py:576-587`). `--no-video`면 `writer`가 `None`이라 그림이 아예 안 나온다. `--frame-dir`는 `--no-video`와 **독립**이어야 한다 — 둘 다 켠 것이 이 하네스의 기본 사용법이다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_frame_dump.py`의 `if __name__` 위에 추가:

```python
class SeatnowArgumentTests(unittest.TestCase):
    """--frame-dir must be independent of --no-video."""

    def _parse(self, argv):
        import seatnow

        return seatnow.build_parser().parse_args(argv)

    def test_frame_dir_defaults_to_none(self):
        args = self._parse(["input.mov"])
        self.assertIsNone(args.frame_dir)

    def test_frame_dir_is_a_path(self):
        args = self._parse(["input.mov", "--frame-dir", "frames/angle1"])
        self.assertEqual(args.frame_dir, Path("frames/angle1"))

    def test_frame_dir_combines_with_no_video(self):
        args = self._parse(["input.mov", "--frame-dir", "frames/a", "--no-video"])
        self.assertEqual(args.frame_dir, Path("frames/a"))
        self.assertTrue(args.no_video)
```

파서 생성 함수는 `seatnow.py:54`의 `build_parser()`다 (확인 완료).

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_frame_dump.SeatnowArgumentTests -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'frame_dir'`

- [ ] **Step 3: 인자를 추가한다**

`seatnow.py:107`의 `--layout` 줄 **위에** 넣는다:

```python
    parser.add_argument(
        "--frame-dir",
        type=Path,
        help="판정한 tick마다 사진 두 장을 이 폴더에 저장한다 "
             "(clean/ = 박스 없음, marked/ = 판정 그려짐). --no-video와 같이 쓴다",
    )
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_frame_dump.SeatnowArgumentTests -v`
Expected: PASS — 3개

- [ ] **Step 5: 저장 호출을 연결한다**

`seatnow.py`의 import 블록(24-40행 근처, `render_frame`을 가져오는 곳 아래)에 추가:

```python
from frame_dump import save_frame_pair
```

`seatnow.py:576-587`의 아래 블록을

```python
                    if writer is not None:
                        writer.write(
                            render_frame(
                                center_frame,
                                analysis,
                                update,
                                debug=args.debug,
                                cadence="fast" if scheduled_fast else None,
                            )
                        )
```

이것으로 바꾼다 — **`render_frame`은 한 번만 부른다.** 둘 다 켰을 때 같은 그림을 두 번 그리면 tick 예산이 이유 없이 늘어난다:

```python
                    rendered = None
                    if writer is not None or args.frame_dir is not None:
                        rendered = render_frame(
                            center_frame,
                            analysis,
                            update,
                            debug=args.debug,
                            cadence="fast" if scheduled_fast else None,
                        )
                    if writer is not None:
                        writer.write(rendered)
                    if args.frame_dir is not None:
                        save_frame_pair(
                            args.frame_dir, center_time, center_frame, rendered
                        )
```

`seatnow.py:336`의 `"log_detections": args.log_detections,` 아래에 추가:

```python
            "frame_dir": str(args.frame_dir) if args.frame_dir else None,
```

- [ ] **Step 6: 전체 테스트로 회귀를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS — 기존 291개 + 신규 13개

- [ ] **Step 7: 실제 영상 5초로 손으로 확인한다**

```bash
./venv/Scripts/python.exe seatnow.py sample_raw/cafe_sample_angle1.mov \
  --no-video --log-detections --sample-seconds 15 --max-samples 2 \
  --frame-dir /tmp/frametest --log /tmp/frametest.jsonl
ls /tmp/frametest/clean /tmp/frametest/marked
```

Expected: 각 폴더에 `t0000.0s.jpg`, `t0015.0s.jpg`. MP4는 없음.
**두 사진을 실제로 열어본다** — `clean/`에 박스가 그려져 있으면 눈가림이 깨진 것이므로 여기서 멈추고 고친다.

- [ ] **Step 8: 커밋**

```bash
git add seatnow.py tests/test_frame_dump.py
git commit -m "feat: seatnow.py --frame-dir - tick마다 사진 두 장

--no-video와 독립이다. 둘 다 켠 것이 검사 하네스의 기본 사용법이다.
render_frame은 한 번만 부른다 - 영상과 사진을 같이 낼 때 두 번 그리면
tick 예산이 이유 없이 늘어난다."
```

---

### Task 3: Codex 명령과 프롬프트 (순수 부분)

**Files:**
- Create: `judge_schema.json`, `judge_frames.py`
- Test: `tests/test_judge_frames.py`

**Interfaces:**
- Consumes: Task 1의 `CLEAN_DIR`
- Produces:
  - `judge_prompt(clean_path: Path) -> str`
  - `build_codex_command(clean_path: Path, schema_path: Path, output_path: Path, codex: str = "codex") -> list[str]`
  - `SCHEMA_PATH: Path` (= `PROJECT_DIR / "judge_schema.json"`)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_judge_frames.py`:

```python
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
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_judge_frames -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'judge_frames'`

- [ ] **Step 3: 스키마를 쓴다**

`judge_schema.json`:

```json
{
  "type": "object",
  "properties": {
    "people_total": {
      "type": "integer",
      "minimum": 0,
      "description": "사진에 보이는 사람의 총 수"
    },
    "people_seated": {
      "type": "integer",
      "minimum": 0,
      "description": "그중 앉아 있는 사람의 수"
    },
    "people_standing": {
      "type": "integer",
      "minimum": 0,
      "description": "그중 서 있거나 걷고 있는 사람의 수"
    },
    "uncertain": {
      "type": "boolean",
      "description": "가려지거나 뒤통수만 보여 사람인지 또는 자세가 확실하지 않은 대상이 하나라도 있으면 참"
    },
    "note": {
      "type": "string",
      "description": "애매했던 점을 한 줄로. 없으면 빈 문자열"
    }
  },
  "required": [
    "people_total",
    "people_seated",
    "people_standing",
    "uncertain",
    "note"
  ],
  "additionalProperties": false
}
```

- [ ] **Step 4: 순수 함수를 구현한다**

`judge_frames.py` (이번 태스크에서는 여기까지만):

```python
"""Count people in each still with Codex, blind to what SeatNow decided.

The harness cannot say "the detector missed people" without something that
knows how many people were actually there.  Codex does that here — but only
if it never sees our answer first.  So each still gets its own throwaway
session (no memory of the previous count to anchor to), the sandbox is
read-only, and the prompt names one file and forbids opening any other.

Codex opens the image itself through its ``view_image`` tool, which is on by
default (``codex features list``).  No API key is involved; the ChatGPT
login the CLI already holds is enough.

    python judge_frames.py frames/angle1 --output judge/angle1
"""

from __future__ import annotations

from pathlib import Path
from typing import List


PROJECT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = PROJECT_DIR / "judge_schema.json"

PROMPT_TEMPLATE = """\
{path} 를 열어라.

이 사진은 카페 CCTV의 한 장면이다. 아래를 판단해 JSON으로만 답하라.

- people_total: 보이는 사람 수
- people_seated: 그중 앉아 있는 사람 수
- people_standing: 그중 서 있거나 걷고 있는 사람 수
- uncertain: 가려지거나 뒤통수만 보여 사람인지 또는 자세가 확실하지 않은 대상이
  하나라도 있으면 true
- note: 애매했던 점을 한 줄로. 없으면 빈 문자열

people_seated 와 people_standing 의 합은 people_total 과 같아야 한다.

**이 사진 파일 외에 다른 어떤 파일도 열지 마라.** 같은 저장소의 다른 사진,
상위 폴더, 로그 파일을 열면 이 판독은 무효가 된다. 너는 이 사진 한 장만 보고
답해야 한다.
"""


def judge_prompt(clean_path: Path) -> str:
    """The counting prompt for one still.

    The last paragraph is not decoration: it is the blinding rule.  Codex
    runs read-only but read-only still reads, and the annotated stills sit
    one directory away from the clean ones.
    """
    return PROMPT_TEMPLATE.format(path=clean_path)


def build_codex_command(
    clean_path: Path,
    schema_path: Path,
    output_path: Path,
    codex: str = "codex",
) -> List[str]:
    """Assemble one throwaway Codex run that answers in schema-shaped JSON."""
    return [
        codex,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        judge_prompt(clean_path),
    ]
```

- [ ] **Step 5: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_judge_frames -v`
Expected: PASS — 15개

- [ ] **Step 6: 커밋**

```bash
git add judge_frames.py judge_schema.json tests/test_judge_frames.py
git commit -m "feat: Codex 판독 명령과 프롬프트 - 눈가림을 명령에 박는다

세션마다 새로 열고(--ephemeral) 읽기 전용으로 돌리고 프롬프트에서
사진 한 장으로 한정한다. 기억이 남으면 앞 장의 숫자에 끌려가고,
marked/ 나 JSONL을 보면 우리 답으로 우리를 채점하게 된다."
```

---

### Task 4: Codex 호출 루프와 실패 처리

**Files:**
- Modify: `judge_frames.py` (실행 부분 추가)
- Test: `tests/test_judge_frames.py` (클래스 추가)

**Interfaces:**
- Consumes: Task 3의 `build_codex_command`, `SCHEMA_PATH`
- Produces:
  - `@dataclass Judgement`: `stem: str`, `people_total: int`, `people_seated: int`, `people_standing: int`, `uncertain: bool`, `note: str`, `error: Optional[str]`
  - `parse_judgement(stem: str, text: str) -> Judgement` — 파싱과 합계 검증
  - `clean_frames(frame_dir: Path) -> list[Path]` — `clean/*.jpg` 정렬
  - `judge_directory(frame_dir, output_dir, codex="codex", timeout=180.0, runner=None) -> list[Judgement]`
  - `runner` 인자는 테스트용 주입점이다 (`tests/test_hwaccel.py`의 probe 주입과 같은 방식)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_judge_frames.py`의 `if __name__` 위에 추가:

```python
import tempfile

from frame_dump import CLEAN_DIR
from judge_frames import Judgement, clean_frames, judge_directory, parse_judgement


GOOD = json.dumps(
    {
        "people_total": 3,
        "people_seated": 2,
        "people_standing": 1,
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
        result = parse_judgement("t0015.0s", f"여기 결과입니다:\n{GOOD}\n")
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
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_judge_frames -v`
Expected: FAIL — `ImportError: cannot import name 'Judgement' from 'judge_frames'`

- [ ] **Step 3: 구현을 추가한다**

`judge_frames.py`의 import를 다음으로 바꾼다:

```python
import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

from frame_dump import CLEAN_DIR
```

파일 끝에 추가:

```python
DEFAULT_TIMEOUT_SECONDS = 180.0
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_REQUIRED = ("people_total", "people_seated", "people_standing", "uncertain", "note")


@dataclass
class Judgement:
    """One still's ground truth, or the reason there is none."""

    stem: str
    people_total: int = 0
    people_seated: int = 0
    people_standing: int = 0
    uncertain: bool = False
    note: str = ""
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        """Whether this may be counted in a score at all."""
        return self.error is None and not self.uncertain


def parse_judgement(stem: str, text: str) -> Judgement:
    """Turn one Codex answer into a Judgement, or into a recorded failure.

    The sum check is here because JSON Schema cannot express it: an answer
    whose parts disagree with its own total is not usable ground truth, and
    silently keeping the total would launder that disagreement into a score.
    """
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        return Judgement(stem=stem, error=f"no JSON object in answer: {text[:120]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return Judgement(stem=stem, error=f"invalid JSON: {exc}")

    missing = [field for field in _REQUIRED if field not in payload]
    if missing:
        return Judgement(stem=stem, error=f"missing fields: {', '.join(missing)}")

    try:
        total = int(payload["people_total"])
        seated = int(payload["people_seated"])
        standing = int(payload["people_standing"])
    except (TypeError, ValueError) as exc:
        return Judgement(stem=stem, error=f"non-integer count: {exc}")

    if seated + standing != total:
        return Judgement(
            stem=stem,
            error=f"parts do not sum to total: {seated}+{standing} != {total}",
        )

    return Judgement(
        stem=stem,
        people_total=total,
        people_seated=seated,
        people_standing=standing,
        uncertain=bool(payload["uncertain"]),
        note=str(payload["note"]),
    )


def clean_frames(frame_dir: Path) -> List[Path]:
    """Every clean still, in time order.  Never touches marked/."""
    clean_dir = Path(frame_dir) / CLEAN_DIR
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"No clean frames directory: {clean_dir}")
    return sorted(clean_dir.glob("*.jpg"))


def run_codex(command: List[str], output_path: Path, timeout: float) -> str:
    """Run one Codex call and return its answer text."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"codex exited {completed.returncode}: {(completed.stderr or '')[:300]}"
        )
    if output_path.exists():
        return output_path.read_text(encoding="utf-8")
    return completed.stdout or ""


def judge_directory(
    frame_dir: Path,
    output_dir: Path,
    codex: str = "codex",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Optional[Callable[[List[str], Path, float], str]] = None,
) -> List[Judgement]:
    """Count people in every clean still, one throwaway session each.

    A failed call costs one still, never the run: with roughly 22 stills per
    angle, throwing the batch away over a single timeout would make the
    harness more fragile than the thing it is measuring.
    """
    call = runner or run_codex
    output_dir = Path(output_dir)
    results: List[Judgement] = []
    for frame in clean_frames(frame_dir):
        stem = frame.stem
        answer_path = output_dir / f"{stem}.json"
        command = build_codex_command(frame, SCHEMA_PATH, answer_path, codex=codex)
        try:
            text = call(command, answer_path, timeout)
        except Exception as exc:  # noqa: BLE001 - any failure costs one still
            judgement = Judgement(stem=stem, error=f"{type(exc).__name__}: {exc}")
        else:
            judgement = parse_judgement(stem, text)
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        answer_path.write_text(
            json.dumps(asdict(judgement), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        results.append(judgement)
        state = judgement.error or (
            f"people={judgement.people_total} uncertain={judgement.uncertain}"
        )
        print(f"[{stem}] {state}", flush=True)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_dir", type=Path, help="seatnow.py --frame-dir 로 만든 폴더")
    parser.add_argument("--output", type=Path, help="판독 결과 폴더 (기본: <frame_dir>/judge)")
    parser.add_argument("--codex", default="codex", help="codex 실행 파일 경로")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="한 사진당 제한 시간(초)"
    )
    args = parser.parse_args(argv)

    output_dir = args.output or (args.frame_dir / "judge")
    results = judge_directory(
        args.frame_dir, output_dir, codex=args.codex, timeout=args.timeout
    )
    failed = sum(result.error is not None for result in results)
    unsure = sum(result.uncertain for result in results if result.error is None)
    print(
        f"\n{len(results)}장 판독: 실패 {failed}장, 불확실 {unsure}장, "
        f"채점 가능 {sum(result.usable for result in results)}장"
    )
    print(f"결과: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_judge_frames -v`
Expected: PASS — 28개

- [ ] **Step 5: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add judge_frames.py tests/test_judge_frames.py
git commit -m "feat: judge_frames.py - 사진마다 Codex를 새로 불러 사람 수를 센다

호출 하나가 실패해도 그 사진만 버리고 계속한다. 22장을 타임아웃 하나로
통째로 버리면 하네스가 재려는 대상보다 더 잘 깨진다.

seated+standing != total 은 스키마로 못 막아서 파서에서 잡는다.
자기 총합과 안 맞는 답은 정답으로 쓸 수 없다."
```

---

### Task 5: `inspect_run.py` — 세 층을 한 줄에

**Files:**
- Create: `inspect_run.py`
- Test: `tests/test_inspect_run.py`

**Interfaces:**
- Consumes: Task 1의 `frame_stem`, Task 4의 `Judgement`, 기존 `verify_seatnow.load_jsonl`
- Produces:
  - `@dataclass Row`: `stem`, `timestamp`, `det_person`, `det_chair`, `det_table`, `pose_seated`, `pose_standing`, `pose_unknown`, `seat_occupied`, `seat_empty`, `seat_unknown`, `seat_ignore`, `truth: Optional[int]`, `excluded: Optional[str]`
  - `Row.pose_total -> int`
  - `Row.detector_gap -> Optional[int]`, `Row.pose_gap -> Optional[int]`
  - `build_rows(records: list[dict], judgements: dict[str, Judgement]) -> list[Row]`
  - `@dataclass Recall`: `layer`, `found`, `truth`, `value: Optional[float]`, `scored_frames`, `excluded_frames`
  - `recall(rows, layer: str) -> Recall` — `layer`는 `"detector"` 또는 `"pose"`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_inspect_run.py`:

```python
"""Unit tests for the three-layer reading table.

Fixtures are hand-written JSONL records: the point of the table is that a
number breaking in one layer is visible as that layer's number, so the tests
build records where exactly one layer is wrong.
"""

from __future__ import annotations

import unittest

from inspect_run import Row, build_rows, recall
from judge_frames import Judgement


def record(timestamp, person=3, chair=7, table=2, seated=2, standing=1, unknown=0,
           occupied=2, empty=0, seat_unknown=1, ignore=0):
    return {
        "timestamp": timestamp,
        "raw_detections": {
            "counts": {"person": person, "chair": chair, "dining table": table}
        },
        "summary": {
            "seated_poses": seated,
            "standing_poses": standing,
            "unknown_poses": unknown,
            "occupied": occupied,
            "empty": empty,
            "unknown": seat_unknown,
            "ignore": ignore,
        },
        "tables": [],
    }


def truth(total=3, uncertain=False, error=None):
    return Judgement(
        stem="ignored",
        people_total=total,
        people_seated=total,
        people_standing=0,
        uncertain=uncertain,
        error=error,
    )


class BuildRowsTests(unittest.TestCase):
    def test_row_stem_matches_the_frame_filename(self):
        rows = build_rows([record(15.0)], {})
        self.assertEqual(rows[0].stem, "t0015.0s")

    def test_judgement_is_matched_by_stem(self):
        rows = build_rows([record(15.0)], {"t0015.0s": truth(total=4)})
        self.assertEqual(rows[0].truth, 4)

    def test_missing_judgement_leaves_truth_empty(self):
        rows = build_rows([record(15.0)], {})
        self.assertIsNone(rows[0].truth)
        self.assertIsNone(rows[0].detector_gap)

    def test_missing_raw_detections_counts_as_zero(self):
        bare = record(15.0)
        del bare["raw_detections"]
        rows = build_rows([bare], {})
        self.assertEqual(rows[0].det_person, 0)

    def test_pose_total_sums_the_three_pose_states(self):
        rows = build_rows([record(15.0, seated=2, standing=1, unknown=1)], {})
        self.assertEqual(rows[0].pose_total, 4)

    def test_gap_is_found_minus_truth(self):
        rows = build_rows([record(15.0, person=1)], {"t0015.0s": truth(total=3)})
        self.assertEqual(rows[0].detector_gap, -2)

    def test_uncertain_judgement_is_marked_excluded(self):
        rows = build_rows([record(15.0)], {"t0015.0s": truth(uncertain=True)})
        self.assertEqual(rows[0].excluded, "uncertain")

    def test_failed_judgement_is_marked_excluded(self):
        rows = build_rows([record(15.0)], {"t0015.0s": truth(error="timeout")})
        self.assertEqual(rows[0].excluded, "error")


class RecallTests(unittest.TestCase):
    def test_perfect_detection_is_one(self):
        rows = build_rows(
            [record(0.0, person=3), record(15.0, person=3)],
            {"t0000.0s": truth(3), "t0015.0s": truth(3)},
        )
        self.assertEqual(recall(rows, "detector").value, 1.0)

    def test_half_the_people_is_one_half(self):
        rows = build_rows(
            [record(0.0, person=2), record(15.0, person=2)],
            {"t0000.0s": truth(4), "t0015.0s": truth(4)},
        )
        self.assertEqual(recall(rows, "detector").value, 0.5)

    def test_over_detection_does_not_exceed_one(self):
        # Recall answers "how many of the people present were found", so a
        # frame with more boxes than people is capped, not credited.
        rows = build_rows([record(0.0, person=5)], {"t0000.0s": truth(3)})
        self.assertEqual(recall(rows, "detector").value, 1.0)

    def test_uncertain_frames_are_left_out_of_the_score(self):
        rows = build_rows(
            [record(0.0, person=3), record(15.0, person=0)],
            {"t0000.0s": truth(3), "t0015.0s": truth(3, uncertain=True)},
        )
        result = recall(rows, "detector")
        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.scored_frames, 1)
        self.assertEqual(result.excluded_frames, 1)

    def test_pose_layer_scores_separately_from_detector(self):
        rows = build_rows(
            [record(0.0, person=3, seated=1, standing=0, unknown=0)],
            {"t0000.0s": truth(3)},
        )
        self.assertEqual(recall(rows, "detector").value, 1.0)
        self.assertAlmostEqual(recall(rows, "pose").value, 1 / 3)

    def test_no_truth_at_all_gives_no_value(self):
        rows = build_rows([record(0.0)], {})
        result = recall(rows, "detector")
        self.assertIsNone(result.value)
        self.assertEqual(result.scored_frames, 0)

    def test_unknown_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            recall(build_rows([record(0.0)], {}), "seats")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_inspect_run -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inspect_run'`

- [ ] **Step 3: 구현을 쓴다**

`inspect_run.py`:

```python
"""Put the three layers of one judgement on one line, side by side.

"Does YOLO work here" cannot be answered, because a person whose lower body
is under a table is not a bug.  "Where does it break first" can be, and that
is what this table is for: raw detection, then pose, then the seat verdict,
in that order, against how many people were actually in the picture.

    python inspect_run.py sample_results/angle1.jsonl --judge frames/angle1/judge

Read it downward, not across.  If the detection column is already wrong the
other two columns are consequences, not findings.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from frame_dump import frame_stem
from judge_frames import Judgement
from seatnow_report import REASON_GROUPS
from verify_seatnow import load_jsonl


PROJECT_DIR = Path(__file__).resolve().parent
LAYERS = ("detector", "pose")

# COCO names the detector emits for the two furniture classes SeatNow uses.
CHAIR_CLASSES = ("chair", "couch", "bench")
TABLE_CLASSES = ("dining table",)


@dataclass
class Row:
    """One judged tick: what was seen, what was inferred, what was decided."""

    stem: str
    timestamp: float
    det_person: int
    det_chair: int
    det_table: int
    pose_seated: int
    pose_standing: int
    pose_unknown: int
    seat_occupied: int
    seat_empty: int
    seat_unknown: int
    seat_ignore: int
    truth: Optional[int] = None
    excluded: Optional[str] = None

    @property
    def pose_total(self) -> int:
        return self.pose_seated + self.pose_standing + self.pose_unknown

    @property
    def detector_gap(self) -> Optional[int]:
        return None if self.truth is None else self.det_person - self.truth

    @property
    def pose_gap(self) -> Optional[int]:
        return None if self.truth is None else self.pose_total - self.truth

    def found(self, layer: str) -> int:
        if layer == "detector":
            return self.det_person
        if layer == "pose":
            return self.pose_total
        raise ValueError(f"Unknown layer: {layer}")


def _counts(record: Dict[str, object]) -> Dict[str, int]:
    raw = record.get("raw_detections") or {}
    return dict(raw.get("counts") or {})  # type: ignore[arg-type]


def build_rows(
    records: List[Dict[str, object]], judgements: Dict[str, Judgement]
) -> List[Row]:
    """Join our own log with the blind counts, matched by frame stem."""
    rows: List[Row] = []
    for record in records:
        timestamp = float(record["timestamp"])  # type: ignore[arg-type]
        stem = frame_stem(timestamp)
        counts = _counts(record)
        summary = record.get("summary") or {}
        judgement = judgements.get(stem)

        truth: Optional[int] = None
        excluded: Optional[str] = None
        if judgement is not None:
            if judgement.error is not None:
                excluded = "error"
            elif judgement.uncertain:
                excluded = "uncertain"
            else:
                truth = judgement.people_total

        rows.append(
            Row(
                stem=stem,
                timestamp=timestamp,
                det_person=int(counts.get("person", 0)),
                det_chair=sum(int(counts.get(name, 0)) for name in CHAIR_CLASSES),
                det_table=sum(int(counts.get(name, 0)) for name in TABLE_CLASSES),
                pose_seated=int(summary.get("seated_poses", 0)),  # type: ignore[union-attr]
                pose_standing=int(summary.get("standing_poses", 0)),  # type: ignore[union-attr]
                pose_unknown=int(summary.get("unknown_poses", 0)),  # type: ignore[union-attr]
                seat_occupied=int(summary.get("occupied", 0)),  # type: ignore[union-attr]
                seat_empty=int(summary.get("empty", 0)),  # type: ignore[union-attr]
                seat_unknown=int(summary.get("unknown", 0)),  # type: ignore[union-attr]
                seat_ignore=int(summary.get("ignore", 0)),  # type: ignore[union-attr]
                truth=truth,
                excluded=excluded,
            )
        )
    return rows


@dataclass
class Recall:
    """How much of what was there got found, at one layer."""

    layer: str
    found: int
    truth: int
    value: Optional[float]
    scored_frames: int
    excluded_frames: int


def recall(rows: List[Row], layer: str) -> Recall:
    """Pooled recall over the frames that have usable ground truth.

    Per-frame found counts are capped at that frame's truth: recall answers
    "how many of the people present were found", so extra boxes in one frame
    must not pay for people missed in another.
    """
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer: {layer}. Expected one of {LAYERS}")
    found = 0
    total = 0
    scored = 0
    excluded = 0
    for row in rows:
        if row.excluded is not None:
            excluded += 1
            continue
        if row.truth is None:
            continue
        scored += 1
        total += row.truth
        found += min(row.found(layer), row.truth)
    value = (found / total) if total else None
    return Recall(
        layer=layer,
        found=found,
        truth=total,
        value=value,
        scored_frames=scored,
        excluded_frames=excluded,
    )


def load_judgements(judge_dir: Optional[Path]) -> Dict[str, Judgement]:
    """Read every written judgement.  A missing directory means none yet."""
    if judge_dir is None or not Path(judge_dir).is_dir():
        return {}
    judgements: Dict[str, Judgement] = {}
    for path in sorted(Path(judge_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("stem", path.stem)
        judgements[payload["stem"]] = Judgement(**payload)
    return judgements
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_inspect_run -v`
Expected: PASS — 15개

- [ ] **Step 5: 커밋**

```bash
git add inspect_run.py tests/test_inspect_run.py
git commit -m "feat: inspect_run.py - 검출/포즈/좌석 세 층을 한 줄에 놓는다

층마다 재현율을 따로 낸다. 검출이 이미 틀렸으면 뒤 두 칸은 발견이
아니라 그 결과이므로, 한 숫자로 뭉치면 어디를 고쳐야 하는지 사라진다.

불확실·실패 판독은 분모에서 빼되 몇 장인지 같이 낸다. 제외가 많으면
그것 자체가 보고할 사실이다."
```

---

### Task 6: 판독표와 요약 출력

**Files:**
- Modify: `inspect_run.py` (렌더링과 `main()` 추가)
- Test: `tests/test_inspect_run.py` (클래스 추가)

**Interfaces:**
- Consumes: Task 5의 `Row`, `Recall`, `recall`, `build_rows`, `load_judgements`
- Produces:
  - `reason_distribution(records: list[dict]) -> dict[str, int]` — `REASON_GROUPS` 기준
  - `render_table(rows: list[Row]) -> str` — 마크다운 표
  - `render_summary(rows, records) -> str`
  - `disagreements(rows: list[Row]) -> list[Row]` — 3단계 진단 대상
  - `main(argv=None) -> int`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_inspect_run.py`의 `if __name__` 위에 추가:

```python
from inspect_run import disagreements, reason_distribution, render_summary, render_table


def record_with_reasons(timestamp, *reason_codes):
    """A record whose seat_report carries one plain table per reason code."""
    payload = record(timestamp)
    payload["seat_report"] = {
        "seats": [
            {
                "seat_id": f"T{index}",
                "kind": "table",
                "capacity": 1,
                "state": "unknown",
                "reason_code": code,
            }
            for index, code in enumerate(reason_codes)
        ]
    }
    return payload


class RenderTableTests(unittest.TestCase):
    def test_every_row_appears(self):
        rows = build_rows([record(0.0), record(15.0)], {})
        table = render_table(rows)
        self.assertIn("t0000.0s", table)
        self.assertIn("t0015.0s", table)

    def test_missing_truth_renders_as_a_blank_to_fill(self):
        # The table must stay usable with no Codex run at all: a person can
        # write the counts into this column by hand.
        table = render_table(build_rows([record(0.0)], {}))
        self.assertIn("___", table)

    def test_excluded_row_says_why(self):
        rows = build_rows([record(0.0)], {"t0000.0s": truth(uncertain=True)})
        self.assertIn("uncertain", render_table(rows))

    def test_disagreement_is_flagged(self):
        rows = build_rows([record(0.0, person=1)], {"t0000.0s": truth(3)})
        self.assertIn("⚠", render_table(rows))

    def test_agreement_is_not_flagged(self):
        rows = build_rows([record(0.0, person=3)], {"t0000.0s": truth(3)})
        self.assertNotIn("⚠", render_table(rows))


class ReasonDistributionTests(unittest.TestCase):
    def test_codes_are_grouped_by_what_fixes_them(self):
        records = [
            record_with_reasons(0.0, "occluded_lower_body", "pose_low_keypoints"),
            record_with_reasons(15.0, "occluded_lower_body"),
        ]
        distribution = reason_distribution(records)
        self.assertEqual(distribution["geometry"], 2)
        self.assertEqual(distribution["model"], 1)

    def test_unseen_groups_are_zero_not_absent(self):
        distribution = reason_distribution([record_with_reasons(0.0, "person_seated")])
        self.assertEqual(distribution["model"], 0)

    def test_unknown_code_does_not_crash(self):
        distribution = reason_distribution([record_with_reasons(0.0, "made_up_code")])
        self.assertEqual(distribution["other"], 1)

    def test_bar_zone_reason_counts_are_added(self):
        # A counted_zone reports {code: count}, not one code per seat
        # (seatnow_report.py:170-182).  Reading only the plain-table shape
        # would silently drop every bar seat's reason.
        payload = record(0.0)
        payload["seat_report"] = {
            "seats": [
                {
                    "seat_id": "BAR",
                    "kind": "counted_zone",
                    "capacity": 3,
                    "occupied": 0,
                    "free": 1,
                    "unknown": 2,
                    "reason_codes": {"occluded_lower_body": 2},
                }
            ]
        }
        self.assertEqual(reason_distribution([payload])["geometry"], 2)

    def test_record_without_a_seat_report_is_skipped(self):
        self.assertEqual(reason_distribution([record(0.0)])["geometry"], 0)


class DisagreementTests(unittest.TestCase):
    def test_only_rows_with_a_gap_are_returned(self):
        rows = build_rows(
            [record(0.0, person=3), record(15.0, person=1)],
            {"t0000.0s": truth(3), "t0015.0s": truth(3)},
        )
        self.assertEqual([row.stem for row in disagreements(rows)], ["t0015.0s"])

    def test_rows_without_truth_are_not_disagreements(self):
        self.assertEqual(disagreements(build_rows([record(0.0)], {})), [])


class RenderSummaryTests(unittest.TestCase):
    def test_both_layer_recalls_appear(self):
        rows = build_rows([record(0.0, person=3)], {"t0000.0s": truth(3)})
        summary = render_summary(rows, [record(0.0)])
        self.assertIn("검출", summary)
        self.assertIn("포즈", summary)

    def test_excluded_count_is_reported(self):
        rows = build_rows([record(0.0)], {"t0000.0s": truth(uncertain=True)})
        self.assertIn("제외", render_summary(rows, [record(0.0)]))
```

사유 코드의 위치는 확인 완료다 — `record["seat_report"]["seats"]`이며 **형태가 두 가지**다 (`seatnow_report.py:150-182`).

| 좌석 종류 | 키 | 값 |
|---|---|---|
| 일반 테이블 (`kind="table"`) | `reason_code` | 코드 문자열 하나 |
| 바 구역 (`kind="counted_zone"`) | `reason_codes` | `{코드: 개수}` 집계 |

레이아웃 없이 돌리는 이번 실행에는 바 구역이 안 나오지만, T14 이후 레이아웃을 주고 다시 돌리면 나온다. 그때 조용히 빠지지 않도록 지금 양쪽을 다 읽는다.

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_inspect_run -v`
Expected: FAIL — `ImportError: cannot import name 'render_table' from 'inspect_run'`

- [ ] **Step 3: 구현을 추가한다**

`inspect_run.py` 끝에 추가:

```python
def reason_distribution(records: List[Dict[str, object]]) -> Dict[str, int]:
    """Count reason codes by the group that fixes them.

    The grouping is the point: "geometry 22% / model 8%" says whether the
    next move is a rescue path or fine-tuning, which one flat list does not.

    Two shapes have to be read, because ``build_seat_report`` emits two: a
    plain table carries one ``reason_code``, while a bar zone carries a
    ``reason_codes`` tally over its seats (``seatnow_report.py:150-182``).
    Reading only the first shape drops every bar seat without a trace.
    """
    lookup = {
        code.value: group
        for group, codes in REASON_GROUPS.items()
        for code in codes
    }
    distribution: Dict[str, int] = {group: 0 for group in REASON_GROUPS}
    distribution["other"] = 0

    def add(code: str, count: int = 1) -> None:
        distribution[lookup.get(code, "other")] += count

    for record in records:
        report = record.get("seat_report") or {}
        for seat in report.get("seats") or []:  # type: ignore[union-attr]
            if seat.get("kind") == "counted_zone":
                for code, count in (seat.get("reason_codes") or {}).items():
                    add(str(code), int(count))
                continue
            code = seat.get("reason_code")
            if code is not None:
                add(str(code))
    return distribution


def disagreements(rows: List[Row]) -> List[Row]:
    """Rows where we and the blind count differ — the diagnosis shortlist."""
    return [
        row
        for row in rows
        if row.excluded is None and row.truth is not None and row.detector_gap != 0
    ]


def _truth_cell(row: Row) -> str:
    if row.excluded is not None:
        return row.excluded
    if row.truth is None:
        return "___"
    return str(row.truth)


def _gap_cell(row: Row) -> str:
    if row.excluded is not None or row.detector_gap is None:
        return ""
    if row.detector_gap == 0:
        return "0"
    return f"{row.detector_gap:+d} ⚠"


def render_table(rows: List[Row]) -> str:
    """The reading table: three layers across, one tick per line."""
    lines = [
        "| 사진 | 검출(사람/의자/책상) | 포즈(앉음/섬/모름) | 좌석(점유/빈/모름/무시) | 실제 | 차이 |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.stem} "
            f"| {row.det_person}/{row.det_chair}/{row.det_table} "
            f"| {row.pose_seated}/{row.pose_standing}/{row.pose_unknown} "
            f"| {row.seat_occupied}/{row.seat_empty}/{row.seat_unknown}/{row.seat_ignore} "
            f"| {_truth_cell(row)} "
            f"| {_gap_cell(row)} |"
        )
    return "\n".join(lines)


def _recall_line(label: str, result: Recall) -> str:
    if result.value is None:
        return f"- **{label} 재현율**: 정답 없음 (채점한 사진 {result.scored_frames}장)"
    return (
        f"- **{label} 재현율**: {result.value:.2f} "
        f"({result.found}/{result.truth}명, 사진 {result.scored_frames}장, "
        f"제외 {result.excluded_frames}장)"
    )


def render_summary(rows: List[Row], records: List[Dict[str, object]]) -> str:
    """The two numbers and the backlog table, under one heading."""
    detector = recall(rows, "detector")
    pose = recall(rows, "pose")
    distribution = reason_distribution(records)
    total_reasons = sum(distribution.values()) or 1

    lines = [
        "## 요약",
        "",
        f"- 판정한 tick: {len(rows)}회",
        _recall_line("검출", detector),
        _recall_line("포즈", pose),
        "",
        "### 사유 코드 분포 (무엇이 고치는가로 묶음)",
        "",
        "| 그룹 | 건수 | 비율 | 고치는 방법 |",
        "|---|---:|---:|---|",
    ]
    how = {
        "install": "카메라를 다시 단다 (코드로 풀지 않는다)",
        "geometry": "구제 경로·좌석 칸 다시 긋기",
        "model": "파인튜닝 또는 imgsz 상향 (plan.md T11)",
        "time": "기다리면 된다 — 할 일 없음",
        "settled": "판정이 끝난 것 — 문제 아님",
        "other": "어휘에 없는 코드 — 확인 필요",
    }
    for group, count in distribution.items():
        lines.append(
            f"| {group} | {count} | {count / total_reasons:.0%} | {how.get(group, '')} |"
        )

    shortlist = disagreements(rows)
    lines += [
        "",
        f"### 진단 대상 {len(shortlist)}장 (3단계에서 이것만 본다)",
        "",
    ]
    if not shortlist:
        lines.append("없음 — 우리 숫자와 실제가 전부 일치한다.")
    else:
        for row in shortlist:
            lines.append(
                f"- `marked/{row.stem}.jpg` — 우리: 사람 {row.det_person}명, "
                f"점유 {row.seat_occupied}석, 모름 {row.seat_unknown}석 "
                f"/ 실제: {row.truth}명 ({row.detector_gap:+d})"
            )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="seatnow.py 가 쓴 JSONL")
    parser.add_argument("--judge", type=Path, help="judge_frames.py 결과 폴더")
    parser.add_argument("--output", type=Path, help="마크다운 저장 경로 (기본: 화면 출력)")
    parser.add_argument("--title", default=None, help="보고서 제목")
    args = parser.parse_args(argv)

    records = load_jsonl(args.log)
    judgements = load_judgements(args.judge)
    rows = build_rows(records, judgements)

    title = args.title or f"# 검출 판독표 — {args.log.name}"
    report = "\n\n".join(
        [title, render_summary(rows, records), "## 판독표", render_table(rows)]
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"판독표: {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest tests.test_inspect_run -v`
Expected: PASS — 29개

- [ ] **Step 5: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS — 기존 291개 + 신규 70개 (frame_dump 13, judge_frames 28, inspect_run 29)

- [ ] **Step 6: 커밋**

```bash
git add inspect_run.py tests/test_inspect_run.py
git commit -m "feat: inspect_run.py 판독표·요약·진단 대상 목록

Codex 판독이 없어도 '실제' 칸이 빈 표가 나온다. 사람이 손으로 채워도
같은 표가 되게 한다 - 자동화가 없어도 하네스는 성립해야 한다.

사유 코드는 무엇이 고치는가로 묶는다. geometry면 구제 경로,
model이면 파인튜닝이다."
```

---

### Task 7: 실제 실행과 기록

**Files:**
- Create: `sample_results/inspect/angle{1..4}.md`
- Modify: `plan.md` (§0-a 상태표, §0-c에 결과 절 추가), `README.md` (도구 목록)

**Interfaces:**
- Consumes: Task 2·4·6의 모든 CLI
- Produces: 설계 문서 §10의 "한 문장"

**주의:** 이 태스크는 코드가 아니라 **측정**이다. 모델을 실제로 돌리므로 각도당 몇 분 걸린다.

- [ ] **Step 1: 네 각도를 돌린다**

```bash
for a in 1 2 3 4; do
  ./venv/Scripts/python.exe seatnow.py sample_raw/cafe_sample_angle${a}.mov \
    --no-video --log-detections \
    --frame-dir frames/angle${a} \
    --log sample_results/angle${a}.jsonl
done
```

Expected: 각도마다 `[  0.0s] tables=... occupied=...` 줄이 흐르고 끝까지 간다.
**중간에 멈추면 거기서 조사한다** — 그 자체가 이 하네스가 찾으려던 결과다.
`--layout`을 주지 않는 것이 의도다 (설계 §2).

- [ ] **Step 2: 사진이 나왔는지 확인한다**

```bash
for a in 1 2 3 4; do
  echo "angle${a}: clean $(ls frames/angle${a}/clean | wc -l) / marked $(ls frames/angle${a}/marked | wc -l)"
done
```

Expected: 각도당 5~7장씩, 합계 약 22장. clean과 marked 장수가 같다.

- [ ] **Step 3: 사람이 먼저 센다 — 눈가림 (설계 §5-2)**

**JSONL과 `marked/`를 열기 전에 한다.** 각도마다 아무 사진이나 하나씩, 총 4~5장을 `clean/`에서 골라 사람 수를 세고 적어둔다.

```bash
mkdir -p sample_results/inspect
```

`sample_results/inspect/human_spotcheck.md`에 적는다:

```markdown
# 사람 대조 (Codex 판독 검증용)

**주의: 이 파일은 JSONL과 marked/ 를 보기 전에 채운다.**

| 사진 | 사람 수 | 메모 |
|---|---:|---|
| frames/angle1/clean/t0015.0s.jpg | ? | |
| frames/angle2/clean/t0030.0s.jpg | ? | |
| frames/angle3/clean/t0015.0s.jpg | ? | |
| frames/angle4/clean/t0045.0s.jpg | ? | |
```

- [ ] **Step 4: Codex에게 세게 한다**

```bash
for a in 1 2 3 4; do
  ./venv/Scripts/python.exe judge_frames.py frames/angle${a} \
    --output frames/angle${a}/judge
done
```

Expected: 사진마다 `[t0015.0s] people=3 uncertain=False` 같은 줄. 마지막에 실패·불확실 장수 요약.

- [ ] **Step 5: Codex 판독을 사람 대조와 맞춰본다 (설계 §5-3)**

Step 3에서 고른 4~5장에 대해 `frames/angle*/judge/<stem>.json`의 `people_total`을 사람이 센 숫자와 비교한다.

**어긋나면 여기서 멈춘다.** Codex 판독 전체를 폐기하고, 표의 `실제` 칸은 사람이 손으로 채운다 (`inspect_run.py`는 판독 없이도 빈 칸 표를 낸다).

- [ ] **Step 6: 판독표를 낸다**

```bash
for a in 1 2 3 4; do
  ./venv/Scripts/python.exe inspect_run.py sample_results/angle${a}.jsonl \
    --judge frames/angle${a}/judge \
    --output sample_results/inspect/angle${a}.md \
    --title "# 검출 판독표 — angle${a}"
done
```

Expected: 각 파일에 요약(재현율 2개 + 사유 분포 + 진단 대상)과 판독표.

- [ ] **Step 7: 어긋난 사진을 Codex와 진단한다 (3단계)**

각 판독표의 "진단 대상" 목록이 비어 있지 않으면:

```bash
orca terminal create --worktree active --title "JUDGE" --command "codex" --focus
```

그 터미널에 붙여넣을 프롬프트:

```
아래 사진들은 우리 좌석 감지 프로그램이 판정한 순간이고, 박스와 판정이 그려져 있다.
각 사진에서 프로그램이 무엇을 놓쳤고 무엇을 잘못 그렸는지 말해달라.
특히 (1) 사람인데 안 잡힌 것 (2) 사람이 아닌데 사람으로 잡힌 것
(3) 앉았는데 서 있다고 본 것.
전부 본 뒤에는 반복되는 패턴이 있으면 그것도 말해달라.

<진단 대상 목록을 여기 붙인다 — inspect_run.py 요약에서 그대로 복사>
```

- [ ] **Step 8: 한 문장으로 결론을 쓴다**

설계 §10-4의 형식이다. `plan.md` §0-c 아래에 절을 추가한다:

```markdown
### 2026-09-01 — 검출 검사 하네스 첫 실행 (T14 완결)

`seatnow.py --frame-dir` + `judge_frames.py` + `inspect_run.py`로 angle1~4를
레이아웃 없이 끝까지 돌린 결과. 판독표: `sample_results/inspect/angle{1..4}.md`

| 각도 | tick | 검출 재현율 | 포즈 재현율 | UNKNOWN geometry | UNKNOWN model |
|---|---:|---:|---:|---:|---:|
| angle1 | | | | | |
| angle2 | | | | | |
| angle3 | | | | | |
| angle4 | | | | | |

**결론 한 문장**: (여기에 §10-4 형식으로 쓴다)

**따라오는 다음 행동**: (파인튜닝 / 구제 경로 / 카메라 각도 중 하나)
```

`plan.md` §0-a 상태표의 T14 줄을 `✅ 완료 (2026-09-01)`로 바꾼다.

- [ ] **Step 9: README에 도구를 추가한다**

`README.md`의 도구 목록에 세 줄을 추가한다 — `bench_decode.py`·`check_edge.py`가 적힌 형식을 그대로 따른다.

- [ ] **Step 10: 커밋**

```bash
git add sample_results/inspect plan.md README.md
git commit -m "docs: 검출 검사 하네스 첫 실행 결과 (T14 완결)

angle1~4를 레이아웃 없이 끝까지 돌렸다. 판정 파이프라인이 실제 영상에서
도는 것을 처음 확인했고, 검출/포즈 두 층의 재현율에 숫자가 박혔다.

<결론 한 문장>"
```

**주의:** `frames/`와 `sample_results/*.jsonl`은 커밋하지 않는다. `.gitignore`에 `frames/`가 없으면 추가한다.

---

## 검토 메모

**설계 문서 대응**

| 설계 절 | 태스크 |
|---|---|
| §3-A 사진 두 장 | Task 1, 2 |
| §3-B Codex 세기 | Task 3, 4 |
| §3-C 판독표 | Task 5, 6 |
| §4 데이터 흐름 | Task 7 |
| §5 눈가림 규칙 | Task 3(프롬프트 테스트), Task 7 Step 3·5 |
| §6 세기/진단 분리 | Task 3(`--ephemeral`), Task 7 Step 7 |
| §7 프레임 방식 유지 | Global Constraints (건드리지 않음) |
| §8 오류 처리 | Task 1 Step 1(저장 실패), Task 4(호출 실패), Task 6(판독 없어도 표) |
| §9 테스트 | Task 1·3·4·5·6 |
| §10 완료 기준 | Task 7 |

**계획 검토에서 고친 것**

계획을 쓸 때 사유 코드가 `record["tables"][].reason_code`에 있다고 가정했는데, 실제로는 `record["seat_report"]["seats"]`이고 일반 테이블과 바 구역의 **형태가 다르다** (`seatnow_report.py:150-182`). 그대로 갔으면 사유 분포가 항상 0으로 나왔을 것이다. Task 6의 구현과 테스트를 양쪽 형태를 다 읽도록 고쳤다.

파서 생성 함수 이름(`build_parser()`, `seatnow.py:54`)도 확인했다. **남은 미확인 사항은 없다.**
