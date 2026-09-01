"""Count people in each still with Codex, blind to what SeatNow decided.

The harness that grades detection cannot say "the detector missed people"
without something that knows how many people were actually there.  Codex does
that here -- but only if it never sees our answer first.  So each still gets
its own throwaway session (no memory of the previous count to anchor to), the
sandbox is read-only, and the prompt names one file and forbids opening any
other.

Codex opens the image itself through its ``view_image`` tool, which is on by
default (``codex features list``).  No API key is involved; the ChatGPT login
the CLI already holds is enough.

    python judge_frames.py frames/angle1 --output frames/angle1/judge
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

from frame_dump import CLEAN_DIR


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


def resolve_codex(codex: str) -> str:
    """Find the Codex CLI once, up front, or say so plainly.

    subprocess does not apply PATHEXT, so a bare ``codex`` never matches the
    npm shim (``codex.CMD``) on Windows.  Resolving here rather than per call
    also keeps one missing tool from being reported as N judging failures.
    """
    resolved = shutil.which(codex)
    if resolved is None:
        raise FileNotFoundError(
            f"Codex CLI를 찾을 수 없다: {codex!r}. "
            "설치되어 있으면 --codex 로 실행 파일 경로를 직접 준다."
        )
    return resolved


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
    codex = resolve_codex(args.codex)
    results = judge_directory(
        args.frame_dir, output_dir, codex=codex, timeout=args.timeout
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
