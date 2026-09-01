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
