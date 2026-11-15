"""git 으로 안 넘어가는 입력(영상·박스 기록·모델)을 워크트리에 붙인다.

입력 묶음(zip)을 한 곳에 풀어 두고, 워크트리마다 이것만 한 번 돌린다:

    python -m checks.link_inputs D:/seatnow_inputs

묶음 안의 파일을 **같은 상대 경로**로 이 워크트리에 붙인다.
(묶음의 sample_raw/x.mp4 → 워크트리의 sample_raw/x.mp4)

- 같은 드라이브면 하드링크라 용량을 더 안 먹는다 — 워크트리를 10개 만들어도 영상은 한 벌.
  다른 드라이브면 복사한다.
- 이미 같은 크기의 파일이 있으면 건너뛴다. 크기가 다르면 덮지 않고 알려준다.
- 하드링크 파일을 고치면 다른 워크트리에서도 바뀐다. 입력은 읽기만 할 것.
- 묶음 맨 위의 안내 파일(README*, 묶음목록*)은 붙이지 않는다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
SKIP_TOP = ("README", "묶음목록")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundle", type=Path, help="입력 묶음(zip)을 푼 폴더")
    p.add_argument("--dest", type=Path, default=PROJECT_DIR, help="기본: 이 워크트리")
    args = p.parse_args(argv)

    bundle = args.bundle.resolve()
    if not bundle.is_dir():
        raise SystemExit(f"폴더가 없다: {bundle}")
    linked = copied = skipped = conflicts = 0
    for src in sorted(bundle.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(bundle)
        if len(rel.parts) == 1 and rel.name.startswith(SKIP_TOP):
            continue
        dst = args.dest / rel
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                skipped += 1
            else:
                conflicts += 1
                print(f"  크기가 다른 파일이 이미 있어 그대로 둠: {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            shutil.copy2(src, dst)
            copied += 1
    print(f"하드링크 {linked} · 복사 {copied} · 이미 있음 {skipped} · 충돌 {conflicts}  → {args.dest}")
    return 1 if conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
