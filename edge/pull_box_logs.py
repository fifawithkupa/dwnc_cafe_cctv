"""박스의 판정 기록을 노트북으로 받아온다 — 인터넷이 없는 매장에서 1주일에 한 번.

    python -m edge.pull_box_logs                        # hugo@uhho.local → results/box_pull/moonq/
    python -m edge.pull_box_logs --host 192.168.0.2     # 이름으로 못 찾을 때

노트북을 매장 공유기에 꽂고 돌린다.  받아오는 것은 박스의
``results/live/YYYY-MM-DD.jsonl`` 뿐이다 — 15초마다 한 줄, **계산값만** 들어 있다
(자리별 판정·사람 자세·처리 시간).  사진·영상은 박스가 애초에 만들지 않는다
(``--no-video``).

* 박스에서 gzip 으로 눌러서 보낸다.  하루치가 100MB 쯤이라 그대로 옮기면 느리다.
* 이미 받은 날은 크기가 같으면 건너뛴다.  아직 쌓이는 중인 오늘 파일은 다음번에
  더 커져 있으므로 그때 다시 받는다.
* **박스 시계와 노트북 시계의 차이를 매번 적는다.**  박스는 인터넷으로 시계를
  맞추는데, 인터넷이 없으면 틀어진 채로 간다 (2026-10-14 문큐에서 약 하루 늦었다).
  기록에 찍힌 시각과 파일 이름의 날짜는 박스 시계 기준이므로, 분석할 때 이 차이로
  바로잡는다.
* 박스의 파일은 지우지 않는다.  박스가 ``SEATNOW_KEEP_DAYS`` (30일) 뒤에 알아서 지운다.
  그래서 한 주를 건너뛰어도 다음 주에 같이 받아올 수 있다.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from edge import tolerant_stdout

PROJECT_DIR = Path(__file__).resolve().parents[1]

#: 박스가 하루에 하나씩 만드는 판정 기록 (``--log-dir`` 의 LiveLogRotator).
DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")
REMOTE_LOG_DIR = "seatnow/results/live"
MANIFEST_NAME = "pulled.json"


def parse_listing(text: str) -> Tuple[float, Dict[str, int]]:
    """박스가 돌려준 목록을 읽는다: 첫 줄은 박스 시계(초), 나머지는 ``이름 크기``."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("박스가 아무것도 돌려주지 않았습니다")
    box_epoch = float(lines[0])
    files: Dict[str, int] = {}
    for line in lines[1:]:
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not DAY_FILE_RE.match(parts[0]):
            continue
        files[parts[0]] = int(parts[1])
    return box_epoch, files


def files_to_pull(remote: Mapping[str, int], manifest_files: Mapping[str, Mapping]) -> List[str]:
    """새 날짜이거나, 지난번에 받은 뒤 더 쌓인 날만 고른다."""
    wanted = []
    for name, size in remote.items():
        known = manifest_files.get(name)
        if known is None or int(known.get("bytes", -1)) != size:
            wanted.append(name)
    return sorted(wanted)


def clock_offset(box_epoch: float, laptop_before: float, laptop_after: float) -> float:
    """박스 시계 − 노트북 시계 (초).  음수면 박스가 늦다.

    박스의 ``date`` 는 ssh 왕복 사이 어딘가에서 찍혔으므로 왕복의 가운데와 비교한다.
    """
    return box_epoch - (laptop_before + laptop_after) / 2.0


def describe_offset(seconds: float) -> str:
    if abs(seconds) < 60:
        return "박스 시계가 맞습니다 (1분 이내)"
    total_minutes = int(round(abs(seconds) / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    amount = f"{hours}시간 {minutes}분" if hours else f"{minutes}분"
    direction = "늦습니다" if seconds < 0 else "빠릅니다"
    return f"박스 시계가 {amount} {direction} — 기록의 시각을 이만큼 바로잡아 읽어야 합니다"


def load_manifest(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}, "pulls": []}


def _ssh_base(user: str, host: str, host_key_alias: Optional[str]) -> List[str]:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    if host_key_alias:
        command += ["-o", f"HostKeyAlias={host_key_alias}"]
    return command + [f"{user}@{host}"]


def _list_remote(ssh: Sequence[str], remote_dir: str) -> str:
    script = (
        f"cd {remote_dir} && date +%s.%N && "
        "for f in 20??-??-??.jsonl; do [ -f \"$f\" ] && stat -c '%n %s' \"$f\"; done; true"
    )
    completed = subprocess.run(
        [*ssh, script], capture_output=True, text=True, timeout=60, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ssh 실패 (코드 {completed.returncode})")
    return completed.stdout


def _pull_one(ssh: Sequence[str], remote_dir: str, name: str, destination: Path) -> int:
    """박스에서 눌러 받아 풀어본 크기를 돌려준다.  깨진 파일은 남기지 않는다."""
    partial = destination.with_name(destination.name + ".part")
    with open(partial, "wb") as handle:
        completed = subprocess.run(
            [*ssh, f"gzip -c {remote_dir}/{name}"],
            stdout=handle, stderr=subprocess.PIPE, timeout=1800, check=False,
        )
    if completed.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(completed.stderr.decode("utf-8", "replace").strip())
    size = 0
    with gzip.open(partial, "rb") as unpacked:
        while True:
            chunk = unpacked.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
    partial.replace(destination)
    return size


def main(argv: Optional[Sequence[str]] = None) -> int:
    tolerant_stdout()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="uhho.local", help="박스 주소 (기본: uhho.local)")
    parser.add_argument("--user", default="hugo")
    parser.add_argument("--host-key-alias", default="uhho.local",
                        help="ssh 가 박스를 알아보는 이름. 매장마다 박스 주소가 달라도 같은 박스로 본다")
    parser.add_argument("--cafe", default="moonq", help="받아둘 폴더 이름 (카페 id)")
    parser.add_argument("--out", type=Path, default=None,
                        help="받을 곳 (기본: results/box_pull/<카페>)")
    parser.add_argument("--remote-dir", default=REMOTE_LOG_DIR)
    args = parser.parse_args(argv)

    out_dir = args.out or PROJECT_DIR / "results" / "box_pull" / args.cafe
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)

    ssh = _ssh_base(args.user, args.host, args.host_key_alias)
    before = time.time()
    try:
        listing = _list_remote(ssh, args.remote_dir)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"❌ 박스에 붙지 못했습니다: {error}")
        print("   · 노트북이 매장 공유기에 (유선으로) 붙어 있나")
        print("   · 이름으로 안 되면 주소로: python -m edge.pull_box_logs --host 192.168.0.2")
        return 1
    after = time.time()
    box_epoch, remote = parse_listing(listing)
    offset = clock_offset(box_epoch, before, after)

    wanted = files_to_pull(remote, manifest["files"])
    print(f"박스에 있는 날: {len(remote)}개 · 새로 받을 날: {len(wanted)}개")
    print(describe_offset(offset))

    pulled = []
    for name in wanted:
        destination = out_dir / f"{name}.gz"
        print(f"  {name} ({remote[name] / 1e6:.0f}MB) 받는 중...", flush=True)
        try:
            size = _pull_one(ssh, args.remote_dir, name, destination)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
            print(f"  ❌ {name} 실패: {error}")
            continue
        manifest["files"][name] = {
            "bytes": size,
            "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        pulled.append(name)
        print(f"  ✅ {name} → {destination.name} ({destination.stat().st_size / 1e6:.1f}MB)")

    manifest["pulls"].append({
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": args.host,
        "box_clock_offset_s": round(offset, 1),
        "pulled": pulled,
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = len(wanted) - len(pulled)
    print(f"받은 곳: {out_dir}")
    if failed:
        print(f"❌ {failed}개 날을 못 받았습니다. 다시 돌리면 그 날만 다시 받습니다.")
        return 1
    print("✅ 끝. 박스의 기록은 그대로 두었습니다 (30일 뒤 박스가 지웁니다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
