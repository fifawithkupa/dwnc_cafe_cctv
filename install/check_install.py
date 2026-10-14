"""설치 당일 연결 확인 — 세 가지를 한 번에 본다.

    python3 -m install.check_install                # 박스에서
    python3 -m install.check_install --shot /tmp/shot.jpg

전부 연결한 직후, **자리를 잡기 전에** 돌린다.  연결이 안 된 상태로 카메라
자리를 고르면 그 시간이 전부 헛일이 되기 때문이다 (문서/2_절차/카페설치당일.md 3장).

보는 것 셋:

1. **카메라가 연결됐나** — 영상이 들어오나, 크기·코덱·초당 장수가 맞나
2. **판정이 도나** — 15초마다 한 줄씩 찍히나, 한 판정이 예산 안에 끝나나
3. **그린 좌석이 판정되나** — 네모가 화면 밖으로 빠지지 않았나
4. **서버로 나가나** — 앱이 읽는 줄이 방금 갱신됐나, 그게 이 카페 줄이 맞나

실패한 줄은 **무엇을 하라**까지 적는다.  이 글을 읽는 사람은 카메라를 들고
사다리 위에 있지, 코드를 읽을 상황이 아니다.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]

#: 4MP 카메라를 쓰기로 했다 (문서/3_배경/SEATNOW_전체정리.md §10).  이보다 작으면
#: 먼 자리의 컵·가방이 안 잡힌다.
CAMERA_MIN_HEIGHT = 1080
#: 판단 주기 15초의 절반.  한 판정이 이 안에 끝나야 다음 판정이 안 밀린다.
TICK_BUDGET_S = 7.5
#: 앱은 45초 넘게 안 바뀌면 지도를 "확인 중"으로 돌린다.  그 절반을 기준으로 본다.
MAX_PUBLISH_AGE_S = 25
#: 모델을 처음 읽는 판정 하나는 느리다.  그걸로 불합격을 내지 않는다.
WARMUP_TICKS = 1

_TICK_RE = re.compile(
    r"\[\s*([\d.]+)s\]\s+tables=(\d+).*?ignore=(\d+).*?inference=(\d+)ms"
)
_HWACCEL_RE = re.compile(r"하드웨어 디코딩 켜짐 \(([^)]+)\)")


@dataclass(frozen=True)
class Tick:
    at_seconds: float
    tables: int
    ignored: int
    inference_ms: int


# ── 1. 카메라 ────────────────────────────────────────────────────────────────


def camera_verdict(stream: Optional[Dict[str, object]]) -> Tuple[bool, List[str]]:
    """영상이 들어오나, 크기·코덱·초당 장수가 쓸 만한가."""
    if not stream:
        return False, [
            "영상이 안 들어온다.",
            "  · PoE 인젝터 전원이 꽂혔나 (어댑터는 인젝터 상자 안에 있다)",
            "  · 긴 랜선이 인젝터의 `POE OUT` 에 꽂혔나 — `LAN IN` 과 바꿔 꽂는 것이 제일 흔하다",
            "  · 카메라는 켜지는 데 1분 걸린다. 방금 꽂았으면 기다렸다 다시",
            "  · 카메라 주소가 바뀌었을 수 있다 → \"카메라 찾아줘\" 로 다시 찾는다",
        ]

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    codec = str(stream.get("codec") or "?")
    fps = float(stream.get("fps") or 0.0)

    lines = [f"영상 들어옴 — {width}x{height}, {codec}, 초당 {fps:g}장"]
    ok = True
    if height < CAMERA_MIN_HEIGHT:
        ok = False
        lines.append(
            f"  ✗ 화면이 너무 작다 (세로 {height}, 적어도 {CAMERA_MIN_HEIGHT} 이어야 한다)."
            " 카메라 설정에서 메인 스트림을 2560x1440 으로 올린다"
        )
    if fps <= 0:
        ok = False
        lines.append("  ✗ 초당 장수가 0 이다. 카메라를 뽑았다 꽂고 1분 뒤 다시 본다")
    return ok, lines


# ── 2. 판정 ──────────────────────────────────────────────────────────────────


def parse_tick_lines(lines: Sequence[str]) -> List[Tick]:
    """로그에서 판정 줄만 골라낸다."""
    ticks: List[Tick] = []
    for line in lines:
        found = _TICK_RE.search(line)
        if found:
            ticks.append(
                Tick(
                    float(found.group(1)),
                    int(found.group(2)),
                    int(found.group(3)),
                    int(found.group(4)),
                )
            )
    return ticks


def find_hwaccel(lines: Sequence[str]) -> Optional[str]:
    for line in lines:
        found = _HWACCEL_RE.search(line)
        if found:
            return found.group(1)
    return None


def judging_verdict(ticks: Sequence[Tick], hwaccel: Optional[str]) -> Tuple[bool, List[str]]:
    """판정이 돌고 있나, 한 판정이 예산 안에 끝나나."""
    if not ticks:
        return False, [
            "판정 줄이 하나도 없다.",
            "  · 서비스가 떠 있나: `systemctl --user status seatnow`",
            "  · 방금 다시 시작했으면 첫 판정까지 1분 걸린다",
            "  · 카메라가 안 들어오면 판정도 안 돈다 → 위 1번부터",
        ]

    lines = [f"판정 돌고 있음 — 최근 {len(ticks)}번, 자리 {ticks[-1].tables}개"]
    ok = True

    judged = list(ticks[WARMUP_TICKS:]) or list(ticks)
    times = sorted(tick.inference_ms for tick in judged)
    slowest = times[-1]
    usual = times[len(times) // 2]
    # 보통값을 같이 보여준다.  재시작 직후 모델을 처음 읽는 판정 하나가 느린데,
    # 가장 느린 값만 보여주면 현장에서 "느려졌다"고 오해한다 (2026-10-13).
    lines.append(
        f"  판정 한 번 보통 {usual / 1000:.1f}초, 가장 느렸을 때 {slowest / 1000:.1f}초"
        f" (한도 {TICK_BUDGET_S}초)"
    )
    if slowest / 1000.0 > TICK_BUDGET_S:
        ok = False
        lines.append(
            f"  ✗ 한도 {TICK_BUDGET_S}초를 넘는다. 이대로 두면 판정이 밀린다."
            " 박스에서 다른 무거운 프로그램이 도는지 본다"
        )

    if hwaccel:
        lines.append(f"  하드웨어 디코딩 켜짐 ({hwaccel})")
    else:
        lines.append(
            "  ⚠ 하드웨어 디코딩이 꺼져 있다 (소프트웨어로 돈다)."
            " 4MP 는 이대로도 예산 안이지만, 설치 기록에 적어두고 집에서 원인을 본다"
        )
    return ok, lines


def seats_verdict(ticks: Sequence[Tick]) -> Tuple[bool, List[str]]:
    """그린 좌석이 실제로 판정되고 있나.

    좌석 네모가 지금 화면과 안 맞으면 전부 "카메라가 못 보는 자리"로 빠지고
    **앱에는 한 자리도 안 나간다.**  판정 줄은 멀쩡해 보이므로 따로 본다
    (2026-10-13 집에서 실제로 겪었다: 카메라를 옮겼더니 12자리가 전부 빠졌다).
    """
    if not ticks:
        return True, []  # 판정 자체가 없으면 2번에서 이미 걸렸다
    last = ticks[-1]
    judged = last.tables - last.ignored
    if last.tables and last.ignored >= last.tables:
        return False, [
            f"✗ 그린 좌석 {last.tables}개가 **전부** 화면 밖으로 빠졌다 — 앱에 한 자리도 안 나간다.",
            "  · 좌석 네모가 지금 카메라 화면과 안 맞는다. 카메라를 움직였으면 당연한 일이다",
            "  · 자리를 다 잡은 **뒤에** 좌석을 다시 그린다 (문서/2_절차/카페설치당일.md 7장)",
            "  · 다른 매장에서 쓰던 좌석 파일이 그대로 들어 있지는 않은지 본다",
        ]
    lines = [f"좌석 {judged}개가 판정되고 있다 (그린 것 {last.tables}개)"]
    ok = True
    if last.ignored:
        ok = False
        lines.append(
            f"  ✗ 그중 {last.ignored}개가 화면 밖으로 빠졌다."
            " 그 자리는 앱에 안 나간다 — 카메라를 돌려 화면 안으로 넣거나, 포기할 자리면 적어둔다"
        )
    return ok, lines


# ── 4. 서버 전송 ─────────────────────────────────────────────────────────────


def publish_verdict(
    row: Optional[Dict[str, object]],
    age_seconds: Optional[float],
    expected_cafe_id: str,
) -> Tuple[bool, List[str]]:
    """앱이 읽는 줄이 방금 갱신됐나, 그게 이 카페 줄이 맞나."""
    if not row:
        return False, [
            f"서버에 `{expected_cafe_id}` 줄이 없다 — 앱에는 아무것도 안 뜬다.",
            "  · 박스 설정의 카페 id 와 서버의 카페 줄이 같은 이름인가",
            "  · 그 카페 줄을 앱 팀이 먼저 넣어야 한다. 없으면 서버가 박스의 쓰기를 전부 거부한다",
            "  · 박스 설정 다섯 값이 다 차 있나 (`deploy/seatnow.env`)",
        ]

    cafe_id = str(row.get("cafe_id") or "?")
    lines = [
        f"서버에 쓰고 있음 — 카페 `{cafe_id}`, 자리 {row.get('total_tables')}개"
        f" (사용중 {row.get('busy_tables')} · 빈자리 {row.get('free_tables')}), 박스 코드 {row.get('box_version')}"
    ]
    ok = True

    if not int(row.get("total_tables") or 0):
        ok = False
        lines.append(
            "  ✗ **자리가 0개로 나가고 있다.** 앱에는 아무 자리도 안 보인다."
            " 위 좌석 줄을 먼저 본다 — 그린 네모가 화면 밖으로 빠졌을 때 이렇게 된다"
        )

    if cafe_id != expected_cafe_id:
        ok = False
        lines.append(
            f"  ✗ **다른 카페 줄에 쓰고 있다.** 지금 `{cafe_id}`, 와야 할 곳은 `{expected_cafe_id}`."
            " 박스 설정의 카페 id 와 서버 `boxes` 줄을 둘 다 같은 값으로 바꾼다"
        )

    if age_seconds is None or age_seconds > MAX_PUBLISH_AGE_S:
        ok = False
        shown = "알 수 없음" if age_seconds is None else f"{age_seconds:.0f}초 전"
        lines.append(
            f"  ✗ 마지막 갱신이 {shown} 이다 (한도 {MAX_PUBLISH_AGE_S}초)."
            " 박스 인터넷이 끊겼거나 전송이 막혔다. 로그에서 `Supabase 전송 실패` 를 찾는다"
        )
    else:
        lines.append(f"  마지막 갱신 {age_seconds:.0f}초 전")

    if str(row.get("status")) == "gap":
        ok = False
        lines.append(
            "  ✗ 카메라가 끊긴 상태로 쏘고 있다 (`gap`). 앱에는 지도 전체가 \"확인 중\" 으로 뜬다."
            " 위 1번 카메라부터 본다"
        )
    return ok, lines


def overall_verdict(results: Sequence[bool]) -> bool:
    return all(results)


# ── 실제로 읽어오는 부분 (여기는 장비가 필요하다) ───────────────────────────


def read_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def hide_password(text: str) -> str:
    """로그·화면에 카메라 비밀번호가 찍히지 않게 한다."""
    return re.sub(r"(rtsp://[^:/@]+):[^@]*@", r"\1:***@", text)


def probe_camera(url: str, timeout: float = 25.0) -> Optional[Dict[str, object]]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-rtsp_transport", "tcp",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,codec_name,avg_frame_rate",
                "-of", "json", url,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout).get("streams") or []
    except json.JSONDecodeError:
        return None
    if not streams:
        return None
    stream = streams[0]
    rate = str(stream.get("avg_frame_rate") or "0/1")
    try:
        top, _, bottom = rate.partition("/")
        fps = float(top) / float(bottom or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "codec": stream.get("codec_name"),
        "fps": fps,
    }


def save_shot(url: str, path: Path, timeout: float = 30.0) -> bool:
    """화면을 눈으로 보려고 한 장 뜬다 (영상이 아니라 정지 사진)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
             "-i", url, "-frames:v", "1", "-q:v", "4", "-y", str(path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and path.exists()


def read_log(lines: int = 200) -> List[str]:
    try:
        result = subprocess.run(
            ["journalctl", "--user", "-u", "seatnow", "-n", str(lines), "-o", "cat"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return result.stdout.splitlines()


def read_live_row(env: Dict[str, str]) -> Tuple[Optional[Dict[str, object]], Optional[float]]:
    """앱이 읽는 줄을 박스와 같은 자격으로 읽는다."""
    url = env.get("SEATNOW_SUPABASE_URL", "").rstrip("/")
    anon = env.get("SEATNOW_SUPABASE_ANON_KEY", "")
    cafe_id = env.get("SEATNOW_CAFE_ID", "")
    if not (url and anon and cafe_id):
        return None, None
    try:
        import requests
    except ImportError:
        return None, None
    try:
        response = requests.get(
            f"{url}/rest/v1/cafe_live",
            params={"cafe_id": f"eq.{cafe_id}", "select": "*"},
            headers={"apikey": anon, "Authorization": f"Bearer {anon}"},
            timeout=15,
        )
    except Exception:
        return None, None
    if response.status_code != 200:
        return None, None
    rows = response.json()
    if not rows:
        return None, None
    row = rows[0]
    age = None
    stamp = str(row.get("updated_at") or "")
    if stamp:
        try:
            from datetime import datetime, timezone

            text = stamp.replace("Z", "+00:00")
            if "." in text:  # 소수점 자리가 6자리를 넘으면 파이썬이 못 읽는다
                head, _, tail = text.partition(".")
                digits = "".join(ch for ch in tail if ch.isdigit())[:6]
                rest = tail[len(digits):] if tail[len(digits):].startswith(("+", "-")) else "+00:00"
                text = f"{head}.{digits}{rest}"
            moment = datetime.fromisoformat(text)
            age = (datetime.now(timezone.utc) - moment).total_seconds()
        except ValueError:
            age = None
    return row, age


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env", type=Path, default=PROJECT_DIR / "deploy" / "seatnow.env")
    parser.add_argument("--shot", type=Path, help="카메라 화면을 이 경로에 한 장 저장한다")
    parser.add_argument("--cafe-id", help="와야 할 카페 id (기본: 박스 설정값)")
    args = parser.parse_args(argv)

    env = read_env(args.env)
    url = env.get("SEATNOW_URL", "")
    expected = args.cafe_id or env.get("SEATNOW_CAFE_ID", "")

    print("= 설치 연결 확인 =", flush=True)
    print(f"  설정 파일: {args.env}")
    print(f"  카메라: {hide_password(url) or '(설정 없음)'}")
    print(f"  카페 id: {expected or '(설정 없음)'}")
    print()

    results: List[bool] = []

    print("1. 카메라 연결")
    ok, lines = camera_verdict(probe_camera(url) if url else None)
    for line in lines:
        print(f"   {line}")
    results.append(ok)
    if args.shot and url:
        if save_shot(url, args.shot):
            print(f"   화면 한 장 저장: {args.shot}  ← 눈으로 본다")
        else:
            print("   화면을 못 떴다")
    print()

    print("2. 판정")
    log = read_log()
    ok, lines = judging_verdict(parse_tick_lines(log), find_hwaccel(log))
    for line in lines:
        print(f"   {hide_password(line)}")
    results.append(ok)
    print()

    print("3. 그린 좌석")
    ok, lines = seats_verdict(parse_tick_lines(log))
    for line in lines or ["(아직 볼 것이 없다 — 좌석을 그리기 전이면 정상)"]:
        print(f"   {line}")
    results.append(ok)
    print()

    print("4. 서버 전송 (앱이 읽는 줄)")
    row, age = read_live_row(env)
    ok, lines = publish_verdict(row, age, expected)
    for line in lines:
        print(f"   {line}")
    results.append(ok)
    print()

    if overall_verdict(results):
        print("=> 셋 다 통과. 카메라 자리를 잡아도 된다.")
        return 0
    print("=> 통과 못 한 줄의 ✗ 를 먼저 해결한다. 자리 잡기는 그 다음이다.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
