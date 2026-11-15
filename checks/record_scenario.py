"""혼자서 30분 시험 영상을 찍는다 — 노트북이 소리로 다음 동작을 불러준다.

문큐처럼 사람 한 명이 손님 역할을 다 해야 할 때 쓴다. 노트북을 카페 공유기에
유선으로 꽂고(노트북 네트워크 설정은 건드리지 않는다 — 공유기가 주소를 준다):

    venv/Scripts/python.exe -m checks.record_scenario --host 192.168.0.2

하는 일:
  1. 박스에서 카메라 주소(deploy/seatnow.env 의 SEATNOW_URL)와 박스 시계를 읽는다
  2. 카메라 한 장을 받아 좌석 이름(T1, BAR6-1 …)을 그린 지도를 띄운다 — 어느 테이블이
     T4 인지 보고 시작하라고
  3. 카메라 영상을 30분 그대로 저장한다 (다시 압축하지 않는다)
  4. 동작이 바뀌기 15초 전과 바뀌는 순간 한국어 음성으로 불러준다
  5. 영상 옆에 대본 파일(.scenario.json)을 남긴다 — 구간마다 "사람 눈으로 본 정답"이
     들어 있어서, 나중에 정답지를 백지에서 만들 필요가 없다

영상은 노트북 sample_raw/ 에만 둔다 (git 제외). 서버로 올리지 않는다.

시험:  venv/Scripts/python.exe -m checks.record_scenario --dry-run --speed 60
       (카메라 없이 가짜 영상으로, 60배 빠르게 대본만 돌려본다)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[1]
SEATS = ["T1", "T2", "T3", "T4", "T5", "T7",
         "BAR6-1", "BAR6-2", "BAR6-3", "BAR6-4", "BAR6-5", "BAR6-6"]
# 동작을 바꾸는 데 걸리는 시간. 구간 앞 이만큼은 채점에서 뺀다.
TRANSITION_SECONDS = 20

# (시작초, 길이초, 부를 말, 그 구간의 정답 {좌석: occupied|ignore}, 무엇을 시험하나)
# 정답에 안 적힌 좌석은 전부 empty.  ignore = 사람 눈으로도 어느 자리라 하기 애매한 것.
STEPS: List[dict] = []


def _step(seconds: int, say: str, truth: Dict[str, str], tests: str) -> None:
    start = sum(s["seconds"] for s in STEPS)
    STEPS.append({"start": start, "seconds": seconds, "say": say,
                  "truth": truth, "tests": tests})


# 녹화 내내 깔아 두는 "자리 맡기 물건" — 혼자 찍어도 늘 서너 자리가 사용중이게.
# 이게 없으면 12자리 중 11자리가 늘 빈자리라 "무조건 빈자리" 방식이 90%로 이긴다.
BG = {"T2": "occupied", "T7": "occupied", "BAR6-5": "occupied"}

_step(180, "화면 밖으로 나가세요. 아무것도 없는 빈 매장, 3분.",
      {}, "빈 매장 기준선 — 흰 테이블 T4 짐 오판, 장식물 오판")
_step(120, "자리 맡기 물건을 까세요. T2 에 가방, T7 의자에 겉옷, 바 5번에 텀블러와 책. 그리고 화면 밖으로.",
      dict(BG), "배경 사용중 깔기 (12번까지 그대로)")
_step(120, "T1 에 노트북을 펴고 앉으세요.",
      {**BG, "T1": "occupied"}, "정상 착석 (하반신 가림이 기본)")
_step(120, "T1 에 노트북과 두 번째 가방을 두고 화면 밖으로.",
      {**BG, "T1": "occupied"}, "짐만 남은 자리")
_step(120, "T1 짐을 챙기세요. 의자는 빼 둔 채로 화면 밖으로.",
      dict(BG), "의자만 빠진 빈자리 — 사용중으로 나오면 오답")
_step(120, "흰 테이블 T4 에 앉아서 엎드리세요.",
      {**BG, "T4": "occupied"}, "엎드림 — 관절점 부족")
_step(120, "T4 에 휴대폰과 지갑만 두고 화면 밖으로.",
      {**BG, "T4": "occupied"}, "작은 물건만 — 탐지기에 지갑 항목 없음")
_step(120, "물건을 챙기세요. T3 의자 하나를 한 걸음 빼서 앉으세요.",
      {**BG, "T3": "occupied"}, "테이블에서 떨어진 의자 — 의자 연결")
_step(120, "T3 의자를 제자리에. T3 와 T4 사이에 서서 휴대폰을 보세요. 앉지 마세요.",
      dict(BG), "서성이기 — 45초 가림 규칙으로 사용중이 될 수 있음(감수한 값)")
_step(120, "카메라와 T5 사이에 서서 T5 를 몸으로 가리세요.",
      dict(BG), "빈자리를 사람이 가림 — 45초 규칙 발동 재료")
_step(120, "바 1번에 앉고, 두 번째 가방은 바 2번 의자에 올려 두세요.",
      {**BG, "BAR6-1": "occupied", "BAR6-2": "occupied"},
      "바 — 옆 의자 짐 (바는 짐을 일부러 안 셈: BAR6-2 는 규칙상 빈자리가 정상)")
_step(120, "가방을 챙기세요. 바 3번과 4번 의자 사이에 걸쳐 앉으세요.",
      {**BG, "BAR6-3": "ignore", "BAR6-4": "ignore"}, "두 자리 걸침")
_step(20, "T3 에 앉으세요. 20초만.", {**BG, "T3": "occupied"}, "짧게 앉기 — 앱이 안 바뀌어야 정상")
_step(60, "일어나서 화면 밖으로. 1분.", dict(BG), "짧게 앉은 뒤 비우기")
_step(20, "다시 T3 에 앉으세요. 20초.", {**BG, "T3": "occupied"}, "짧게 앉기")
_step(60, "일어나서 화면 밖으로. 1분.", dict(BG), "짧게 앉은 뒤 비우기")
_step(120, "T3 에 앉으세요. 2분.", {**BG, "T3": "occupied"}, "잠깐 비움 준비")
_step(30, "화면 밖으로. 30초만.", {**BG, "T3": "ignore"},
      "잠깐 비움 — 앱이 계속 사용중이어야 정상(사람 눈으론 빈자리)")
_step(60, "다시 T3 에 앉으세요. 1분.", {**BG, "T3": "occupied"}, "잠깐 비운 뒤 복귀")
_step(60, "T2 가방을 치우고 화면 밖으로.", {"T7": "occupied", "BAR6-5": "occupied"}, "떠남 — 빈자리 복귀 속도")
_step(60, "T7 겉옷을 치우고 화면 밖으로.", {"BAR6-5": "occupied"}, "떠남")
_step(60, "바 5번 텀블러와 책을 치우고 화면 밖으로.", {}, "떠남")
_step(150, "허락을 받았으면 불을 30초 끄고 다시 켜세요. 아니면 화면 밖에서 기다리세요.",
      {}, "조명 변화 — 10월 15일 18시 10분 일괄 변화 확인 / 빈 매장 마무리")

TOTAL_SECONDS = sum(s["seconds"] for s in STEPS)


def say(text: str, enabled: bool) -> None:
    """윈도우 한국어 음성(Heami)으로 읽는다. 기다리지 않는다 — 시간이 밀리면 안 되니까."""
    print(f"  >> {text}", flush=True)
    if not enabled:
        return
    safe = text.replace("'", "''")
    script = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              "try { $s.SelectVoice('Microsoft Heami Desktop') } catch {}; "
              f"$s.Speak('{safe}')")
    subprocess.Popen(["powershell", "-NoProfile", "-Command", script],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ssh_box(host: str, user: str, alias: str, command: str) -> str:
    out = subprocess.run(
        ["ssh", "-o", f"HostKeyAlias={alias}", "-o", "ConnectTimeout=8",
         f"{user}@{host}", command],
        capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise SystemExit(f"박스 접속 실패: {out.stderr.strip()}")
    return out.stdout.strip()


def camera_url_from_box(host: str, user: str, alias: str) -> str:
    line = ssh_box(host, user, alias,
                   "grep '^SEATNOW_URL=' ~/seatnow/deploy/seatnow.env")
    url = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not url.startswith("rtsp://"):
        raise SystemExit(f"박스 설정에서 카메라 주소를 못 읽었다: {line!r}")
    return url


def seat_map(url: str, layout: Path, out_jpg: Path, dry_run: bool) -> None:
    """카메라 한 장에 좌석 이름을 그려서 띄운다. 노트북에만 남는다."""
    import cv2

    raw = out_jpg.with_suffix(".raw.jpg")
    src = (["-f", "lavfi", "-i", "testsrc=size=2560x1440"] if dry_run
           else ["-rtsp_transport", "tcp", "-i", url])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src,
                    "-frames:v", "1", str(raw)], check=True, timeout=30)
    image = cv2.imread(str(raw))
    raw.unlink(missing_ok=True)
    data = json.loads(layout.read_text(encoding="utf-8"))
    for table in data["tables"]:
        x1, y1, x2, y2 = map(int, table["box"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 4)
        for seat in table.get("seats") or []:
            sx1, sy1, sx2, sy2 = map(int, seat["box"])
            cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (255, 160, 0), 3)
            cv2.putText(image, str(seat["id"]), (sx1 + 6, sy1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 160, 0), 4)
        cv2.putText(image, table["name"], (x1 + 8, y1 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 220, 255), 6)
    cv2.imwrite(str(out_jpg), image)
    print(f"좌석 지도: {out_jpg}")
    if not dry_run:
        subprocess.Popen(["cmd", "/c", "start", "", str(out_jpg)])


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="192.168.0.2", help="박스 주소")
    p.add_argument("--user", default="hugo")
    p.add_argument("--host-key-alias", default="uhho.local")
    p.add_argument("--url", help="카메라 주소를 직접 줄 때 (안 주면 박스에서 읽음)")
    p.add_argument("--layout", type=Path, default=PROJECT_DIR / "layouts" / "moonq.json")
    p.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "sample_raw")
    p.add_argument("--lead", type=int, default=60,
                   help="녹화 시작 뒤 첫 동작까지 여유(초) — 노트북에서 걸어 나갈 시간")
    p.add_argument("--dry-run", action="store_true", help="카메라 없이 가짜 영상으로 시험")
    p.add_argument("--speed", type=float, default=1.0, help="시험할 때만: 몇 배 빠르게")
    p.add_argument("--mute", action="store_true", help="음성 끄기")
    p.add_argument("--map-only", action="store_true",
                   help="좌석 지도만 띄우고 끝낸다 (어디가 T1 인지 먼저 볼 때)")
    p.add_argument("--free", type=float, metavar="분",
                   help="대본 없이 이만큼(분) 녹화만 한다. 정답은 나중에 영상을 보고 단다")
    args = p.parse_args(argv)
    if args.speed != 1.0 and not args.dry_run:
        raise SystemExit("--speed 는 --dry-run 에서만 쓴다")

    now = dt.datetime.now()
    stamp = now.strftime("%Y-%m-%d_%H%M")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    video = args.out_dir / f"moonq_{stamp}_scenario.mp4"
    sidecar = video.with_suffix(".scenario.json")

    box_clock = None
    if args.dry_run:
        url = "testsrc"
    else:
        url = args.url or camera_url_from_box(args.host, args.user, args.host_key_alias)
        box_clock = ssh_box(args.host, args.user, args.host_key_alias, "date -Iseconds")
        print(f"박스 시계 {box_clock} / 노트북 시계 {dt.datetime.now().isoformat(timespec='seconds')}")
    seat_map(url, args.layout, video.with_suffix(".seatmap.jpg"), args.dry_run)
    if args.map_only:
        return 0

    steps = [] if args.free else STEPS
    duration = int(args.free * 60) if args.free else args.lead + TOTAL_SECONDS
    if args.dry_run:
        src = ["-re", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=5"]
        rec_seconds = duration / args.speed
    else:
        src = ["-rtsp_transport", "tcp", "-i", url]
        rec_seconds = duration
    ffmpeg = ["ffmpeg", "-y", "-loglevel", "error", *src, "-t", f"{rec_seconds:.1f}",
              "-an", "-c", "copy" if not args.dry_run else "libx264",
              "-movflags", "+frag_keyframe+empty_moov", str(video)]

    record = {
        "video": video.name, "layout": str(args.layout.relative_to(PROJECT_DIR)),
        "seats": SEATS, "lead_seconds": args.lead,
        "transition_seconds": TRANSITION_SECONDS,
        "laptop_start": None, "box_clock_at_setup": box_clock,
        "laptop_clock_at_setup": now.isoformat(timespec="seconds"),
        "steps": [{**s, "video_start": args.lead + s["start"]} for s in steps],
        "mode": "free" if args.free else "scripted",
        "note": "truth 에 없는 좌석은 empty. 구간 앞 transition_seconds 는 채점 제외.",
    }

    if args.free:
        say(f"녹화를 시작합니다. {args.free:.0f}분.", not args.mute)
    else:
        say(f"녹화를 시작합니다. {args.lead}초 안에 화면 밖으로 나가세요.", not args.mute)
    proc = subprocess.Popen(ffmpeg, stdin=subprocess.DEVNULL)
    t0 = time.monotonic()
    record["laptop_start"] = dt.datetime.now().isoformat(timespec="seconds")
    sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def wait_until(video_second: float) -> bool:
        while (time.monotonic() - t0) * args.speed < video_second:
            if proc.poll() is not None:
                return False
            time.sleep(0.05)
        return True

    try:
        for i, step in enumerate(record["steps"]):
            begin = step["video_start"]
            if begin - 15 > 0 and not wait_until(begin - 15):
                break
            say(f"15초 뒤. {step['say']}", not args.mute)
            if not wait_until(begin):
                break
            left = (duration - begin) / 60
            print(f"[{begin // 60:02d}:{begin % 60:02d}] {i + 1}/{len(steps)} "
                  f"{step['tests']}  (남은 {left:.0f}분)", flush=True)
            say(f"지금. {step['say']}", not args.mute)
        wait_until(duration)
    except KeyboardInterrupt:
        print("\n중단 — 지금까지 찍은 영상은 남는다.")
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=30)

    ok = video.exists() and video.stat().st_size > 0
    say("녹화가 끝났습니다." if ok else "녹화 파일이 없습니다. 확인하세요.", not args.mute)
    if ok:
        mb = video.stat().st_size / 1e6
        print(f"\n영상: {video}  ({mb:.0f}MB)\n대본: {sidecar}")
        if proc.returncode not in (0, None, 255) and not args.dry_run:
            print(f"주의: 녹화가 코드 {proc.returncode} 로 끝났다 — 길이를 확인할 것")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
