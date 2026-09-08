"""Summarise a live run: was the tick on time, did memory grow, who used the CPU.

    python -m edge.live_report results/live/run_4mp_h265.jsonl \
        --samples results/live/sample_4mp_h265.csv

Reads the JSONL the live loop writes (one record per tick, with ``tick`` /
``process`` / ``live`` fields) and, optionally, the CSV ``edge/live_sampler.py``
wrote beside it.  Prints the numbers the 24/7 decision is made on and writes
them next to the log as ``*_report.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "p95": None, "max": None, "min": None}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "mean": round(sum(values) / len(values), 3),
        "p95": round(p95, 3),
        "max": round(max(values), 3),
        "min": round(min(values), 3),
    }


def memory_trend(points: Sequence[Tuple[float, float]], skip: int = 0) -> Dict[str, Optional[float]]:
    """Linear fit of (elapsed seconds, MB) → MB per hour, plus first/last/max."""
    pts = list(points)[skip:]
    if not pts:
        return {"first": None, "last": None, "max": None, "mb_per_hour": None, "points": 0}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(pts)
    slope_per_s = 0.0
    if n >= 2:
        mx = sum(xs) / n
        my = sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        if var > 0:
            slope_per_s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    return {
        "first": ys[0],
        "last": ys[-1],
        "max": max(ys),
        "mb_per_hour": round(slope_per_s * 3600.0, 3),
        "points": n,
    }


def summarize_records(records: Sequence[dict], interval_seconds: float) -> dict:
    budget = interval_seconds / 2.0
    ticks = [r for r in records if "tick" in r]
    durations = [float(r["tick"]["duration_s"]) for r in ticks]
    steady = durations[1:] if len(durations) > 1 else durations
    inference = [float(r.get("inference_ms", 0.0)) for r in ticks][1:]
    late = [float(r["tick"].get("late_s", 0.0)) for r in ticks]
    ages = [float(r["tick"].get("burst_age_s", 0.0)) for r in ticks]
    rss_points = [
        (float(r["process"]["elapsed_s"]), float(r["process"]["rss_mb"]))
        for r in ticks
        if r.get("process", {}).get("rss_mb") is not None
    ]
    ffmpeg_points = [
        (float(r["process"]["elapsed_s"]), float(r["process"]["ffmpeg_rss_mb"]))
        for r in ticks
        if r.get("process", {}).get("ffmpeg_rss_mb") is not None
    ]
    last = ticks[-1] if ticks else {}
    return {
        "ticks": len(ticks),
        "interval_seconds": interval_seconds,
        "budget_s": budget,
        "first_tick_s": durations[0] if durations else None,
        "tick_s": {**_stats(steady), "over_budget": sum(1 for d in steady if d > budget)},
        "inference_ms": _stats(inference),
        "late_s": _stats(late),
        "burst_age_s": _stats(ages),
        "skipped_slots": int(last.get("tick", {}).get("skipped_total", 0)) if last else 0,
        "reconnects": int(last.get("live", {}).get("reconnects", 0)) if last else 0,
        "hwaccel_suspect": any(bool(r.get("live", {}).get("hwaccel_suspect")) for r in ticks),
        "decode_fps": _stats([float(r["live"].get("decode_fps", 0.0)) for r in ticks][1:]),
        "rss_mb": memory_trend(rss_points, skip=min(2, max(0, len(rss_points) - 2))),
        "rss_mb_all": memory_trend(rss_points),
        "ffmpeg_rss_mb": memory_trend(ffmpeg_points, skip=min(2, max(0, len(ffmpeg_points) - 2))),
        "elapsed_s": float(last.get("process", {}).get("elapsed_s", 0.0)) if last else 0.0,
        # 전송이 꺼져 있었으면 None. 숫자가 0 인 것과 아예 안 보낸 것은 다르다.
        "publish": (
            {
                "sent": int(last["publish"].get("sent", 0)),
                "failed": int(last["publish"].get("failed", 0)),
                "last_error": last["publish"].get("last_error"),
            }
            if last and isinstance(last.get("publish"), dict)
            else None
        ),
    }


def summarize_samples(path: Path) -> dict:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    def column(name: str) -> List[float]:
        out: List[float] = []
        for row in rows:
            value = (row.get(name) or "").strip()
            if value:
                try:
                    out.append(float(value))
                except ValueError:
                    pass
        return out

    cores = {"total": _stats(column("cpu_total_cores"))}
    for key in ("py", "dec", "pub", "mtx"):
        cores[key] = _stats(column(f"{key}_cores"))
    return {
        "samples": len(rows),
        "cores": cores,
        "mem_used_mb": _stats(column("mem_used_mb")),
        "mem_avail_mb": _stats(column("mem_avail_mb")),
        "swap_used_mb": _stats(column("swap_used_mb")),
        "temp_c": _stats(column("temp_c")),
        "cpu_mhz": _stats(column("cpu_mhz")),
        "rss_mb": {key: _stats(column(f"{key}_rss_mb")) for key in ("py", "dec", "pub", "mtx")},
    }


def load_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _fmt(value: Optional[float], unit: str = "", digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}{unit}"


def render(summary: dict, samples: Optional[dict], name: str) -> str:
    t = summary["tick_s"]
    lines = [f"## 실시간 측정 — {name}", ""]
    lines.append(f"- 틱 {summary['ticks']}개, {summary['elapsed_s'] / 60:.0f}분, 주기 {summary['interval_seconds']:g}초, 합격선 {summary['budget_s']:.1f}초")
    lines.append(f"- 첫 틱 {_fmt(summary['first_tick_s'], 's')} (버린다)")
    lines.append(
        f"- 틱 시간 (2번째부터): 평균 {_fmt(t['mean'], 's')} · p95 {_fmt(t['p95'], 's')} · 최대 {_fmt(t['max'], 's')} · "
        f"합격선 초과 {t['over_budget']}개"
    )
    lines.append(f"- 추론: 평균 {_fmt(summary['inference_ms']['mean'], 'ms', 0)} · 최대 {_fmt(summary['inference_ms']['max'], 'ms', 0)}")
    lines.append(
        f"- 늦게 시작한 틱: 최대 {_fmt(summary['late_s']['max'], 's')} · 건너뛴 슬롯 {summary['skipped_slots']} · "
        f"재연결 {summary['reconnects']} · 프레임 신선도 최대 {_fmt(summary['burst_age_s']['max'], 's')}"
    )
    lines.append(f"- 디코딩 fps 평균 {_fmt(summary['decode_fps']['mean'], '', 1)} · 하드웨어 디코딩 의심 {'있음 ⚠️' if summary['hwaccel_suspect'] else '없음'}")
    r = summary["rss_mb"]
    lines.append(
        f"- 메모리(판정 프로세스): 처음 {_fmt(summary['rss_mb_all']['first'], 'MB', 0)} → 마지막 {_fmt(r['last'], 'MB', 0)} · "
        f"최대 {_fmt(r['max'], 'MB', 0)} · **추세 {_fmt(r['mb_per_hour'], 'MB/시간', 0)}** (워밍업 2틱 제외)"
    )
    f = summary["ffmpeg_rss_mb"]
    lines.append(f"- 메모리(ffmpeg 디코더): 마지막 {_fmt(f['last'], 'MB', 0)} · 추세 {_fmt(f['mb_per_hour'], 'MB/시간', 0)}")
    p = summary.get("publish")
    if p is None:
        lines.append("- Supabase 전송: 꺼짐")
    else:
        lines.append(
            f"- Supabase 전송 성공 {p['sent']}회 · 실패 {p['failed']}회"
            + (f" · 마지막 오류: {p['last_error']}" if p["last_error"] else "")
        )
    if samples:
        c = samples["cores"]
        lines.append("")
        lines.append("| 누가 | CPU 코어 평균 | 최대 |")
        lines.append("|---|---:|---:|")
        for key, label in (("total", "박스 전체"), ("py", "판정(파이썬)"), ("dec", "디코더(ffmpeg)"), ("pub", "가짜 카메라 송출"), ("mtx", "mediamtx")):
            lines.append(f"| {label} | {_fmt(c[key]['mean'])} | {_fmt(c[key]['max'])} |")
        lines.append("")
        lines.append(
            f"- 박스 메모리 사용: 최대 {_fmt(samples['mem_used_mb']['max'], 'MB', 0)} · 남은 최소 {_fmt(samples['mem_avail_mb']['min'], 'MB', 0)} · "
            f"스왑 최대 {_fmt(samples['swap_used_mb']['max'], 'MB', 0)}"
        )
        lines.append(f"- CPU 온도 최대 {_fmt(samples['temp_c']['max'], '°C', 0)} · 클럭 평균 {_fmt(samples['cpu_mhz']['mean'], 'MHz', 0)}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path, help="run_*.jsonl written by the live loop")
    parser.add_argument("--samples", type=Path, help="sample_*.csv written by edge/live_sampler.py")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)
    try:  # Windows consoles default to cp949, which cannot print "—" or "°"
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    records = load_records(args.log)
    summary = summarize_records(records, args.interval)
    samples = summarize_samples(args.samples) if args.samples and args.samples.exists() else None
    name = args.name or args.log.stem
    text = render(summary, samples, name)
    print(text)
    out = args.log.with_name(args.log.stem + "_report.json")
    out.write_text(json.dumps({"summary": summary, "samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
