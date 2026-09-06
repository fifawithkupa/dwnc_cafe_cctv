#!/usr/bin/env python3
"""Sample box-wide and per-process CPU/memory every N seconds into a CSV.

CPU is *cores used over the interval* from /proc deltas (not ps's lifetime
average), so a 2-core box saturates at 2.0.  Processes are found by command
line pattern each sample so restarts are followed.
"""
import os, re, sys, time, subprocess

OUT = sys.argv[1]
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
CLK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")
PATTERNS = {
    "py": r"engine\.seatnow rtsp",
    "dec": r"ffmpeg -nostdin -v warning",
    "pub": r"stream_loop -1 -i results",
    "mtx": r"tools/mediamtx",
}


def find_pids():
    found = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "sampler.py" in cmd:
            continue
        for key, pat in PATTERNS.items():
            if key not in found and re.search(pat, cmd):
                found[key] = int(pid)
    return found


def proc_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])  # utime + stime
    except OSError:
        return None


def proc_rss_mb(pid):
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * PAGE / (1024 * 1024)
    except OSError:
        return None


def total_ticks():
    with open("/proc/stat") as f:
        parts = f.readline().split()
    vals = list(map(int, parts[1:]))
    idle = vals[3] + vals[4]
    return sum(vals), idle


def mem():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) // 1024
    return info["MemTotal"] - info["MemAvailable"], info["MemAvailable"], info.get("SwapTotal", 0) - info.get("SwapFree", 0)


def temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000
    except OSError:
        return None


def cpu_mhz():
    try:
        vals = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
        return sum(vals) / len(vals)
    except Exception:
        return None


with open(OUT, "w") as out:
    out.write("ts,load1,mem_used_mb,mem_avail_mb,swap_used_mb,cpu_total_cores,"
              + ",".join(f"{k}_cores,{k}_rss_mb" for k in PATTERNS) + ",temp_c,cpu_mhz\n")
    prev = {}
    prev_total, prev_idle = total_ticks()
    prev_t = time.monotonic()
    while True:
        time.sleep(INTERVAL)
        now = time.monotonic()
        dt = now - prev_t
        prev_t = now
        pids = find_pids()
        tot, idle = total_ticks()
        ncpu = os.cpu_count() or 1
        busy = ((tot - prev_total) - (idle - prev_idle)) / max(1, (tot - prev_total)) * ncpu
        prev_total, prev_idle = tot, idle
        used, avail, swap = mem()
        load1 = open("/proc/loadavg").read().split()[0]
        cols = [time.strftime("%Y-%m-%dT%H:%M:%S"), load1, str(used), str(avail), str(swap), f"{busy:.2f}"]
        for key in PATTERNS:
            pid = pids.get(key)
            ticks = proc_ticks(pid) if pid else None
            if pid and ticks is not None and prev.get(key, (None, None))[0] == pid:
                cores = (ticks - prev[key][1]) / CLK / dt
                cols.append(f"{cores:.2f}")
            else:
                cols.append("")
            rss = proc_rss_mb(pid) if pid else None
            cols.append(f"{rss:.0f}" if rss is not None else "")
            prev[key] = (pid, ticks)
        t = temp_c()
        m = cpu_mhz()
        cols.append(f"{t:.0f}" if t is not None else "")
        cols.append(f"{m:.0f}" if m is not None else "")
        out.write(",".join(cols) + "\n")
        out.flush()
