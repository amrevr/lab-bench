#!/usr/bin/env python3
"""GPU sampler built on nvidia-smi (no root, stdlib only, Python 3.6+).

  python3 gpu_monitor.py --out results/gpu.csv --interval 0.5 [-- command args ...]

With a command it samples while the command runs and exits with its status. Without one it samples
until Ctrl-C / SIGTERM. `--summary results/gpu.csv` prints per-GPU peak/mean utilisation, memory,
power and clocks.
"""
import argparse
import csv
import signal
import subprocess
import sys
import time

FIELDS = ["utilization.gpu", "utilization.memory", "memory.used", "memory.total",
          "power.draw", "temperature.gpu", "clocks.sm", "clocks.mem"]


def sample():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index," + ",".join(FIELDS),
                        "--format=csv,noheader,nounits"], capture_output=True, universal_newlines=True)
    return [[x.strip() for x in line.split(",")] for line in r.stdout.strip().splitlines()]


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    gpus = sorted({r["gpu"] for r in rows})
    print("gpu  util_mean util_max  mem_max_MiB  power_max_W  sm_clock_min_MHz")
    for g in gpus:
        rs = [r for r in rows if r["gpu"] == g]
        f = lambda k: [float(r[k]) for r in rs if r[k] not in ("", "[N/A]", "N/A")]
        u, m, p, c = f("utilization.gpu"), f("memory.used"), f("power.draw"), f("clocks.sm")
        print("%-4s %9.1f %8.0f %12.0f %12.0f %17.0f" % (g, sum(u) / max(len(u), 1), max(u or [0]),
              max(m or [0]), max(p or [0]), min(c or [0])))


def main():
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gpu.csv")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--summary")
    a = ap.parse_args(argv)
    if a.summary:
        return summarize(a.summary)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    proc = subprocess.Popen(cmd) if cmd else None
    t0 = time.time()
    try:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "gpu"] + FIELDS)
            while not stop and (proc is None or proc.poll() is None):
                now = time.time() - t0
                for row in sample():
                    w.writerow(["%.2f" % now] + row)
                fh.flush()
                time.sleep(a.interval)
    except KeyboardInterrupt:
        pass
    if proc:
        proc.wait()
    summarize(a.out)
    return proc.returncode if proc else 0


if __name__ == "__main__":
    sys.exit(main())
