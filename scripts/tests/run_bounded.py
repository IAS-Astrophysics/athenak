#!/usr/bin/env python3
"""Run one local build/test with a wall deadline and a process-group RSS budget."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--memory-gb', type=float, default=20)
    parser.add_argument('--gpu-allocations', action='store_true',
                        help='use the Kokkos allocation guard instead of a virtual-space cap')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not (0 < args.memory_gb <= 24) or args.seconds <= 0:
        parser.error('command, positive deadline and memory budget <=24 decimal GB required')
    budget = int(args.memory_gb * 1e9)
    if args.gpu_allocations:
        tracked_budget = int(os.environ.get('KOKKOS_MEMORY_BUDGET_BYTES', '0'))
        if not os.environ.get('KOKKOS_TOOLS_LIBS') or not (0 < tracked_budget <= 24e9):
            parser.error('--gpu-allocations requires the Kokkos guard and budget <=24 GB')

    def limits():
        os.setsid()
        # Per-process address-space cap supplements aggregate resident-memory monitoring.
        # Keep extra headroom below the requested 30 GB total cap.
        if sys.platform != 'darwin' and not args.gpu_allocations:
            resource.setrlimit(resource.RLIMIT_AS, (26_000_000_000, 26_000_000_000))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    started = time.monotonic()
    proc = subprocess.Popen(command, preexec_fn=limits)
    peak = 0
    reason = None
    descendants = {proc.pid}

    def stop_tree():
        for pid in sorted(descendants, reverse=True):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    try:
        while proc.poll() is None:
            rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='],
                                           universal_newlines=True)
            table = [tuple(map(int, row.split())) for row in rows.splitlines()
                     if len(row.split()) == 3]
            changed = True
            while changed:
                added = {pid for pid, ppid, _ in table if ppid in descendants}
                changed = not added.issubset(descendants)
                descendants.update(added)
            rss = sum(rss_kb*1024 for pid, _, rss_kb in table if pid in descendants)
            peak = max(peak, rss)
            if rss > budget:
                reason = 'aggregate RSS budget exceeded'
            elif time.monotonic()-started > args.seconds:
                reason = 'wall deadline exceeded'
            if reason:
                stop_tree()
                break
            time.sleep(0.1)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            stop_tree()
            proc.wait()
    report = dict(command=command, returncode=code, peak_rss_bytes=peak,
                  memory_budget_bytes=budget, seconds=time.monotonic()-started, stopped=reason)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return code if code >= 0 else 128-code


if __name__ == '__main__':
    raise SystemExit(main())
