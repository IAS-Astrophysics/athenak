#!/usr/bin/env python3
"""Summarize bounded Aurora runs; never count failed/incomplete cases as timings."""
import argparse
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
a = p.parse_args()
report = {}
for case in sorted(a.directory.iterdir()):
    if not case.is_dir():
        continue
    logs = case / 'rank-0.log'
    if not logs.exists():
        logs = case / 'run.log'
    if not logs.exists():
        continue
    text = logs.read_text(errors='replace')
    resources = [json.loads(f.read_text()) for f in sorted(case.glob('resources-*.json'))]
    memories = [json.loads(f.read_text()) for f in sorted(case.glob('memory-*.json'))]
    pairs = re.findall(r'elapsed=([\d.eE+-]+)\s+cycle=(\d+)', text)
    times = {int(c): float(t) for t, c in pairs}
    row = dict(resources=resources, memory=memories, final_cycle=max(times, default=None),
               successful=bool(resources) and all(r['returncode'] == 0 and not r['stopped']
                                                   for r in resources))
    row['aggregate_peak_rss_bytes_upper_bound'] = sum(r['peak_rss_bytes'] for r in resources)
    if row['successful'] and 10 in times and 50 in times:
        row['warm_seconds_per_step_10_to_50'] = (times[50] - times[10]) / 40
    report[case.name] = row
(a.directory / 'benchmark-summary.json').write_text(json.dumps(report, indent=2)+'\n')
for name, r in report.items():
    print(name, 'PASS' if r['successful'] else 'INCOMPLETE/FAIL',
          r.get('warm_seconds_per_step_10_to_50', 'no warm timing'))
