#!/usr/bin/env python3
"""Build non-invasive staging manifests for selected Aurora run outputs.

The scanner only inventories candidate files and writes planning metadata.  It
does not copy, move, delete, compress, or modify simulation outputs.
"""

import argparse
import csv
import fnmatch
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOTS = [
    "/home/hzhu/scratch2/hzhu/acc/cbd/tilted_large",
    "/home/hzhu/scratch2/hzhu/acc/bondi/adi",
    "/home/hzhu/scratch2/hzhu/acc/bondi/cooling",
]
DEFAULT_OUTPUT_DIR = "staging_manifests"
TRANSFER_BUDGET_BYTES = 90 * 1000**4
CATEGORY_ORDER = [
    "restart",
    "restart_rank_local",
    "mhd_w_bcc",
    "slice",
    "torque_excluded",
    "am_excluded",
    "history",
    "parfile_input",
    "npz",
    "png",
    "other_athdf",
    "other_candidate",
]
TRANSFER_CATEGORIES = {
    "restart",
    "restart_rank_local",
    "mhd_w_bcc",
    "slice",
    "history",
    "parfile_input",
    "npz",
    "png",
}


def log(message):
    print(f"[aurora-scan] {message}", file=sys.stderr, flush=True)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)}")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def append_jsonl(path, records):
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def read_jsonl(path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_find(root, args, discovery, run_dir):
    if not root.exists():
        return
    cmd = [
        "find",
        str(root),
        *args,
        "-printf",
        "%p\t%s\t%T@\\0",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data, err = proc.communicate()
    if err:
        text = err.decode("utf-8", errors="replace").strip()
        if text:
            log(f"find warning for {root}: {text}")
    if proc.returncode not in (0, 1):
        log(f"find returned {proc.returncode} for {root}")
    for item in data.split(b"\0"):
        if not item:
            continue
        try:
            raw_path, raw_size, raw_mtime = item.rsplit(b"\t", 2)
        except ValueError:
            log(f"skipping unparsable find row under {root!s}")
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        try:
            relpath = os.path.relpath(path, run_dir)
        except ValueError:
            relpath = path
        yield {
            "run_dir": str(run_dir),
            "path": path,
            "relpath": relpath,
            "size": int(raw_size),
            "mtime": float(raw_mtime),
            "discovery": discovery,
        }


def discover_runs(roots, run_glob, max_runs):
    runs = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            log(f"missing root: {root_path}")
            continue
        for child in root_path.iterdir():
            if child.is_dir() and fnmatch.fnmatch(child.name, run_glob):
                runs.append(child)
    runs = sorted(runs, key=lambda p: str(p))
    if max_runs is not None:
        runs = runs[:max_runs]
    return runs


def parse_parfile(path):
    blocks = defaultdict(dict)
    if not path.exists():
        return blocks
    current = None
    block_re = re.compile(r"^\s*<([^>\n]+)>\s*$")
    key_re = re.compile(r"^\s*([A-Za-z0-9_./-]+)\s*=\s*(.*?)\s*$")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            block_match = block_re.match(line)
            if block_match:
                current = block_match.group(1).strip()
                continue
            key_match = key_re.match(line)
            if current and key_match:
                blocks[current][key_match.group(1).strip()] = (
                    key_match.group(2).strip()
                )
    return blocks


def as_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def as_bool(value):
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in ("true", "t", "yes", "1"):
        return True
    if lowered in ("false", "f", "no", "0"):
        return False
    return None


def inspect_dynbbh(repo_root):
    path = repo_root / "src" / "pgen" / "dynbbh.cpp"
    info = {
        "path": str(path),
        "found": path.exists(),
        "omega_formula": None,
        "period_formula": None,
    }
    if not path.exists():
        return info
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"bbh\.om\s*=\s*std::pow\(bbh\.sep,\s*-1\.5\)", text):
        info["omega_formula"] = "bbh.om = pow(sep, -1.5)"
        info["period_formula"] = "2*pi*sep^1.5"
    return info


def infer_orbit_model(run_dir, par, dynbbh_info):
    problem = par.get("problem", {})
    sep = as_float(problem.get("sep"))
    use_traj_table = as_bool(problem.get("use_traj_table"))
    notes = []
    if use_traj_table:
        return {
            "period": None,
            "omega": None,
            "sep": sep,
            "use_traj_table": True,
            "confidence": "low",
            "method": (
                "ambiguous: use_traj_table=true; "
                "analytic sep^-1.5 orbit not assumed"
            ),
            "notes": ["trajectory table runs need explicit time-to-orbit review"],
        }
    if sep is None:
        return {
            "period": None,
            "omega": None,
            "sep": None,
            "use_traj_table": use_traj_table,
            "confidence": "none",
            "method": "missing problem/sep in parfile",
            "notes": ["cannot infer binary orbital period"],
        }
    if dynbbh_info.get("period_formula") != "2*pi*sep^1.5":
        notes.append("dynbbh.cpp formula was not recognized by the scanner")
        confidence = "medium"
    else:
        confidence = "high"
    omega = sep ** -1.5
    period = 2.0 * math.pi / omega
    return {
        "period": period,
        "omega": omega,
        "sep": sep,
        "use_traj_table": bool(use_traj_table),
        "confidence": confidence,
        "method": "orbit = simulation_time / (2*pi*sep^1.5)",
        "notes": notes,
    }


def output_blocks(par):
    blocks = {}
    basename = par.get("job", {}).get("basename")
    for block, values in par.items():
        if not block.startswith("output"):
            continue
        out_id = values.get("id")
        variable = values.get("variable")
        file_type = values.get("file_type")
        if not out_id:
            out_id = variable or file_type or block
        blocks[out_id] = {
            "block": block,
            "id": out_id,
            "variable": variable,
            "file_type": file_type,
            "dt": as_float(values.get("dt")),
            "single_file_per_rank": as_bool(values.get("single_file_per_rank")),
        }
    return basename, blocks


def output_name_from_file(path):
    name = Path(path).name
    parts = name.split(".")
    if len(parts) >= 4 and parts[-1] == "xdmf" and parts[-2] == "athdf":
        return ".".join(parts[1:-3]) if len(parts) > 4 else parts[1]
    if len(parts) >= 3 and parts[-1].startswith("athdf"):
        return ".".join(parts[1:-2]) if len(parts) > 3 else parts[1]
    if len(parts) >= 3 and parts[-1].startswith("rst"):
        return "rst"
    if len(parts) >= 3:
        return ".".join(parts[1:-1])
    return Path(path).stem


def last_numeric_token(name):
    stem = name
    for suffix in (".xdmf", ".athdf", ".rst", ".npz", ".png", ".hst"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    matches = re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", stem)
    if not matches:
        return None
    token = matches[-1]
    try:
        return float(token), token
    except ValueError:
        return None


def token_set_for_output_name(path):
    output = output_name_from_file(path).lower()
    return {token for token in re.split(r"[^a-z0-9]+", output) if token}


def classify(record, par, basename, outputs):
    relpath = record["relpath"]
    path = record["path"]
    name = Path(path).name
    lower = name.lower()
    output_name = output_name_from_file(path)
    tokens = token_set_for_output_name(path)
    category = "other_candidate"
    selection_reason = None
    include_by_rule = False

    if record["discovery"] == "restart":
        rank_match = re.search(r"(^|/)rank_[0-9]+/", relpath)
        category = "restart_rank_local" if rank_match else "restart"
    elif record["discovery"] == "athdf":
        if "slice" in tokens or "slice" in output_name.lower():
            category = "slice"
            include_by_rule = True
            selection_reason = "all slice athdf outputs are included"
        elif "torque" in tokens:
            category = "torque_excluded"
        elif tokens.intersection({"am", "angular", "angular_momentum"}):
            category = "am_excluded"
        elif "mhd_w_bcc" in output_name.lower() or "mhd_w_bcc" in lower:
            category = "mhd_w_bcc"
        else:
            category = "other_athdf"
    elif record["discovery"] == "history":
        category = "history"
        include_by_rule = True
        selection_reason = "history files are included"
    elif record["discovery"] == "parfile_input":
        category = "parfile_input"
        include_by_rule = True
        selection_reason = "run input/configuration files are included"
    elif record["discovery"] == "npz":
        category = "npz"
        include_by_rule = True
        selection_reason = "all npz analysis products are included"
    elif record["discovery"] == "png":
        category = "png"
        include_by_rule = True
        selection_reason = "all png diagnostic figures are included"

    numeric = last_numeric_token(name)
    index = numeric[0] if numeric else None
    index_token = numeric[1] if numeric else None
    dt = None
    time = None
    time_method = None
    ambiguity = []
    if category in ("restart", "restart_rank_local"):
        rst_blocks = [v for v in outputs.values() if v.get("file_type") == "rst"]
        if len(rst_blocks) == 1:
            dt = rst_blocks[0].get("dt")
        elif len(rst_blocks) > 1:
            ambiguity.append("multiple restart output blocks")
        if dt is not None and index is not None:
            time = index * dt
            time_method = "filename checkpoint index multiplied by restart dt"
        elif index is not None:
            time_method = "filename checkpoint index only; restart dt unavailable"
    elif category == "mhd_w_bcc":
        block = outputs.get(output_name) or outputs.get("mhd_w_bcc")
        if block:
            dt = block.get("dt")
        if dt is not None and index is not None:
            time = index * dt
            time_method = f"filename output index multiplied by {output_name} dt"
        elif index is not None:
            time_method = "filename output index only; output dt unavailable"

    result = dict(record)
    result.update({
        "category": category,
        "output_name": output_name,
        "output_tokens": sorted(tokens),
        "index": index,
        "index_token": index_token,
        "dt": dt,
        "time": time,
        "time_method": time_method,
        "ambiguity": ambiguity,
        "include_by_rule": include_by_rule,
        "selection_reason": selection_reason,
        "selected": False,
    })
    return result


def checkpoint_key(record):
    if record.get("index_token") is not None:
        return record["index_token"]
    name = Path(record["path"]).name
    return name


def select_nearest_groups(groups, orbit_model, orbit_stride):
    period = orbit_model.get("period")
    selected = {}
    selections = []
    if not groups:
        return selected, selections, "no groups available"
    sortable = []
    for key, members in groups.items():
        sample = members[0]
        time = sample.get("time")
        index = sample.get("index")
        order_value = time if time is not None else index
        sortable.append((order_value is None, order_value, key, members))
    sortable.sort(
        key=lambda item: (
            item[0],
            item[1] if item[1] is not None else 0,
            item[2],
        )
    )
    if period is None:
        key = sortable[-1][2]
        selected[key] = "last available group only; orbital period unavailable"
        selections.append({
            "target_orbit": None,
            "selected_group": key,
            "selected_orbit": None,
            "selected_time": sortable[-1][3][0].get("time"),
            "mismatch_orbits": None,
            "reason": selected[key],
        })
        return selected, selections, "period unavailable"

    orbit_groups = []
    for _, _, key, members in sortable:
        time = members[0].get("time")
        if time is None:
            continue
        orbit = time / period
        orbit_groups.append((orbit, key, members))
    if not orbit_groups:
        key = sortable[-1][2]
        selected[key] = "last available group only; no physical times parsed"
        selections.append({
            "target_orbit": None,
            "selected_group": key,
            "selected_orbit": None,
            "selected_time": sortable[-1][3][0].get("time"),
            "mismatch_orbits": None,
            "reason": selected[key],
        })
        return selected, selections, "no physical times parsed"

    max_orbit = max(item[0] for item in orbit_groups)
    target = 0.0
    while target <= max_orbit + orbit_stride * 0.5:
        nearest = min(orbit_groups, key=lambda item: abs(item[0] - target))
        mismatch = abs(nearest[0] - target)
        key = nearest[1]
        reason = f"nearest available group to target orbit {target:g}"
        if key not in selected:
            selected[key] = reason
        selections.append({
            "target_orbit": target,
            "selected_group": key,
            "selected_orbit": nearest[0],
            "selected_time": nearest[2][0].get("time"),
            "mismatch_orbits": mismatch,
            "reason": reason,
        })
        target += orbit_stride

    last_key = max(orbit_groups, key=lambda item: item[0])[1]
    selected[last_key] = "last available group"
    if all(s["selected_group"] != last_key or s["reason"] != "last available group"
           for s in selections):
        last = [item for item in orbit_groups if item[1] == last_key][0]
        selections.append({
            "target_orbit": "last",
            "selected_group": last_key,
            "selected_orbit": last[0],
            "selected_time": last[2][0].get("time"),
            "mismatch_orbits": None,
            "reason": "last available group",
        })
    return selected, selections, None


def inventory_run(run_dir):
    log(f"inventory {run_dir}")
    records = []
    bin_dir = run_dir / "bin"
    rst_dir = run_dir / "rst"
    records.extend(run_find(
        bin_dir,
        ["-maxdepth", "1", "-type", "f", "-name", "*.athdf*"],
        "athdf",
        run_dir,
    ))
    records.extend(run_find(
        rst_dir,
        ["-maxdepth", "2", "-type", "f", "-name", "*.rst*"],
        "restart",
        run_dir,
    ))
    for place in (run_dir, bin_dir):
        records.extend(run_find(
            place,
            [
                "-maxdepth",
                "1",
                "-type",
                "f",
                "(",
                "-name",
                "*.hst",
                "-o",
                "-iname",
                "*history*",
                ")",
            ],
            "history",
            run_dir,
        ))
    records.extend(run_find(
        run_dir,
        [
            "-maxdepth",
            "2",
            "-type",
            "f",
            "(",
            "-name",
            "parfile.par",
            "-o",
            "-name",
            "*.athinput",
            "-o",
            "-name",
            "*.par",
            "-o",
            "-name",
            "*.in",
            "-o",
            "-iname",
            "*config*",
            "-o",
            "-iname",
            "*setup*",
            ")",
        ],
        "parfile_input",
        run_dir,
    ))
    records.extend(run_find(
        run_dir,
        ["-type", "f", "-name", "*.npz"],
        "npz",
        run_dir,
    ))
    records.extend(run_find(
        run_dir,
        ["-type", "f", "-iname", "*.png"],
        "png",
        run_dir,
    ))
    seen = set()
    deduped = []
    for record in records:
        key = record["path"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def load_or_create_raw(run_dir, cache_dir, refresh_cache):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_dir).strip("/"))
    cache_path = cache_dir / f"{safe}.raw.jsonl"
    if cache_path.exists() and not refresh_cache:
        log(f"reuse raw cache {cache_path}")
        return list(read_jsonl(cache_path))
    records = inventory_run(run_dir)
    write_jsonl(cache_path, records)
    return records


def classify_and_select_run(run_dir, raw_records, repo_root, orbit_stride):
    par_path = run_dir / "parfile.par"
    par = parse_parfile(par_path)
    dynbbh_info = inspect_dynbbh(repo_root)
    orbit_model = infer_orbit_model(run_dir, par, dynbbh_info)
    basename, outputs = output_blocks(par)
    classified = [
        classify(record, par, basename, outputs)
        for record in raw_records
    ]
    for record in classified:
        period = orbit_model.get("period")
        if period and record.get("time") is not None:
            record["orbit"] = record["time"] / period
        else:
            record["orbit"] = None

    mhd_groups = defaultdict(list)
    restart_groups = defaultdict(list)
    for record in classified:
        if record["category"] == "mhd_w_bcc":
            mhd_groups[checkpoint_key(record)].append(record)
        elif record["category"] in ("restart", "restart_rank_local"):
            restart_groups[checkpoint_key(record)].append(record)

    selected_mhd, mhd_selection, mhd_issue = select_nearest_groups(
        mhd_groups, orbit_model, orbit_stride)
    selected_rst, rst_selection, rst_issue = select_nearest_groups(
        restart_groups, orbit_model, orbit_stride)

    for record in classified:
        key = checkpoint_key(record)
        if record.get("include_by_rule"):
            record["selected"] = True
        elif record["category"] == "mhd_w_bcc" and key in selected_mhd:
            record["selected"] = True
            record["selection_reason"] = selected_mhd[key]
        elif (
            record["category"] in ("restart", "restart_rank_local")
            and key in selected_rst
        ):
            record["selected"] = True
            record["selection_reason"] = selected_rst[key]

    rst_rank_local = any(r["category"] == "restart_rank_local" for r in classified)
    rst_group_sizes = [len(v) for v in restart_groups.values()]
    rst_ambiguous = bool(rst_issue) or (rst_rank_local and len(set(rst_group_sizes)) > 1)
    summary = build_run_summary(
        run_dir,
        classified,
        orbit_model,
        outputs,
        mhd_selection,
        rst_selection,
        mhd_issue,
        rst_issue,
        rst_rank_local,
        rst_ambiguous,
        par_path.exists(),
    )
    return classified, [r for r in classified if r["selected"]], summary


def build_run_summary(run_dir, records, orbit_model, outputs, mhd_selection,
                      rst_selection, mhd_issue, rst_issue, rst_rank_local,
                      rst_ambiguous, has_parfile):
    counts = defaultdict(int)
    sizes = defaultdict(int)
    selected_counts = defaultdict(int)
    selected_sizes = defaultdict(int)
    for record in records:
        category = record["category"]
        counts[category] += 1
        sizes[category] += record.get("size") or 0
        if record.get("selected"):
            selected_counts[category] += 1
            selected_sizes[category] += record.get("size") or 0

    selected = [r for r in records if r.get("selected")]
    latest_rst = latest_selected(selected, {"restart", "restart_rank_local"})
    latest_mhd = latest_selected(selected, {"mhd_w_bcc"})
    missing = []
    if not has_parfile:
        missing.append("parfile.par")
    if counts["mhd_w_bcc"] == 0:
        missing.append("mhd_w_bcc")
    if counts["restart"] + counts["restart_rank_local"] == 0:
        missing.append("restart")
    return {
        "run_dir": str(run_dir),
        "has_parfile": has_parfile,
        "orbit_period": orbit_model.get("period"),
        "orbit_omega": orbit_model.get("omega"),
        "orbit_sep": orbit_model.get("sep"),
        "orbit_confidence": orbit_model.get("confidence"),
        "orbit_method": orbit_model.get("method"),
        "orbit_notes": "; ".join(orbit_model.get("notes") or []),
        "single_file_per_rank_detected": rst_rank_local,
        "restart_grouping_ambiguous": rst_ambiguous,
        "missing_expected_outputs": ";".join(missing),
        "mhd_selection_issue": mhd_issue or "",
        "restart_selection_issue": rst_issue or "",
        "latest_restart_selected": latest_rst.get("path") if latest_rst else "",
        "latest_mhd_w_bcc_selected": latest_mhd.get("path") if latest_mhd else "",
        "latest_restart_orbit": latest_rst.get("orbit") if latest_rst else None,
        "latest_mhd_w_bcc_orbit": latest_mhd.get("orbit") if latest_mhd else None,
        "counts": dict(counts),
        "sizes": dict(sizes),
        "selected_counts": dict(selected_counts),
        "selected_sizes": dict(selected_sizes),
        "selected_total_bytes": sum(selected_sizes.values()),
        "mhd_selection": mhd_selection,
        "restart_selection": rst_selection,
        "output_blocks": outputs,
    }


def latest_selected(records, categories):
    candidates = [r for r in records if r["category"] in categories]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (
        r.get("time") is not None,
        r.get("time") if r.get("time") is not None else -1,
        r.get("index") if r.get("index") is not None else -1,
        not r["path"].endswith(".xdmf"),
        r["path"],
    ))


def rel_for_rsync(path):
    roots = ["/home/hzhu/scratch2/hzhu/acc"]
    for root in roots:
        try:
            return os.path.relpath(path, root), root
        except ValueError:
            pass
    return path.lstrip("/"), "/"


def write_run_summary_csv(path, summaries):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "run_dir",
            "has_parfile",
            "orbit_period",
            "orbit_omega",
            "orbit_sep",
            "orbit_confidence",
            "orbit_method",
            "orbit_notes",
            "single_file_per_rank_detected",
            "restart_grouping_ambiguous",
            "missing_expected_outputs",
            "mhd_selection_issue",
            "restart_selection_issue",
            "latest_restart_selected",
            "latest_restart_orbit",
            "latest_mhd_w_bcc_selected",
            "latest_mhd_w_bcc_orbit",
            "selected_total_bytes",
        ]
        for category in CATEGORY_ORDER:
            fields.extend([
                f"count_{category}",
                f"bytes_{category}",
                f"selected_count_{category}",
                f"selected_bytes_{category}",
            ])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {field: summary.get(field, "") for field in fields}
            for category in CATEGORY_ORDER:
                row[f"count_{category}"] = summary["counts"].get(category, 0)
                row[f"bytes_{category}"] = summary["sizes"].get(category, 0)
                row[f"selected_count_{category}"] = (
                    summary["selected_counts"].get(category, 0)
                )
                row[f"selected_bytes_{category}"] = (
                    summary["selected_sizes"].get(category, 0)
                )
            writer.writerow(row)


def write_outputs(output_dir, raw_all, classified_all, selected_all, summaries):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "raw_inventory.jsonl", raw_all)
    write_jsonl(output_dir / "classified_inventory.jsonl", classified_all)
    write_jsonl(output_dir / "selected_transfer_manifest.jsonl", selected_all)
    source_roots = set()
    with (output_dir / "selected_transfer_manifest.txt").open(
            "w", encoding="utf-8") as handle:
        for record in selected_all:
            relpath, root = rel_for_rsync(record["path"])
            source_roots.add(root)
            handle.write(relpath + "\n")
    write_run_summary_csv(output_dir / "run_summary.csv", summaries)
    write_manifest_metadata(output_dir, source_roots)


def write_manifest_metadata(output_dir, source_roots):
    meta = {
        "rsync_files_from_source_roots": sorted(source_roots),
        "rsync_note": (
            "selected_transfer_manifest.txt is relative to "
            "/home/hzhu/scratch2/hzhu/acc when all selected paths are under it"
        ),
    }
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--run-glob", default="run*")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--orbit-stride", type=float, default=100.0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir
    cache_dir = output_dir / "raw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.roots, args.run_glob, args.max_runs)
    log(f"found {len(runs)} run(s)")
    summaries = []
    source_roots = set()
    raw_path = output_dir / "raw_inventory.jsonl"
    classified_path = output_dir / "classified_inventory.jsonl"
    selected_path = output_dir / "selected_transfer_manifest.jsonl"
    selected_txt_path = output_dir / "selected_transfer_manifest.txt"
    for path in (raw_path, classified_path, selected_path, selected_txt_path):
        path.write_text("", encoding="utf-8")
    for index, run_dir in enumerate(runs, 1):
        log(f"run {index}/{len(runs)}: {run_dir}")
        raw_records = load_or_create_raw(run_dir, cache_dir, args.refresh_cache)
        classified, selected, summary = classify_and_select_run(
            run_dir, raw_records, repo_root, args.orbit_stride)
        append_jsonl(raw_path, raw_records)
        append_jsonl(classified_path, classified)
        append_jsonl(selected_path, selected)
        with selected_txt_path.open("a", encoding="utf-8") as handle:
            for record in selected:
                relpath, root = rel_for_rsync(record["path"])
                source_roots.add(root)
                handle.write(relpath + "\n")
        summaries.append(summary)

    write_run_summary_csv(output_dir / "run_summary.csv", summaries)
    write_manifest_metadata(output_dir, source_roots)

    report_script = repo_root / "scripts" / "staging" / "report_aurora_scan.py"
    if report_script.exists():
        subprocess.run([sys.executable, str(report_script), str(output_dir)], check=False)
    log(f"wrote manifests under {output_dir}")
    log("dry-run mode: no simulation data was copied or modified")


if __name__ == "__main__":
    main()
