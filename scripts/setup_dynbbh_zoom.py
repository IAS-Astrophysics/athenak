#!/usr/bin/env python3
"""Generate portable restart overlays for a dynbbh staged AMR/excision zoom.

Only the first input is a complete deck. Later inputs override zoom controls,
leaving physics, grid layout, radiation quadrature, and output counters in the
checkpoint untouched. No checkpoint parsing, scheduler assumptions, or launches.
"""

import argparse
import json
import math
from pathlib import Path
import re
import shlex


def read_input(text):
    blocks = {}
    block = None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"<([^>]+)>", line)
        if match:
            block = match[1]
            if block == "par_end":
                break
            if block in blocks:
                raise ValueError(f"duplicate input block: {block}")
            blocks[block] = {}
        elif block is not None and "=" in line:
            key, value = line.split("=", 1)
            blocks[block][key.strip()] = value.strip()
        else:
            raise ValueError(f"unrecognized input line: {line}")
    return blocks


def render(blocks):
    return "\n\n".join(
        f"<{block}>\n" + "\n".join(f"{key} = {value}" for key, value in params.items())
        for block, params in blocks.items()
    ) + "\n"


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate(schedule):
    required = {"max_level", "max_nmb_per_rank", "excision_radii", "horizon_fraction",
                "shrink_start_time", "shrink_timescale", "restart_dt", "stages"}
    if set(schedule) != required:
        raise ValueError(f"schedule keys must be exactly {sorted(required)}")
    maximum = integer(schedule["max_level"], "max_level", 1)
    integer(schedule["max_nmb_per_rank"], "max_nmb_per_rank", 1)
    for key in ("horizon_fraction", "shrink_start_time",
                "shrink_timescale", "restart_dt"):
        positive(schedule[key], key)
    if schedule["horizon_fraction"] > 1:
        raise ValueError("horizon_fraction must be <= 1")
    if len(schedule["excision_radii"]) != 2:
        raise ValueError("excision_radii must contain two positive radii")
    for radius in schedule["excision_radii"]:
        positive(radius, "excision radius")
    stages = schedule["stages"]
    if len(stages) < 3:
        raise ValueError("need coarse, pre-shrink refined, and shrink stages")
    last_time, names = 0.0, set()
    for stage in stages:
        keys = {"name", "end_time", "tracker_level", "tracker_radius", "regions"}
        if set(stage) != keys:
            raise ValueError("each stage needs name, end_time, tracker_level, "
                             "tracker_radius, and regions")
        name = stage["name"]
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name) or name in names:
            raise ValueError("stage names must be unique simple directory names")
        names.add(name)
        if positive(stage["end_time"], "end_time") <= last_time:
            raise ValueError("stage end times must increase strictly from zero")
        last_time = stage["end_time"]
        if integer(stage["tracker_level"], "tracker_level") > maximum:
            raise ValueError("tracker_level exceeds max_level")
        positive(stage["tracker_radius"], "tracker_radius")
        if len(stage["regions"]) > 16:
            raise ValueError("at most 16 COM regions are supported")
        for radius, level in stage["regions"]:
            positive(radius, "COM radius")
            if integer(level, "COM level") > maximum:
                raise ValueError("COM level exceeds max_level")
    if stages[0]["tracker_level"] != 0 or any(level for _, level in stages[0]["regions"]):
        raise ValueError("the first stage must be root-level only")
    start = schedule["shrink_start_time"]
    before = [i for i, stage in enumerate(stages) if stage["end_time"] == start]
    if not before or before[0] == 0:
        raise ValueError("shrink_start_time must end a separate pre-shrink refined stage")
    final_policy = {k: stages[-1][k]
                    for k in ("tracker_level", "tracker_radius", "regions")}
    if final_policy["tracker_level"] != maximum:
        raise ValueError("final tracker_level must equal max_level")
    for stage in stages[before[0]:]:
        if any(stage[k] != value for k, value in final_policy.items()):
            raise ValueError("resolve before shrinking; keep the final AMR policy "
                             "thereafter")
    if last_time <= start + schedule["shrink_timescale"]:
        raise ValueError("include settling time after the shrink finishes")


def generate(input_path, schedule, output_dir):
    validate(schedule)
    input_path = Path(input_path).resolve()
    base = read_input(input_path.read_text())
    for block in ("mesh", "meshblock", "mhd", "adm", "problem"):
        if block not in base:
            raise ValueError(f"base input must contain <{block}>")
    if "radiation" in base:
        raise ValueError("dynbbh zoom uses <dyn_radiation>, not <radiation>")
    if base.get("dyn_radiation", {}).get("geometry", "adm") != "adm":
        raise ValueError("dyn_radiation/geometry must be adm")
    if base["problem"].get("unresolved_sink", "false").lower() in ("true", "1"):
        raise ValueError("disable unresolved_sink: its independent MHD-only sink "
                         "does not follow the excision shrink schedule")
    for name, params in base.items():
        if re.fullmatch(r"refinement\d+", name):
            raise ValueError("remove static <refinementN> regions from the base input")
        if name.startswith("amr_criterion") and params.get("method") != "user":
            raise ValueError("remove non-user AMR criteria before staged zoom")
    base = {k: v for k, v in base.items() if not k.startswith("amr_criterion")}
    base["amr_criterion0"] = {"method": "user"}
    # Moving to separate run directories must not invalidate a trajectory path.
    problem = base["problem"]
    if problem.get("use_traj_table", "false").lower() in ("true", "1"):
        trajectory = Path(problem["traj_file"]).expanduser()
        if not trajectory.is_absolute():
            trajectory = input_path.parent / trajectory
        if not trajectory.is_file():
            raise ValueError(f"trajectory does not exist: {trajectory}")
        problem["traj_file"] = str(trajectory.resolve())
    for name, params in base.items():
        if name.startswith("output"):
            params.pop("file_number", None)
            params.pop("last_time", None)
    restart_blocks = [name for name, p in base.items() if p.get("file_type") == "rst"]
    if len(restart_blocks) > 1:
        raise ValueError("use only one restart output block")
    restart_block = next(iter(restart_blocks), None)
    if restart_block is None:
        number = 1
        while f"output{number}" in base:
            number += 1
        restart_block = f"output{number}"
        base[restart_block] = {"file_type": "rst"}
    # Presence of dcycle overrides dt even when dcycle is negative.
    base[restart_block].pop("dcycle", None)
    base[restart_block]["dt"] = schedule["restart_dt"]
    base.setdefault("mesh_refinement", {}).update(
        refinement="adaptive", num_levels=schedule["max_level"] + 1,
        max_nmb_per_rank=schedule["max_nmb_per_rank"])
    base.setdefault("coord", {}).update(
        excise="true", excision_scheme="puncture", require_resolved_horizon="false",
        excise_1_rad=schedule["excision_radii"][0],
        excise_2_rad=schedule["excision_radii"][1],
        excise_to_horizon="false", excise_cap_to_horizon="false",
        excise_shrink_to_horizon="true",
        excise_horizon_fraction=schedule["horizon_fraction"],
        excise_shrink_start_time=schedule["shrink_start_time"],
        excise_shrink_timescale=schedule["shrink_timescale"])
    # Decks previously run without excision may omit these required parameters.
    for excision_floor, mhd_floor in (("dexcise", "dfloor"), ("pexcise", "pfloor")):
        if excision_floor not in base["coord"]:
            base["coord"][excision_floor] = base["mhd"][mhd_floor]
    base.setdefault("time", {}).update(nlim=-1)
    base["problem"]["amr_condition"] = "tracker"
    count = max([len(s["regions"]) for s in schedule["stages"]] + [
        int(m[1]) + 1 for key in problem if (m := re.fullmatch(r"radius_(\d+)_rad", key))
    ])
    if count > 16:
        raise ValueError("base input contains unsupported COM region index")
    result = []
    for stage in schedule["stages"]:
        policy = {"amr_condition": "tracker"}
        for hole in (1, 2):
            policy[f"tracker_{hole}_reflevel"] = stage["tracker_level"]
            policy[f"tracker_{hole}_rad"] = stage["tracker_radius"]
        for i in range(count):
            radius, level = stage["regions"][i] if i < len(stage["regions"]) else (1, 0)
            policy[f"radius_{i}_rad"] = radius
            policy[f"radius_{i}_reflevel"] = level
        overlay = {"job": {"basename": stage["name"]},
                   "time": {"tlim": stage["end_time"], "nlim": -1}, "problem": policy}
        if not result:
            for name, params in overlay.items():
                base.setdefault(name, {}).update(params)
            overlay = base
        result.append((stage["name"], render(overlay)))
    output_dir = Path(output_dir).resolve()
    # Refuse reuse, including after a partial run: never overwrite checkpoints.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, text in result:
        directory = output_dir / name
        directory.mkdir()
        (directory / "stage.athinput").write_text(text)
    (output_dir / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    commands = []
    for index, (name, _) in enumerate(result):
        directory = shlex.quote(str(output_dir / name))
        deck = shlex.quote(str(output_dir / name / "stage.athinput"))
        restart = "" if index == 0 else " -r /absolute/path/to/previous-stage-final.rst"
        commands.append(f'$LAUNCHER "$ATHENA"{restart} -i {deck} -d {directory}')
    (output_dir / "RUN.txt").write_text(
        "Set ATHENA to the dyn_grmhd/dynbbh executable and LAUNCHER to e.g. srun\n"
        "(or empty for serial). Run stages in order, selecting each final checkpoint\n"
        "explicitly. Confirm the previous run reached its tlim, not a walltime stop.\n"
        "For a within-stage resume use -r and -d only: omit -i, preserving counters.\n"
        "Keep the radiation model, angular grid, mesh and meshblock sizes unchanged.\n"
        "Check local horizon resolution BEFORE advancing to the shrinking stage.\n\n"
        + "\n\n".join(commands) + "\n")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = generate(args.input, json.loads(args.schedule.read_text()),
                          args.output_dir)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Created {result}; see RUN.txt and docs/dynbbh_zoom.md before launching.")


if __name__ == "__main__":
    main()
