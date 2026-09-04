"""End-to-end restart zoom smoke test, with and without ADM radiation.

Can also run directly: python dynbbh_zoom_common.py --athena /path/to/athena
Logs/checkpoints are retained in the printed temporary directory on failure.
"""

import argparse
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "setup_zoom", ROOT / "scripts/setup_dynbbh_zoom.py")
ZOOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ZOOM)


def run_zoom_regression(athena=None, keep_artifacts=False):
    athena = Path(athena or "./athena").resolve()
    work = Path(tempfile.mkdtemp(prefix="dynbbh-zoom-"))
    print(f"Zoom test artifacts: {work}", flush=True)
    schedule = {
        "max_level": 2, "max_nmb_per_rank": 512, "excision_radii": [2.0, 2.0],
        "horizon_fraction": 0.8, "shrink_start_time": 0.6,
        "shrink_timescale": 0.2, "restart_dt": 0.1, "stages": [],
    }
    for i, name in enumerate(("outer", "inner", "resolve", "shrink", "settle")):
        schedule["stages"].append({
            "name": name, "end_time": (i + 1)/5, "tracker_level": min(i, 2),
            "tracker_radius": 3.0, "regions": [[10.0, min(i, 1)]],
        })
    base_path = ROOT / "tst/inputs/dynbbh_radiation_cbd.athinput"
    base = ZOOM.read_input(base_path.read_text())
    base["mesh_refinement"].update(refinement_interval=1, ncycle_check=1)
    base["coord"].update(smooth_excision="true")
    for name in ("output1", "output2"):
        base[name].update(dt=0.1, dcycle=1)
    base["output3"] = {"file_type": "bin", "variable": "mhd_w", "dt": 0.1,
                       "dcycle": 1}
    for radiation in (False, True):
        model = "radiation" if radiation else "mhd"
        deck = {name: dict(params) for name, params in base.items()}
        if not radiation:
            del deck["dyn_radiation"]
            deck["output1"].update(variable="mhd_w", id="mhd_w")
        source = work / f"{model}.athinput"
        source.write_text(ZOOM.render(deck))
        plan = ZOOM.generate(source, schedule, work / model)
        restart, counts, previous_cycle = None, [], 0
        for stage in schedule["stages"]:
            directory = plan / stage["name"]
            command = [str(athena), "-i", str(directory / "stage.athinput"),
                       "-d", str(directory)]
            if restart:
                command += ["-r", str(restart)]
            log = directory / "run.log"
            with log.open("w") as stream:
                subprocess.run(command, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            text = log.read_text()
            assert "nan" not in text.lower(), log
            times = re.findall(r"time=\s*([\d.eE+-]+)", text)
            assert times and np.isclose(float(times[-1]), stage["end_time"]), text[-2000:]
            checkpoints = sorted((directory / "rst").glob("*.rst"))
            assert checkpoints, directory
            if stage["name"] == "shrink":
                split = plan / "shrink_split"
                split.mkdir()
                with (split / "interrupted.log").open("w") as stream:
                    subprocess.run([
                        str(athena), "-r", str(restart),
                        "-i", str(directory / "stage.athinput"), "-d", str(split),
                        f"time/nlim={previous_cycle + 1}",
                    ], stdout=stream, stderr=subprocess.STDOUT, check=True)
                interrupted = (split / "interrupted.log").read_text()
                stop = float(re.findall(r"time=\s*([\d.eE+-]+)", interrupted)[-1])
                assert schedule["shrink_start_time"] < stop < stage["end_time"]
                partial = sorted((split / "rst").glob("*.rst"))[-1]
                with (split / "resumed.log").open("w") as stream:
                    # No input overlay on resume. nlim only removes the artificial
                    # test interruption; a real walltime restart does not need it.
                    subprocess.run([
                        str(athena), "-r", str(partial), "-d", str(split), "time/nlim=-1",
                    ], stdout=stream, stderr=subprocess.STDOUT, check=True)
                fields = ("mhd_w_bcc", "rad_coord") if radiation else ("mhd_w_bcc",)
                for field in fields:
                    reference = sorted((directory / "tab").glob(f"*.{field}.*.tab"))[-1]
                    resumed = sorted((split / "tab").glob(f"*.{field}.*.tab"))[-1]
                    # Startup C2P adds an inversion, so allow double roundoff,
                    # not physical changes from a repeated projection/missing mask.
                    original, continued = np.loadtxt(reference), np.loadtxt(resumed)
                    np.testing.assert_allclose(original, continued,
                                               rtol=2e-12, atol=1e-18)
                    print(f"{model} mid-shrink restart {field}: max absolute difference "
                          f"{np.max(np.abs(original - continued)):.3e}", flush=True)
            previous_cycle = int(re.findall(r"cycle=\s*(\d+)", text)[-1])
            restart = checkpoints[-1]
            # Fresh plus resumed runs must all produce finite, nontrivial fluid/radiation.
            for field in ("mhd_w_bcc", "rad_coord") if radiation else ("mhd_w_bcc",):
                tables = sorted((directory / "tab").glob(f"*.{field}.*.tab"))
                assert tables, (directory, field)
                data = np.loadtxt(tables[-1], comments="#", ndmin=2)
                assert np.all(np.isfinite(data)), tables[-1]
                assert np.max(data[:, 3]) > 0, tables[-1]
                assert np.min(data[:, 3]) >= -1e-12, tables[-1]
            # Inspect the actual mesh, not merely requested target levels.
            from dynbbh_metric_common import _read_binary_mesh, _level_at
            dumps = sorted((directory / "bin").glob("*.mhd_w.*.bin"))
            _, blocks = _read_binary_mesh(dumps[-1])
            counts.append(len(blocks))
            if stage["name"] == "outer":
                assert max(level for level, _ in blocks) == 0
            if stage["name"] in ("resolve", "shrink", "settle"):
                # sep=4; the holes move only slightly during this short run.
                assert _level_at(blocks, (2, 0, 0)) == 2, blocks
                assert _level_at(blocks, (-2, 0, 0)) == 2, blocks
        assert counts[0] < counts[1] < counts[2], counts
        print(f"{model}: five-stage zoom passed; block counts {counts}", flush=True)
    if not keep_artifacts:
        shutil.rmtree(work)
    return work


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--athena", required=True)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()
    run_zoom_regression(args.athena, args.keep_artifacts)
