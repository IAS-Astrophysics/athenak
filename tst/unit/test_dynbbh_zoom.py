"""Host-only zoom-plan validation; no solver, scheduler or numpy required."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "zoom", ROOT / "scripts/setup_dynbbh_zoom.py")
ZOOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ZOOM)


class ZoomPlanTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "inputs/dyn_grmhd/dynbbh_zoom_schedule.json"
        self.plan = json.loads(path.read_text())
        self.base = ROOT / "tst/inputs/dynbbh_radiation_cbd.athinput"

    def test_sparse_restart_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "zoom"
            ZOOM.generate(self.base, self.plan, out)
            fresh = ZOOM.read_input((out / "outer/stage.athinput").read_text())
            later = ZOOM.read_input((out / "inner/stage.athinput").read_text())
            self.assertIn("dyn_radiation", fresh)
            self.assertEqual(fresh["mesh_refinement"]["num_levels"], "6")
            self.assertEqual(fresh["time"]["nlim"], "-1")
            self.assertEqual(fresh["coord"]["excise_horizon_fraction"], "0.8")
            restart = next(p for p in fresh.values() if p.get("file_type") == "rst")
            self.assertEqual(restart["dt"], "20.0")
            self.assertNotIn("dcycle", restart)
            self.assertEqual(set(later), {"job", "time", "problem"})
            self.assertEqual(later["problem"]["tracker_1_reflevel"], "2")
            with self.assertRaises(FileExistsError):
                ZOOM.generate(self.base, self.plan, out)

    def test_invalid_schedules(self):
        for key, value in (("horizon_fraction", 0), ("horizon_fraction", 1.1),
                           ("horizon_fraction", float("nan")), ("max_level", 1.5),
                           ("shrink_start_time", 1000), ("shrink_start_time", 1550),
                           ("restart_dt", -1), ("shrink_timescale", 10000)):
            plan = copy.deepcopy(self.plan)
            plan[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ZOOM.validate(plan)
        for i, key, value in ((0, "tracker_level", 1), (2, "tracker_level", 3),
                              (4, "regions", [[30, 0]]), (1, "name", "../unsafe")):
            plan = copy.deepcopy(self.plan)
            plan["stages"][i][key] = value
            with self.subTest(stage=i, key=key), self.assertRaises(ValueError):
                ZOOM.validate(plan)

    def test_sink_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "base.athinput"
            source.write_text(self.base.read_text().replace("unresolved_sink = false",
                                                            "unresolved_sink = true"))
            out = Path(temporary) / "zoom"
            with self.assertRaisesRegex(ValueError, "unresolved_sink"):
                ZOOM.generate(source, self.plan, out)
            self.assertFalse(out.exists())

    def test_trajectory_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory = root / "orbit.traj"
            trajectory.write_text("# The solver validates the table data.\n")
            base = ZOOM.read_input(self.base.read_text())
            base["problem"].update(use_traj_table="true", traj_file="orbit.traj",
                                   radius_7_rad="3", radius_7_reflevel="4")
            base["output4"] = {"file_type": "rst", "dcycle": "9", "dt": "1",
                               "file_number": "42", "last_time": "99"}
            source = root / "base.athinput"
            source.write_text(ZOOM.render(base))
            out = ZOOM.generate(source, self.plan, root / "plan with spaces")
            fresh = ZOOM.read_input((out / "outer/stage.athinput").read_text())
            self.assertEqual(fresh["problem"]["traj_file"], str(trajectory.resolve()))
            self.assertEqual(fresh["problem"]["radius_7_reflevel"], "0")
            self.assertEqual(fresh["output4"], {"file_type": "rst", "dt": "20.0"})
            self.assertIn('"$ATHENA"', (out / "RUN.txt").read_text())


if __name__ == "__main__":
    unittest.main()
