"""Unit tests for the maintained circumbinary-disk post-processor."""

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[3] / "scripts" / "cbd_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("cbd_diagnostics", SCRIPT)
cbd = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cbd
SPEC.loader.exec_module(cbd)


def _dump(variables, arrays):
    block = cbd.BinaryBlock(
        (0, 0, 0, 0), (0.0, 2.0, -0.5, 0.5, -0.5, 0.5), (1, 1, 2),
        {name: np.asarray(value, dtype=float).reshape(1, 1, 2)
         for name, value in zip(variables, arrays)})
    parameters = {
        "job": {"basename": "unit"},
        "meshblock": {"nx1": "2", "nx2": "1", "nx3": "1"},
        "adm": {"dynamic": "true"},
        "problem": {"sep": "4", "q": "2", "a1": "0.1", "a2": "0.2"},
    }
    return cbd.BinaryDump((Path("bin/unit.bin"),), 1.0, 2, tuple(variables),
                          parameters, [block])


def test_block_scoped_input_parser():
    parsed = cbd.parse_athena_input("""
        <mhd>
        gamma = 1.4 # gas
        <problem>
        gamma = 2.0
        q = 3
    """)
    assert parsed["mhd"]["gamma"] == "1.4"
    assert parsed["problem"]["gamma"] == "2.0"
    assert parsed["problem"]["q"] == "3"


def test_radial_helpers():
    indices, valid = cbd.radial_bin_indices(
        np.array([-1.0, 0.0, 0.99, 1.0, 2.0, np.nan]), 0.0, 1.0, 2)
    np.testing.assert_array_equal(indices[:5], [-1, 0, 0, 1, 2])
    np.testing.assert_array_equal(valid, [False, True, True, True, False, False])
    np.testing.assert_allclose(cbd.annulus_areas([0.0, 1.0, 2.0]),
                               [math.pi, 3.0*math.pi])


def test_volume_integrals_use_densitized_mass_and_coordinate_volume_once():
    primitive = _dump(("dens", "press"), ([2.0, 4.0], [0.2, 0.4]))
    conserved = _dump(("dens",), ([3.0, 5.0],))
    angular = _dump(
        ("Jx", "Jy", "Jz", "JEMx", "JEMy", "JEMz"),
        ([1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]))
    torque = _dump(("Tx", "Ty", "Tz"), ([1, 2], [3, 4], [5, 6]))
    summary, profile = cbd.analyze_volume_dump(
        primitive, conserved, angular, torque, 0.0, 2.0, 1.0, 0.0,
        "cylindrical")
    np.testing.assert_allclose(profile["rest_mass"], [3.0, 5.0])
    np.testing.assert_allclose(profile["rho_coordinate_volume_mean"], [2.0, 4.0])
    assert summary["disk_rest_mass"] == 8.0
    assert summary["characteristic_radius"] == 1.125
    assert summary["density_max_radius"] == 1.5
    np.testing.assert_allclose(
        [summary["Jgas_x"], summary["Jgas_y"], summary["Jgas_z"]], [2, 4, 6])
    np.testing.assert_allclose(
        [summary["torque_x"], summary["torque_y"], summary["torque_z"]], [3, 7, 11])

    selected, _ = cbd.analyze_volume_dump(
        primitive, conserved, angular, torque, 0.0, 1.0, 1.0, 0.0,
        "cylindrical")
    assert selected["disk_rest_mass"] == 3.0
    np.testing.assert_allclose(
        [selected["Jgas_x"], selected["Jgas_y"], selected["Jgas_z"]], [1, 2, 3])
    np.testing.assert_allclose(
        [selected["torque_x"], selected["torque_y"], selected["torque_z"]],
        [1, 3, 5])


def test_current_surface_history_group_is_recovered_despite_label_truncation():
    surface = "r100"
    record = {"time": 2.0, "dt": 0.1}
    for n, quantity in enumerate(cbd.SURFACE_QUANTITIES):
        record[f"{quantity}_{surface}"[:10]] = float(n)
    rows = cbd.extract_surface_fluxes([record], source="test.hst")
    assert len(rows) == len(cbd.SURFACE_QUANTITIES)
    assert rows[0] == {
        "source": "test.hst", "time": 2.0, "dt": 0.1,
        "surface": "r100", "quantity": "mdot", "value": 0.0,
    }
    assert rows[-1]["quantity"] == "area"
    assert rows[-1]["value"] == 16.0
