"""Binary-to-HDF5 coordinates must describe the actual output cell subset."""
import importlib.util

import h5py
import numpy as np
import pytest

from .overhaul_utils import ROOT, run_case

spec = importlib.util.spec_from_file_location("bin_convert", ROOT / "vis/python/bin_convert.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


@pytest.mark.parametrize("subset", ["full", "ghost", "slice"])
def test_binary_coordinates(tmp_path, subset):
    extra = "\n<output1>\nfile_type=bin\nvariable=z4c\n"
    if subset == "ghost":
        extra += "ghost_zones=true\n"
    if subset == "slice":
        extra += "slice_x1=0.7\nslice_x2=0.4\nslice_x3=0.2\n"
    run_case(tmp_path, extra, slices=False)
    path = next((tmp_path / "bin").glob("*.bin"))
    data = converter.read_binary(str(path))
    output = tmp_path / "converted.athdf"
    converter.write_athdf(str(output), data)
    with h5py.File(output) as f:
        for axis in range(1, 4):
            if subset == "ghost":
                expected = np.arange(-4, 13)/8
            elif subset == "slice":
                index = int((0.7, 0.4, 0.2)[axis-1]*8)
                expected = np.arange(index, index+2)/8
            else:
                expected = np.arange(9)/8
            np.testing.assert_allclose(f[f"x{axis}f"][0], expected, atol=1e-14)
            np.testing.assert_allclose(f[f"x{axis}v"][0], (expected[1:]+expected[:-1])/2)


def test_telegraph_reflection(tmp_path):
    extra = """
<problem>
pgen_name=z4c_superposed_punctures
<z4c>
telegraph_lapse=true
<time>
nlim=2
cfl_number=0.01
<mesh>
ix1_bc=reflect
ox1_bc=reflect
ix2_bc=reflect
ox2_bc=reflect
ix3_bc=reflect
ox3_bc=reflect
<output1>
file_type=bin
variable=z4c
ghost_zones=true
"""
    run_case(tmp_path, extra, slices=False)
    path = sorted((tmp_path / "bin").glob("*.bin"))[-1]
    data = converter.read_binary(str(path))
    assert np.max(np.abs(data["mb_data"]["z4c_Bx"])) > 1e-10
    for component, letter in enumerate("xyz"):
        field = data["mb_data"]["z4c_B"+letter][0]
        for axis in range(3):
            values = np.moveaxis(field, 2-axis, 0)
            sign = -1 if axis == component else 1
            # Check face interiors, avoiding corner update-order dependence.
            np.testing.assert_allclose(values[:4, 4:12, 4:12],
                                       sign*values[7:3:-1, 4:12, 4:12], atol=1e-14)
            np.testing.assert_allclose(values[12:, 4:12, 4:12],
                                       sign*values[11:7:-1, 4:12, 4:12], atol=1e-14)
