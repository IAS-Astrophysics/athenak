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
