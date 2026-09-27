"""HDF5 import contracts, including noncubic layout and invalid input rejection."""
import os
import numpy as np
import pytest
import h5py

from .overhaul_utils import run_case, table


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "initial.h5"
    x, y, z = [np.linspace(-1, 2, n) for n in (13, 15, 17)]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    phi = 1 + 0.01*xx + 0.02*yy**2 + 0.03*zz**3
    data = np.zeros((6, 1, len(z), len(y), len(x)))
    data[[0, 3, 5], 0] = phi
    with h5py.File(path, "w") as f:
        for name, coord in zip(("x1v", "x2v", "x3v"), (x, y, z)):
            f[name] = coord[None, :]
        f["metric"] = data
        f["extrin"] = np.zeros_like(data)
    return path


def import_data(directory, source, success=True):
    exe = os.environ.get("ATHENA_ID_EXE")
    if not exe:
        pytest.skip("Set ATHENA_ID_EXE to a PROBLEM=id_solve build")
    return run_case(directory, f"\n<problem>\nid_filename={source}\n",
                    executable=exe, success=success)


def test_noncubic_polynomial(tmp_path, source):
    import_data(tmp_path / "run", source)
    data = table(tmp_path / "run")
    expected = 1+0.01*data["x1v"]+0.02*0.5625**2+0.03*0.5625**3
    for name in ("adm_gxx", "adm_gyy", "adm_gzz"):
        np.testing.assert_allclose(data[name], expected, rtol=3e-14)


@pytest.mark.parametrize("fault", ["rank", "shape", "coordinates", "coverage", "nan"])
def test_invalid_hdf5(tmp_path, source, fault):
    with h5py.File(source, "r+") as f:
        if fault in ("rank", "shape"):
            del f["extrin"]
            f["extrin"] = np.zeros((6, 1) if fault == "rank" else (6, 1, 13, 15, 17))
        elif fault == "coordinates":
            f["x1v"][0, 2] = f["x1v"][0, 1]
        elif fault == "coverage":
            f["x1v"][:] += 10
        else:
            f["metric"][:] = np.nan
    result = import_data(tmp_path / "run", source, success=False)
    assert "id_solve:" in result.stderr
