"""Compare split kernels with a saved executable from before the kernel change."""
import os

import numpy as np
import pytest

from .overhaul_utils import run_case, table


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("telegraph", [False, True])
@pytest.mark.parametrize("output", ["z4c", "con"])
def test_kernel_equivalence(tmp_path, order, telegraph, output):
    baseline = os.environ.get("ATHENA_REFERENCE_EXE")
    if not baseline:
        pytest.skip("Set ATHENA_REFERENCE_EXE to the pre-split-kernel executable")
    settings = f"""
<problem>
pgen_name=z4c_superposed_punctures
punc_1_rest_mass=0.1
punc_2_rest_mass=0.2
punc_1_velocity_x1=0.2
punc_2_velocity_x1=-0.1
<z4c>
spatial_order={order}
telegraph_lapse={str(telegraph).lower()}
<time>
nlim=4
cfl_number=0.01
<output1>
variable={output}
"""
    run_case(tmp_path / "reference", settings, executable=baseline)
    run_case(tmp_path / "split", settings)
    actual, expected = table(tmp_path / "split"), table(tmp_path / "reference")
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-11, atol=2e-13,
                                   err_msg=key)
