"""Z4c overhaul feature regressions; set ATHENA_OVERHAUL_EXE to the binary."""
import numpy as np
import pytest

from .overhaul_utils import run_case, table


def test_superposed_static(tmp_path):
    run_case(tmp_path, """
<problem>
pgen_name=z4c_superposed_punctures
punc_1_rest_mass=0.4
punc_2_rest_mass=0.6
punc_1_center_x1=-1
punc_2_center_x1=2
""")
    data = table(tmp_path)
    x, y, z = data["x1v"], 0.5625, 0.5625
    r1 = np.sqrt((x+1)**2+y*y+z*z)
    r2 = np.sqrt((x-2)**2+y*y+z*z)
    expected = (1+0.2/r1)**4 + (1+0.3/r2)**4 - 1
    for name in ("adm_gxx", "adm_gyy", "adm_gzz"):
        np.testing.assert_allclose(data[name], expected, rtol=1e-13)
    for name in ("adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz"):
        assert np.max(np.abs(data[name])) == 0


@pytest.mark.parametrize("velocity", [-1.0, 1.0, 1.1])
def test_superposed_invalid_velocity(tmp_path, velocity):
    result = run_case(tmp_path, f"""
<problem>
pgen_name=z4c_superposed_punctures
punc_1_velocity_x1={velocity}
""", success=False)
    assert "Invalid puncture" in result.stderr


def test_superposed_boost_reversal(tmp_path):
    fields = []
    for sign in (-1, 1):
        directory = tmp_path / str(sign)
        run_case(directory, f"""
<problem>
pgen_name=z4c_superposed_punctures
punc_1_velocity_x1={sign*0.3}
punc_2_velocity_x1={sign*-0.2}
""")
        fields.append(table(directory))
    for name in ("adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kzz"):
        np.testing.assert_allclose(fields[0][name], -fields[1][name], atol=1e-14)
    np.testing.assert_allclose(fields[0]["adm_gxx"], fields[1]["adm_gxx"])
