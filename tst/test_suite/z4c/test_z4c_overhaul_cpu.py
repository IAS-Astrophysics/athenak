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


@pytest.mark.parametrize("tau,kappa", [(0, 1), (-1, 1), (1, -1)])
def test_invalid_telegraph(tmp_path, tau, kappa):
    result = run_case(tmp_path, f"""
<z4c>
telegraph_lapse=true
telegraph_tau={tau}
telegraph_kappa={kappa}
""", success=False)
    assert "Telegraph lapse requires" in result.stderr


def test_telegraph_lapse(tmp_path):
    outputs = []
    for enabled in (False, True):
        directory = tmp_path / str(enabled)
        run_case(directory, f"""
<problem>
pgen_name=z4c_superposed_punctures
<time>
nlim=2
cfl_number=0.01
<z4c>
telegraph_lapse={str(enabled).lower()}
<output1>
variable=z4c
""")
        outputs.append(table(directory))
    assert np.max(np.abs(outputs[0]["z4c_Bx"])) == 0
    assert np.max(np.abs(outputs[1]["z4c_Bx"])) > 1e-10
    assert np.max(np.abs(outputs[1]["z4c_alpha"]-outputs[0]["z4c_alpha"])) > 1e-14


@pytest.mark.parametrize("order,ng", [(2, 2), (4, 3), (6, 4)])
def test_spatial_order_ghost_independence(tmp_path, order, ng):
    outputs = []
    for ghosts in (ng, 4):
        directory = tmp_path / str(ghosts)
        run_case(directory, f"""
<mesh>
nghost={ghosts}
<problem>
amp=1e-6
<time>
nlim=3
cfl_number=0.01
<z4c>
spatial_order={order}
<output1>
variable=z4c
""")
        outputs.append(table(directory))
    for name in outputs[0]:
        if name.startswith("z4c_"):
            np.testing.assert_allclose(outputs[0][name], outputs[1][name],
                                       rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("order,ghosts", [(3, 4), (6, 2)])
def test_invalid_spatial_order(tmp_path, order, ghosts):
    result = run_case(tmp_path, f"""
<mesh>
nghost={ghosts}
<z4c>
spatial_order={order}
""", success=False)
    assert "spatial_order must be" in result.stderr


@pytest.mark.parametrize("order,ghosts", [(2, 2), (4, 3), (6, 4)])
def test_flat_curvature(tmp_path, order, ghosts):
    run_case(tmp_path, f"""
<mesh>
nghost={ghosts}
<z4c>
spatial_order={order}
<output1>
variable=z4c_diag
""")
    data = table(tmp_path)
    for name, values in data.items():
        if name.startswith("z4c_"):
            assert np.max(np.abs(values)) < 1e-12


def test_super_poynting_metric_norm(tmp_path):
    fields = []
    for output in ("adm", "z4c_diag"):
        directory = tmp_path / output
        run_case(directory, f"""
<problem>
pgen_name=z4c_superposed_punctures
punc_1_velocity_x1=0.3
punc_2_velocity_x1=-0.2
<output1>
variable={output}
""")
        fields.append(table(directory))
    g, diag = fields
    n = len(g["x1v"])
    matrices = []
    for prefix, data in (("adm_g", g), ("z4c_E", diag), ("z4c_B", diag)):
        tensor = np.zeros((n, 3, 3))
        for a, x in enumerate("xyz"):
            for b, y in enumerate("xyz"):
                tensor[:, a, b] = data[prefix+"".join(sorted(x+y))]
        matrices.append(tensor)
    metric, electric, magnetic = matrices
    inverse = np.linalg.inv(metric)
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
    expected_p = -np.einsum("abc,nbd,nde,nce->na", epsilon, electric,
                            inverse, magnetic)/np.sqrt(np.linalg.det(metric))[:, None]
    actual_p = np.array([diag["z4c_P"+a] for a in "xyz"]).T
    np.testing.assert_allclose(actual_p, expected_p, rtol=1e-11, atol=1e-16)
    expected_norm = np.sqrt(np.einsum("na,nab,nb->n", actual_p, metric, actual_p))
    assert expected_norm.max() > 1e-8
    np.testing.assert_allclose(diag["z4c_Pnorm"], expected_norm, rtol=1e-12)
    assert np.max(np.abs(expected_norm-np.linalg.norm(actual_p, axis=1))) > 1e-8
    # Weyl tensors must be symmetric and trace-free in the physical metric.
    np.testing.assert_allclose(np.einsum("nab,nab->n", inverse, electric), 0, atol=1e-13)
    np.testing.assert_allclose(np.einsum("nab,nab->n", inverse, magnetic), 0, atol=1e-13)


def test_single_curvature_output(tmp_path):
    run_case(tmp_path, "\n<output1>\nvariable=z4c_Pnorm\n")
    assert np.max(np.abs(table(tmp_path)["z4c_Pnorm"])) < 1e-12


@pytest.mark.parametrize("cap", [0, 1])
def test_refinement_cap(tmp_path, cap):
    run_case(tmp_path, f"""
<problem>
pgen_name=z4c_superposed_punctures
<mesh_refinement>
refinement=adaptive
num_levels=3
max_nmb_per_rank=64
<z4c_amr>
method=chi
chi_min=2
max_ref_lev={cap}
<time>
nlim=3
cfl_number=0.01
""")
    data = table(tmp_path)
    assert len(np.unique(data["x1v"])) == 8*2**cap


def test_cartesian_interpolation(tmp_path):
    result = run_case(tmp_path, """
<problem>
pgen_name=z4c_interpolation
<mesh>
nx1=16
""")
    assert "PASS Cartesian polynomial interpolation" in result.stdout


def test_cce_harmonics(tmp_path):
    (tmp_path / "cce").mkdir()
    run_case(tmp_path, """
<problem>
pgen_name=z4c_interpolation
check_cce=true
<mesh>
nx1=16
x1min=-1
x1max=1
x2min=-1
x2max=1
x3min=-1
x3max=1
<cce>
rin_0=0.2
rout_0=0.4
num_l_modes=2
num_radial_modes=3
""")
    path = next((tmp_path / "cce").glob("*.bin"))
    with path.open("rb") as f:
        nr, lmax = np.fromfile(f, dtype=np.int32, count=2)
        time, rin, rout = np.fromfile(f, dtype=np.float64, count=3)
        values = np.fromfile(f, dtype=np.float64).reshape(2, nr, 10, (lmax+1)**2)
    radii = (rin+rout)/2-(rout-rin)/2*np.cos(np.pi*np.arange(1, nr+1)/(nr+1))
    # alpha=1+y: Y00 is sqrt(4*pi); Im(a_11)=r*sqrt(2*pi/3).
    np.testing.assert_allclose(values[0, :, 0, 0], np.sqrt(4*np.pi), atol=1e-12)
    np.testing.assert_allclose(values[1, :, 0, 3], radii*np.sqrt(2*np.pi/3), atol=1e-12)
    np.testing.assert_allclose(values[0, :, 0, 1:], 0, atol=1e-12)
