#!/usr/bin/env python3
"""Generate residual-Z4c instability-isolation input decks."""

from pathlib import Path
from typing import Dict


OUT = Path("/home/hzhu/athenak_tde/inputs/tde/aurora")


def section(name: str, values: Dict[str, object]) -> str:
    lines = [f"<{name}>"]
    for key, value in values.items():
        lines.append(f"{key} = {value}")
    return "\n".join(lines)


def base_deck(
    basename: str,
    comment: str,
    *,
    xmin: float = -4.8,
    xmax: float = 4.8,
    ymin: float = None,
    ymax: float = None,
    zmin: float = None,
    zmax: float = None,
    nx: int = 256,
    ny: int = None,
    nz: int = None,
    star_x: float = 1.0,
    boost_x: float = -0.1,
    tlim: float = 4.0,
    use_background: bool = True,
    gauge: str = "full",
    kappa1: float = 0.2,
    mesh_refinement: str = "",
    refined_regions: str = "",
    star_refine: bool = False,
    star_refine_level: int = 0,
    star_refine_radius_factor: float = 1.5,
    amr_rho_slope_refine: bool = False,
    bh_mass: float = 0.0,
    coord_minkowski: bool = True,
    excise_history: bool = False,
    bh_exclusion_radius: float = 0.0,
    bh_refine_radius: float = 0.0,
    bh_derefine_radius: float = 0.0,
    bh_refine_level: int = -1,
) -> str:
    ymin = xmin if ymin is None else ymin
    ymax = xmax if ymax is None else ymax
    zmin = xmin if zmin is None else zmin
    zmax = xmax if zmax is None else zmax
    ny = nx if ny is None else ny
    nz = nx if nz is None else nz
    if gauge == "frozen":
        evolve_gauge = evolve_lapse = evolve_shift = "false"
        lapse_advect = shift_gamma = shift_advect = shift_eta = 0.0
    elif gauge == "lapse":
        evolve_gauge = evolve_lapse = "true"
        evolve_shift = "false"
        lapse_advect = 1.0
        shift_gamma = shift_advect = shift_eta = 0.0
    elif gauge == "shift":
        evolve_gauge = evolve_shift = "true"
        evolve_lapse = "false"
        lapse_advect = 0.0
        shift_gamma = 1.0
        shift_advect = 1.0
        shift_eta = 2.0
    elif gauge == "full":
        evolve_gauge = evolve_lapse = evolve_shift = "true"
        lapse_advect = shift_gamma = shift_advect = 1.0
        shift_eta = 2.0
    else:
        raise ValueError(f"unknown gauge {gauge}")

    mesh = section("mesh", {
        "nghost": 4,
        "nx1": nx,
        "x1min": xmin,
        "x1max": xmax,
        "ix1_bc": "outflow",
        "ox1_bc": "outflow",
        "nx2": ny,
        "x2min": ymin,
        "x2max": ymax,
        "ix2_bc": "outflow",
        "ox2_bc": "outflow",
        "nx3": nz,
        "x3min": zmin,
        "x3max": zmax,
        "ix3_bc": "outflow",
        "ox3_bc": "outflow",
    })
    z4c = section("z4c", {
        "use_analytic_background": str(use_background).lower(),
        "evolve_gauge_residual": evolve_gauge,
        "evolve_lapse_residual": evolve_lapse,
        "evolve_shift_residual": evolve_shift,
        "diss": 0.5,
        "damp_kappa1": kappa1,
        "damp_kappa2": 0.0,
        "lapse_harmonic": 0.0,
        "lapse_oplog": 2.0,
        "lapse_advect": lapse_advect,
        "shift_Gamma": shift_gamma,
        "shift_advect": shift_advect,
        "shift_eta": shift_eta,
        "history_excise_ks_horizon": str(excise_history).lower(),
    })
    problem = section("problem", {
        "pgen_name": "z4c_tov_ks",
        "user_hist": "true",
        "metric_diag_history": "true",
        "bh_mass": bh_mass,
        "bh_spin": 0.0,
        "bh_center_x1": 0.0,
        "bh_center_x2": 0.0,
        "bh_center_x3": 0.0,
        "use_direct_z4c_background": "true",
        "force_minkowski_metric": "false",
        "star_center_x1": star_x,
        "star_center_x2": 0.0,
        "star_center_x3": 0.0,
        "star_boost_x": boost_x,
        "star_boost_y": 0.0,
        "star_boost_z": 0.0,
        "isotropic": "true",
        "rhoc": "1.236298759450e-04",
        "kappa": "3.639390619285e-05",
        "npoints": 200000,
        "dr": "1.0e-5",
        "rho_cut": "1.0e-16",
        "excision_damp_rate": 0.0 if bh_mass == 0.0 else 50.0,
        "excision_atmo_density": "1.0e-16",
        "excision_atmo_energy": "1.0e-22",
        "amr_rho_slope_refine": str(amr_rho_slope_refine).lower(),
        "amr_bh_exclusion_radius": bh_exclusion_radius,
        "amr_bh_refine_radius": bh_refine_radius,
        "amr_bh_derefine_radius": bh_derefine_radius,
        "amr_bh_refine_level": bh_refine_level,
        "amr_star_refine": str(star_refine).lower(),
        "amr_star_refine_radius_factor": star_refine_radius_factor if star_refine else 0.0,
        "amr_star_derefine_radius_factor": 2.0 if star_refine else 0.0,
        "amr_star_refine_level": star_refine_level,
        "b_norm": 0.0,
        "pcut": 0.04,
        "use_pcut_rel": "true",
        "magindex": 1.0,
    })

    parts = [
        f"# {comment}",
        "",
        section("job", {"basename": basename}),
        "",
        mesh,
        "",
        section("meshblock", {"nx1": 32, "nx2": 32, "nx3": 32}),
    ]
    if mesh_refinement:
        parts.extend(["", mesh_refinement])
    if refined_regions:
        parts.extend(["", refined_regions])
    parts.extend([
        "",
        section("time", {
            "evolution": "dynamic",
            "integrator": "rk2",
            "cfl_number": 0.2,
            "nlim": -1,
            "tlim": tlim,
            "ndiag": 20,
        }),
        "",
        section("mhd", {
            "evolution": "dynamic",
            "eos": "ideal",
            "dyn_eos": "ideal",
            "dyn_error": "reset_floor",
            "reconstruct": "wenoz",
            "rsolver": "llf",
            "fofc_method": "llf",
            "gamma": "1.3333333333333333",
            "dfloor": "1.0e-16",
            "pfloor": "1.0e-22",
            "dthreshold": 1.0,
            "gamma_max": 10.0,
            "fofc": "false",
        }),
        "",
        "<adm>",
        "",
        section("coord", {
            "general_rel": "true",
            "is_dynamical": "true",
            "minkowski": str(coord_minkowski).lower(),
            "a": 0.0,
            "excise": "false",
        }),
        "",
        z4c,
        "",
        problem,
        "",
        section("output1", {
            "file_type": "hst",
            "dt": 0.05,
            "data_format": "%20.15e",
        }),
        "",
        section("output2", {
            "file_type": "log",
            "dcycle": 20,
        }),
        "",
        section("output3", {
            "file_type": "bin",
            "variable": "con",
            "id": "xy_con",
            "dt": 0.1,
            "slice_x3": 0.0,
        }),
        "",
        section("output4", {
            "file_type": "bin",
            "variable": "con",
            "id": "xz_con",
            "dt": 0.1,
            "slice_x2": 0.0,
        }),
        "",
        section("output5", {
            "file_type": "bin",
            "variable": "con",
            "id": "yz_con",
            "dt": 0.1,
            "slice_x1": star_x,
        }),
    ])
    return "\n".join(parts) + "\n"


def write(name: str, text: str) -> None:
    path = OUT / name
    path.write_text(text, encoding="utf-8")
    print(path)


def generate_gauge_ladder() -> None:
    prefix = "z4c_tov_ks_n3_minkowski_boosted_residual_uniform"
    common = {
        "comment": "Gauge-ladder residual-Z4c case: boosted n=3 TOV star, larger uniform Minkowski box.",
        "xmin": -4.8,
        "xmax": 4.8,
        "nx": 256,
        "star_x": 1.0,
        "boost_x": -0.1,
        "tlim": 4.0,
        "use_background": True,
    }
    for suffix, gauge in [
        ("frozengauge", "frozen"),
        ("lapseonly", "lapse"),
        ("shiftonly", "shift"),
        ("fullgauge_bigbox", "full"),
    ]:
        basename = f"{prefix}_{suffix}_aurora"
        write(f"{basename}.athinput", base_deck(basename, gauge=gauge, **common))


def smr_regions() -> str:
    return "\n\n".join([
        section("refined_region1", {
            "level": 1,
            "x1min": -1.2,
            "x1max": 2.4,
            "x2min": -1.2,
            "x2max": 1.2,
            "x3min": -1.2,
            "x3max": 1.2,
        }),
    ])


def generate_amr_isolation() -> None:
    common = {
        "comment": "AMR-isolation residual-Z4c case: boosted n=3 TOV star, Minkowski full gauge.",
        "xmin": -4.8,
        "xmax": 4.8,
        "nx": 128,
        "star_x": 1.0,
        "boost_x": -0.1,
        "tlim": 4.0,
        "use_background": True,
        "gauge": "full",
    }
    smr_mesh = section("mesh_refinement", {
        "refinement": "static",
        "num_levels": 2,
        "refinement_interval": 2,
        "max_nmb_per_rank": 256,
    })
    dyn_mesh = section("mesh_refinement", {
        "refinement": "adaptive",
        "num_levels": 2,
        "refinement_interval": 2,
        "max_nmb_per_rank": 256,
    }) + "\n\n" + section("amr_criterion0", {"method": "user"})
    basename = "z4c_tov_ks_n3_minkowski_boosted_residual_smr_fullgauge_bigbox_aurora"
    write(f"{basename}.athinput", base_deck(
        basename, mesh_refinement=smr_mesh, refined_regions=smr_regions(),
        star_refine=False, **common))
    basename = "z4c_tov_ks_n3_minkowski_boosted_residual_amr_fullgauge_bigbox_aurora"
    write(f"{basename}.athinput", base_deck(
        basename, mesh_refinement=dyn_mesh, refined_regions=smr_regions(),
        star_refine=True, star_refine_level=1, star_refine_radius_factor=1.5,
        amr_rho_slope_refine=True, **common))


def schwarzschild_regions() -> str:
    return "\n\n".join([
        section("refined_region1", {
            "level": 1,
            "x1min": 1.6,
            "x1max": 14.4,
            "x2min": -6.4,
            "x2max": 6.4,
            "x3min": -6.4,
            "x3max": 6.4,
        }),
        section("refined_region2", {
            "level": 2,
            "x1min": 4.8,
            "x1max": 11.2,
            "x2min": -3.2,
            "x2max": 3.2,
            "x3min": -3.2,
            "x3max": 3.2,
        }),
        section("refined_region3", {
            "level": 3,
            "x1min": 6.4,
            "x1max": 9.6,
            "x2min": -1.6,
            "x2max": 1.6,
            "x3min": -1.6,
            "x3max": 1.6,
        }),
        section("refined_region4", {
            "level": 1,
            "x1min": -3.2,
            "x1max": 3.2,
            "x2min": -3.2,
            "x2max": 3.2,
            "x3min": -3.2,
            "x3max": 3.2,
        }),
    ])


def schwarzschild_regions_to_level(max_star_level: int, star_x: float = 8.0) -> str:
    regions = []
    boxes = [
        (1, star_x - 6.4, star_x + 6.4, -6.4, 6.4),
        (2, star_x - 3.2, star_x + 3.2, -3.2, 3.2),
        (3, star_x - 1.6, star_x + 1.6, -1.6, 1.6),
        (4, star_x - 0.8, star_x + 0.8, -0.8, 0.8),
    ]
    for level, xlo, xhi, ylo, yhi in boxes:
        if level <= max_star_level:
            regions.append(section(f"refined_region{len(regions) + 1}", {
                "level": level,
                "x1min": xlo,
                "x1max": xhi,
                "x2min": ylo,
                "x2max": yhi,
                "x3min": ylo,
                "x3max": yhi,
            }))
    regions.append(section(f"refined_region{len(regions) + 1}", {
        "level": 1,
        "x1min": -3.2,
        "x1max": 3.2,
        "x2min": -3.2,
        "x2max": 3.2,
        "x3min": -3.2,
        "x3max": 3.2,
    }))
    return "\n\n".join(regions)


def generate_schwarzschild_reintro() -> None:
    mesh = section("mesh_refinement", {
        "refinement": "static",
        "num_levels": 4,
        "refinement_interval": 2,
        "max_nmb_per_rank": 512,
    })
    common = {
        "comment": "Schwarzschild reintroduction residual-Z4c case: fixed SMR, star far from BH.",
        "xmin": -12.8,
        "xmax": 25.6,
        "nx": 128,
        "star_x": 8.0,
        "tlim": 4.0,
        "use_background": True,
        "mesh_refinement": mesh,
        "refined_regions": schwarzschild_regions(),
        "bh_mass": 1.0,
        "coord_minkowski": False,
        "excise_history": True,
        "bh_exclusion_radius": 8.0,
        "bh_refine_radius": 2.4,
        "bh_derefine_radius": 3.2,
        "bh_refine_level": 1,
    }
    for motion, boost_x in [("static", 0.0), ("boosted", -0.1)]:
        for gauge_suffix, gauge in [("frozengauge", "frozen"), ("fullgauge", "full")]:
            basename = f"z4c_tov_ks_n3_schwarzschild_{motion}_residual_smr_{gauge_suffix}_reintro_aurora"
            write(f"{basename}.athinput", base_deck(
                basename, boost_x=boost_x, gauge=gauge, **common))


def generate_bridge_grid() -> None:
    # Fixed-SMR ladder around the stable Schwarzschild reintroduction geometry.
    for label, root_dx, nx, max_star_level in [
        ("dx0025", 0.2, 192, 3),
        ("dx00125", 0.2, 192, 4),
    ]:
        mesh = section("mesh_refinement", {
            "refinement": "static",
            "num_levels": max_star_level + 1,
            "refinement_interval": 2,
            "max_nmb_per_rank": 1024,
        })
        basename = f"z4c_tov_ks_n3_schwarzschild_boosted_residual_smr_fullgauge_{label}_bridge_aurora"
        write(f"{basename}.athinput", base_deck(
            basename,
            "Bridge-grid Schwarzschild residual-Z4c case: fixed SMR refinement ladder.",
            xmin=-12.8, xmax=25.6, ymin=-6.4, ymax=6.4, zmin=-6.4, zmax=6.4,
            nx=nx, ny=64, nz=64,
            star_x=8.0, boost_x=-0.1, tlim=4.0, use_background=True,
            gauge="full", mesh_refinement=mesh,
            refined_regions=schwarzschild_regions_to_level(max_star_level, star_x=8.0),
            bh_mass=1.0, coord_minkowski=False, excise_history=True,
            bh_exclusion_radius=8.0, bh_refine_radius=2.4,
            bh_derefine_radius=3.2, bh_refine_level=1))

    # Production geometry with controlled changes to star refinement and AMR mode.
    template_path = OUT / "z4c_tov_ks_n3_schwarzschild_resgauge_full_kappa020_hi2n_aurora.athinput"
    template = template_path.read_text(encoding="utf-8")
    for label, star_level, num_levels, refinement in [
        ("prodgrid_level3", 3, 4, "adaptive"),
        ("prodgrid_level2", 2, 3, "adaptive"),
        ("prodgrid_level4_static_smr", 4, 5, "static"),
    ]:
        basename = f"z4c_tov_ks_n3_schwarzschild_resgauge_full_{label}_bridge_aurora"
        text = template
        text = replace_assignment(text, "basename", basename)
        text = replace_assignment(text, "tlim", 4.0)
        text = replace_assignment(text, "dt", 0.1)
        text = replace_assignment(text, "refinement", refinement)
        text = replace_assignment(text, "num_levels", num_levels)
        text = replace_assignment(text, "amr_star_refine_level", star_level)
        if star_level < 4:
            text = remove_refined_region(text, "refined_region4")
        if star_level < 3:
            text = remove_refined_region(text, "refined_region3")
        if refinement == "static":
            text = replace_assignment(text, "amr_star_refine", "false")
            text = replace_assignment(text, "amr_rho_slope_refine", "false")
        write(f"{basename}.athinput", text)

    basename = "z4c_tov_ks_n3_schwarzschild_resgauge_full_prodgrid_level4_static_smr_debugdiag_aurora"
    text = template
    text = replace_assignment(text, "basename", basename)
    text = replace_assignment(text, "tlim", 3.3)
    text = replace_assignment(text, "dt", 0.05)
    text = replace_assignment(text, "refinement", "static")
    text = replace_assignment(text, "num_levels", 5)
    text = replace_assignment(text, "amr_star_refine", "false")
    text = replace_assignment(text, "amr_rho_slope_refine", "false")
    text = text.replace(
        "history_excise_ks_horizon = true\n",
        "history_excise_ks_horizon = true\n"
        "debug_reductions = true\n"
        "debug_reduction_stride = 20\n")
    write(f"{basename}.athinput", text)


def insert_after_assignment(text: str, key: str, insertion: str) -> str:
    prefix = f"{key} = "
    lines = []
    inserted = False
    for line in text.splitlines():
        lines.append(line)
        if not inserted and line.strip().startswith(prefix):
            lines.extend(insertion.splitlines())
            inserted = True
    if not inserted:
        raise ValueError(f"did not find assignment for {key}")
    return "\n".join(lines) + "\n"


def generate_gauge_matter_ablation() -> None:
    template_path = OUT / "z4c_tov_ks_n3_schwarzschild_resgauge_full_prodgrid_level4_static_smr_bridge_aurora.athinput"
    template = template_path.read_text(encoding="utf-8")
    cases = [
        ("frozengauge", {
            "evolve_gauge_residual": "false",
            "evolve_lapse_residual": "false",
            "evolve_shift_residual": "false",
            "lapse_advect": 0.0,
            "shift_Gamma": 0.0,
            "shift_advect": 0.0,
            "shift_eta": 0.0,
        }),
        ("lapseonly", {
            "evolve_gauge_residual": "true",
            "evolve_lapse_residual": "true",
            "evolve_shift_residual": "false",
            "lapse_advect": 1.0,
            "shift_Gamma": 0.0,
            "shift_advect": 0.0,
            "shift_eta": 0.0,
        }),
        ("shiftonly", {
            "evolve_gauge_residual": "true",
            "evolve_lapse_residual": "false",
            "evolve_shift_residual": "true",
            "lapse_advect": 0.0,
            "shift_Gamma": 1.0,
            "shift_advect": 1.0,
            "shift_eta": 2.0,
        }),
        ("lapse_noadvect", {"lapse_advect": 0.0}),
        ("shift_noadvect", {"shift_advect": 0.0}),
        ("static_star", {"star_boost_x": 0.0}),
    ]
    for suffix, replacements in cases:
        basename = f"z4c_tov_ks_n3_schwarzschild_resgauge_full_prodgrid_level4_static_smr_{suffix}_ablation_aurora"
        text = template
        text = replace_assignment(text, "basename", basename)
        text = replace_assignment(text, "tlim", 4.0)
        for key, value in replacements.items():
            text = replace_assignment(text, key, value)
        write(f"{basename}.athinput", text)

    basename = "z4c_tov_ks_n3_schwarzschild_resgauge_full_prodgrid_level4_static_smr_zero_tmunu_ablation_aurora"
    text = template
    text = replace_assignment(text, "basename", basename)
    text = replace_assignment(text, "tlim", 4.0)
    text = insert_after_assignment(text, "fofc", "zero_tmunu_feedback = true")
    text = insert_after_assignment(text, "rho_cut", "zero_tmunu = true")
    write(f"{basename}.athinput", text)


def replace_assignment(text: str, key: str, value: object) -> str:
    prefix = f"{key} = "
    lines = []
    replaced = False
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            lines.append(f"{key} = {value}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        raise ValueError(f"did not find assignment for {key}")
    return "\n".join(lines) + "\n"


def remove_refined_region(text: str, region_name: str) -> str:
    lines = text.splitlines()
    out = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped == f"<{region_name}>":
            skipping = True
            continue
        if skipping and stripped.startswith("<") and stripped.endswith(">"):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out) + "\n"


def generate_damping_sweep() -> None:
    template_path = OUT / "z4c_tov_ks_n3_schwarzschild_resgauge_full_kappa020_hi2n_aurora.athinput"
    template = template_path.read_text(encoding="utf-8")
    for label, kappa in [
        ("kappa000", 0.0),
        ("kappa020", 0.2),
        ("kappa050", 0.5),
        ("kappa100", 1.0),
        ("kappa200", 2.0),
    ]:
        basename = f"z4c_tov_ks_n3_schwarzschild_resgauge_full_{label}_hi2n_sweep_aurora"
        text = template
        text = replace_assignment(text, "basename", basename)
        text = replace_assignment(text, "tlim", 4.0)
        text = replace_assignment(text, "damp_kappa1", kappa)
        write(f"{basename}.athinput", text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generate_gauge_ladder()
    generate_amr_isolation()
    generate_schwarzschild_reintro()
    generate_bridge_grid()
    generate_gauge_matter_ablation()
    generate_damping_sweep()


if __name__ == "__main__":
    main()
