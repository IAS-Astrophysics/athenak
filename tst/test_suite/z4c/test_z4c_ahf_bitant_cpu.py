"""
Regression test for bitant-aware apparent-horizon finding (FastFlow).

A single Brill-Lindquist puncture at the origin has an apparent horizon that is exactly
symmetric under reflection about z=0, and for M=1 in isotropic coordinates it sits at
r = M/2 with area = 16*pi*M^2 and irreducible mass M_irr = 1. This test evolves it on a
full domain and on a bitant (z>=0, reflecting) domain at matching resolution, and checks
that the horizon area and mass agree between the two runs and match the analytic answer.

Without bitant support, every surface point with z<0 falls outside the domain, so
IndicesAndWeights reports point_exist=false and FastFlow silently drops the point from
both the surface integrals and the spectral flow update. The result is roughly half the
true area and a mass low by a factor of ~sqrt(2) -- and, because convergence is only
tested via |mass_prev - mass| < mass_tol, the finder still reports the horizon as found.
"""

import glob
import math
import os

import pytest
import test_suite.testutils as testutils

# Column layout of <basename>.horizon_summary_0.txt (see FastFlow::Write in fastflow.cpp):
# 0:iter 1:time 2:mass 3:Sx 4:Sy 5:Sz 6:S 7:area 8:hrms 9:hmean 10:meanradius 11:minradius
IMASS, ISX, ISY, IAREA = 2, 3, 4, 7

# The full and bitant runs use identical MeshBlock geometry for z>0 (z=0 is a MeshBlock
# boundary in the full run), so they should agree far more tightly than either agrees with
# the continuum answer. The only sources of disagreement are last-ulp asymmetry in the
# Gauss-Legendre roots and floating-point reduction ordering.
RTOL_FULL_VS_HALF = 1.0e-6

# Absolute check against Schwarzschild. Deliberately loose -- at dx=0.125 with r_AH=0.5
# this is a coarsely resolved horizon. Its job is to catch the half-the-sphere-is-missing
# failure (a ~29% error in mass, ~50% in area), not to measure convergence order.
ATOL_SCHWARZSCHILD = 0.10

_SUMMARY_GLOB = "*.horizon_summary_*.txt"
_SHAPE_GLOB = "*.horizon_shape_*.txt"


def _cleanup_horizon_files():
    # FastFlow opens the summary file with fopen(..., "a") and only writes the header when
    # the file does not already exist, so stale files from a previous run would be
    # appended to rather than replaced. testutils.cleanup() only removes tab/ and *.dat,
    # so these have to be handled explicitly.
    for pattern in (_SUMMARY_GLOB, _SHAPE_GLOB):
        for path in glob.glob(pattern):
            os.remove(path)


def _run_and_read(athinput, basename):
    _cleanup_horizon_files()
    testutils.run(athinput, [])
    path = f"{basename}.horizon_summary_0.txt"
    if not os.path.exists(path):
        pytest.fail(f"horizon summary file {path} not produced")
    with open(path) as fh:
        rows = [ln for ln in fh if ln.strip() and not ln.startswith("#")]
    if not rows:
        pytest.fail(f"horizon summary file {path} contains no data rows")
    return [float(x) for x in rows[-1].split()]


def test_z4c_ahf_bitant():
    try:
        full = _run_and_read("inputs/z4c_ahf_bitant_full.athinput", "ahf_full")
        half = _run_and_read("inputs/z4c_ahf_bitant_half.athinput", "ahf_half")

        # 1. Absolute: Schwarzschild M=1 gives area = 16*pi and M_irr = 1. This is the
        #    check that catches the missing-hemisphere bug (mass ~0.707, area ~8*pi).
        for label, row in (("full", full), ("bitant", half)):
            area_ratio = row[IAREA] / (16.0 * math.pi)
            if abs(area_ratio - 1.0) > ATOL_SCHWARZSCHILD:
                pytest.fail(
                    f"{label} domain: horizon area {row[IAREA]:g} is "
                    f"{area_ratio:.4g}x the Schwarzschild value 16*pi. A ratio near 0.5 "
                    f"means surface points are being dropped from the integrals."
                )
            if abs(row[IMASS] - 1.0) > ATOL_SCHWARZSCHILD:
                pytest.fail(
                    f"{label} domain: horizon mass {row[IMASS]:g} differs from the "
                    f"Schwarzschild value 1.0 by more than {ATOL_SCHWARZSCHILD:g}. "
                    f"A value near 0.707 means half the surface is missing."
                )

        # 2. Relative: identical block geometry for z>0, so these should track closely.
        for idx, name in ((IMASS, "mass"), (IAREA, "area")):
            f, h = full[idx], half[idx]
            if abs(f - h) > RTOL_FULL_VS_HALF * abs(f):
                pytest.fail(
                    f"horizon {name} disagrees between full and bitant domains: "
                    f"full={f:.15g} bitant={h:.15g} "
                    f"(rel. diff {abs(f - h) / abs(f):.3g}, tol {RTOL_FULL_VS_HALF:g}). "
                    f"Bitant-aware horizon finding (mirrored-coordinate interpolation "
                    f"and per-component reflection parity in FastFlow::MetricInterp) is "
                    f"likely broken."
                )

        # 3. Symmetry: a z-reflection-symmetric spacetime has Sx = Sy = 0 exactly, so a
        #    nonzero value here means the reflection parity table is wrong.
        for idx, name in ((ISX, "Sx"), (ISY, "Sy")):
            if abs(half[idx]) > 1.0e-8:
                pytest.fail(
                    f"bitant domain: spin component {name} = {half[idx]:g}, but it must "
                    f"vanish by reflection symmetry. Check the z-reflection parity signs "
                    f"applied to the extrinsic curvature in FastFlow::MetricInterp."
                )
    finally:
        _cleanup_horizon_files()
        testutils.cleanup()
