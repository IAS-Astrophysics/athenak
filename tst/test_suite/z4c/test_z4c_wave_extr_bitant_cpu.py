"""
Regression test for bitant-aware Weyl-scalar (Psi4) wave extraction.

A single boosted puncture (boost confined to x1, no z-velocity/spin component) is exactly
symmetric under reflection about z=0. This test evolves it on a full domain and on a
bitant (z>=0, reflecting) domain at matching resolution, and checks that the extracted
l=2 mode coefficients of r*Psi4 agree between the two runs. This exercises SphericalGrid's
bitant-aware interpolation (mirrored-coordinate lookup for angles with z<0) and the
imaginary-part sign correction applied in Z4c::WaveExtr.
"""

import os
import shutil

import pytest
import test_suite.testutils as testutils

# Absolute+relative tolerance (numpy.isclose-style: |full-half| <= ATOL + RTOL*|full|).
# An absolute floor is needed because several l=2 components are near-zero junk/gauge
# content for this symmetric setup, where a purely relative comparison is meaningless.
ATOL = 2.0e-4
RTOL = 8.0e-2

_REAL = "rpsi4_real_0004.txt"
_IMAG = "rpsi4_imag_0004.txt"


def _read_last_row(path):
    with open(path) as fh:
        rows = [ln for ln in fh if ln.strip() and not ln.startswith("#")]
    return [float(x) for x in rows[-1].split()]


def _check_close(label, full, half):
    # columns 1..5 are the l=2 modes (2,-2)..(2,2); column 0 is time
    for i in range(1, 6):
        f, h = full[i], half[i]
        if abs(f - h) > ATOL + RTOL * abs(f):
            pytest.fail(
                f"{label} part of l=2 mode index {i} disagrees between full and bitant "
                f"domains: full={f:g} bitant={h:g} (tol={ATOL + RTOL * abs(f):g}). "
                f"Bitant-aware wave extraction (SphericalGrid mirrored-coordinate "
                f"interpolation / Z4c::WaveExtr sign correction) is likely broken."
            )


def test_z4c_wave_extr_bitant():
    try:
        testutils.run("inputs/z4c_boosted_bitant_full.athinput", [])
        if not os.path.exists(f"waveforms/{_REAL}"):
            pytest.fail(f"waveform file waveforms/{_REAL} not produced (full domain)")
        full_real = _read_last_row(f"waveforms/{_REAL}")
        full_imag = _read_last_row(f"waveforms/{_IMAG}")
        shutil.rmtree("waveforms")

        testutils.run("inputs/z4c_boosted_bitant_half.athinput", [])
        if not os.path.exists(f"waveforms/{_REAL}"):
            pytest.fail(f"waveform file waveforms/{_REAL} not produced (bitant domain)")
        half_real = _read_last_row(f"waveforms/{_REAL}")
        half_imag = _read_last_row(f"waveforms/{_IMAG}")

        _check_close("real", full_real, half_real)
        _check_close("imaginary", full_imag, half_imag)
    finally:
        if os.path.exists("waveforms"):
            shutil.rmtree("waveforms")
        testutils.cleanup()
