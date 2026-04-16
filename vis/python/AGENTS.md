# Directory Guide

## Role
Python-side readers and plotting helpers for AthenaK outputs. This is the first place to look when a workflow needs post-processing without rebuilding the C++ code.

## Important Files
- `athena_read.py`: canonical Python readers for AthenaK tables, histories, HDF5 dumps, and other output formats; test code imports this directly.
- `plot_sts_diffusion_benchmark.py`: plots the explicit-vs-STS diffusion benchmark from
  the CSV outputs in `doc/data/sts_diffusion/` and writes publication-ready PDF/PNG
  figures under `doc/figures/sts_diffusion/`.
- `plot_slice.py`: large, feature-rich 2D plotting entrypoint for binary dumps, including many derived GR/MHD quantities.
- `plot_hst.py`, `plot_mesh.py`, `plot_tab.py`: lighter plotting scripts for histories, mesh structure, and tabular output.
- `bin_convert.py`, `make_athdf.py`, `cartgrid.py`: conversion and grid reshaping helpers used when downstream tools expect different formats.
- `calculate_tori_equil.py`, `calculate_tori_magnetization.py`, `calculate_tori_rpeak.py`: torus analysis helpers for GR accretion setups.

## Read This Next
- For data ingestion bugs, read `athena_read.py` first.
- For visualization changes, start in the specific `plot_*.py` script before touching the shared readers.
- For the benchmark note that consumes the generated figures, read `doc/AGENTS.md`.
