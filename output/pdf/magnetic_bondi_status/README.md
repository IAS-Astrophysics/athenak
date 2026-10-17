# Magnetized binary Bondi status report

The eight-page PDF and editable AASTeX 7.0.1 source describe the actual magnetic
pilot, including initial conditions, rotating-frame velocities, boundary and
hard-excision options, cyclic masks and magnetic EMF coupling, stage dwell times,
physical exposure, new matched-phase accretion and shell-mass diagnostics, and
numerical/boundary limitations. Five numbered illustrations and two tables are
included. The anonymous AASTeX option avoids inventing author affiliations or
contact information for an internal status record.

Status and analysis cutoffs are distinct: the live log used for the timeline
ends at10208.05 M, while quantitative finest-snapshot analysis ends at9903.62 M.
A later PBS check in live-status.txt confirms8939546 running at10402.45 M on
2026-10-17 at19:29 UTC. Those are dated observations, not a promise of future
running state. Only one complete zoom roundtrip is available. All physical
snapshots analyzed here belong to the new MHD chain, not the old hydro chain.

## Read / compile

Open magnetic_bondi_status.pdf, or edit magnetic_bondi_status.tex. Keep the
figures directory beside the source. A current TeX Live installation with
AASTeX 7.0.1 is required. From this directory run:

    latexmk -pdf -interaction=nonstopmode -halt-on-error magnetic_bondi_status.tex

This is a multi-file figure-bearing manuscript, compiled with installed TeX
because the built-in standalone compiler does not support extra project files.
The final build has no undefined references, overfull boxes, or stuck floats.
All eight pages were rendered and visually checked. Auxiliary build products
are ignored; figure PDFs/PNGs and the final PDF are deliverables.

## Reproduce the analysis in the repository

    python3 scripts/analyze_magnetic_inflow.py --base build-mac-zoom-opt/aurora-magnetic --out output/pdf/magnetic_bondi_status/figures
    python3 scripts/make_magnetic_report_figures.py

The scripts intentionally record the seven completed physical jobs through
8938971. To update later, extend their explicit chronological job lists, download
only validated finest snapshots, and revise the comparison windows/visit labels
before claiming multi-cycle convergence. No automated future-equilibrium claim
is implied by rerunning the current startup-versus-first-return audit.

The flux calculation uses saved ADM alpha/shift/spatial metric and stored Wv.
It imports only geometry/flux helpers from the old diagnostic, not its hydro
normalization, zero-field gate, or old phase schedule. Shell assignment uses
identical restored topology and center-based boundaries. Comparison means use
phases50,100,200 M via within-visit interpolation. Full results and source paths
are in figures/inflow_equilibrium.json. No continuous conservation residual is
inferred across frozen intervals.

Raw snapshots are excluded from this small source package. Data manifests,
input-used.athinput, actual saved-header excision settings, source/executable
hashes, schedule.json, live evidence, and plotted diagnostics are included.
