//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file azimuthal_average.cpp
//! \brief writes azimuthally averaged data on an (r, theta) grid in binary VTK format.
//!
//! Performance optimisation (fused interpolation)
//! -----------------------------------------------
//! The original implementation called InterpolateToSphere() in a double loop
//! (nr radii x nout_vars variables), producing nr*nout_vars separate Kokkos kernel
//! launches each followed by a device->host sync.  For typical parameters
//! (nr=128, nout_vars=14) that is 1792 launches/syncs per output step.
//!
//! This version fuses the per-radius interpolation tables (interp_indcs and
//! interp_wghts) from all SphericalSurface objects into two flat Kokkos arrays at
//! construction time.  LoadOutputData then runs ONE Kokkos kernel per output
//! variable over all nr*nphi*ntheta sample points simultaneously, reducing kernel
//! launches from nr*nout_vars to nout_vars and eliminating per-radius syncs.

#include "utils/spherical_surface.hpp"

#include <sys/stat.h>  // mkdir

#include <cmath>
#include <cstdio>  // snprintf
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "outputs.hpp"

AzimuthalAverageOutput::AzimuthalAverageOutput(ParameterInput *pin, Mesh *pm,
                                               OutputParameters op)
    : BaseTypeOutput(pin, pm, op),
      fused_indcs("fused_indcs", 1, 4),
      fused_wghts("fused_wghts", 1, 1, 3) {
  mkdir("azavg", 0755);

  // Cache mesh index parameters needed inside the interpolation kernel.
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  ng_ = indcs.ng;
  is_ = indcs.is;
  js_ = indcs.js;
  ks_ = indcs.ks;
  adaptive_ = pm->adaptive;

  nr = pin->GetInteger(op.block_name, "nr");
  ntheta = pin->GetOrAddInteger(op.block_name, "ntheta", 32);
  nphi = pin->GetOrAddInteger(op.block_name, "nphi", 2 * ntheta);
  rmin = pin->GetReal(op.block_name, "rmin");
  rmax = pin->GetReal(op.block_name, "rmax");
  log_spacing = pin->GetOrAddBoolean(op.block_name, "log_spacing", false);

  // Build radial grid
  radii.resize(nr);
  if (log_spacing) {
    Real log_rmin = std::log10(rmin);
    Real log_rmax = std::log10(rmax);
    Real dlog_r = (log_rmax - log_rmin) / (nr - 1);
    for (int i = 0; i < nr; ++i) {
      radii[i] = std::pow(10.0, log_rmin + i * dlog_r);
    }
  } else {
    Real dr = (rmax - rmin) / (nr - 1);
    for (int i = 0; i < nr; ++i) {
      radii[i] = rmin + i * dr;
    }
  }

  // Create one SphericalSurface per radius to build interpolation tables.
  surfaces.reserve(nr);
  for (int i = 0; i < nr; ++i) {
    surfaces.push_back(
        std::make_unique<SphericalSurface>(pm->pmb_pack, ntheta, radii[i],
                                           0.0, 0.0, 0.0, nphi));
  }

  // Extract theta grid from the first surface (identical for all radii).
  theta_grid.resize(ntheta);
  for (int j = 0; j < ntheta; ++j)
    theta_grid[j] = surfaces[0]->polar_pos.h_view(j, 0);

  // Flatten per-radius tables into fused arrays.  For non-AMR runs the
  // SphericalSurface objects are then freed to recover ~1 MB/surface.
  BuildFusedArrays();
  if (!adaptive_)
    surfaces.clear();
}

// ── Destructor ────────────────────────────────────────────────────────────────

AzimuthalAverageOutput::~AzimuthalAverageOutput() {
  surfaces.clear();
}

// ── BuildFusedArrays ─────────────────────────────────────────────────────────
// Copy interp_indcs and interp_wghts from all nr surfaces into flat arrays.
// Flat index layout: idx = ir * (nphi*ntheta) + ip * ntheta + it,
// matching the phi-outer, theta-inner ordering of InitializeAngleAndWeights.

void AzimuthalAverageOutput::BuildFusedArrays() {
  int nang_per_r   = nphi * ntheta;
  int total_angles = nr * nang_per_r;

  Kokkos::realloc(fused_indcs, total_angles, 4);
  Kokkos::realloc(fused_wghts, total_angles, 2 * ng_, 3);

  for (int ir = 0; ir < nr; ++ir) {
    int offset = ir * nang_per_r;
    auto &surf = surfaces[ir];
    for (int n = 0; n < nang_per_r; ++n) {
      for (int k = 0; k < 4; ++k)
        fused_indcs.h_view(offset + n, k) = surf->interp_indcs.h_view(n, k);
      for (int i = 0; i < 2 * ng_; ++i)
        for (int k = 0; k < 3; ++k)
          fused_wghts.h_view(offset + n, i, k) =
              surf->interp_wghts.h_view(n, i, k);
    }
  }

  fused_indcs.template modify<HostMemSpace>();
  fused_indcs.template sync<DevExeSpace>();
  fused_wghts.template modify<HostMemSpace>();
  fused_wghts.template sync<DevExeSpace>();
}

// ── LoadOutputData ────────────────────────────────────────────────────────────

void AzimuthalAverageOutput::LoadOutputData(Mesh *pm) {
  // For AMR runs rebuild the per-rank interpolation tables before every output.
  if (adaptive_) {
    for (auto &surf : surfaces) {
      surf->SetInterpolationIndices();
      surf->SetInterpolationWeights();
    }
    BuildFusedArrays();
  }

  int nout_vars    = outvars.size();
  int nang_per_r   = nphi * ntheta;
  int total_angles = nr * nang_per_r;

  Kokkos::realloc(outarray, nout_vars, 1, 1, nr, ntheta);

  if (out_params.contains_derived)
    ComputeDerivedVariable(out_params.variable, pm);

  // Temporary buffer for one variable across all angles.
  DualArray1D<Real> fused_vals("fused_vals", total_angles);

  auto f_indcs = fused_indcs.d_view;
  auto f_wghts = fused_wghts.d_view;
  auto f_vals  = fused_vals.d_view;
  int ng = ng_, is = is_, js = js_, ks = ks_;

  for (int n = 0; n < nout_vars; ++n) {
    int  v   = outvars[n].data_index;
    auto val = *(outvars[n].data_ptr);   // Kokkos View — safe to capture

    // Single kernel over ALL nr*nphi*ntheta angles (was nr separate kernels).
    par_for("azavg_interp", DevExeSpace(), 0, total_angles - 1,
        KOKKOS_LAMBDA(int idx) {
          int ii0 = f_indcs(idx, 0);
          int ii1 = f_indcs(idx, 1);
          int ii2 = f_indcs(idx, 2);
          int ii3 = f_indcs(idx, 3);
          if (ii0 == -1) {
            f_vals(idx) = 0.0;
          } else {
            Real sum = 0.0;
            for (int i = 0; i < 2 * ng; ++i)
              for (int j = 0; j < 2 * ng; ++j)
                for (int k = 0; k < 2 * ng; ++k)
                  sum += f_wghts(idx, i, 0) * f_wghts(idx, j, 1) *
                         f_wghts(idx, k, 2) *
                         val(ii0, v,
                             ii3 - (ng - k - ks) + 1,
                             ii2 - (ng - j - js) + 1,
                             ii1 - (ng - i - is) + 1);
            f_vals(idx) = sum;
          }
        });

    // One device->host sync per variable (was one per radius per variable).
    fused_vals.template modify<DevExeSpace>();
    fused_vals.template sync<HostMemSpace>();

    // Phi average into outarray.
    for (int ir = 0; ir < nr; ++ir) {
      int r_offset = ir * nang_per_r;
      for (int jt = 0; jt < ntheta; ++jt) {
        Real sum = 0.0;
        for (int ip = 0; ip < nphi; ++ip)
          sum += fused_vals.h_view(r_offset + ip * ntheta + jt);
        outarray(n, 0, 0, ir, jt) = sum;
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  int count = nout_vars * nr * ntheta;
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  for (int n = 0; n < nout_vars; ++n)
    for (int ir = 0; ir < nr; ++ir)
      for (int jt = 0; jt < ntheta; ++jt)
        outarray(n, 0, 0, ir, jt) /= static_cast<Real>(nphi);
}

void AzimuthalAverageOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  bool big_end = IsBigEndian();

#if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
#endif
    char fname[BUFSIZ];
    std::snprintf(fname, BUFSIZ, "azavg/%s.%s.%05d.vtk",
                  out_params.file_basename.c_str(),
                  out_params.file_id.c_str(), out_params.file_number);

    std::ofstream ofile(fname, std::ios::binary);

    // VTK header
    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "# AthenaK azimuthal average at time=" << pm->time
          << " cycle=" << pm->ncycle
          << " rmin=" << rmin << " rmax=" << rmax
          << " ntheta=" << ntheta << " nphi=" << nphi
          << std::endl;
    ofile << "BINARY" << std::endl;
    ofile << "DATASET STRUCTURED_GRID" << std::endl;
    ofile << "DIMENSIONS " << nr << " " << ntheta << " 1" << std::endl;

    int npoints = nr * ntheta;
    ofile << "POINTS " << npoints << " float\n";

    // Write grid points in meridional plane: (r*sin(theta), 0, r*cos(theta))
    for (int jt = 0; jt < ntheta; ++jt) {
      Real theta = theta_grid[jt];
      for (int ir = 0; ir < nr; ++ir) {
        float pt[3] = {static_cast<float>(radii[ir] * std::sin(theta)),
                       0.0f,
                       static_cast<float>(radii[ir] * std::cos(theta))};
        if (!big_end) {
          Swap4Bytes(&pt[0]);
          Swap4Bytes(&pt[1]);
          Swap4Bytes(&pt[2]);
        }
        ofile.write(reinterpret_cast<char *>(pt), 3 * sizeof(float));
      }
    }

    // Field data: time and cycle
    float t = static_cast<float>(pm->time);
    if (!big_end) { Swap4Bytes(&t); }
    ofile << "\nFIELD FieldData 2\n";
    ofile << "TIME 1 1 float\n";
    ofile.write(reinterpret_cast<char *>(&t), sizeof(float));

    ofile << "\nCYCLE 1 1 int\n";
    int cycle = pm->ncycle;
    if (!big_end) { Swap4Bytes(&cycle); }
    ofile.write(reinterpret_cast<char *>(&cycle), sizeof(int));

    // Point data: one scalar per output variable
    ofile << "\nPOINT_DATA " << npoints << std::endl;

    int nout_vars = outvars.size();
    for (int n = 0; n < nout_vars; ++n) {
      ofile << "SCALARS " << outvars[n].label << " float 1" << std::endl;
      ofile << "LOOKUP_TABLE default" << std::endl;
      // Write in same order as points: theta outer, r inner
      for (int jt = 0; jt < ntheta; ++jt) {
        for (int ir = 0; ir < nr; ++ir) {
          float d = static_cast<float>(outarray(n, 0, 0, ir, jt));
          if (!big_end) { Swap4Bytes(&d); }
          ofile.write(reinterpret_cast<char *>(&d), sizeof(float));
        }
      }
    }

#if MPI_PARALLEL_ENABLED
  }
#endif

  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
