//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file azimuthal_average.cpp
//! \brief writes azimuthally averaged data on an (r, theta) grid in binary VTK format.
//!        For each radius, data is interpolated onto a (theta, phi) grid using
//!        SphericalSurface, then averaged over the phi direction.

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
    : BaseTypeOutput(pin, pm, op) {
  mkdir("azavg", 0755);

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

  // Create a SphericalSurface for each radius (with independent nphi)
  for (int i = 0; i < nr; ++i) {
    surfaces.push_back(
        std::make_unique<SphericalSurface>(pm->pmb_pack, ntheta, radii[i],
                                           0.0, 0.0, 0.0, nphi));
  }

  // Extract theta grid from the first surface (identical for all radii)
  theta_grid.resize(ntheta);
  for (int j = 0; j < ntheta; ++j) {
    theta_grid[j] = surfaces[0]->polar_pos.h_view(j, 0);
  }
}

AzimuthalAverageOutput::~AzimuthalAverageOutput() {
  surfaces.clear();
}

void AzimuthalAverageOutput::LoadOutputData(Mesh *pm) {
  // If AMR is enabled, reset interpolation for all surfaces
  if (pm->adaptive) {
    for (auto &surf : surfaces) {
      surf->SetInterpolationIndices();
      surf->SetInterpolationWeights();
    }
  }

  int nout_vars = outvars.size();

  // outarray dimensions: (nout_vars, 1, 1, nr, ntheta)
  Kokkos::realloc(outarray, nout_vars, 1, 1, nr, ntheta);

  if (out_params.contains_derived) {
    ComputeDerivedVariable(out_params.variable, pm);
  }

  // For each radius, interpolate each variable and average over phi
  for (int ir = 0; ir < nr; ++ir) {
    auto &surf = surfaces[ir];

    for (int n = 0; n < nout_vars; ++n) {
      surf->InterpolateToSphere(outvars[n].data_index, *(outvars[n].data_ptr));

      // Average over phi for each theta index
      // Grid layout: flat index = i_phi * ntheta + j_theta
      for (int jt = 0; jt < ntheta; ++jt) {
        Real sum = 0.0;
        for (int ip = 0; ip < nphi; ++ip) {
          sum += surf->interp_vals.h_view(ip * ntheta + jt);
        }
        outarray(n, 0, 0, ir, jt) = sum;
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  // Each rank contributes zeros for points it doesn't own; sum to rank 0
  int count = nout_vars * nr * ntheta;
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  // Divide by nphi to complete the average (after MPI reduce)
  for (int n = 0; n < nout_vars; ++n) {
    for (int ir = 0; ir < nr; ++ir) {
      for (int jt = 0; jt < ntheta; ++jt) {
        outarray(n, 0, 0, ir, jt) /= static_cast<Real>(nphi);
      }
    }
  }
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
