//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_shells.cpp
//! \brief writes spherically integrated data over multiple radial shells

#include <sys/stat.h>  // mkdir

#include <cstdio>  // snprintf
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "outputs.hpp"
#include "geodesic-grid/spherical_grid.hpp"

SphericalShellsOutput::SphericalShellsOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op)
    : BaseTypeOutput(pin, pm, op) {
  mkdir("sphshell", 0755);

  // Read parameters for shell configuration
  nr = pin->GetInteger(op.block_name, "nr");
  rmin = pin->GetReal(op.block_name, "rmin");
  rmax = pin->GetReal(op.block_name, "rmax");
  int nlev = pin->GetOrAddInteger(op.block_name, "nlev", 4);
  
  // Optional logarithmic spacing
  log_spacing = pin->GetOrAddBoolean(op.block_name, "log_spacing", false);
  
  // Note: Spheres are always centered at the origin (0,0,0)
  
  // Initialize array of radial faces (nr+1 values from rmin to rmax)
  radii_faces.resize(nr + 1);
  if (log_spacing) {
    // Logarithmic spacing for faces
    Real log_rmin = std::log10(rmin);
    Real log_rmax = std::log10(rmax);
    Real dlog_r = (log_rmax - log_rmin) / nr;
    for (int i = 0; i <= nr; ++i) {
      radii_faces[i] = std::pow(10.0, log_rmin + i * dlog_r);
    }
  } else {
    // Linear spacing for faces
    Real dr = (rmax - rmin) / nr;
    for (int i = 0; i <= nr; ++i) {
      radii_faces[i] = rmin + i * dr;
    }
  }
  
  // Initialize array of radial centers (nr values, midpoints between faces)
  radii.resize(nr);
  if (log_spacing) {
    // Geometric mean for cell centers on log grid
    for (int i = 0; i < nr; ++i) {
      radii[i] = std::sqrt(radii_faces[i] * radii_faces[i + 1]);
    }
  } else {
    // Arithmetic mean for cell centers on linear grid
    for (int i = 0; i < nr; ++i) {
      radii[i] = (radii_faces[i] + radii_faces[i + 1]) / 2.0;
    }
  }

  // Create SphericalGrid objects for each radius
  for (int i = 0; i < nr; ++i) {
    // Adjust center for each sphere
    Real rad_adj = radii[i];
    spheres.push_back(std::make_unique<SphericalGrid>(pm->pmb_pack, nlev, rad_adj));
  }
}

SphericalShellsOutput::~SphericalShellsOutput() {
  spheres.clear();
}

void SphericalShellsOutput::LoadOutputData(Mesh *pm) {
  // If AMR is enabled we need to reset the grids
  if (pm->adaptive) {
    for (auto& sphere : spheres) {
      sphere->SetInterpolationIndices();
      sphere->SetInterpolationWeights();
    }
  }

  int nout_vars = outvars.size();
  
  // Allocate output array: (nradii, nvars)
  Kokkos::realloc(outarray, nout_vars, 1, 1, nr, 1);

  // Calculate derived variables, if required
  if (out_params.contains_derived) {
    ComputeDerivedVariable(out_params.variable, pm);
  }

  // For each radius, interpolate and integrate over the shell
  for (int ir = 0; ir < nr; ++ir) {
    auto& sphere = spheres[ir];
    
    // Get precomputed radial cell center and faces
    Real r0 = radii[ir];           // Cell center radius
    Real r1 = radii_faces[ir];     // Inner face
    Real r2 = radii_faces[ir + 1]; // Outer face
    
    // Compute radial volume factor: integral of r^2 dr from r1 to r2
    // = (r2^3 - r1^3) / 3
    Real radial_factor = (r2 * r2 * r2 - r1 * r1 * r1) / 3.0;
    
    for (int n = 0; n < nout_vars; ++n) {
      // Interpolate data to this spherical surface at radius r0
      // Note the geodesic spherical grid uses an inclusive range for the variables
      // so we pass the same index for vs and ve (start and end).
      // This is unlike in the SphericalSurface class.
      sphere->InterpolateToSphere(outvars[n].data_index, outvars[n].data_index, *(outvars[n].data_ptr));
      
      // Integrate over the shell using proper volume element
      // dV = r^2 * dOmega * dr
      // For shell: integral = solid_angle * radial_factor * value(r0)
      Real integrated_value = 0.0;
      
      for (int iang = 0; iang < sphere->nangles; ++iang) {
        Real solid_angle = sphere->solid_angles.h_view(iang);
        Real value = sphere->interp_vals.h_view(iang, 0);
        integrated_value += value * solid_angle * radial_factor;
      }
      
      outarray(n, 0, 0, ir, 0) = integrated_value;
    }
  }

#if MPI_PARALLEL_ENABLED
  // Sum contributions from all ranks
  // Note that InterpolateToSphere will set zero values for points not owned by current rank
  int count = nout_vars * nr;
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void SphericalShellsOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
#if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
#endif
    // Assemble filename
    char fname[BUFSIZ];
    std::snprintf(fname, BUFSIZ, "sphshell/%s.%s.%05d.dat",
                  out_params.file_basename.c_str(),
                  out_params.file_id.c_str(), out_params.file_number);

    // Open file
    std::ofstream ofile(fname);
    
    // Write header
    ofile << "# Spherically integrated data over radial shells" << std::endl;
    ofile << "# Time = " << pm->time << std::endl;
    ofile << "# Cycle = " << pm->ncycle << std::endl;
    ofile << "# Center: origin (0, 0, 0)" << std::endl;
    ofile << "# Number of radii: " << nr << std::endl;
    ofile << "# Radii range: " << rmin << " to " << rmax;
    if (log_spacing) {
      ofile << " (logarithmic spacing)" << std::endl;
    } else {
      ofile << " (linear spacing)" << std::endl;
    }
    
    // Write column headers
    ofile << "# Columns: radius";
    int nout_vars = outvars.size();
    for (int n = 0; n < nout_vars; ++n) {
      ofile << " " << outvars[n].label;
    }
    ofile << std::endl;
    
    // Set output precision
    ofile << std::scientific << std::setprecision(8);
    
    // Write data for each radius
    for (int ir = 0; ir < nr; ++ir) {
      ofile << radii[ir];
      for (int n = 0; n < nout_vars; ++n) {
        ofile << " " << outarray(n, 0, 0, ir, 0);
      }
      ofile << std::endl;
    }
    
    ofile.close();

#if MPI_PARALLEL_ENABLED
  }
#endif

  // Increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
