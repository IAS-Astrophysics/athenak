//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_surface.cpp
//! \brief writes data on a SphericalSurface sub-grid in binary VTK format
//!
//! Parameters of an <output> block with file_type = sph:
//!   radius | radii | (nradii, r_min, r_max, r_spacing)  surface radii, see ParseRadii
//!   ntheta, nphi, theta_spacing, theta_centering        angular grid, see
//!                                                       ParseAngleOptions
//!   xc, yc, zc                                          center of the surfaces
//!   weights                                             write the surface element of
//!                                                       each point (default true)
//!
//! Points are written with the radius varying fastest, then theta, then phi. theta
//! increases with its index, phi starts at 0.

#include "utils/spherical_surface.hpp"

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cmath>
#include <cstdio>  // snprintf
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "outputs.hpp"

namespace {
//! \fn std::vector<Real> ParseRadii(ParameterInput *pin, const OutputParameters &op)
//! \brief Build the list of surface radii requested by an <output> block.
//! Accepted formats:
//!   radius = 100.0                  (single surface)
//!   radii  = 10.0, 20.0, 50.0       (explicit list)
//!   r_min, r_max, nradii, r_spacing (linear or log-spaced list)
std::vector<Real> ParseRadii(ParameterInput *pin, const OutputParameters &op) {
  std::vector<Real> rad;
  bool has_radius = pin->DoesParameterExist(op.block_name, "radius");
  bool has_radii  = pin->DoesParameterExist(op.block_name, "radii");
  bool has_range  = pin->DoesParameterExist(op.block_name, "nradii");

  if (static_cast<int>(has_radius) + static_cast<int>(has_radii) +
      static_cast<int>(has_range) != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Output block '" << op.block_name << "' must specify"
              << "exactly one of 'radius', 'radii', or 'nradii' (with r_min/r_max)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (has_radius) {
    rad.push_back(pin->GetReal(op.block_name, "radius"));
  } else if (has_radii) {
    std::string list = pin->GetString(op.block_name, "radii");
    std::replace(list.begin(), list.end(), ',', ' ');
    std::istringstream iss(list);
    Real r;
    while (iss >> r) {
      rad.push_back(r);
    }
    if (rad.empty()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Could not parse any radii in output block '"
                << op.block_name << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
  } else {
    int nradii = pin->GetInteger(op.block_name, "nradii");
    Real rmin  = pin->GetReal(op.block_name, "r_min");
    Real rmax  = pin->GetReal(op.block_name, "r_max");
    std::string spacing = pin->GetOrAddString(op.block_name, "r_spacing", "linear");

    // Check that the input parameters for the list fulfill certain conditions.
    if (nradii < 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "nradii must be >= 1 in output block '"
                << op.block_name << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (spacing.compare("log") == 0 && rmin <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "r_min must be > 0 for r_spacing=log in output block '"
                << op.block_name << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (rmin >= rmax) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "r_min = " << rmin << " must be smaller than r_max = "
                << rmax << " in output block '" << op.block_name << "'" << std::endl;
      exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nradii; ++i) {
      // if nradii == 1, then single surface sits at r_min
      Real f = (nradii > 1) ? static_cast<Real>(i)/static_cast<Real>(nradii - 1) : 0.0;
      if (spacing.compare("log") == 0) {
        rad.push_back(rmin * std::pow(rmax/rmin, f));
      } else if (spacing.compare("linear") == 0) {
        rad.push_back(rmin + (rmax - rmin) * f);
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Unrecognized r_spacing = '" << spacing
                  << "' in output block '" << op.block_name
                  << "' (must be 'linear' or 'log')" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  for (auto r : rad) {
    if (r <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Surface radii must be > 0, found " << r
                << " in output block '" << op.block_name << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  return rad;
}

//! \fn SphericalSurface::AngleOptions ParseAngleOptions(...)
//! \brief Build the angular grid description requested by an <output> block.
//! Accepted parameters:
//!   ntheta           number of points along theta (default 32)
//!   nphi             number of points along phi (default 2*ntheta)
//!   theta_spacing    'uniform_costheta' (default) or 'uniform_theta'
//!   theta_centering  'node' (default) or 'cell'
SphericalSurface::AngleOptions ParseAngleOptions(ParameterInput *pin,
                                                 const OutputParameters &op,
                                                 int ntheta) {
  SphericalSurface::AngleOptions aopt;
  aopt.nphi = pin->GetOrAddInteger(op.block_name, "nphi", 2*ntheta);

  std::string spacing = pin->GetOrAddString(op.block_name, "theta_spacing",
                                            "uniform_costheta");
  if (spacing.compare("uniform_theta") == 0) {
    aopt.theta_spacing = SphThetaSpacing::uniform_theta;
  } else if (spacing.compare("uniform_costheta") == 0) {
    aopt.theta_spacing = SphThetaSpacing::uniform_costheta;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unrecognized theta_spacing = '" << spacing
              << "' in output block '" << op.block_name
              << "' (must be 'uniform_theta' or 'uniform_costheta')" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string centering = pin->GetOrAddString(op.block_name, "theta_centering", "node");
  if (centering.compare("cell") == 0) {
    aopt.theta_centering = SphThetaCentering::cell;
  } else if (centering.compare("node") == 0) {
    aopt.theta_centering = SphThetaCentering::node;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unrecognized theta_centering = '" << centering
              << "' in output block '" << op.block_name
              << "' (must be 'cell' or 'node')" << std::endl;
    exit(EXIT_FAILURE);
  }
  return aopt;
}

//! \brief names of the angular grid layout, written to the file header so that the grid
//! a file was dumped on can be identified without the input file
const char *SpacingName(SphThetaSpacing spacing) {
  return (spacing == SphThetaSpacing::uniform_theta) ? "uniform_theta"
                                                     : "uniform_costheta";
}
const char *CenteringName(SphThetaCentering centering) {
  return (centering == SphThetaCentering::cell) ? "cell" : "node";
}
} // namespace

SphericalSurfaceOutput::SphericalSurfaceOutput(ParameterInput *pin, Mesh *pm,
                                               OutputParameters op)
    : BaseTypeOutput(pin, pm, op) {
  mkdir("sph", 0755);

  std::vector<Real> rad = ParseRadii(pin, op);
  int ntheta = pin->GetOrAddInteger(op.block_name, "ntheta", 32);
  SphericalSurface::AngleOptions aopt = ParseAngleOptions(pin, op, ntheta);
  Real xc = pin->GetOrAddReal(op.block_name, "xc", 0.0);
  Real yc = pin->GetOrAddReal(op.block_name, "yc", 0.0);
  Real zc = pin->GetOrAddReal(op.block_name, "zc", 0.0);
  // the quadrature weights are only useful when integrating over the surface, so they
  // can be switched off if not needed
  dump_weights = pin->GetOrAddBoolean(op.block_name, "weights", true);
  psurf = new SphericalSurface(pm->pmb_pack, ntheta, rad, xc, yc, zc, aopt);
}

SphericalSurfaceOutput::~SphericalSurfaceOutput() { delete psurf; }

void SphericalSurfaceOutput::LoadOutputData(Mesh *pm) {
  // If AMR is enabled we need to reset the CartesianGrid
  if (pm->adaptive) {
    psurf->SetInterpolationIndices();
    psurf->SetInterpolationWeights();
  }

  int nout_vars = outvars.size();
  Kokkos::realloc(outarray, nout_vars, 1, 1, 1, psurf->npoints);

  // Calculate derived variables, if required
  if (out_params.contains_derived) {
    ComputeDerivedVariable(out_params.variable, pm);
  }

  for (int n = 0; n < nout_vars; ++n) {
    psurf->InterpolateToSphere(outvars[n].data_index, *(outvars[n].data_ptr));
    auto v_slice = Kokkos::subview(outarray, n, 0, 0, 0, Kokkos::ALL);
    Kokkos::deep_copy(v_slice, psurf->interp_vals.h_view);
  }
#if MPI_PARALLEL_ENABLED
  // Note that InterpolateToSphere will set zero values for points not owned by
  // current rank
  int count = nout_vars * psurf->npoints;
  if (0 == global_variable::my_rank) {
    MPI_Reduce(MPI_IN_PLACE, outarray.data(), count, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(outarray.data(), outarray.data(), count, MPI_ATHENA_REAL,
               MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void SphericalSurfaceOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  bool big_end = IsBigEndian();

#if MPI_PARALLEL_ENABLED
  if (0 == global_variable::my_rank) {
#endif
    int nradii = psurf->nradii;
    int nangles = psurf->nangles;

    // Assemble filename
    char fname[BUFSIZ];
    if (nradii == 1) {
      std::snprintf(fname, BUFSIZ, "sph/%s.r=%.2f.%s.%05d.vtk",
                    out_params.file_basename.c_str(), psurf->radii.h_view(0),
                    out_params.file_id.c_str(), out_params.file_number);
    } else {
      std::snprintf(fname, BUFSIZ, "sph/%s.r=%.2f-%.2f.%s.%05d.vtk",
                    out_params.file_basename.c_str(), psurf->radii.h_view(0),
                    psurf->radii.h_view(nradii-1), out_params.file_id.c_str(),
                    out_params.file_number);
    }

    // Open file
    std::ofstream ofile(fname, std::ios::binary);

    ofile << "# vtk DataFile Version 3.0" << std::endl;
    ofile << "# AthenaK data at time=" << pm->time
          << " cycle=" << pm->ncycle << " nradii=" << nradii
          << " rmin=" << psurf->radii.h_view(0)
          << " rmax=" << psurf->radii.h_view(nradii-1)
          << " xc=" << psurf->xc << " yc=" << psurf->yc << " zc=" << psurf->zc
          << " theta_spacing=" << SpacingName(psurf->theta_spacing)
          << " theta_centering=" << CenteringName(psurf->theta_centering)
          << std::endl;
    ofile << "BINARY" << std::endl;
    ofile << "DATASET STRUCTURED_GRID" << std::endl;
    // radius varies fastest, then theta, then phi
    ofile << "DIMENSIONS " << nradii << " " << psurf->ntheta
          << " " << psurf->nphi << std::endl;
    ofile << "POINTS " << psurf->npoints << " float\n";

    for (int i = 0; i < psurf->nangles; ++i) {
      for (int r = 0; r < nradii; ++r) {
        float dline[3] = {static_cast<float>(psurf->radii.h_view(r)),
                          static_cast<float>(psurf->polar_pos.h_view(i, 0)),
                          static_cast<float>(psurf->polar_pos.h_view(i, 1))};
        if (!big_end) {
          Swap4Bytes(&dline[0]);
          Swap4Bytes(&dline[1]);
          Swap4Bytes(&dline[2]);
        }

        ofile.write(reinterpret_cast<char *>(&dline[0]), 3 * sizeof(float));
      }
    }

    float t = static_cast<float>(pm->time);
    if (!big_end) {
      Swap4Bytes(&t);
    }
    ofile << "\nFIELD FieldData 3\n";
    ofile << "TIME 1 1 float\n";
    ofile.write(reinterpret_cast<char *>(&t), sizeof(float));

    ofile << "\nCYCLE 1 1 int\n";
    int cycle = pm->ncycle;
    if (!big_end) {
      Swap4Bytes(&cycle);
    }
    ofile.write(reinterpret_cast<char *>(&cycle), sizeof(int));

    // write the radii explicitely
    ofile << "\nRADII 1 " << nradii << " float\n";
    for (int r = 0; r < nradii; ++r) {
      float rr = static_cast<float>(psurf->radii.h_view(r));
      if (!big_end) {
        Swap4Bytes(&rr);
      }
      ofile.write(reinterpret_cast<char *>(&rr), sizeof(float));
    }

    // every block below starts on a new line, since the preceding one ends in binary data
    ofile << "\nPOINT_DATA " << psurf->npoints;
    if (dump_weights) {
      ofile << "\nSCALARS weights float 1" << std::endl;
      ofile << "LOOKUP_TABLE default" << std::endl;
      for (int i = 0; i < psurf->nangles; ++i) {
        for (int r = 0; r < nradii; ++r) {
          Real rad = psurf->radii.h_view(r);
          // int_weights is the solid angle of the point, r^2*dOmega its surface element
          float d = rad * rad * psurf->int_weights.h_view(i);
          if (!big_end) {
            Swap4Bytes(&d);
          }
          ofile.write(reinterpret_cast<char *>(&d), sizeof(float));
        }
      }
    }

    int nout_vars = outvars.size();
    for (int n = 0; n < nout_vars; ++n) {
      ofile << "\nSCALARS " << outvars[n].label << " float 1" << std::endl;
      ofile << "LOOKUP_TABLE default" << std::endl;
      for (int i = 0; i < psurf->nangles; ++i) {
        for (int r = 0; r < nradii; ++r) {
          float d = outarray(n, 0, 0, 0, r*nangles + i);
          if (!big_end) {
            Swap4Bytes(&d);
          }
          ofile.write(reinterpret_cast<char *>(&d), sizeof(float));
        }
      }
    }

#if MPI_PARALLEL_ENABLED
  }
#endif

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
