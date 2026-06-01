//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fastflow.cpp
//! \brief Implementation of the apparent horizon finder class
//!        based on the fast-flow algorithm of Gundlach:1997us and Alcubierre:1998rq

#include <math.h> // NAN
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "fastflow.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "compact_object_tracker.hpp"
#include "utils/spherical_harm.hpp"
#include "utils/lagrange_interpolator.hpp"
#include "z4c.hpp"
#include "coordinates/cell_locations.hpp"
#include "utils/inline_interpolator.hpp"

//----------------------------------------------------------------------------------------
//! \fn FastFlow::FastFlow(MeshBlockPack *pmbp, ParameterInput * pin, int n)
//! \brief Constructor for FastFlow class.
FastFlow::FastFlow(MeshBlockPack *pmbp, ParameterInput *pin, int n):
  pmbp(pmbp),
  pin(pin),
  Y0("Y0",1,1), Ys("Ys",1,1), Yc("Yc",1,1),
  dY0dth("dY0dth",1,1), dYcdth("dYcdth",1,1),
  dYsdth("dYsdth",1,1), dYcdph("dYcdph",1,1),
  dYsdph("dYsdph",1,1), dY0dth2("dY0dth2",1,1),
  dYcdth2("dYcdth2",1,1), dYcdthdph("dYcdthdph",1,1),
  dYsdth2("dYsdth2",1,1), dYsdthdph("dYsdthdph",1,1),
  dYcdph2("dYcdph2",1,1), dYsdph2("dYsdph2",1,1),
  a0("a0",1), ac("ac",1), as("as",1),
  rr("rr",1), rr_dth("rr_dth",1), rr_dph("rr_dph",1),
  rho("rho",1), dg("dg",1,1,1,1,1), g_interp("g_interp",1,1),
  K_interp("K_interp",1,1), dg_interp("dg_interp",1,1) {
  nh = n; // The n-th horizon
  std::string n_str = std::to_string(nh);

  // Read parameter input variables.
  nhorizon = pin->GetOrAddInteger("fastflow", "num_horizons", 1); // Number of horizons
  ntheta = pin->GetOrAddInteger("fastflow", "ntheta", 5); // Number of points theta

  lmax = pin->GetOrAddInteger("fastflow", "lmax", 4);
  lmax1 = lmax + 1;

  // Flow parameters
  flow_iterations = pin->GetOrAddInteger("fastflow", "flow_iterations_" + n_str, 100);
  flow_alpha_beta_const = pin->GetOrAddReal("fastflow",
                                            "flow_alpha_beta_const_" + n_str, 1.0);
  flow_function = pin->GetOrAddString("fastflow", "flow", "standard");
  if (flow_function.compare("expansion") == 0) {
    flowflag = 1;
  } else if (flow_function.compare("standard") == 0) {
    flowflag = 2;
  } else if (flow_function.compare("shear") == 0) {
    flowflag = 3;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Invalid flow function specified!\n"
                 "(Options are expansion, standard, shear)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Convergence parameters
  hmean_tol = pin->GetOrAddReal("fastflow", "hmean_tol_" + n_str, 100.);
  mass_tol = pin->GetOrAddReal("fastflow", "mass_tol_" + n_str, 1e-2);

  // Output booleans
  verbose = pin->GetOrAddBoolean("fastflow", "verbose", false);
  output_ylm = pin->GetOrAddBoolean("fastflow", "output_ylm", false);
  output_grid = pin->GetOrAddBoolean("fastflow", "output_grid", false);

  root = pin->GetOrAddInteger("fastflow", "mpi_root", 0);
  merger_distance = pin->GetOrAddReal("fastflow", "merger_distance", 0.1);
  use_stored_metric_drvts = pin->GetBoolean("fastflow", "store_metric_drvts");

  // Initial guess
  initial_radius = pin->GetOrAddReal("fastflow", "initial_radius_" + n_str, 1.0);
  rr_min = -1.0;

  expand_guess = pin->GetOrAddReal("fastflow", "expand_guess", 1.0);

  // Center
  center[0] = pin->GetOrAddReal("fastflow", "center_x_" + n_str, 0.0);
  center[1] = pin->GetOrAddReal("fastflow", "center_y_" + n_str, 0.0);
  center[2] = pin->GetOrAddReal("fastflow", "center_z_" + n_str, 0.0);

  // Punctures
  npunct = pin->GetOrAddInteger("fastflow", "npunct", 0); // Number of punctures
  use_puncture = pin->GetOrAddInteger("fastflow", "use_puncture_" + n_str, -1);

  if (use_puncture >= 0) {
    // Center is determined on the fly during the initial guess
    // to follow the chosen puncture.
    if (use_puncture >= npunct) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " : punc = " << use_puncture << " > npunct = " << npunct << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  wait_until_punc_are_close = pin->GetOrAddBoolean("fastflow",
                                            "wait_until_punc_are_close_" + n_str, 0);
  use_puncture_massweighted_center = pin->GetOrAddBoolean("fastflow",
                                     "use_puncture_massweighted_center_" + n_str, 0);

  // Timer
  start_time = pin->GetOrAddReal("fastflow", "start_time_" + n_str,
                                                 std::numeric_limits<double>::max());
  stop_time = pin->GetOrAddReal("fastflow", "stop_time_" + n_str, -1.0);

  // Grid and quadrature weights.
  gl_grid = new GaussLegendreGrid(pmbp, ntheta, 1.0); // unit-sphere
  nangles = gl_grid->nangles;

  // Points for spherical harmonics l >= 1.
  lmpoints = lmax1 * lmax1;

  // Reallocate for the coefficients.
  Kokkos::realloc(a0, lmax1);
  Kokkos::realloc(ac, lmpoints);
  Kokkos::realloc(as, lmpoints);

  // Reallocate for the spherical harmonics.
  // The spherical grid is the same for all surfaces.
  Kokkos::realloc(Y0, nangles, lmax1);
  Kokkos::realloc(Yc, nangles, lmpoints);
  Kokkos::realloc(Ys, nangles, lmpoints);

  Kokkos::realloc(dY0dth, nangles, lmax1);
  Kokkos::realloc(dYcdth, nangles, lmpoints);
  Kokkos::realloc(dYsdth, nangles, lmpoints);
  Kokkos::realloc(dYcdph, nangles, lmpoints);
  Kokkos::realloc(dYsdph, nangles, lmpoints);

  Kokkos::realloc(dY0dth2, nangles, lmax1);
  Kokkos::realloc(dYcdth2, nangles, lmpoints);
  Kokkos::realloc(dYcdthdph, nangles, lmpoints);
  Kokkos::realloc(dYsdth2, nangles, lmpoints);
  Kokkos::realloc(dYsdthdph, nangles, lmpoints);
  Kokkos::realloc(dYcdph2, nangles, lmpoints);
  Kokkos::realloc(dYsdph2, nangles, lmpoints);

  ComputeSphericalHarmonics();

  // Fields on the sphere.
  Kokkos::realloc(rr, nangles);
  Kokkos::realloc(rr_dth, nangles);
  Kokkos::realloc(rr_dph, nangles);

  // Allocate the arrays holding the interpolated values.
  Kokkos::realloc(g_interp, (NSPMETRIC), nangles);
  Kokkos::realloc(K_interp, (NEXCURV), nangles);
  Kokkos::realloc(dg_interp, (NDRVSSPMETRIC), nangles);

  // Allocate memory for the array holding the metric derivatives
  auto &indcs = pmbp->pmesh->mb_indcs;
  int nmb = pmbp->nmb_thispack;
  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = indcs.nx2 + 2 * (indcs.ng);
  int ncells3 = indcs.nx3 + 2 * (indcs.ng);
  Kokkos::realloc(dg, nmb, (NDRVSSPMETRIC), ncells3, ncells2, ncells1);

  // Array computed in surface integrals.
  Kokkos::realloc(rho, nangles);

  // Flag points existing on this mesh.
  Kokkos::realloc(havepoint, nangles);

  // Initialize horizon properties to NAN.
  for (int v = 0; v < hnvar; ++v) {
    ah_prop[v] = NAN;
  }

  // Prepare output
  ofname_summary = pin->GetString("job", "basename") + ".";
  ofname_summary += pin->GetOrAddString("fastflow", "horizon_file_summary_" + n_str,
                                                          "horizon_summary_" + n_str);
  ofname_summary += ".txt";

  ofname_shape = pin->GetString("job", "basename") + ".";
  ofname_shape += pin->GetOrAddString("fastflow", "horizon_file_shape_" + n_str,
                                                            "horizon_shape_" + n_str);
  ofname_shape += ".txt";

  if (verbose) {
    ofname_verbose = pin->GetString("job", "basename") + ".";
    ofname_verbose += pin->GetOrAddString("fastflow", "horizon_verbose_" + n_str,
                                                          "horizon_verbose_" + n_str);
    ofname_verbose += ".txt";
  }

  if (output_ylm) {
    ofname_ylm = pin->GetString("job", "basename") + ".";
    ofname_ylm += pin->GetOrAddString("fastflow", "horizon_ylm_" + n_str,
                                                              "horizon_ylm_" + n_str);
    ofname_ylm += ".txt";
  }

  if (output_grid) {
    ofname_grid = pin->GetString("job", "basename") + ".";
    ofname_grid += pin->GetOrAddString("fastflow", "horizon_grid_" + n_str,
                                                             "horizon_grid_" + n_str);
    ofname_grid += ".txt";
  }

  #if MPI_PARALLEL_ENABLED
    ioproc = (root == global_variable::my_rank);
  #else
    ioproc = true;
  #endif

  if (ioproc) {
    // Summary file
    bool new_file = true;
    if (access(ofname_summary.c_str(), F_OK) == 0) {
      new_file = false;
    }
    pofile_summary = fopen(ofname_summary.c_str(), "a");
    if (NULL == pofile_summary) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "Could not open file '" << pofile_summary << "' for writing!" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (new_file) {
      fprintf(pofile_summary, "# 1:iter 2:time 3:mass 4:Sx 5:Sy 6:Sz 7:S 8:area"
                               "9:hrms 10:hmean 11:meanradius 12:minradius\n");
      fflush(pofile_summary);
    }

    if (output_grid) {
      pofile_grid = fopen(ofname_grid.c_str(), "w");
      if (NULL == pofile_grid) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "Could not open file '" << pofile_grid << "' for writing!" << std::endl;
        exit(EXIT_FAILURE);
      }

      // Header
      fprintf(pofile_grid, "# 1:Theta\t2:Phi\t3:Weight\n");

      // Print grid to the output file.
      for (int p = 0; p < nangles; ++p) {
        const Real theta = gl_grid->polar_pos.h_view(p,0);
        const Real phi   = gl_grid->polar_pos.h_view(p,1);
        const Real weight = gl_grid->int_weights.h_view(p);
        fprintf(pofile_grid, "%.15e %.15e %.15e\n", theta, phi, weight);
      }
      fclose(pofile_grid);
    }

    if (output_ylm) {
      pofile_ylm = fopen(ofname_ylm.c_str(), "w");
      if (NULL == pofile_ylm) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl
          << "Could not open file '" << pofile_ylm << "' for writing!" << std::endl;
        exit(EXIT_FAILURE);
      }

      // Header
      fprintf(pofile_ylm, "# 1:Theta\t2:Phi\t3:l\t4:m\t5:Y0\t6:Yc\t7:Ys\t8:dY0dth\t"
                   "9:dYcdth\t10:dYsdth\t11:dYcdphi\t12:dYsdphi\t13:dY0dth2\t"
                   "14:dYcdth2\t15:dYsdth2\t16:dYcdph2\t17:dYsdph2\t18:dYcdthdphi\t"
                   "19:dYsdthdphi\n");

      for (int l = 0; l <= lmax; ++l) {
        for (int m = 0; m <= l; ++m) {
          for (int p = 0; p < nangles; ++p) {
            const Real theta = gl_grid->polar_pos.h_view(p,0);
            const Real phi   = gl_grid->polar_pos.h_view(p,1);

            if (m == 0) {
              fprintf(pofile_ylm, "%.15e %.15e %d %d %.15e %.15e %.15e %.15e %.15e %.15e"
                                "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                      theta, phi, l, m, Y0.h_view(p,l), 0.0, 0.0,
                      dY0dth.h_view(p,l), 0.0, 0.0, 0.0, 0.0,
                      dY0dth2.h_view(p,l), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            } else {
              const int l1 = lmindex(l,m,lmax);
              fprintf(pofile_ylm, "%.15e %.15e %d %d %.15e %.15e %.15e %.15e %.15e %.15e"
                                "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                      theta, phi, l, m,
                      0.0,
                      Yc.h_view(p,l1),
                      Ys.h_view(p,l1),
                      0.0,
                      dYcdth.h_view(p,l1),
                      dYsdth.h_view(p,l1),
                      dYcdph.h_view(p,l1),
                      dYsdph.h_view(p,l1),
                      0.0,
                      dYcdth2.h_view(p,l1),
                      dYsdth2.h_view(p,l1),
                      dYcdph2.h_view(p,l1),
                      dYsdph2.h_view(p,l1),
                      dYcdthdph.h_view(p,l1),
                      dYsdthdph.h_view(p,l1)
                      );
            }
          }
        }
      }
      fclose(pofile_ylm);
    }

    if (verbose) {
      pofile_verbose = fopen(ofname_verbose.c_str(), "a");
      if (NULL == pofile_verbose) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "Could not open file '" << pofile_verbose << "' for writing!" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::FastFlow()
//! \brief Destructor for FastFlow class
FastFlow::~FastFlow() {
  // Delete Gauss-Legendre grid
  delete gl_grid;

  // Close files
  if (ioproc) {
    fclose(pofile_summary);
    if (verbose) {
      fclose(pofile_verbose);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::Write(int iter, Real time)
//! \brief Output summary and shape file, for each horizon.
void FastFlow::Write(int iter, Real time) {
  if (ioproc) {
    if ((time < start_time) || (time > stop_time)) return;
    if (wait_until_punc_are_close && !(PuncAreClose())) return;

    // Summary file
    fprintf(pofile_summary, "%d %g ", iter, time);
    fprintf(pofile_summary, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e",
        ah_prop[hmass],
        ah_prop[hSx],
        ah_prop[hSy],
        ah_prop[hSz],
        ah_prop[hS],
        ah_prop[harea],
        ah_prop[hhrms],
        ah_prop[hhmean],
        ah_prop[hmeanradius],
        ah_prop[hminradius]);
    fprintf(pofile_summary, "\n");
    fflush(pofile_summary);

    if (ah_found) {
      // Shape file (coefficients).
      pofile_shape = fopen(ofname_shape.c_str(), "a");
      if (NULL == pofile_shape) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "Could not open file '" << pofile_shape << "' for writing!" << std::endl;
        exit(EXIT_FAILURE);
      }
      fprintf(pofile_shape, "# iter = %d, Time = %g\n",iter,time);
      for (int l = 0; l <= lmax; l++) {
        fprintf(pofile_shape,"%e ", a0.h_view(l));

        for (int m = 1; m <= l; m++) {
          int l1 = lmindex(l,m,lmax);
          fprintf(pofile_shape,"%e ",ac.h_view(l1));
          fprintf(pofile_shape,"%e ",as.h_view(l1));
        }
      }
      fprintf(pofile_shape,"\n");
      fclose(pofile_shape);
    }
  }

  // This is needed on all ranks.
  if (ah_found && (time_first_found < 0)) {
    std::string parname {"time_first_found_" + std::to_string(nh)};
    time_first_found = time;
    pin->SetReal("fastflow", parname, time_first_found);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::Find(int iter, Real time)
//! \brief Search for the horizons
void FastFlow::Find(int iter, Real time) {
  if ((time < start_time) || (time > stop_time)) return;
  if (wait_until_punc_are_close && !(PuncAreClose())) return;
  if (verbose && ioproc) {
    fprintf(pofile_verbose, "time=%.4f, cycle=%d\n", time, iter);
  }

  InitialGuess();
  FastFlowLoop();

  // Retain `last_a0` in restart: this serves as primary ini. guess.
  if (ah_found) {
    std::string parname;
    parname = "last_a0_" + std::to_string(nh); // nh: horizon index

    pin->SetReal("fastflow", parname, last_a0);

    parname = "ah_found_a0_" + std::to_string(nh);
    pin->SetBoolean("fastflow", parname, ah_found);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::InitialGuess()
//! \brief Initial guess for spectral coefs of horizon n
void FastFlow::InitialGuess() {
  // Reset Coefficients to Zero
  Kokkos::deep_copy(a0.h_view, 0.0);
  Kokkos::deep_copy(ac.h_view, 0.0);
  Kokkos::deep_copy(as.h_view, 0.0);

  if (use_puncture >= 0) {
    // Update the center to the puncture position
    center[0] = pmbp->pz4c->ptracker[use_puncture]->GetPos(0);
    center[1] = pmbp->pz4c->ptracker[use_puncture]->GetPos(1);
    center[2] = pmbp->pz4c->ptracker[use_puncture]->GetPos(2);

    // Update a0
    // For single BH in isotropic coordinates: horizon radius=m/2
    // but make sure it can surround all punctures comfortably, i.e.
    // make radius a bit larger than half the distance between any of the punctures
    Real largedist = PuncMaxDistance(use_puncture);
    Real mass = pmbp->pz4c->ptracker[use_puncture]->GetMass();
    if (ah_found && last_a0 > 0) {
      a0.h_view(0) = last_a0 * expand_guess;
    } else {
      a0.h_view(0) = Kokkos::fmax(0.5 * mass, Kokkos::fmin(mass, 0.5 * largedist));
      a0.h_view(0) *= Kokkos::sqrt(4.0 * M_PI);
    }

    // Sync to device
    a0.template modify<HostMemSpace>();
    a0.template sync<DevExeSpace>();
    ac.template modify<HostMemSpace>();
    ac.template sync<DevExeSpace>();
    as.template modify<HostMemSpace>();
    as.template sync<DevExeSpace>();
    return;
  }

  if (use_puncture_massweighted_center) {
    // Update the center based on the mass-weighted distance
    Real pos[3];
    PuncWeightedMassCentralPoint(&pos[0], &pos[1], &pos[2]);
    center[0] = pos[0];
    center[1] = pos[1];
    center[2] = pos[2];
  }

  // Take a0 either from previous or from input value
  if (ah_found && last_a0 > 0) {
    a0.h_view(0) = last_a0 * expand_guess;
  } else {
    a0.h_view(0) = Kokkos::sqrt(4.0 * M_PI) * initial_radius;
  }
  // Sync to device
  a0.template modify<HostMemSpace>();
  a0.template sync<DevExeSpace>();
  ac.template modify<HostMemSpace>();
  ac.template sync<DevExeSpace>();
  as.template modify<HostMemSpace>();
  as.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::MetricDerivatives(Real time)
//! \brief Compute drvts of ADM metric at MB level.
template <int NGHOST>
void FastFlow::MetricDerivatives(Real time) {
  // Check whether derivatives have to be computed
  if (use_stored_metric_drvts) return;
  if((time < start_time) || (time > stop_time)) return;
  if (wait_until_punc_are_close && !(PuncAreClose())) return;

  // Explicitely capture the variables for the Kokkos kernel.
  auto &adm = pmbp->padm->adm;
  auto &dg_ = dg;
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  par_for("FastFlow_metric_derivatives",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Grid spacing
    Real idx[] = {1.0 / size.d_view(m).dx1, 1.0 / size.d_view(m).dx2,
                  1.0 / size.d_view(m).dx3};

    // x-derivative
    dg_(m,D1S11,k,j,i) = Dx<NGHOST>(0, idx, adm.g_dd, m, 0, 0, k, j, i);
    dg_(m,D1S12,k,j,i) = Dx<NGHOST>(0, idx, adm.g_dd, m, 0, 1, k, j, i);
    dg_(m,D1S13,k,j,i) = Dx<NGHOST>(0, idx, adm.g_dd, m, 0, 2, k, j, i);
    dg_(m,D1S22,k,j,i) = Dx<NGHOST>(0, idx, adm.g_dd, m, 1, 1, k, j, i);
    dg_(m,D1S23,k,j,i) = Dx<NGHOST>(0, idx, adm.g_dd, m, 1, 2, k, j, i);
    dg_(m,D1S33,k,j,i) = Dx<NGHOST>(0, idx, adm.g_dd, m, 2, 2, k, j, i);

    // y-derivative
    dg_(m,D2S11,k,j,i) = Dx<NGHOST>(1, idx, adm.g_dd, m, 0, 0, k, j, i);
    dg_(m,D2S12,k,j,i) = Dx<NGHOST>(1, idx, adm.g_dd, m, 0, 1, k, j, i);
    dg_(m,D2S13,k,j,i) = Dx<NGHOST>(1, idx, adm.g_dd, m, 0, 2, k, j, i);
    dg_(m,D2S22,k,j,i) = Dx<NGHOST>(1, idx, adm.g_dd, m, 1, 1, k, j, i);
    dg_(m,D2S23,k,j,i) = Dx<NGHOST>(1, idx, adm.g_dd, m, 1, 2, k, j, i);
    dg_(m,D2S33,k,j,i) = Dx<NGHOST>(1, idx, adm.g_dd, m, 2, 2, k, j, i);

    // z-derivative
    dg_(m,D3S11,k,j,i) = Dx<NGHOST>(2, idx, adm.g_dd, m, 0, 0, k, j, i);
    dg_(m,D3S12,k,j,i) = Dx<NGHOST>(2, idx, adm.g_dd, m, 0, 1, k, j, i);
    dg_(m,D3S13,k,j,i) = Dx<NGHOST>(2, idx, adm.g_dd, m, 0, 2, k, j, i);
    dg_(m,D3S22,k,j,i) = Dx<NGHOST>(2, idx, adm.g_dd, m, 1, 1, k, j, i);
    dg_(m,D3S23,k,j,i) = Dx<NGHOST>(2, idx, adm.g_dd, m, 1, 2, k, j, i);
    dg_(m,D3S33,k,j,i) = Dx<NGHOST>(2, idx, adm.g_dd, m, 2, 2, k, j, i);
  });

  return;
}
template void FastFlow::MetricDerivatives<2>(Real time);
template void FastFlow::MetricDerivatives<3>(Real time);
template void FastFlow::MetricDerivatives<4>(Real time);

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::MetricInterp(MeshBlock *pmb)
//! \brief Interpolate metric on the surface n.
//!        Flag here the surface points contained (on this rank).
template <int NGHOST>
void FastFlow::MetricInterp() {
  // In MetricInterp() we'll flag the surface points on this mesh
  // default to 0 (no points).
  Kokkos::deep_copy(havepoint.d_view, 0.0);

  // Set metric interpolated on the surface to 0.0.
  Kokkos::deep_copy(g_interp, 0.0);
  Kokkos::deep_copy(K_interp, 0.0);
  Kokkos::deep_copy(dg_interp, 0.0);

  // Set some necessary variables used
  // for the interpolation.
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const Real xc = center[0];
  const Real yc = center[1];
  const Real zc = center[2];

  // Explicitely capture the variables for the Kokkos kernel.
  auto &polar_pos = gl_grid->polar_pos;
  auto &u_adm = pmbp->padm->u_adm;
  auto &dg_ = dg;
  auto &gi_ = g_interp;
  auto &Ki_ = K_interp;
  auto &dgi_ = dg_interp;
  auto &havepoint_ = havepoint;
  auto &rr_ = rr;

  // Resolve the host pointers on the
  // ADM indices.
  int gind[NSPMETRIC] = {
    pmbp->padm->I_ADM_GXX,
    pmbp->padm->I_ADM_GXY,
    pmbp->padm->I_ADM_GXZ,
    pmbp->padm->I_ADM_GYY,
    pmbp->padm->I_ADM_GYZ,
    pmbp->padm->I_ADM_GZZ
  };

  int Kind[NEXCURV] = {
    pmbp->padm->I_ADM_KXX,
    pmbp->padm->I_ADM_KXY,
    pmbp->padm->I_ADM_KXZ,
    pmbp->padm->I_ADM_KYY,
    pmbp->padm->I_ADM_KYZ,
    pmbp->padm->I_ADM_KZZ
  };

  par_for("FastFlow_interpolate", DevExeSpace(), 0, nangles-1,
  KOKKOS_LAMBDA(int p) {
    Real theta = polar_pos.d_view(p,0);
    Real phi = polar_pos.d_view(p,1);

    // Global coordinates of the surface.
    Real pos[3];
    Real x = xc + rr_(p) * Kokkos::sin(theta) * Kokkos::cos(phi);
    Real y = yc + rr_(p) * Kokkos::sin(theta) * Kokkos::sin(phi);
    Real z = zc + rr_(p) * Kokkos::cos(theta);
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;

    // Compute interpolation indices and weights (inline).
    auto ind_and_wghts = IndicesAndWeights<NGHOST>(indcs, size, pos, nmb);

    // Set havepoint flag.
    havepoint_.d_view(p) = ind_and_wghts.point_exist;
    if (ind_and_wghts.point_exist) {
      // Metric
      for (int a = 0; a < NSPMETRIC; ++a) {
        gi_(a,p) = InterpolateLagrange<NGHOST>(u_adm, gind[a], indcs, ind_and_wghts);
      }

      // Extrinsic curvature
      for (int b = 0; b < NEXCURV; ++b) {
        Ki_(b,p) = InterpolateLagrange<NGHOST>(u_adm, Kind[b], indcs, ind_and_wghts);
      }

      // Metric derivatives
      for (int c = 0; c < NDRVSSPMETRIC; ++c) {
        dgi_(c,p) = InterpolateLagrange<NGHOST>(dg_, c, indcs, ind_and_wghts);
      }
    }
  });

  // Sync back to host.
  havepoint.template modify<DevExeSpace>();
  havepoint.template sync<HostMemSpace>();
}
template void FastFlow::MetricInterp<2>();
template void FastFlow::MetricInterp<3>();
template void FastFlow::MetricInterp<4>();

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::FastFlowLoop()
//! \brief Fast Flow loop for horizon n.
void FastFlow::FastFlowLoop() {
  ah_found = false;

  Real meanradius = a0.h_view(0) / Kokkos::sqrt(4.0*M_PI);
  Real mass = 0;
  Real mass_prev = 0;
  Real area = 0;
  Real hrms = 0;
  Real hmean = 0;
  Real Sx = 0;
  Real Sy = 0;
  Real Sz = 0;
  Real S = 0;
  bool failed = false;

  if (verbose && ioproc) {
    fprintf(pofile_verbose, "\nSearching for horizon %d\n", nh);
    fprintf(pofile_verbose, "center = (%f, %f, %f)\n", center[0], center[1], center[2]);
    fprintf(pofile_verbose, "r_mean = %f\n", meanradius);
    fprintf(pofile_verbose, " iter      area            mass         meanradius"
                   "       minradius        hmean            Sx              Sy"
                   "              Sz             S\n");
  }

  for (int k = 0; k < flow_iterations; k++) {
    fastflow_iter = k;

    // Step 1: Compute radius r = a_lm Y_lm.
    RadiiFromSphericalHarmonics();

    // Step 2: Interpolate metric onto the surface.
    auto &indcs = pmbp->pmesh->mb_indcs;
    switch (indcs.ng) {
      case 2: MetricInterp<2>();
              break;
      case 3: MetricInterp<3>();
              break;
      case 4: MetricInterp<4>();
              break;
    }

    // Step 3: Compute the surface integrals.
    SurfaceIntegrals();

    area  = integrals[iarea];
    hrms  = integrals[ihrms]/area;
    hmean = integrals[ihmean];
    Sx = integrals[iSx] / (8 * M_PI);
    Sy = integrals[iSy] / (8 * M_PI);
    Sz = integrals[iSz] / (8 * M_PI);
    S  = Kokkos::sqrt(SQR(Sx) + SQR(Sy) + SQR(Sz));

    meanradius = a0.h_view(0) / Kokkos::sqrt(4.0 * M_PI);

    // Step 4: Check that we get a finite result.
    if (!(Kokkos::isfinite(area))) {
      if (verbose && ioproc) {
        fprintf(pofile_verbose, "Failed, Area not finite\n");
        fflush(pofile_verbose);
      }
      failed = true;
      break;
    }

    if (!(Kokkos::isfinite(hmean))) {
      if (verbose && ioproc) {
        fprintf(pofile_verbose, "Failed, hmean not finite\n");
        fflush(pofile_verbose);
      }
      failed = true;
      break;
    }

    // Irreducible mass
    mass_prev = mass;
    mass = Kokkos::sqrt(area / (16.0 * M_PI));

    if (verbose && ioproc) {
      fprintf(pofile_verbose, "%3d %15.7e %15.7e %15.7e %15.7e %15.7e"
                              " %15.7e %15.7e %15.7e %15.7e\n",
              k, area, mass, meanradius, rr_min, hmean, Sx, Sy, Sz, S);
      fflush(pofile_verbose);
    }

    if (Kokkos::fabs(hmean) > hmean_tol) {
      if (verbose && ioproc) {
        fprintf(pofile_verbose, "Failed, hmean > %f\n", hmean_tol);
        fflush(pofile_verbose);
      }
      failed = true;
      break;
     }

    if (meanradius < 0.) {
      if (verbose && ioproc) {
        fprintf(pofile_verbose, "Failed, meanradius < 0\n");
        fflush(pofile_verbose);
      }
      failed = true;
      break;
    }

    // Check to prevent horizon radius blow up and mass = 0
    if (mass < 1.0e-10) {
      if (verbose && ioproc) {
        fprintf(pofile_verbose, "Failed mass < 1e-10\n");
        fflush(pofile_verbose);
      }
      failed = true;
      break;
    }

    // End flow when mass difference is small
    if (Kokkos::fabs(mass_prev-mass) < mass_tol) {
      ah_found = true;
      break;
    }

    // Step 5: Find new spectral components.
    UpdateFlowSpectralComponents();
  }

  if (ah_found) {
    last_a0 = a0.h_view(0);

    ah_prop[harea] = area;
    ah_prop[hcoarea] = integrals[icoarea];
    ah_prop[hhrms] = hrms;
    ah_prop[hhmean] = hmean;
    ah_prop[hmeanradius] = meanradius;
    ah_prop[hminradius] = rr_min;
    ah_prop[hSx] = Sx;
    ah_prop[hSy] = Sy;
    ah_prop[hSz] = Sz;
    ah_prop[hS]  = S;
    ah_prop[hmass] = Kokkos::sqrt( SQR(mass) + 0.25*SQR(S/mass) ); // Christodoulu mass
  }

  if (verbose && ioproc) {
    if (ah_found) {
      fprintf(pofile_verbose, "Found horizon %d\n", nh);
      fprintf(pofile_verbose, " mass_irr = %f\n", mass);
      fprintf(pofile_verbose, " meanradius = %f\n", meanradius);
      fprintf(pofile_verbose, " minradius = %f\n", rr_min);
      fprintf(pofile_verbose, " hrms = %f\n", hrms);
      fprintf(pofile_verbose, " hmean = %f\n", hmean);
      fprintf(pofile_verbose, " Sx = %f\n", Sx);
      fprintf(pofile_verbose, " Sy = %f\n", Sy);
      fprintf(pofile_verbose, " Sz = %f\n", Sz);
      fprintf(pofile_verbose, " S  = %f\n", S);
    } else if (!failed && !ah_found) {
      fprintf(pofile_verbose, "Failed, reached max iterations %d\n", flow_iterations);
    }
    fflush(pofile_verbose);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::UpdateFlowSpectralComponents()
//! \brief Find new spectral components with fast-flow.
void FastFlow::UpdateFlowSpectralComponents() {
  const Real alpha = flow_alpha_beta_const;
  const Real beta = 0.5 * flow_alpha_beta_const;
  const Real A = alpha / (lmax * lmax1) + beta;
  const Real B = beta / alpha;

  Real *ABfac = new Real[lmax1];
  Real *spec0 = new Real[lmax1];
  Real *specc = new Real[lmpoints];
  Real *specs = new Real[lmpoints];

  // Step 1: Initialize coefficients.
  for (int l = 0; l <= lmax; l++) {
    spec0[l] = 0;
    ABfac[l] = A / (1.0 + B * l * (l + 1));

    for (int m = 1; m <= l; m++) {
      int l1 = lmindex(l,m,lmax);
      specc[l1] = 0;
      specs[l1] = 0;
    }
  }

  // Step 2: Build the local sums.
  for (int p = 0; p < nangles; p++) {
    if (!havepoint.h_view(p)) continue;

    const Real drho = gl_grid->int_weights.h_view(p) * rho.h_view(p);

    for (int l = 0; l <= lmax; l++) {
      spec0[l] += drho * Y0.h_view(p,l);

      for (int m = 1; m <= l; m++) {
        int l1 = lmindex(l,m,lmax);
        specc[l1] += drho * Yc.h_view(p,l1);
        specs[l1] += drho * Ys.h_view(p,l1);
      }
    }
  }

  // Step 3: Communicate the results across ranks.
  #if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,spec0,lmax1,   MPI_ATHENA_REAL,MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,specc,lmpoints,MPI_ATHENA_REAL,MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,specs,lmpoints,MPI_ATHENA_REAL,MPI_SUM, MPI_COMM_WORLD);
  #endif

  // Step 4: Update the spectral coefficients.
  for (int l = 0; l <= lmax; l++) {
    a0.h_view(l) -= ABfac[l] * spec0[l];

    for (int m = 1; m <= l; m++) {
      int l1 = lmindex(l,m,lmax);
      ac.h_view(l1) -= ABfac[l] * specc[l1];
      as.h_view(l1) -= ABfac[l] * specs[l1];
    }
  }

  delete[] ABfac;
  delete[] spec0;
  delete[] specc;
  delete[] specs;

  // Sync to back to device.
  a0.template modify<HostMemSpace>();
  a0.template sync<DevExeSpace>();
  ac.template modify<HostMemSpace>();
  ac.template sync<DevExeSpace>();
  as.template modify<HostMemSpace>();
  as.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::RadiiFromSphericalHarmonics()
//! \brief Compute the radius of the surface.
void FastFlow::RadiiFromSphericalHarmonics() {
  // Reset the radii on the surface to zero.
  Kokkos::deep_copy(rr, 0.0);
  Kokkos::deep_copy(rr_dth, 0.0);
  Kokkos::deep_copy(rr_dph, 0.0);

  // Explicitely capture the variables for the Kokkos kernel.
  auto &rr_ = rr;
  auto &rr_dth_ = rr_dth;
  auto &rr_dph_ = rr_dph;
  auto &lmax_ = lmax;
  auto &a0_ = a0;
  auto &ac_ = ac;
  auto &as_ = as;
  auto &Y0_ = Y0;
  auto &Yc_ = Yc;
  auto &Ys_ = Ys;
  auto &dY0dth_ = dY0dth;
  auto &dYcdth_ = dYcdth;
  auto &dYsdth_ = dYsdth;
  auto &dYcdph_ = dYcdph;
  auto &dYsdph_ = dYsdph;

  // Step 1: Compute the radii from the spherical harmonics.
  par_for("FastFlow_sphradii_compute", DevExeSpace(), 0, nangles-1,
  KOKKOS_LAMBDA(int p) {
    for (int l = 0; l <= lmax_; l++){
      rr_(p) += a0_.d_view(l) * Y0_.d_view(p,l);
      rr_dth_(p) += a0_.d_view(l) * dY0dth_.d_view(p,l);

      for (int m = 1; m <= l; m++){
        int l1 = lmindex(l,m,lmax_);
        rr_(p) += ac_.d_view(l1) * Yc_.d_view(p,l1) + as_.d_view(l1) * Ys_.d_view(p,l1);
        rr_dth_(p) += ac_.d_view(l1) * dYcdth_.d_view(p,l1) +
                      as_.d_view(l1) * dYsdth_.d_view(p,l1);
        rr_dph_(p) += ac_.d_view(l1) * dYcdph_.d_view(p,l1) +
                      as_.d_view(l1) * dYsdph_.d_view(p,l1);
      }
    }
  });

  // Step 2: Compute the global minimum.
  rr_min = std::numeric_limits<Real>::infinity();
  Kokkos::parallel_reduce("FastFlow_sphradii",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nangles-1),
  KOKKOS_LAMBDA(const int &p, Real &lmin) {
    lmin = Kokkos::min(lmin, rr_(p));
  }, Kokkos::Min<Real>(rr_min));
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::SurfaceIntegrals()
//! \brief Compute expansion, surface element and spin integrand on surface n.
//!        Needs metric and extr. curv. interpolated on the surface.
//!        Performs local sums and MPI reduce.
void FastFlow::SurfaceIntegrals() {
  const Real min_rp = 1e-10;

  // Initialize integrals
  for (int v = 0; v < invar; v++) {
    integrals[v] = 0.0;
  }

  Kokkos::deep_copy(rho.d_view, 0.0); // Initialize rho

  // Explicitely capture the variables for the Kokkos kernel.
  auto &polar_pos = gl_grid->polar_pos;
  auto &int_weights = gl_grid->int_weights;
  auto &havepoint_ = havepoint;
  auto &lmax_ = lmax;
  auto &flowflag_ = flowflag;

  // **INTERPOLATED SPACETIME**
  auto &gi_ = g_interp;
  auto &Ki_ = K_interp;
  auto &dgi_ = dg_interp;

  // **RADII ON THE SPHERE**
  auto &rr_ = rr;
  auto &rr_dth_ = rr_dth;
  auto &rr_dph_ = rr_dph;

  auto &rho_ = rho;

  // **EXPANSION COEFFICIENTS**
  auto &a0_ = a0;
  auto &as_ = as;
  auto &ac_ = ac;

  // **FIRST DERIVATIVES SPHERICAL HARMONICS**
  auto &dY0dth_ = dY0dth;
  auto &dYcdth_ = dYcdth;
  auto &dYsdth_ = dYsdth;
  auto &dYcdph_ = dYcdph;
  auto &dYsdph_ = dYsdph;

  // **SECOND DERIVATIVES SPHERICAL HARMONICS**
  auto &dY0dth2_ = dY0dth2;
  auto &dYcdth2_ = dYcdth2;
  auto &dYsdth2_ = dYsdth2;
  auto &dYcdph2_ = dYcdph2;
  auto &dYsdph2_ = dYsdph2;
  auto &dYcdthdph_ = dYcdthdph;
  auto &dYsdthdph_ = dYsdthdph;

  // Indices mapping
  int gmap[3][3] = {
    {S11, S12, S13},
    {S12, S22, S23},
    {S13, S23, S33}
  };

  int Kmap[3][3] = {
    {K11, K12, K13},
    {K12, K22, K23},
    {K13, K23, K33}
  };

  int dgmap[3][3][3] = {
    // Derivative in x1.
    {{D1S11, D1S12, D1S13},
     {D1S12, D1S22, D1S23},
     {D1S13, D1S23, D1S33}},

     // Derivative in x2.
    {{D2S11, D2S12, D2S13},
     {D2S12, D2S22, D2S23},
     {D2S13, D2S23, D2S33}},

     // Derivative in x3.
    {{D3S11, D3S12, D3S13},
     {D3S12, D3S22, D3S23},
     {D3S13, D3S23, D3S33}},
  };

  // Loop over surface points
  Kokkos::parallel_reduce("FastFlow_surfintegrals",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nangles-1),
  KOKKOS_LAMBDA(const int &p,
                Real& area,
                Real& coarea,
                Real& hrms,
                Real& hmean,
                Real& Sx,
                Real& Sy,
                Real& Sz) {
    // Derivatives of (r,theta,phi) w.r.t (x,y,z)
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> drdi;
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> dthetadi;
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> dphidi;

    AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> drdidj;
    AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> dthetadidj;
    AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> dphididj;

    // Derivatives of F
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> dFdi;
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> dFdi_u; // upper index
    AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> dFdidj;

    // Inverse metric
    AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> ginv;

    // Normal
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> R;

    // dx^adth , dx^a/dph
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> dXdth;
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> dXdph;

    // Flat-space coordinate rotational KV
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> phix;
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> phiy;
    AthenaPointTensor<Real, TensorSymm::NONE, NDIM, 1> phiz;

    AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> nnF;

    if (havepoint_.d_view(p)) {
      Real const theta = polar_pos.d_view(p,0);
      Real const sinth = Kokkos::sin(theta);
      Real const costh = Kokkos::cos(theta);

      Real const phi = polar_pos.d_view(p,1);
      Real const sinph = Kokkos::sin(phi);
      Real const cosph = Kokkos::cos(phi);

      // -----------------------
      // Calculate the expansion
      // -----------------------

      // Determinant of 3-metric
      Real detg = adm::SpatialDet(
        gi_(gmap[0][0],p), gi_(gmap[0][1],p), gi_(gmap[0][2],p),
        gi_(gmap[1][1],p), gi_(gmap[1][2],p), gi_(gmap[2][2],p)
      );

      // Inverse metric
      adm::SpatialInv(
        1.0/detg,
        gi_(gmap[0][0],p), gi_(gmap[0][1],p), gi_(gmap[0][2],p),
        gi_(gmap[1][1],p), gi_(gmap[1][2],p), gi_(gmap[2][2],p),
        &ginv(0,0), &ginv(0,1),  &ginv(0,2),
        &ginv(1,1), &ginv(1,2) , &ginv(2,2)
      );

      // Trace of K
      Real TrK = adm::Trace(1.0/detg,
            gi_(gmap[0][0],p), gi_(gmap[0][1],p), gi_(gmap[0][2],p),
            gi_(gmap[1][1],p), gi_(gmap[1][2],p), gi_(gmap[2][2],p),
            Ki_(Kmap[0][0],p), Ki_(Kmap[0][1],p), Ki_(Kmap[0][2],p),
            Ki_(Kmap[1][1],p), Ki_(Kmap[1][2],p), Ki_(Kmap[2][2],p));

      // Local coordinates of the surface (re-used below)
      Real const xp = rr_(p) * sinth * cosph;
      Real const yp = rr_(p) * sinth * sinph;
      Real const zp = rr_(p) * costh;

      Real const rp   = Kokkos::sqrt(xp * xp + yp * yp + zp * zp);
      Real const rhop = Kokkos::sqrt(xp * xp + yp * yp);

      if (rp < min_rp) {
        // Do not stop the code, just FastFlow failing.
        // Stop the thread and catch the NANs in FastFlow later.
        return;
      }

      Real const _divrp = 1.0 / rp;
      Real const _divrp3 = SQR(_divrp) * _divrp;
      Real const _divrhop = 1.0 / rhop;

      // First derivatives of (r,theta,phi) with respect to (x,y,z)
      drdi(0) = xp * _divrp;
      drdi(1) = yp * _divrp;
      drdi(2) = zp * _divrp;

      dthetadi(0) = zp * xp * (SQR(_divrp) * _divrhop);
      dthetadi(1) = zp * yp * (SQR(_divrp) * _divrhop);
      dthetadi(2) = -rhop * SQR(_divrp);

      dphidi(0) = -yp * SQR(_divrhop);
      dphidi(1) = xp * SQR(_divrhop);
      dphidi(2) = 0.0;

      // Second derivatives of (r,theta,phi) with respect to (x,y,z)
      drdidj(0,0) = _divrp - xp * xp * _divrp3;
      drdidj(0,1) = - xp * yp * _divrp3;
      drdidj(0,2) = - xp * zp * _divrp3;
      drdidj(1,1) = _divrp - yp * yp * _divrp3;
      drdidj(1,2) = - yp * zp * _divrp3;
      drdidj(2,2) = _divrp - zp * zp * _divrp3;

      dthetadidj(0,0) = zp*(-2.0*xp*xp*xp*xp-xp*xp*yp*yp+yp*yp*yp*yp+zp*zp*yp*yp)
                          *(SQR(_divrp)*SQR(_divrp)*SQR(_divrhop)*_divrhop);
      dthetadidj(0,1) = -xp*yp*zp*(3.0*xp*xp+3.0*yp*yp+zp*zp)*(SQR(_divrp)
                           *SQR(_divrp)*SQR(_divrhop)*_divrhop);
      dthetadidj(0,2) = xp*(xp*xp+yp*yp-zp*zp)*(SQR(_divrp)*(SQR(_divrp)*_divrhop));
      dthetadidj(1,1) = zp*(-2.0*yp*yp*yp*yp-yp*yp*xp*xp+xp*xp*xp*xp+zp*zp*xp*xp)
                          *(SQR(_divrp)*SQR(_divrp)*SQR(_divrhop)*_divrhop);
      dthetadidj(1,2) = yp*(xp*xp+yp*yp-zp*zp)*(SQR(_divrp)*(SQR(_divrp)*_divrhop));
      dthetadidj(2,2) = 2.0*zp*rhop/(rp*rp*rp*rp);

      dphididj(0,0) = 2.0 * yp * xp * (SQR(_divrhop) * SQR(_divrhop));
      dphididj(0,1) = (yp * yp - xp * xp) * (SQR(_divrhop) * SQR(_divrhop));
      dphididj(0,2) = 0.0;
      dphididj(1,1) = - 2.0 * yp * xp * (SQR(_divrhop) * SQR(_divrhop));
      dphididj(1,2) = 0.0;
      dphididj(2,2) = 0.0;

      // Compute first and second derivatives of F.
      for (int a = 0; a < NDIM; ++a) {
        // First derivative of F.
        dFdi(a) = drdi(a);

        for (int l = 0; l <= lmax_; l++) {
          dFdi(a) -= a0_.d_view(l) * dthetadi(a) * dY0dth_.d_view(p,l);

          for (int m = 1; m <= l; m++) {
            const int l1 = lmindex(l,m,lmax_);
            dFdi(a) -=
              ac_.d_view(l1) * (dthetadi(a) * dYcdth_.d_view(p,l1) +
              dphidi(a) * dYcdph_.d_view(p,l1)) +
              as_.d_view(l1) * (dthetadi(a) * dYsdth_.d_view(p,l1) +
              dphidi(a) * dYsdph_.d_view(p,l1));
          }
        }

        // Second derivative of F.
        for (int b = 0; b < NDIM; ++b) {
          dFdidj(a,b) = drdidj(a,b);

          for (int l = 0; l <= lmax_; l++) {
            dFdidj(a,b) -= a0_.d_view(l)*(dthetadidj(a,b) * dY0dth_.d_view(p,l)
                            + dthetadi(a) * dthetadi(b) * dY0dth2_.d_view(p,l));

            for (int m = 1; m <= l; m++) {
              int l1 = lmindex(l,m,lmax_);
              dFdidj(a,b) -= ac_.d_view(l1) * (dthetadidj(a,b) * dYcdth_.d_view(p,l1)
                + dthetadi(a) * (dthetadi(b) * dYcdth2_.d_view(p,l1)
                + dphidi(b) * dYcdthdph_.d_view(p,l1))
                + dphididj(a,b) * dYcdph_.d_view(p,l1)
                + dphidi(a) * (dthetadi(b) * dYcdthdph_.d_view(p,l1)
                + dphidi(b) * dYcdph2_.d_view(p,l1)))
                + as_.d_view(l1) * (dthetadidj(a,b) * dYsdth_.d_view(p,l1)
                + dthetadi(a) * (dthetadi(b) * dYsdth2_.d_view(p,l1)
                + dphidi(b) * dYsdthdph_.d_view(p,l1))
                + dphididj(a,b) * dYsdph_.d_view(p,l1)
                + dphidi(a) * (dthetadi(b) * dYsdthdph_.d_view(p,l1)
                + dphidi(b) * dYsdph2_.d_view(p,l1)));
            }
          }
        }
      }

      // Compute dFdi with the index up.
      for (int a = 0; a < NDIM; ++a) {
        dFdi_u(a) = 0;

        for (int b = 0; b < NDIM; ++b) {
          dFdi_u(a) += ginv(a,b) * dFdi(b);
        }
      }

      // Compute norm of dFdi.
      Real norm = 0;
      for (int a = 0; a < NDIM; ++a) {
        norm += dFdi_u(a) * dFdi(a);
      }

      Real u = (norm > 0) ? Kokkos::sqrt(norm) : 0.0;

      // Covariant Hessian: nabla_a nabla_b F = d_a d_b F - Gamma^c_{ab} d_c F.
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
          nnF(a,b) = dFdidj(a,b);
          for (int d = 0; d < NDIM; ++d) {
            nnF(a,b) -= 0.5 * dFdi_u(d) *
                        (dgi_(dgmap[a][b][d],p) + dgi_(dgmap[b][a][d],p)
                          - dgi_(dgmap[d][a][b],p));
          }
          nnF(b,a) = nnF(a,b);
        }
      }

      // Compute symmetric tensor for expansion.
      Real d2F = 0.0, dFdadFdbKab = 0.0, dFdadFdbFdadb = 0.0;
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
          d2F += ginv(a,b) * nnF(a,b);
          dFdadFdbFdadb += dFdi_u(a) * dFdi_u(b) * nnF(a,b);
          dFdadFdbKab += dFdi_u(a) * dFdi_u(b) * Ki_(Kmap[a][b],p);
        }
      }

      // Expansion: Theta = div(s) = (1/u) nabla^2 F + (1/u^3) dF^a dF^b K_ab
      //                            - (1/u^3) dF^a dF^b nabla_a nabla_b F - K
      Real divu = (norm > 0) ? 1.0 / u : 0.0;
      Real H = d2F * divu + dFdadFdbKab * (divu * divu)
               - dFdadFdbFdadb * (divu * divu * divu) - TrK;

      // Normal vector.
      for (int a = 0; a < NDIM; ++a) {
        R(a) = dFdi_u(a) * divu;
      }

      if (flowflag_ == 1) {
        rho_.d_view(p) = H;
      } else if (flowflag_ == 2) {
        rho_.d_view(p) = H * u;
      } else {
        // Compute the shear for the flow function.
        Real sigma_para = (
          ginv(0,0) + ginv(1,1) + ginv(2,2)
          - ginv(0,0) * SQR(drdi(0))
          - ginv(1,1) * SQR(drdi(1))
          - ginv(2,2) * SQR(drdi(2))
          - 2 * (ginv(0,1) * drdi(0) * drdi(1)
              + ginv(0,2) * drdi(0) * drdi(2)
              + ginv(1,2) * drdi(1) * drdi(2))
          - SQR(R(0)) - SQR(R(1)) - SQR(R(2))
          + SQR(R(0)) * SQR(drdi(0))
          + SQR(R(1)) * SQR(drdi(1))
          + SQR(R(2)) * SQR(drdi(2))
          + 2 * (R(0) * R(1) * drdi(0) * drdi(1)
              + R(0) * R(2) * drdi(0) * drdi(2)
              + R(1) * R(2) * drdi(1) * drdi(2)));

        Real sigma = 2 * SQR(rp) * (1 / sigma_para);
        rho_.d_view(p) = H * u * sigma;
      }

      // ---------------
      // Surface Element
      // ---------------

      // Derivatives of (x,y,z) vs (thetas, phi)

      // dr/dtheta, dr/dphi
      Real const drdt = rr_dth_(p);
      Real const drdp = rr_dph_(p);

      // Derivatives of (x,y,z) with respect to theta
      dXdth(0) = (drdt * sinth + rr_(p) * costh) * cosph;
      dXdth(1) = (drdt * sinth + rr_(p) * costh) * sinph;
      dXdth(2) = drdt * costh - rr_(p) * sinth;

      // Derivatives of (x,y,z) with respect to phi
      dXdph(0) = (drdp * cosph - rr_(p) * sinph) * sinth;
      dXdph(1) = (drdp * sinph + rr_(p) * cosph) * sinth;
      dXdph(2) = drdp * costh;

      // Induced metric on the horizon
      Real h11 = 0.;
      Real h12 = 0.;
      Real h22 = 0.;
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
          h11 += dXdth(a) * dXdth(b) * gi_(gmap[a][b],p);
          h12 += dXdth(a) * dXdph(b) * gi_(gmap[a][b],p);
          h22 += dXdph(a) * dXdph(b) * gi_(gmap[a][b],p);
        }
      }

      // Determinant of the induced metric
      Real deth = h11 * h22 - h12 * h12;
      if (deth < 0.) deth = 0.0;

      // --------------
      // Spin integrand
      // --------------

      // Flat-space coordinate rotational KV
      phix(0) =  0;
      phix(1) = -zp; // -(z-zc);
      phix(2) =  yp; // (y-yc);
      phiy(0) =  zp; // (z-zc);
      phiy(1) =  0;
      phiy(2) = -xp; // -(x-xc);
      phiz(0) = -yp; // -(y-yc);
      phiz(1) =  xp; // (x-xc);
      phiz(2) =  0;

      // Integrand of spin
      Real intSx = 0;
      Real intSy = 0;
      Real intSz = 0;
      for (int a = 0; a < NDIM; ++a) {
        for (int b = 0; b < NDIM; ++b) {
          intSx += phix(a) * R(b) * Ki_(Kmap[a][b],p);
          intSy += phiy(a) * R(b) * Ki_(Kmap[a][b],p);
          intSz += phiz(a) * R(b) * Ki_(Kmap[a][b],p);
        }
      }

      // ----------
      // Local sums
      // ----------
      const Real wght = int_weights.d_view(p);
      const Real da = wght * Kokkos::sqrt(deth) / sinth;

      area   += da;
      coarea += wght * rr_(p) * rr_(p);
      hrms   += da * H * H;
      hmean  += da * H;
      Sx     += da * intSx;
      Sy     += da * intSy;
      Sz     += da * intSz;
    }
  }, Kokkos::Sum<Real>(integrals[iarea]),
     Kokkos::Sum<Real>(integrals[icoarea]),
     Kokkos::Sum<Real>(integrals[ihrms]),
     Kokkos::Sum<Real>(integrals[ihmean]),
     Kokkos::Sum<Real>(integrals[iSx]),
     Kokkos::Sum<Real>(integrals[iSy]),
     Kokkos::Sum<Real>(integrals[iSz]));

  #if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,integrals,invar,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  #endif

  // Sync rho back to host.
  rho.template modify<DevExeSpace>();
  rho.template sync<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::ComputeSphericalHarmonics()
//! \brief Compute spherical harmonics for grid of size ntheta*nphi.
//!        Results are used for all horizons.
void FastFlow::ComputeSphericalHarmonics() {
  const Real sqrt2 = Kokkos::sqrt(2.0);

  // Explicitely capture the variables for the Kokkos kernel.
  auto &polar_pos = gl_grid->polar_pos;
  auto &lmax_ = lmax;

  // **SCALAR SPHERICAL HARMONICS**
  auto &Y0_ = Y0;
  auto &Ys_ = Ys;
  auto &Yc_ = Yc;

  // **FIRST DERIVATIVES SPHERICAL HARMONICS**
  auto &dY0dth_ = dY0dth;
  auto &dYcdth_ = dYcdth;
  auto &dYsdth_ = dYsdth;
  auto &dYcdph_ = dYcdph;
  auto &dYsdph_ = dYsdph;

  // **SECOND DERIVATIVES SPHERICAL HARMONICS**
  auto &dY0dth2_ = dY0dth2;
  auto &dYcdth2_ = dYcdth2;
  auto &dYsdth2_ = dYsdth2;
  auto &dYcdph2_ = dYcdph2;
  auto &dYsdph2_ = dYsdph2;
  auto &dYcdthdph_ = dYcdthdph;
  auto &dYsdthdph_ = dYsdthdph;

  // Loop over all angles.
  par_for("FastFlow_sphharmonics", DevExeSpace(), 0, nangles-1,
  KOKKOS_LAMBDA(int p) {
    const Real theta = polar_pos.d_view(p,0);
    const Real phi   = polar_pos.d_view(p,1);

    for (int l = 0; l <= lmax_; ++l) {
      for (int m = 0; m <= l; ++m) {
        Real YlmR, YlmI;
        Real YlmRdth, YlmIdth, YlmRdphi, YlmIdphi;
        Real YlmRdth2, YlmIdth2, YlmRdphi2, YlmIdphi2, YlmRdthdphi, YlmIdthdphi;

        SphericalHarmSecondDerivs(&YlmR, &YlmI,
                                  &YlmRdth, &YlmIdth, &YlmRdphi, &YlmIdphi,
                                  &YlmRdth2, &YlmIdth2, &YlmRdphi2, &YlmIdphi2,
                                  &YlmRdthdphi, &YlmIdthdphi, l, m, theta, phi);

        if (m == 0) { // m = 0 spherical harmonics
          Y0_.d_view(p,l) = YlmR;
          dY0dth_.d_view(p,l) = YlmRdth;
          dY0dth2_.d_view(p,l) = YlmRdth2;
        } else { // m > 0 spherical harmonics
          const int l1 = lmindex(l,m,lmax_);
          Yc_.d_view(p,l1) = sqrt2 * YlmR;
          Ys_.d_view(p,l1) = sqrt2 * YlmI;

          dYcdth_.d_view(p,l1) = sqrt2 * YlmRdth;
          dYsdth_.d_view(p,l1) = sqrt2 * YlmIdth;
          dYcdph_.d_view(p,l1) = sqrt2 * YlmRdphi;
          dYsdph_.d_view(p,l1) = sqrt2 * YlmIdphi;

          dYcdth2_.d_view(p,l1) = sqrt2 * YlmRdth2;
          dYsdth2_.d_view(p,l1) = sqrt2 * YlmIdth2;
          dYcdph2_.d_view(p,l1) = sqrt2 * YlmRdphi2;
          dYsdph2_.d_view(p,l1) = sqrt2 * YlmIdphi2;
          dYsdthdph_.d_view(p,l1) = sqrt2 * YlmIdthdphi;
          dYcdthdph_.d_view(p,l1) = sqrt2 * YlmRdthdphi;
        }
      }
    }
  });

  // Sync the spherical harmonics to Host.
  // (OS): Although there is a lot to sync, this
  //       is okay, since the function is only called
  //       once, when the constructor is instantiated.
  //       This will come in handy to speed up the rest
  //       of the code.
  // **SCALARS**
  Y0.template modify<DevExeSpace>();
  Y0.template sync<HostMemSpace>();
  Yc.template modify<DevExeSpace>();
  Yc.template sync<HostMemSpace>();
  Ys.template modify<DevExeSpace>();
  Ys.template sync<HostMemSpace>();

  // **FIRST DERIVATIVES**
  dY0dth.template modify<DevExeSpace>();
  dY0dth.template sync<HostMemSpace>();
  dYcdth.template modify<DevExeSpace>();
  dYcdth.template sync<HostMemSpace>();
  dYsdth.template modify<DevExeSpace>();
  dYsdth.template sync<HostMemSpace>();
  dYcdph.template modify<DevExeSpace>();
  dYcdph.template sync<HostMemSpace>();
  dYsdph.template modify<DevExeSpace>();
  dYsdph.template sync<HostMemSpace>();

  // **SECOND DERIVATIVES**
  dY0dth2.template modify<DevExeSpace>();
  dY0dth2.template sync<HostMemSpace>();
  dYcdth2.template modify<DevExeSpace>();
  dYcdth2.template sync<HostMemSpace>();
  dYsdth2.template modify<DevExeSpace>();
  dYsdth2.template sync<HostMemSpace>();
  dYcdph2.template modify<DevExeSpace>();
  dYcdph2.template sync<HostMemSpace>();
  dYsdph2.template modify<DevExeSpace>();
  dYsdph2.template sync<HostMemSpace>();
  dYcdthdph.template modify<DevExeSpace>();
  dYcdthdph.template sync<HostMemSpace>();
  dYsdthdph.template modify<DevExeSpace>();
  dYsdthdph.template sync<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn Real FastFlow::PuncMaxDistance()
//! \brief Max Euclidean distance between punctures.
Real FastFlow::PuncMaxDistance() {
  Real maxdist = 0.0;
  for (int pix = 0; pix < npunct; ++pix) {
    Real xp = pmbp->pz4c->ptracker[pix]->GetPos(0);
    Real yp = pmbp->pz4c->ptracker[pix]->GetPos(1);
    Real zp = pmbp->pz4c->ptracker[pix]->GetPos(2);

    for (int p = pix+1; p < npunct; ++p) {
      Real x = pmbp->pz4c->ptracker[p]->GetPos(0);
      Real y = pmbp->pz4c->ptracker[p]->GetPos(1);
      Real z = pmbp->pz4c->ptracker[p]->GetPos(2);
      maxdist = Kokkos::fmax(maxdist,
                Kokkos::sqrt(SQR(x - xp) + SQR(y - yp) + SQR(z - zp)));
    }
  }
  return maxdist;
}

//----------------------------------------------------------------------------------------
//! \fn Real FastFlow::PuncMaxDistance(const int pix)
//! \brief Max Euclidean distance from puncture pix to other punctures.
Real FastFlow::PuncMaxDistance(const int pix) {
  Real xp = pmbp->pz4c->ptracker[pix]->GetPos(0);
  Real yp = pmbp->pz4c->ptracker[pix]->GetPos(1);
  Real zp = pmbp->pz4c->ptracker[pix]->GetPos(2);
  Real maxdist = 0.0;

  for (int p = 0; p < npunct; ++p) {
    if (p==pix) continue;
    Real x = pmbp->pz4c->ptracker[p]->GetPos(0);
    Real y = pmbp->pz4c->ptracker[p]->GetPos(1);
    Real z = pmbp->pz4c->ptracker[p]->GetPos(2);
    maxdist = Kokkos::fmax(maxdist,
              Kokkos::sqrt(SQR(x - xp) + SQR(y - yp) + SQR(z - zp)));
  }

  return maxdist;
}

//----------------------------------------------------------------------------------------
//! \fn Real FastFlow::PuncSumMasses()
//! \brief Return sum of puncture's intial masses.
Real FastFlow::PuncSumMasses() {
  Real mass = 0.0;
  for (int p = 0; p < npunct; ++p) {
    mass += pmbp->pz4c->ptracker[p]->GetMass();
  }
  return mass;
}

//----------------------------------------------------------------------------------------
//! \fn void FastFlow::PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc)
//! \brief Return mss-weighted center of puncture positions.
void FastFlow::PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc) {
  Real sumx = 0.0; // sum of m_i*x_i
  Real sumy = 0.0;
  Real sumz = 0.0;
  Real divsum = 0.0; // sum of m_i to later divide by
  for (int p = 0; p < npunct; p++) {
    Real x = pmbp->pz4c->ptracker[p]->GetPos(0);
    Real y = pmbp->pz4c->ptracker[p]->GetPos(1);
    Real z = pmbp->pz4c->ptracker[p]->GetPos(2);
    Real m = pmbp->pz4c->ptracker[p]->GetMass();
    sumx += m*x;
    sumy += m*y;
    sumz += m*z;
    divsum += m;
  }
  divsum = 1.0/divsum;
  *xc = sumx * divsum;
  *yc = sumy * divsum;
  *zc = sumz * divsum;
}

//----------------------------------------------------------------------------------------
//! \fn bool FastFlow::PuncAreClose()
//! \brief Check when the maximal distance between all punctures is below threshold.
bool FastFlow::PuncAreClose() {
  Real const mass = PuncSumMasses();
  Real const maxdist = PuncMaxDistance();
  return (maxdist < merger_distance * mass);
}
