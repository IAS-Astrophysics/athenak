//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ahf.cpp
//! \brief Implementation of the apparent horizon finder class
//!        based on the fast-flow algorithm of Gundlach:1997us and Alcubierre:1998rq

#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <unistd.h>
#include <cmath> // NAN

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "ahf.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "compact_object_tracker.hpp"
#include "geodesic-grid/spherical_grid.hpp" 
#include "utils/spherical_harm.hpp"
#include "z4c.hpp"

//----------------------------------------------------------------------------------------
//! \fn AHF::AHF(MeshBlockPack *pmbp, ParameterInput * pin, int n)
//! \brief Constructor for AHF class
AHF::AHF(MeshBlockPack *pmbp, ParameterInput *pin, int n):
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
  rho("rho",1)
{
  nh = n; // The n-th horizon
  std::string n_str = std::to_string(nh);

  // Read parameter input variables
  nhorizon = pin->GetOrAddInteger("ahf", "num_horizons", 1); // Number of horizons
  ntheta = pin->GetOrAddInteger("ahf", "ntheta", 5); // Number of points theta

  lmax = pin->GetOrAddInteger("ahf", "lmax", 4);
  lmax1 = lmax + 1;

  // Flow parameters
  flow_iterations = pin->GetOrAddInteger("ahf", "flow_iterations_" + n_str, 100);
  flow_alpha_beta_const = pin->GetOrAddReal("ahf", "flow_alpha_beta_const_" + n_str, 1.0);

  // Convergence parameters
  hmean_tol = pin->GetOrAddReal("ahf", "hmean_tol_" + n_str, 100.);
  mass_tol = pin->GetOrAddReal("ahf", "mass_tol_" + n_str, 1e-2);

  // Output booleans
  verbose = pin->GetOrAddBoolean("ahf", "verbose", false);
  output_ylm = pin->GetOrAddBoolean("ahf", "output_ylm", false);
  output_grid = pin->GetOrAddBoolean("ahf", "output_grid", false);

  root = pin->GetOrAddInteger("ahf", "mpi_root", 0);
  merger_distance = pin->GetOrAddReal("ahf", "merger_distance", 0.1);
  use_stored_metric_drvts = pin->GetBoolean("z4c", "store_metric_drvts");

  // Initial guess
  initial_radius = pin->GetOrAddReal("ahf", "initial_radius_" + n_str, 1.0);
  rr_min = -1.0;

  expand_guess = pin->GetOrAddReal("ahf", "expand_guess", 1.0);

  // Center
  center[0] = pin->GetOrAddReal("ahf", "center_x_" + n_str, 0.0);
  center[1] = pin->GetOrAddReal("ahf", "center_y_" + n_str, 0.0);
  center[2] = pin->GetOrAddReal("ahf", "center_z_" + n_str, 0.0);

  compute_every_iter = 1; // (OS): Is this covered by the task triggers?

  // Punctures
  npunct = pin->GetOrAddInteger("ahf", "npunct", 0); // Number of punctures
  use_puncture = pin->GetOrAddInteger("ahf", "use_puncture_" + n_str, -1);

  if (use_puncture >= 0) {
    // Center is determined on the fly during the initial guess
    // to follow the chosen puncture
    if (use_puncture >= npunct) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << " : punc = " << use_puncture << " > npunct = " << npunct << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  wait_until_punc_are_close = pin->GetOrAddBoolean("ahf", "wait_until_punc_are_close_" + n_str, 0);

  // Timer
  start_time = pin->GetOrAddReal("ahf", "start_time_" + n_str, std::numeric_limits<double>::max());
  stop_time = pin->GetOrAddReal("ahf", "stop_time_" + n_str, -1.0);

  // Grid and quadrature weights
  gl_grid = new GaussLegendreGrid(pmbp, ntheta, initial_radius);
  nangles = gl_grid->nangles;

  // Points for spherical harmonics l >= 1
  lmpoints = lmax1 * lmax1;

  // Reallocate for the coefficients
  Kokkos::realloc(a0, lmax1); 
  Kokkos::realloc(ac, lmpoints);
  Kokkos::realloc(as, lmpoints);

  // Reallocate for the spherical harmonics
  // The spherical grid is the same for all surfaces
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

  // Fields on the sphere
  Kokkos::realloc(rr, nangles);
  Kokkos::realloc(rr_dth, nangles);
  Kokkos::realloc(rr_dph, nangles);

  // Array computed in surface integrals
  Kokkos::realloc(rho, nangles);

  // Flag points existing on this mesh
  Kokkos::realloc(havepoint, nangles);

  // Initialize horizon properties to NAN
  for (int v = 0; v < hnvar; ++v) {
    ah_prop[v] = NAN;
  }

  // Prepare output
  ofname_summary = pin->GetString("job", "basename") + ".";
  ofname_summary += pin->GetOrAddString("ahf", "horizon_file_summary_" + n_str, "horizon_summary_" + n_str);
  ofname_summary += ".txt";

  ofname_shape = pin->GetString("job", "basename") + ".";
  ofname_shape += pin->GetOrAddString("ahf", "horizon_file_shape_" + n_str, "horizon_shape_" + n_str);
  ofname_shape += ".txt";

  if (verbose) {
    ofname_verbose = pin->GetString("job", "basename") + ".";
    ofname_verbose += pin->GetOrAddString("ahf", "horizon_verbose_" + n_str, "horizon_verbose_" + n_str);
    ofname_verbose += ".txt";
  }

  if (output_ylm) {
    ofname_ylm = pin->GetString("job", "basename") + ".";
    ofname_ylm += pin->GetOrAddString("ahf", "horizon_ylm_" + n_str, "horizon_ylm_" + n_str);
    ofname_ylm += ".txt";
  }

  if (output_grid) {
    ofname_grid = pin->GetString("job", "basename") + ".";
    ofname_grid += pin->GetOrAddString("ahf", "horizon_grid_" + n_str, "horizon_grid_" + n_str);
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
      fprintf(pofile_summary, "# 1:iter 2:time 3:mass 4:Sx 5:Sy 6:Sz 7:S 8:area 9:hrms 10:hmean 11:meanradius 12:minradius\n");
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

      // Print grid to the output file
      for (int p = 0; p < nangles; ++p) {
        const Real theta = gl_grid->polar_pos.h_view(p,0);
        const Real phi   = gl_grid->polar_pos.h_view(p,1);
        const Real weight = gl_grid->int_weights.h_view(p);
        fprintf(pofile_grid, "%e %e %e\n", theta, phi, weight);
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
      fprintf(pofile_ylm, "# 1:Theta\t2:Phi\t3:l\t4:m\t5:Y0\t6:Yc\t7:Ys\t8:dY0dth\t9:dYcdth\t10:dYsdth\t"
                   "11:dYcdphi\t12:dYsdphi\t13:dY0dth2\t14:dYcdth2\t15:dYsdth2\t16:dYcdph2\t17:dYsdph2\t" 
                   "18:dYcdthdphi\t19:dYsdthdphi\n");

      for (int l = 0; l <= lmax; ++l) {
        for (int m = 0; m <= l; ++m) {
          for (int p = 0; p < nangles; ++p) {
            const Real theta = gl_grid->polar_pos.h_view(p,0);
            const Real phi   = gl_grid->polar_pos.h_view(p,1);

            if (m == 0){
              fprintf(pofile_ylm, "%e %e %d %d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",
                      theta, phi, l, m, Y0.h_view(p,l), 0.0, 0.0,
                      dY0dth.h_view(p,l), 0.0, 0.0, 0.0, 0.0,
                      dY0dth2.h_view(p,l), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            } else {
              const int l1 = lmindex(l,m);
              fprintf(pofile_ylm, "%e %e %d %d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",
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
} // (DONE)

//----------------------------------------------------------------------------------------
//! \fn void AHF::AHF()
//! \brief Destructor for AHF class
AHF::~AHF() {
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
//! \fn void AHF::ComputeSphericalHarmonics()
//! \brief Compute spherical harmonics for grid of size ntheta*nphi.
//!        Results are used for all horizons.
void AHF::ComputeSphericalHarmonics()
{
  const Real sqrt2 = Kokkos::sqrt(2.0);

  // Initialize all arrays
  Kokkos::deep_copy(Y0.h_view, 0.0);
  Kokkos::deep_copy(Yc.h_view, 0.0);
  Kokkos::deep_copy(Ys.h_view, 0.0);

  Kokkos::deep_copy(dY0dth.h_view, 0.0);
  Kokkos::deep_copy(dYcdth.h_view, 0.0);
  Kokkos::deep_copy(dYsdth.h_view, 0.0);
  Kokkos::deep_copy(dYcdph.h_view, 0.0);
  Kokkos::deep_copy(dYsdph.h_view, 0.0);

  Kokkos::deep_copy(dY0dth2.h_view, 0.0);
  Kokkos::deep_copy(dYcdth2.h_view, 0.0);
  Kokkos::deep_copy(dYcdthdph.h_view, 0.0);
  Kokkos::deep_copy(dYsdth2.h_view, 0.0);
  Kokkos::deep_copy(dYsdthdph.h_view, 0.0);
  Kokkos::deep_copy(dYcdph2.h_view, 0.0);
  Kokkos::deep_copy(dYsdph2.h_view, 0.0);

  // Loop over angles
  for(int p = 0; p < nangles; ++p){
    const Real theta = gl_grid->polar_pos.h_view(p,0);
    const Real phi   = gl_grid->polar_pos.h_view(p,1);

    for(int l = 0; l <= lmax; ++l){
      for(int m = 0; m <= l; ++m){
        Real YlmR, YlmI;
        Real YlmRdth, YlmIdth, YlmRdphi, YlmIdphi;
        Real YlmRdth2, YlmIdth2, YlmRdphi2, YlmIdphi2, YlmRdthdphi, YlmIdthdphi;

        SphericalHarmSecondDerivs(&YlmR, &YlmI,
                                  &YlmRdth, &YlmIdth, &YlmRdphi, &YlmIdphi,
                                  &YlmRdth2, &YlmIdth2, &YlmRdphi2, &YlmIdphi2,
                                  &YlmRdthdphi, &YlmIdthdphi, l, m, theta, phi);
        
        if (m == 0) { // m = 0 spherical harmonics
          Y0.h_view(p,l) = YlmR;
          dY0dth.h_view(p,l) = YlmRdth;
          dY0dth2.h_view(p,l) = YlmRdth2;
        }
        else { // m > 0 spherical harmonics
          const int l1 = lmindex(l,m);
          Yc.h_view(p,l1) = sqrt2 * YlmR;
          Ys.h_view(p,l1) = sqrt2 * YlmI;

          dYcdth.h_view(p,l1) = sqrt2 * YlmRdth;
          dYsdth.h_view(p,l1) = sqrt2 * YlmIdth;
          dYcdph.h_view(p,l1) = sqrt2 * YlmRdphi;
          dYsdph.h_view(p,l1) = sqrt2 * YlmIdphi;

          dYcdth2.h_view(p,l1)   = sqrt2 * YlmRdth2;
          dYcdthdph.h_view(p,l1) = sqrt2 * YlmRdthdphi;
          dYsdth2.h_view(p,l1)   = sqrt2 * YlmIdth2;
          dYsdthdph.h_view(p,l1) = sqrt2 * YlmIdthdphi;
          dYcdph2.h_view(p,l1)   = sqrt2 * YlmRdphi2;
          dYsdph2.h_view(p,l1)   = sqrt2 * YlmIdphi2;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn int AHF::lmindex(const int l, const int m)
//! \brief Multipolar single index (l,m) -> index
int AHF::lmindex(const int l, const int m)
{
  return l * lmax1 + m;
} // (DONE)

