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
#include "coordinates/coordinates.hpp"
#include "compact_object_tracker.hpp"
#include "geodesic-grid/spherical_grid.hpp" 
#include "utils/spherical_harm.hpp"
#include "utils/lagrange_interpolator.hpp"
#include "z4c.hpp"
#include "coordinates/cell_locations.hpp"
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
  rho("rho",1), dg("dg",1,1,1,1,1), g_interp("g_interp",1,1),
  K_interp("K_interp",1,1), dg_interp("dg_interp",1,1)
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
  use_puncture_massweighted_center = pin->GetOrAddBoolean("ahf", "use_puncture_massweighted_center_" + n_str, 0);

  // Timer
  start_time = pin->GetOrAddReal("ahf", "start_time_" + n_str, std::numeric_limits<double>::max());
  stop_time = pin->GetOrAddReal("ahf", "stop_time_" + n_str, -1.0);

  // Grid and quadrature weights
  gl_grid = new GaussLegendreGrid(pmbp, ntheta, 1.0); // unit-sphere
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

  // Allocate the arrays holding the interpolated values
  /* g_interp.resize(6);
  K_interp.resize(6);
  for (int d = 0; d < g_interp.size(); ++d) {
    Kokkos::realloc(g_interp[d], nangles);
    Kokkos::realloc(K_interp[d], nangles);
  }

  dg_interp.resize(18);
  for (int d = 0; d < dg_interp.size(); ++d) {
    Kokkos::realloc(dg_interp[d], nangles);
  } */
  Kokkos::realloc(g_interp, 6, nangles);
  Kokkos::realloc(K_interp, 6, nangles);
  Kokkos::realloc(dg_interp, 18, nangles);

  // Allocate memory for the array holding the metric derivatives
  auto &indcs = pmbp->pmesh->mb_indcs;
  int nmb = pmbp->nmb_thispack;
  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = indcs.nx2 + 2 * (indcs.ng);
  int ncells3 = indcs.nx3 + 2 * (indcs.ng);
  Kokkos::realloc(dg, nmb, 18, ncells3, ncells2, ncells1);
  
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
      fprintf(pofile_ylm, "# 1:Theta\t2:Phi\t3:l\t4:m\t5:Y0\t6:Yc\t7:Ys\t8:dY0dth\t9:dYcdth\t10:dYsdth\t"
                   "11:dYcdphi\t12:dYsdphi\t13:dY0dth2\t14:dYcdth2\t15:dYsdth2\t16:dYcdph2\t17:dYsdph2\t" 
                   "18:dYcdthdphi\t19:dYsdthdphi\n");

      for (int l = 0; l <= lmax; ++l) {
        for (int m = 0; m <= l; ++m) {
          for (int p = 0; p < nangles; ++p) {
            const Real theta = gl_grid->polar_pos.h_view(p,0);
            const Real phi   = gl_grid->polar_pos.h_view(p,1);

            if (m == 0){
              fprintf(pofile_ylm, "%.15e %.15e %d %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                      theta, phi, l, m, Y0(p,l), 0.0, 0.0,
                      dY0dth(p,l), 0.0, 0.0, 0.0, 0.0,
                      dY0dth2(p,l), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            } else {
              const int l1 = lmindex(l,m);
              fprintf(pofile_ylm, "%.15e %.15e %d %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
                      theta, phi, l, m,
                      0.0,
                      Yc(p,l1),
                      Ys(p,l1),
                      0.0,
                      dYcdth(p,l1),
                      dYsdth(p,l1),
                      dYcdph(p,l1),
                      dYsdph(p,l1),
                      0.0,
                      dYcdth2(p,l1),
                      dYsdth2(p,l1),
                      dYcdph2(p,l1),
                      dYsdph2(p,l1),
                      dYcdthdph(p,l1),
                      dYsdthdph(p,l1)
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
//! \fn void AHF::Write(int iter, Real time)
//! \brief Output summary and shape file, for each horizon
void AHF::Write(int iter, Real time)
{
  if (ioproc) {
    if((time < start_time) || (time > stop_time)) return;
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

    if (ah_found)
    {
      // Shape file (coefficients)
      pofile_shape = fopen(ofname_shape.c_str(), "a");
      if (NULL == pofile_shape)
      {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "Could not open file '" << pofile_shape << "' for writing!" << std::endl;
        exit(EXIT_FAILURE);
      }
      fprintf(pofile_shape, "# iter = %d, Time = %g\n",iter,time);
      for(int l = 0; l <= lmax; l++) {
        fprintf(pofile_shape,"%e ", a0(l));
    
        for(int m = 1; m <= l; m++){
          int l1 = lmindex(l,m);
          fprintf(pofile_shape,"%e ",ac(l1));
          fprintf(pofile_shape,"%e ",as(l1));
        }
      }
      fprintf(pofile_shape,"\n");
      fclose(pofile_shape);
    }
  }

  // This is needed on all ranks.
  if (ah_found && (time_first_found < 0))
  {
    std::string parname {"time_first_found_" + std::to_string(nh)};
    time_first_found = time;
    pin->SetReal("ahf", parname, time_first_found);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::Find(int iter, Real time)
//! \brief Search for the horizons
void AHF::Find(int iter, Real time)
{
  if((time < start_time) || (time > stop_time)) return;
  if (wait_until_punc_are_close && !(PuncAreClose())) return;
  if (verbose && ioproc) {
    fprintf(pofile_verbose, "time=%.4f, cycle=%d\n", time, iter);
  }

  InitialGuess();
  FastFlowLoop();

  // Retain `last_a0` in restart: this serves as primary ini. guess.
  if (ah_found)
  {
    std::string parname;
    parname = "last_a0_" + std::to_string(nh); // nh: horizon index

    pin->SetReal("ahf", parname, last_a0);

    parname = "ah_found_a0_" + std::to_string(nh);
    pin->SetBoolean("ahf", parname, ah_found);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::InitialGuess()
//! \brief Initial guess for spectral coefs of horizon n
void AHF::InitialGuess()
{  
  // Reset Coefficients to Zero
  Kokkos::deep_copy(a0, 0.0); 
  Kokkos::deep_copy(ac, 0.0);
  Kokkos::deep_copy(as, 0.0);
  
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
      a0(0) = last_a0 * expand_guess;
    } else {
      a0(0) = std::max(0.5 * mass, std::min(mass, 0.5 * largedist));
      a0(0) *= std::sqrt(4.0 * M_PI);
    }
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
    a0(0) = last_a0 * expand_guess;
  } else {
    a0(0) = std::sqrt(4.0 * M_PI) * initial_radius;
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool AHF::MetricDerivatives(Real time)
//! \brief compute drvts of ADM metric at MB level
template <int NGHOST>
bool AHF::MetricDerivatives(Real time)
{ 
  // Check whether derivatives have to be computed
  if (use_stored_metric_drvts) return false;
  if((time < start_time) || (time > stop_time)) return false;
  if (wait_until_punc_are_close && !(PuncAreClose())) return false;

  AthenaTensor<Real, TensorSymm::SYM2, NDIM, 2> adm_g_dd; // 3-metric
  adm_g_dd.InitWithShallowSlice(pmbp->padm->u_adm, adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ);
  
  // MeshBlock variables
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  par_for("AHF_metric_derivatives",DevExeSpace(),0,nmb-1,ks,ke+1,js,je+1,is,ie+1, 
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Grid spacing
    Real idx[] = {1.0 / size.d_view(m).dx1, 1.0 / size.d_view(m).dx2, 1.0 / size.d_view(m).dx3};

    // x-derivative
    dg(m,DX_GXX,k,j,i) = Dx<NGHOST>(0, idx, adm_g_dd, m, 0, 0, k, j, i);
    dg(m,DX_GXY,k,j,i) = Dx<NGHOST>(0, idx, adm_g_dd, m, 0, 1, k, j, i);
    dg(m,DX_GXZ,k,j,i) = Dx<NGHOST>(0, idx, adm_g_dd, m, 0, 2, k, j, i);
    dg(m,DX_GYY,k,j,i) = Dx<NGHOST>(0, idx, adm_g_dd, m, 1, 1, k, j, i);
    dg(m,DX_GYZ,k,j,i) = Dx<NGHOST>(0, idx, adm_g_dd, m, 1, 2, k, j, i);
    dg(m,DX_GZZ,k,j,i) = Dx<NGHOST>(0, idx, adm_g_dd, m, 2, 2, k, j, i);
    
    // y-derivative
    dg(m,DY_GXX,k,j,i) = Dx<NGHOST>(1, idx, adm_g_dd, m, 0, 0, k, j, i);
    dg(m,DY_GXY,k,j,i) = Dx<NGHOST>(1, idx, adm_g_dd, m, 0, 1, k, j, i);
    dg(m,DY_GXZ,k,j,i) = Dx<NGHOST>(1, idx, adm_g_dd, m, 0, 2, k, j, i);
    dg(m,DY_GYY,k,j,i) = Dx<NGHOST>(1, idx, adm_g_dd, m, 1, 1, k, j, i);
    dg(m,DY_GYZ,k,j,i) = Dx<NGHOST>(1, idx, adm_g_dd, m, 1, 2, k, j, i);
    dg(m,DY_GZZ,k,j,i) = Dx<NGHOST>(1, idx, adm_g_dd, m, 2, 2, k, j, i);
    
    // z-derivative
    dg(m,DZ_GXX,k,j,i) = Dx<NGHOST>(2, idx, adm_g_dd, m, 0, 0, k, j, i);
    dg(m,DZ_GXY,k,j,i) = Dx<NGHOST>(2, idx, adm_g_dd, m, 0, 1, k, j, i);
    dg(m,DZ_GXZ,k,j,i) = Dx<NGHOST>(2, idx, adm_g_dd, m, 0, 2, k, j, i);
    dg(m,DZ_GYY,k,j,i) = Dx<NGHOST>(2, idx, adm_g_dd, m, 1, 1, k, j, i);
    dg(m,DZ_GYZ,k,j,i) = Dx<NGHOST>(2, idx, adm_g_dd, m, 1, 2, k, j, i);
    dg(m,DZ_GZZ,k,j,i) = Dx<NGHOST>(2, idx, adm_g_dd, m, 2, 2, k, j, i);
  });

  return true;
}
template bool AHF::MetricDerivatives<2>(Real time);
template bool AHF::MetricDerivatives<3>(Real time);
template bool AHF::MetricDerivatives<4>(Real time);

//----------------------------------------------------------------------------------------
//! \fn void AHF::MetricInterp(MeshBlock *pmb)
//! \brief Interpolate metric on the surface n
//!        Flag here the surface points contained (on this rank)
void AHF::MetricInterp()
{ 
  // Interpolate metric and extrinsic curvature to sphere
  /* for (int c = 0; c < g_interp.size(); ++c) {
    gl_grid->InterpolateToSphere(g_idx[c], pmbp->padm->u_adm);
    Kokkos::deep_copy(g_interp[c], gl_grid->interp_vals);
    
    gl_grid->InterpolateToSphere(K_idx[c], pmbp->padm->u_adm);
    Kokkos::deep_copy(K_interp[c], gl_grid->interp_vals);
  }

  // Interpolate metric derivatives to sphere
  for (int c = 0; c < dg_interp.size(); ++c) {
    gl_grid->InterpolateToSphere(c, dg);
    Kokkos::deep_copy(dg_interp[c], gl_grid->interp_vals);
  } */
  
  // Set havepoint flag
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const Real xc = center[0];
  const Real yc = center[1];
  const Real zc = center[2];
  Real pos[3];

  for (int p = 0; p < nangles; ++p) {
    Real theta = gl_grid->polar_pos.h_view(p,0);
    Real phi = gl_grid->polar_pos.h_view(p,1);

    // Global coordinates of the surface
    Real x = xc + rr(p) * Kokkos::sin(theta) * Kokkos::cos(phi);
    Real y = yc + rr(p) * Kokkos::sin(theta) * Kokkos::sin(phi);
    Real z = zc + rr(p) * Kokkos::cos(theta);
    pos[0] = x;
    pos[1] = y;
    pos[2] = z;

    // Set havepoint flag
    for (int m = 0; m < nmb; ++m) {
      if ((size.h_view(m).x1min <= x) && (x <= size.h_view(m).x1max) &&
          (size.h_view(m).x2min <= y) && (y <= size.h_view(m).x2max) &&
          (size.h_view(m).x3min <= z) && (z <= size.h_view(m).x3max)) {
            havepoint(p) += 1;
      }
    }

    // Initialize interpolator at point
    auto *S = new LagrangeInterpolator(pmbp, pos);

    if (S->point_exist) {
      g_interp(GXX,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_GXX);
      g_interp(GXY,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_GXY);
      g_interp(GXZ,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_GXZ);
      g_interp(GYY,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_GYY);
      g_interp(GYZ,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_GYZ);
      g_interp(GZZ,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_GZZ);

      K_interp(KXX,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_KXX);
      K_interp(KXY,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_KXY);
      K_interp(KXZ,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_KXZ);
      K_interp(KYY,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_KYY);
      K_interp(KYZ,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_KYZ);
      K_interp(KZZ,p) = S->Interpolate(pmbp->padm->u_adm, pmbp->padm->I_ADM_KZZ);

      dg_interp(DX_GXX,p) = S->Interpolate(dg, DX_GXX);
      dg_interp(DX_GXY,p) = S->Interpolate(dg, DX_GXY);
      dg_interp(DX_GXZ,p) = S->Interpolate(dg, DX_GXZ);
      dg_interp(DX_GYY,p) = S->Interpolate(dg, DX_GYY);
      dg_interp(DX_GYZ,p) = S->Interpolate(dg, DX_GYZ);
      dg_interp(DX_GZZ,p) = S->Interpolate(dg, DX_GZZ);

      dg_interp(DY_GXX,p) = S->Interpolate(dg, DY_GXX);
      dg_interp(DY_GXY,p) = S->Interpolate(dg, DY_GXY);
      dg_interp(DY_GXZ,p) = S->Interpolate(dg, DY_GXZ);
      dg_interp(DY_GYY,p) = S->Interpolate(dg, DY_GYY);
      dg_interp(DY_GYZ,p) = S->Interpolate(dg, DY_GYZ);
      dg_interp(DY_GZZ,p) = S->Interpolate(dg, DY_GZZ);

      dg_interp(DZ_GXX,p) = S->Interpolate(dg, DZ_GXX);
      dg_interp(DZ_GXY,p) = S->Interpolate(dg, DZ_GXY);
      dg_interp(DZ_GXZ,p) = S->Interpolate(dg, DZ_GXZ);
      dg_interp(DZ_GYY,p) = S->Interpolate(dg, DZ_GYY);
      dg_interp(DZ_GYZ,p) = S->Interpolate(dg, DZ_GYZ);
      dg_interp(DZ_GZZ,p) = S->Interpolate(dg, DZ_GZZ);
    }
    fprintf(fgi, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n", theta, phi,
              g_interp(GXX,p),g_interp(GXY,p),g_interp(GXZ,p),
              g_interp(GYY,p),g_interp(GYZ,p),g_interp(GZZ,p));
    fprintf(fKi, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n", theta, phi,
              K_interp(GXX,p),K_interp(GXY,p),K_interp(GXZ,p),
              K_interp(GYY,p),K_interp(GYZ,p),K_interp(GZZ,p));
    fprintf(fdgi, "%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n", theta, phi,
              dg_interp(DX_GXX,p),dg_interp(DX_GXY,p),dg_interp(DX_GXZ,p),
              dg_interp(DX_GYY,p),dg_interp(DX_GYZ,p),dg_interp(DX_GZZ,p),
              dg_interp(DY_GXX,p),dg_interp(DY_GXY,p),dg_interp(DY_GXZ,p),
              dg_interp(DY_GYY,p),dg_interp(DY_GYZ,p),dg_interp(DY_GZZ,p),
              dg_interp(DZ_GXX,p),dg_interp(DZ_GXY,p),dg_interp(DZ_GXZ,p),
              dg_interp(DZ_GYY,p),dg_interp(DZ_GYZ,p),dg_interp(DZ_GZZ,p));
    delete S;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::FastFlowLoop()
//! \brief Fast Flow loop for horizon n
void AHF::FastFlowLoop()
{
  ah_found = false;

  Real meanradius = a0(0) / Kokkos::sqrt(4.0*M_PI);
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
    fprintf(pofile_verbose, " iter      area            mass         meanradius       minradius        hmean            Sx              Sy              Sz             S\n");
  }

  for(int k = 0; k < flow_iterations; k++){
    fastflow_iter = k;

    // Compute radius r = a_lm Y_lm
    RadiiFromSphericalHarmonics();

    // In MetricInterp() we'll flag the surface points on this mesh
    // default to 0 (no points)
    Kokkos::deep_copy(havepoint, 0.0);

    // Set metric interpolated on the surface to 0.0
    Kokkos::deep_copy(g_interp, 0.0);
    Kokkos::deep_copy(K_interp, 0.0);
    Kokkos::deep_copy(dg_interp, 0.0);

    // Interpolate metric on surface
    MetricInterp();
    SurfaceIntegrals();

    area  = integrals[iarea];
    hrms  = integrals[ihrms]/area;
    hmean = integrals[ihmean];
    Sx = integrals[iSx] / (8 * M_PI);
    Sy = integrals[iSy] / (8 * M_PI);
    Sz = integrals[iSz] / (8 * M_PI);
    S  = Kokkos::sqrt(SQR(Sx) + SQR(Sy) + SQR(Sz));

    meanradius = a0(0) / Kokkos::sqrt(4.0 * M_PI);

    // Check we get a finite result
    if (!(std::isfinite(area))) {
      if (verbose && ioproc) {
        fprintf(pofile_verbose, "Failed, Area not finite\n");
        fflush(pofile_verbose);
      }
      failed = true;
      break;
    }
 
    if (!(std::isfinite(hmean))) {
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
      fprintf(pofile_verbose, "%3d %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e %15.7e\n",
              k, area, mass, meanradius, rr_min, hmean, Sx, Sy, Sz, S);
      fflush(pofile_verbose);
    }

    if (std::fabs(hmean) > hmean_tol) {
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
    if (std::fabs(mass_prev-mass) < mass_tol) {
      ah_found = true;
      break;
    }

    // Find new spectral components
    UpdateFlowSpectralComponents();
  }

  if (ah_found) {
    last_a0 = a0(0);

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
    ah_prop[hmass] = std::sqrt( SQR(mass) + 0.25*SQR(S/mass) ); // Christodoulu mass
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
    } 
    else if (!failed && !ah_found) {
      fprintf(pofile_verbose, "Failed, reached max iterations %d\n", flow_iterations);
    }
    fflush(pofile_verbose);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::UpdateFlowSpectralComponents()
//! \brief Find new spectral components with fast flow
void AHF::UpdateFlowSpectralComponents()
{
  const Real alpha = flow_alpha_beta_const;
  const Real beta = 0.5 * flow_alpha_beta_const;
  const Real A = alpha / (lmax * lmax1) + beta;
  const Real B = beta / alpha;
  
  Real *ABfac = new Real[lmax1];
  Real *spec0 = new Real[lmax1];
  Real *specc = new Real[lmpoints];
  Real *specs = new Real[lmpoints];
  
  // Initialize coefficients to zero
  for(int l = 0; l <= lmax; l++) {
    spec0[l] = 0;
    ABfac[l] = A / (1.0 + B * l * (l + 1));
    
    for(int m = 1; m <= l; m++) {
      int l1 = lmindex(l,m);
      specc[l1] = 0;
      specs[l1] = 0;
    }
  }

  // Local sums
  for(int p = 0; p < nangles; p++){
    if (!havepoint(p)) continue;

    const Real drho = gl_grid->int_weights.h_view(p) * rho(p);
    
    for(int l = 0; l <= lmax; l++) {
      spec0[l] += drho * Y0(p,l);
    
      for(int m = 1; m <= l; m++) { 
        int l1 = lmindex(l,m);
        specc[l1] += drho * Yc(p,l1);
        specs[l1] += drho * Ys(p,l1);
      }
    }
  }
   
  // MPI reduce
  #if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, spec0, lmax1,    MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, specc, lmpoints, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, specs, lmpoints, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  #endif

  // Update the coefficients
  for(int l = 0; l <= lmax; l++) {
    a0(l) -= ABfac[l] * spec0[l];

    for(int m = 1; m <= l; m++){
      int l1 = lmindex(l,m);
      ac(l1) -= ABfac[l] * specc[l1];
      as(l1) -= ABfac[l] * specs[l1];
    }
  }

  delete[] ABfac;
  delete[] spec0;
  delete[] specc;
  delete[] specs;
} // (OS): Sync DualArray? or change to Host

//----------------------------------------------------------------------------------------
//! \fn void AHF::RadiiFromSphericalHarmonics()
//! \brief Compute the radius of the surface
void AHF::RadiiFromSphericalHarmonics()
{
  Kokkos::deep_copy(rr, 0.0);
  Kokkos::deep_copy(rr_dth, 0.0);
  Kokkos::deep_copy(rr_dph, 0.0);
  
  rr_min = std::numeric_limits<Real>::infinity();
  for(int p = 0; p < nangles; p++){
    for(int l = 0; l <= lmax; l++){
      rr(p) += a0(l) * Y0(p,l);
      rr_dth(p) += a0(l) * dY0dth(p,l);
    
      for(int m = 1; m <= l; m++){
        int l1 = lmindex(l,m);
        rr(p) += ac(l1) * Yc(p,l1) + as(l1) * Ys(p,l1);
        rr_dth(p) += ac(l1) * dYcdth(p,l1) + as(l1) * dYsdth(p,l1);
        rr_dph(p) += ac(l1) * dYcdph(p,l1) + as(l1) * dYsdph(p,l1);
      }
    }
    rr_min = std::min(rr_min, rr(p));
  }
} // (OS): Sync DualArray? or change to Host

//----------------------------------------------------------------------------------------
//! \fn void AHF::SurfaceIntegrals()
//! \brief Compute expansion, surface element and spin integrand on surface n
//!        Needs metric and extr. curv. interpolated on the surface
//!        Performs local sums and MPI reduce
void AHF::SurfaceIntegrals()
{
  const Real min_rp = 1e-10;

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

  // Initialize integrals
  for(int v = 0; v < invar; v++){
    integrals[v] = 0.0;
  }

  Kokkos::deep_copy(rho, 0.0); // Initialize rho
  
  // Loop over surface points
  for(int p = 0; p < nangles; p++){
    Real const theta = gl_grid->polar_pos.h_view(p,0);
    Real const sinth = Kokkos::sin(theta);
    Real const costh = Kokkos::cos(theta);

    if (!havepoint(p)) continue;
    
    Real const phi = gl_grid->polar_pos.h_view(p,1);
    Real const sinph = Kokkos::sin(phi);
    Real const cosph = Kokkos::cos(phi);

    // -----------------------
    // Calculate the expansion
    // -----------------------

    // Determinant of 3-metric
    Real detg = adm::SpatialDet(
      g_interp(GXX,p), g_interp(GXY,p), g_interp(GXZ,p),
      g_interp(GYY,p), g_interp(GYZ,p), g_interp(GZZ,p)
    );

    // Inverse metric
    adm::SpatialInv(
      1.0/detg,
      g_interp(GXX,p), g_interp(GXY,p), g_interp(GXZ,p),
      g_interp(GYY,p), g_interp(GYZ,p), g_interp(GZZ,p),
      &ginv(0,0), &ginv(0,1),  &ginv(0,2),
      &ginv(1,1), &ginv(1,2) , &ginv(2,2)
    );

    // Trace of K
    Real TrK = adm::Trace(1.0/detg,
          g_interp(GXX,p), g_interp(GXY,p), g_interp(GXZ,p),
          g_interp(GYY,p), g_interp(GYZ,p), g_interp(GZZ,p),
          K_interp(KXX,p), K_interp(KXY,p), K_interp(KXZ,p),
          K_interp(KYY,p), K_interp(KYZ,p), K_interp(KZZ,p));

    // Local coordinates of the surface (re-used below)
    Real const xp = rr(p) * sinth * cosph;
    Real const yp = rr(p) * sinth * sinph;
    Real const zp = rr(p) * costh;

    Real const rp   = Kokkos::sqrt(xp * xp + yp * yp + zp * zp);
    Real const rhop = Kokkos::sqrt(xp * xp + yp * yp);

    if (rp < min_rp) {
      // Do not stop the code, just AHF failing
      // break the loop and catch the nans in AHF later.
      break;
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

    dthetadidj(0,0) = zp*(-2.0*xp*xp*xp*xp-xp*xp*yp*yp+yp*yp*yp*yp+zp*zp*yp*yp)*(SQR(_divrp)*SQR(_divrp)*SQR(_divrhop)*_divrhop);
    dthetadidj(0,1) = -xp*yp*zp*(3.0*xp*xp+3.0*yp*yp+zp*zp)*(SQR(_divrp)*SQR(_divrp)*SQR(_divrhop)*_divrhop);
    dthetadidj(0,2) = xp*(xp*xp+yp*yp-zp*zp)*(SQR(_divrp)*(SQR(_divrp)*_divrhop));
    dthetadidj(1,1) = zp*(-2.0*yp*yp*yp*yp-yp*yp*xp*xp+xp*xp*xp*xp+zp*zp*xp*xp)*(SQR(_divrp)*SQR(_divrp)*SQR(_divrhop)*_divrhop);
    dthetadidj(1,2) = yp*(xp*xp+yp*yp-zp*zp)*(SQR(_divrp)*(SQR(_divrp)*_divrhop));
    dthetadidj(2,2) = 2.0*zp*rhop/(rp*rp*rp*rp);

    dphididj(0,0) = 2.0 * yp * xp * (SQR(_divrhop) * SQR(_divrhop));
    dphididj(0,1) = (yp * yp - xp * xp) * (SQR(_divrhop) * SQR(_divrhop));
    dphididj(0,2) = 0.0;
    dphididj(1,1) = - 2.0 * yp * xp * (SQR(_divrhop) * SQR(_divrhop));
    dphididj(1,2) = 0.0;
    dphididj(2,2) = 0.0;

    // Compute first derivatives of F
    for (int a = 0; a < NDIM; ++a) {
      dFdi(a) = drdi(a);

      for(int l = 0; l <= lmax; l++){
        dFdi(a) -= a0(l) * dthetadi(a) * dY0dth(p,l);
        
        for(int m = 1; m <= l; m++){
          const int l1 = lmindex(l,m);
          dFdi(a) -=
            ac(l1) * (dthetadi(a) * dYcdth(p,l1) + dphidi(a) * dYcdph(p,l1)) +
            as(l1) * (dthetadi(a) * dYsdth(p,l1) + dphidi(a) * dYsdph(p,l1));
        }
      }
    }

    // Compute second derivatives of F
    for (int a = 0; a < NDIM; ++a) {
      for (int b = 0; b < NDIM; ++b) {
        dFdidj(a,b) = drdidj(a,b);

        for(int l = 0;l <= lmax; l++){
          dFdidj(a,b) -= a0(l)*(dthetadidj(a,b)*dY0dth(p,l) + dthetadi(a)*dthetadi(b)*dY0dth2(p,l));
          
          for(int m = 1; m <= l; m++){
            int l1 = lmindex(l,m);
            dFdidj(a,b) -= ac(l1)*(dthetadidj(a,b)*dYcdth(p,l1)
              + dthetadi(a)*(dthetadi(b)*dYcdth2(p,l1) + dphidi(b)*dYcdthdph(p,l1))
              + dphididj(a,b)*dYcdph(p,l1)
              + dphidi(a)*(dthetadi(b)*dYcdthdph(p,l1) + dphidi(b)*dYcdph2(p,l1)))
              + as(l1)*(dthetadidj(a,b)*dYsdth(p,l1)
              + dthetadi(a)*(dthetadi(b)*dYsdth2(p,l1) + dphidi(b)*dYsdthdph(p,l1))
              + dphididj(a,b)*dYsdph(p,l1)
              + dphidi(a)*(dthetadi(b)*dYsdthdph(p,l1) + dphidi(b)*dYsdph2(p,l1)));
          }
        }
      }
    }

    for (int a = 0; a < NDIM; ++a) {
      dFdi_u(a) = 0;
      
      for (int b = 0; b < NDIM; ++b) {
        dFdi_u(a) += ginv(a,b) * dFdi(b); // Compute dFdi with the index up
        nnF(a,b) = dFdidj(a,b); // Compute nabla_a nabla_b F
      }
    }

    Real norm = 0;
    for (int a = 0; a < NDIM; ++a){
       norm += dFdi_u(a) * dFdi(a); // Compute norm of dFdi
    }

    Real u = (norm > 0) ? std::sqrt(norm) : 0.0;

    nnF(0,0) -= 0.5*(dFdi_u(0)*dg_interp(DX_GXX,p) + dFdi_u(1)*(2.0*dg_interp(DX_GXY,p)-dg_interp(DY_GXX,p)) 
                + dFdi_u(2)*(2.0*dg_interp(DX_GXZ,p)-dg_interp(DZ_GXX,p)));
    nnF(0,1) -= 0.5*(dFdi_u(0)*dg_interp(DY_GXX,p) + dFdi_u(1)*dg_interp(DX_GYY,p) 
                + dFdi_u(2)*(dg_interp(DX_GYZ,p)+dg_interp(DY_GXZ,p)-dg_interp(DZ_GXY,p)));
    nnF(0,2) -= 0.5*(dFdi_u(0)*dg_interp(DZ_GXX,p) + dFdi_u(1)*(dg_interp(DX_GYZ,p)
                +dg_interp(DZ_GXY,p)-dg_interp(DY_GXZ,p)) + dFdi_u(2)*dg_interp(DX_GZZ,p));
    nnF(1,1) -= 0.5*(dFdi_u(0)*(2.0*dg_interp(DY_GXY,p)-dg_interp(DX_GYY,p)) + dFdi_u(1)*dg_interp(DY_GYY,p) 
                + dFdi_u(2)*(2.0*dg_interp(DY_GYZ,p)-dg_interp(DZ_GYY,p)));
    nnF(1,2) -= 0.5*(dFdi_u(0)*(2.0*dg_interp(DY_GXZ,p)+dg_interp(DZ_GXY,p)-dg_interp(DX_GYZ,p)) 
                + dFdi_u(1)*dg_interp(DZ_GYY,p) + dFdi_u(2)*dg_interp(DY_GZZ,p));
    nnF(2,2) -= 0.5*(dFdi_u(0)*(2.0*dg_interp(DZ_GXZ,p)-dg_interp(DX_GZZ,p)) + dFdi_u(1)*(2.0*dg_interp(DZ_GYZ,p)
                -dg_interp(DY_GZZ,p)) + dFdi_u(2)*dg_interp(DZ_GZZ,p));

    Real d2F = 0.; // Compute d2F = g^{ab} nabla_a nabla_b F
    Real dFdadFdbKab = 0.; // Compute dFd^a dFd^b Kab
    Real dFdadFdbFdadb = 0.; // Compute dFd^a dFd^b Fdadb
    for (int a = 0; a < NDIM; ++a) {
      for (int b = 0; b < NDIM; ++b) {
        d2F += ginv(a,b)*nnF(a,b);
        dFdadFdbFdadb += dFdi_u(a) * dFdi_u(b) * nnF(a,b);
      }
    }

    dFdadFdbKab += dFdi_u(0) * dFdi_u(0) * K_interp(KXX,p);
    dFdadFdbKab += dFdi_u(0) * dFdi_u(1) * K_interp(KXY,p);
    dFdadFdbKab += dFdi_u(0) * dFdi_u(2) * K_interp(KXZ,p);
    dFdadFdbKab += dFdi_u(1) * dFdi_u(0) * K_interp(KXY,p);
    dFdadFdbKab += dFdi_u(1) * dFdi_u(1) * K_interp(KYY,p);
    dFdadFdbKab += dFdi_u(1) * dFdi_u(2) * K_interp(KYZ,p);
    dFdadFdbKab += dFdi_u(2) * dFdi_u(0) * K_interp(KXZ,p);
    dFdadFdbKab += dFdi_u(2) * dFdi_u(1) * K_interp(KYZ,p);
    dFdadFdbKab += dFdi_u(2) * dFdi_u(2) * K_interp(KZZ,p);
    
    // Expansion & rho = H * u * sigma (sigma=1)
    Real divu = (norm > 0) ? 1.0 / u : 0.0;
    Real H = d2F * divu + dFdadFdbKab * (divu * divu) - dFdadFdbFdadb * (divu * divu * divu) - TrK;

    rho(p) = H * u;

    // Normal vector
    for (int a = 0; a < NDIM; ++a) {
      R(a) = dFdi_u(a) * divu;
    }
    
    // ---------------
    // Surface Element
    // ---------------

    // Derivatives of (x,y,z) vs (thetas, phi)

    // dr/dtheta, dr/dphi
    Real const drdt = rr_dth(p);
    Real const drdp = rr_dph(p);
    
    // Derivatives of (x,y,z) with respect to theta
    dXdth(0) = (drdt * sinth + rr(p) * costh) * cosph;
    dXdth(1) = (drdt * sinth + rr(p) * costh) * sinph;
    dXdth(2) = drdt * costh - rr(p) * sinth;

    // Derivatives of (x,y,z) with respect to phi
    dXdph(0) = (drdp * cosph - rr(p) * sinph) * sinth;
    dXdph(1) = (drdp * sinph + rr(p) * cosph) * sinth;
    dXdph(2) = drdp * costh;

    // Induced metric on the horizon
    Real h11 = 0.;
    Real h12 = 0.; 
    Real h22 = 0.;
    h11 += dXdth(0) * dXdth(0) * g_interp(GXX,p);
    h11 += dXdth(0) * dXdth(1) * g_interp(GXY,p);
    h11 += dXdth(0) * dXdth(2) * g_interp(GXZ,p);
    h11 += dXdth(1) * dXdth(0) * g_interp(GXY,p);
    h11 += dXdth(1) * dXdth(1) * g_interp(GYY,p);
    h11 += dXdth(1) * dXdth(2) * g_interp(GYZ,p);
    h11 += dXdth(2) * dXdth(0) * g_interp(GXZ,p);
    h11 += dXdth(2) * dXdth(1) * g_interp(GYZ,p);
    h11 += dXdth(2) * dXdth(2) * g_interp(GZZ,p);
    
    h12 += dXdth(0) * dXdph(0) * g_interp(GXX,p);
    h12 += dXdth(0) * dXdph(1) * g_interp(GXY,p);
    h12 += dXdth(0) * dXdph(2) * g_interp(GXZ,p);
    h12 += dXdth(1) * dXdph(0) * g_interp(GXY,p);
    h12 += dXdth(1) * dXdph(1) * g_interp(GYY,p);
    h12 += dXdth(1) * dXdph(2) * g_interp(GYZ,p);
    h12 += dXdth(2) * dXdph(0) * g_interp(GXZ,p);
    h12 += dXdth(2) * dXdph(1) * g_interp(GYZ,p);
    h12 += dXdth(2) * dXdph(2) * g_interp(GZZ,p);
    
    h22 += dXdph(0) * dXdph(0) * g_interp(GXX,p);
    h22 += dXdph(0) * dXdph(1) * g_interp(GXY,p);
    h22 += dXdph(0) * dXdph(2) * g_interp(GXZ,p);
    h22 += dXdph(1) * dXdph(0) * g_interp(GXY,p);
    h22 += dXdph(1) * dXdph(1) * g_interp(GYY,p);
    h22 += dXdph(1) * dXdph(2) * g_interp(GYZ,p);
    h22 += dXdph(2) * dXdph(0) * g_interp(GXZ,p);
    h22 += dXdph(2) * dXdph(1) * g_interp(GYZ,p);
    h22 += dXdph(2) * dXdph(2) * g_interp(GZZ,p);

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
    intSx += phix(0) * R(0) * K_interp(KXX,p);
    intSx += phix(0) * R(1) * K_interp(KXY,p);
    intSx += phix(0) * R(2) * K_interp(KXZ,p);
    intSx += phix(1) * R(0) * K_interp(KXY,p);
    intSx += phix(1) * R(1) * K_interp(KYY,p);
    intSx += phix(1) * R(2) * K_interp(KYZ,p);
    intSx += phix(2) * R(0) * K_interp(KXZ,p);
    intSx += phix(2) * R(1) * K_interp(KYZ,p);
    intSx += phix(2) * R(2) * K_interp(KZZ,p);
    
    intSy += phiy(0) * R(0) * K_interp(KXX,p);
    intSy += phiy(0) * R(1) * K_interp(KXY,p);
    intSy += phiy(0) * R(2) * K_interp(KXZ,p);
    intSy += phiy(1) * R(0) * K_interp(KXY,p);
    intSy += phiy(1) * R(1) * K_interp(KYY,p);
    intSy += phiy(1) * R(2) * K_interp(KYZ,p);
    intSy += phiy(2) * R(0) * K_interp(KXZ,p);
    intSy += phiy(2) * R(1) * K_interp(KYZ,p);
    intSy += phiy(2) * R(2) * K_interp(KZZ,p);
    
    intSz += phiz(0) * R(0) * K_interp(KXX,p);
    intSz += phiz(0) * R(1) * K_interp(KXY,p);
    intSz += phiz(0) * R(2) * K_interp(KXZ,p);
    intSz += phiz(1) * R(0) * K_interp(KXY,p);
    intSz += phiz(1) * R(1) * K_interp(KYY,p);
    intSz += phiz(1) * R(2) * K_interp(KYZ,p);
    intSz += phiz(2) * R(0) * K_interp(KXZ,p);
    intSz += phiz(2) * R(1) * K_interp(KYZ,p);
    intSz += phiz(2) * R(2) * K_interp(KZZ,p);

    // Local sums
    // ----------
    const Real wght = gl_grid->int_weights.h_view(p);
    const Real da = wght * std::sqrt(deth) / sinth;

    integrals[iarea]   += da;
    integrals[icoarea] += wght * SQR(rr(p));
    integrals[ihrms]   += da * SQR(H);
    integrals[ihmean]  += da * H;
    integrals[iSx]     += da * intSx;
    integrals[iSy]     += da * intSy;
    integrals[iSz]     += da * intSz;
  }
  
  #if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, integrals, invar, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  #endif
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::ComputeSphericalHarmonics()
//! \brief Compute spherical harmonics for grid of size ntheta*nphi.
//!        Results are used for all horizons.
void AHF::ComputeSphericalHarmonics()
{
  const Real sqrt2 = Kokkos::sqrt(2.0);

  // Initialize all arrays
  Kokkos::deep_copy(Y0, 0.0);
  Kokkos::deep_copy(Yc, 0.0);
  Kokkos::deep_copy(Ys, 0.0);

  Kokkos::deep_copy(dY0dth, 0.0);
  Kokkos::deep_copy(dYcdth, 0.0);
  Kokkos::deep_copy(dYsdth, 0.0);
  Kokkos::deep_copy(dYcdph, 0.0);
  Kokkos::deep_copy(dYsdph, 0.0);

  Kokkos::deep_copy(dY0dth2, 0.0);
  Kokkos::deep_copy(dYcdth2, 0.0);
  Kokkos::deep_copy(dYcdthdph, 0.0);
  Kokkos::deep_copy(dYsdth2, 0.0);
  Kokkos::deep_copy(dYsdthdph, 0.0);
  Kokkos::deep_copy(dYcdph2, 0.0);
  Kokkos::deep_copy(dYsdph2, 0.0);

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
          Y0(p,l) = YlmR;
          dY0dth(p,l) = YlmRdth;
          dY0dth2(p,l) = YlmRdth2;
        }
        else { // m > 0 spherical harmonics
          const int l1 = lmindex(l,m);
          Yc(p,l1) = sqrt2 * YlmR;
          Ys(p,l1) = sqrt2 * YlmI;

          dYcdth(p,l1) = sqrt2 * YlmRdth;
          dYsdth(p,l1) = sqrt2 * YlmIdth;
          dYcdph(p,l1) = sqrt2 * YlmRdphi;
          dYsdph(p,l1) = sqrt2 * YlmIdphi;

          dYcdth2(p,l1)   = sqrt2 * YlmRdth2;
          dYcdthdph(p,l1) = sqrt2 * YlmRdthdphi;
          dYsdth2(p,l1)   = sqrt2 * YlmIdth2;
          dYsdthdph(p,l1) = sqrt2 * YlmIdthdphi;
          dYcdph2(p,l1)   = sqrt2 * YlmRdphi2;
          dYsdph2(p,l1)   = sqrt2 * YlmIdphi2;
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
}

//----------------------------------------------------------------------------------------
//! \fn Real AHF::PuncMaxDistance()
//! \brief Max Euclidean distance between punctures
Real AHF::PuncMaxDistance() {
  Real maxdist = 0.0;
  for (int pix = 0; pix < npunct; ++pix) {
    Real xp = pmbp->pz4c->ptracker[pix]->GetPos(0);
    Real yp = pmbp->pz4c->ptracker[pix]->GetPos(1);
    Real zp = pmbp->pz4c->ptracker[pix]->GetPos(2);

    for (int p = pix+1; p < npunct; ++p) {
      Real x = pmbp->pz4c->ptracker[p]->GetPos(0);
      Real y = pmbp->pz4c->ptracker[p]->GetPos(1);
      Real z = pmbp->pz4c->ptracker[p]->GetPos(2);
      maxdist = std::max(maxdist, std::sqrt(SQR(x - xp) + SQR(y - yp) + SQR(z - zp)));
    }
  }
  return maxdist;
}

//----------------------------------------------------------------------------------------
//! \fn Real AHF::PuncMaxDistance(const int pix)
//! \brief Max Euclidean distance from puncture pix to other punctures
Real AHF::PuncMaxDistance(const int pix) {
  Real xp = pmbp->pz4c->ptracker[pix]->GetPos(0);
  Real yp = pmbp->pz4c->ptracker[pix]->GetPos(1);
  Real zp = pmbp->pz4c->ptracker[pix]->GetPos(2);
  Real maxdist = 0.0;

  for (int p = 0; p < npunct; ++p) {
    if (p==pix) continue;
    Real x = pmbp->pz4c->ptracker[p]->GetPos(0);
    Real y = pmbp->pz4c->ptracker[p]->GetPos(1);
    Real z = pmbp->pz4c->ptracker[p]->GetPos(2);
    maxdist = std::max(maxdist, std::sqrt(SQR(x - xp) + SQR(y - yp) + SQR(z - zp)));
  }

  return maxdist;
}

//----------------------------------------------------------------------------------------
//! \fn Real AHF::PuncSumMasses()
//! \brief Return sum of puncture's intial masses
Real AHF::PuncSumMasses() {
  Real mass = 0.0;
  for (int p = 0; p < npunct; ++p) {
    mass += pmbp->pz4c->ptracker[p]->GetMass();
  }
  return mass;
}

//----------------------------------------------------------------------------------------
//! \fn void AHF::PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc)
//! \brief Return mss-weighted center of puncture positions
void AHF::PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc) {
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
//! \fn int AHF::PuncAreClose()
//! \brief Check when the maximal distance between all punctures is below threshold
bool AHF::PuncAreClose() {
  Real const mass = PuncSumMasses();
  Real const maxdist = PuncMaxDistance();
  return (maxdist < merger_distance * mass);
}