//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_puncture_inflow.cpp
//  \brief Problem generator for a radial relativistic hydrodynamical inflow accreting
//  onto a single, non-spinning black hole puncture in isotropic coordinates.
//
//  This problem generator is designed to test the general relativistic flux calculation
//  in a Z4c evolved spacetime. It initializes the metric based on the puncture formalism
//  and sets up a spherically symmetric radial inflow of fluid. It registers a
//  user-defined history function to integrate and record fluxes of mass, momentum,
//  and energy through a spherical surface.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>

// AthenaXXX headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "outputs/outputs.hpp"
#include "utils/surface_grid.hpp"
#include "utils/flux_generalized.hpp" // Assumes this header is in the include path

// Local to this file
namespace {

// Forward declarations for functions in this file
void InflowHistory(HistoryData *pdata, Mesh *pm);
void PrintAnalytic(ParameterInput *pin, Mesh *pm);
} // namespace

// Forward declarations for problem-specific functions
void InitializeADMAndInflow(MeshBlockPack *pmbp, ParameterInput *pin);
void RefinementCondition(MeshBlockPack* pmbp);


//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Top-level function to set up the problem
//----------------------------------------------------------------------------------------
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // --- Z4c, AMR, and MHD setup ---
  user_ref_func  = RefinementCondition;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "This test must be run with Z4c, but no <z4c> block "
              << "was found in the input file." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "This test requires MHD, but no <mhd> block "
              << "was found in the input file." << std::endl;
    exit(EXIT_FAILURE);
  }

  // --- Setup for Flux Integration ---
  user_hist_func = &InflowHistory; // Register our custom history function

  const Real analysis_radius = pin->GetReal("problem", "analysis_radius");
  const int ntheta = pin->GetOrAddInteger("problem", "flux_ntheta", 64);
  const int nphi = pin->GetOrAddInteger("problem", "flux_nphi", 128);

  // Define a function that returns a constant radius for the surface grid
  auto r_func_fixed = [analysis_radius](Real, Real){ return analysis_radius; };

  // Create the surface grid object and add it to our static vector
  this->surface_grids.push_back(std::make_unique<SphericalSurfaceGrid>(
      pmbp, ntheta, nphi, r_func_fixed, "flux_sphere"));

  // --- Call the combined initialization function for metric and fluid ---
  InitializeADMAndInflow(pmbp, pin);

  // --- Z4c evolution variable setup ---
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin); break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin); break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin); break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp); break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp); break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp); break;
  }
  std::cout<<"One Puncture with GRH inflow initialized."<<std::endl;

  // Print the analytical flux values for comparison
  PrintAnalytic(pin, pmy_mesh_);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void InitializeADMAndInflow(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM variables for a puncture and the hydro radial inflow.
//----------------------------------------------------------------------------------------
void InitializeADMAndInflow(MeshBlockPack *pmbp, ParameterInput *pin) {
  // --- Capture variables and parameters for the Kokkos kernel ---
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;

  // Puncture parameters
  Real ADM_mass = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.);

  // Inflow parameters
  const Real rho_inf = pin->GetReal("problem", "rho_inf");
  const Real p_inf = pin->GetReal("problem", "p_inf");
  const Real inflow_speed = pin->GetReal("problem", "inflow_speed");
  const Real gamma_ad = pmbp->pmhd->peos->eos_data.gamma;

  // Capture Kokkos views for device access
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  auto &w0 = pmbp->pmhd->w0;
  auto &bcc0 = pmbp->pmhd->bcc0;

  par_for("pgen_puncture_inflow_init",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // --- Calculate cell-centered Cartesian coordinates ---
    Real &x1min = size.d_view(m).x1min; Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min; Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min; Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    x1v -= center_x1;
    x2v -= center_x2;
    x3v -= center_x3;

    Real r = sqrt(x1v*x1v + x2v*x2v + x3v*x3v);
    if (r < 1.0e-12) r = 1.0e-12; // Avoid singularity at the origin

    // --- Set ADM Metric Components (Isotropic Coordinates for a non-spinning BH) ---
    // Extrinsic curvature K_ij is initialized to zero by default.
    const Real psi = 1.0 + 0.5 * ADM_mass / r;
    const Real psi4 = psi*psi*psi*psi;
    adm.psi4(m,k,j,i) = psi4;

    adm.g_dd(m,0,0,k,j,i) = psi4; // g_xx
    adm.g_dd(m,0,1,k,j,i) = 0.0;  // g_xy
    adm.g_dd(m,0,2,k,j,i) = 0.0;  // g_xz
    adm.g_dd(m,1,1,k,j,i) = psi4; // g_yy
    adm.g_dd(m,1,2,k,j,i) = 0.0;  // g_yz
    adm.g_dd(m,2,2,k,j,i) = psi4; // g_zz

    // --- Set Fluid and Field Initial Conditions (Radial Inflow) ---
    w0(m, IDN, k, j, i) = rho_inf;
    w0(m, IPR, k, j, i) = p_inf;
    w0(m, IEN, k, j, i) = p_inf / (gamma_ad - 1.0);

    // Set radial velocity pointing towards the origin
    w0(m, IVX, k, j, i) = -inflow_speed * x1v / r;
    w0(m, IVY, k, j, i) = -inflow_speed * x2v / r;
    w0(m, IVZ, k, j, i) = -inflow_speed * x3v / r;

    // Set magnetic field to zero for a simple hydro setup
    bcc0(m, IBX, k, j, i) = 0.0;
    bcc0(m, IBY, k, j, i) = 0.0;
    bcc0(m, IBZ, k, j, i) = 0.0;
  });
  pmbp->pmhd->peos->PrimToCons(w0, bcc0, pmbp->pmhd->u0, is, ie, js, je, ks, ke);
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCondition(MeshBlockPack* pmbp)
//! \brief Sets the refinement condition for AMR.
//----------------------------------------------------------------------------------------
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

// Anonymous namespace for file-local functions
namespace {
//----------------------------------------------------------------------------------------
//! \fn void InflowHistory(HistoryData *pdata, Mesh *pm)
//! \brief The user-defined history function that calls the general flux integrator.
//----------------------------------------------------------------------------------------
void InflowHistory(HistoryData *pdata, Mesh *pm) {
    ProblemGenerator *pgen = pm->pgen.get();
    if (pgen->surface_grids.empty()) {
        pdata->nhist = 0;
        return;
    }
    MeshBlockPack *pmbp = pm->pmb_pack;

    // Convert the vector of unique_ptrs to a vector of raw pointers for the function.
    std::vector<SphericalSurfaceGrid*> surf_raw_ptrs;
    surf_raw_ptrs.reserve(pgen->surface_grids.size());
    for(const auto& s : pgen->surface_grids) {
        surf_raw_ptrs.push_back(s.get());
    }

    // Call the generalized flux calculator from "utils/flux_generalized.cpp"
    TorusFluxes_General(pdata, pmbp, surf_raw_ptrs);
}

//----------------------------------------------------------------------------------------
//! \fn PrintAnalytic(ParameterInput *pin, Mesh *pm)
//! \brief Calculates and prints the exact analytical fluxes for the initial data.
//----------------------------------------------------------------------------------------
void PrintAnalytic(ParameterInput *pin, Mesh *pm) {
    // This function should only execute on the root process.
    if (global_variable::my_rank == 0) {
        // Read parameters needed for analytical calculation
        const Real rho_inf = pin->GetReal("problem", "rho_inf");
        const Real p_inf = pin->GetReal("problem", "p_inf");
        const Real inflow_speed = pin->GetReal("problem", "inflow_speed");
        const Real adm_mass = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.0);
        const Real gamma_ad = pm->pmb_pack->pmhd->peos->eos_data.gamma;
        const Real analysis_radius = pin->GetReal("problem", "analysis_radius");
        const Real Ra = analysis_radius; // shorthand

        std::cout << "---------------------------------------------------------" << std::endl;
        std::cout << "--- Analytical Fluxes (Exact for Initial Conditions) ---" << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;

        // --- Calculate derived quantities at the analysis radius ---
        const Real psi = 1.0 + 0.5 * adm_mass / Ra;
        const Real v_sq_proper = psi*psi*psi*psi * inflow_speed*inflow_speed;

        if (v_sq_proper >= 1.0) {
            std::cout << "ERROR: Fluid velocity is superluminal at analysis radius." << std::endl;
            return;
        }
        const Real W = 1.0 / sqrt(1.0 - v_sq_proper);
        const Real h = 1.0 + (p_inf / rho_inf) * (gamma_ad / (gamma_ad - 1.0));
        const Real H = rho_inf * h;

        // --- Calculate exact analytical fluxes ---
        // These formulas are derived to match exactly what the numerical integrator in
        // flux_generalized.cpp calculates for this specific coordinate system and setup.
        const Real mdot_analytic = 4.0 * M_PI * Ra*Ra * pow(psi, 10) * rho_inf * W * inflow_speed;
        const Real edot_analytic = 4.0 * M_PI * Ra*Ra * pow(psi, 4) * H * W*W * inflow_speed;
        const Real pdot_analytic = 0.0; // Zero by spherical symmetry

        // --- Print Comparison ---
        std::cout << std::scientific << std::setprecision(8);
        std::cout << "Analysis Sphere Radius: " << Ra << std::endl << std::endl;

        std::cout << "Expected Mass Flux (mdot):         " << mdot_analytic << std::endl;
        std::cout << "Expected Energy Flux (edot_fluid): " << edot_analytic << std::endl;
        std::cout << "Expected Momentum Flux (pdot):     " << pdot_analytic << std::endl;

        std::cout << "---------------------------------------------------------" << std::endl;
        std::cout << "NOTE: These are the exact fluxes for the initial data slice." << std::endl;
        std::cout << "The numerical fluxes should match these values at t=0." << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;
    }
}
} // namespace
