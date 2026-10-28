//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for spherical blast wave problem.
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

#include <cmath>

#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>
#include <limits>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "units/units.hpp"
#include "pgen/fluids/reactive_bubble.hpp"
#include "nuclear/microphysics.hpp"
#include "nuclear/nuclear.hpp"

using nuclear::Thermo;

namespace {
  Real g_grav;
  int max_iter_nuc;
  int n_refactor_nuc;
  Real e_atol_nuc;
  Real e_rtol_nuc;
  Real y_atol_nuc;
  Real y_rtol_nuc;
  Real dt_nuc_factor;  // dt_nuc = dt_nuc_factor * min(e/|edot|)
  void SetReactiveSources(Mesh* pm, const Real bdt);
  void HydrostaticBCs(Mesh* pm);

  // Isentropic hydrostatic state of the vertical ghost cells, tabulated in
  // UserProblem and used by HydrostaticBCs. Index j is the ghost-cell index used in
  // the BC kernel (inner: cell j, outer: cell je+1+j).
  constexpr int NGHOST_MAX = 8;
  struct GhostAtm {
    Real rho_lo[NGHOST_MAX], pres_lo[NGHOST_MAX];
    Real rho_hi[NGHOST_MAX], pres_hi[NGHOST_MAX];
  };
  GhostAtm ghost_atm;
}


struct Atm{ 
  Real rho; 
  Real pres;
  Real temp;
  Real yloc;
};


// Advances the hydrostatic pressure profile by one step, using burn() with
// eos_input_ps (fixed entropy s_base and composition xn[]) to get rho(p,s)
// at each RK4 stage.
Real Rk4PressureIntegrationStep(const Real &pres, const Real &s_base, const Real &g_grav,
    const Real &dy, const Real xn[NSPECIES],
    Primitive::EOS<Primitive::IdealGas, Primitive::ResetFloor> &eos) {
  Thermo stage;
  for (int n = 0; n < NSPECIES; n++) stage.xn[n] = xn[n];

  stage.p = pres; stage.s = s_base;
  nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_ps, stage, eos);
  Real k1 = stage.rho*g_grav;

  stage.p = pres + k1*dy/2.0; stage.s = s_base;
  nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_ps, stage, eos);
  Real k2 = stage.rho*g_grav;

  stage.p = pres + k2*dy/2.0; stage.s = s_base;
  nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_ps, stage, eos);
  Real k3 = stage.rho*g_grav;

  stage.p = pres + k3*dy; stage.s = s_base;
  nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_ps, stage, eos);
  Real k4 = stage.rho*g_grav;

  Real pres_int = pres + dy *(k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
  return pres_int;
}


//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for reactive bubble problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  max_iter_nuc     = pin->GetInteger("problem", "max_iter_nuc");
  n_refactor_nuc   = pin->GetOrAddInteger("problem", "nuc_refactor", 3);
  // Gravity is applied by SourceTerms::ConstantAccel (<hydro_srcterms>); use the same
  // value for the hydrostatic profile. const_accel_val < 0 points downward.
  g_grav           = pin->GetReal("hydro_srcterms", "const_accel_val");
  int accel_dir    = pin->GetInteger("hydro_srcterms", "const_accel_dir");
  if ((pmy_mesh_->two_d && accel_dir != 2) || (pmy_mesh_->three_d && accel_dir != 3)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "const_accel_dir must be 2 in 2D and 3 in 3D" << std::endl;
    exit(EXIT_FAILURE);
  }
  e_atol_nuc       = pin-> GetReal("problem", "e_atol");
  e_rtol_nuc       = pin-> GetReal("problem", "e_rtol"); 
  y_atol_nuc       = pin-> GetReal("problem", "y_atol");
  y_rtol_nuc       = pin-> GetReal("problem", "y_rtol");
  // Substep size of the network: dt_nuc = dt_nuc_factor * min(e/|edot|), evaluated in
  // SetReactiveSources at the burned state and used for all the substeps of the next
  // hydro step (each substep changes e by <~ dt_nuc_factor)
  dt_nuc_factor    = pin->GetOrAddReal("problem", "dt_nuc_factor", 0.1);
  // Nuclear burning is switched on/off with <problem>/user_nuc_srcs, which is read
  // by the ProblemGenerator constructor before UserProblem is called.
  user_srcs_nuc_func = &SetReactiveSources;
  // Applied on faces with <mesh> ix2_bc/ox2_bc (2D) or ix3_bc/ox3_bc (3D) = user
  user_bcs_func = &HydrostaticBCs;

  // Initial guess of the network substep size, used until the first burn sets it from
  // e/|edot| in SetReactiveSources (also needed on restarts)
  if (pmy_mesh_->pmb_pack->pnuc != nullptr) {
    auto &mesh_indcs = pmy_mesh_->mesh_indcs;
    int npts_nuc = pmy_mesh_->three_d
                 ? std::min({mesh_indcs.nx1, mesh_indcs.nx2, mesh_indcs.nx3})
                 : std::min(mesh_indcs.nx1, mesh_indcs.nx2);
    pmy_mesh_->pmb_pack->pnuc->dt_nuc = (1.5e-4)*(64.0/static_cast<Real>(npts_nuc));
  }

  Mesh *pm  = pmy_mesh_;
  MeshBlockPack *pmbp = pm->pmb_pack;
  bool two_d   = pm->two_d;
  bool three_d = pm->three_d;

  // number of vertical cells at the (uniform, non-AMR) root resolution
  // similar to cshock.cpp
  Real dy;
  Real ymin;
  const int NFACT = 10;
  int npts;

  if (two_d) {
    npts = pm->mesh_indcs.nx2; 
  } else if (three_d) {
    npts = pm->mesh_indcs.nx3;
  }

  if(two_d) {
    ymin = pm->mesh_size.x2min;
    dy  = (pm->mesh_size.x2max - pm->mesh_size.x2min) / static_cast<Real>(NFACT*npts);
  } else if (three_d) {
    ymin = pm->mesh_size.x3min;
    dy  = (pm->mesh_size.x3max - pm->mesh_size.x3min) / static_cast<Real>(NFACT*npts);
  }
  
  //Inputs
  Real rho_base    = pin->GetReal("problem", "rho_base");
  Real temp_base   = pin->GetReal("problem", "temp_base");
  // Set perturb = false to keep the unperturbed isentropic atmosphere
  bool perturb     = pin->GetOrAddBoolean("problem", "perturb", true);
  // Length scale of the perturbation [cm]: T = T0*(1 + 0.6*(1 + tanh(4 - r/sigma)))
  // (Zingale et al. 2019, Eq. 89, Table 9)
  Real sigma       = pin->GetOrAddReal("problem", "sigma", 2.56e5);
  // Vertical position of the bubble centre [cm] (x2 in 2D, x3 in 3D); default: centre
  // of the domain in that direction
  Real yc_default  = two_d ? 0.5*(pm->mesh_size.x2min + pm->mesh_size.x2max)
                           : 0.5*(pm->mesh_size.x3min + pm->mesh_size.x3max);
  Real y_bubble    = pin->GetOrAddReal("problem", "y0", yc_default);

  // --------------------Initialize Hydro variables -----------------
  if (pmbp->phydro != nullptr) {
    // Define the total number of species
    int nfluid = pmbp->phydro->nhydro;
    int nscalars = pmbp->phydro->nscalars;

    // Define the initial mass fractions in the entire domain:
    DualArray1D<Real> X_mass("species", nscalars);
    X_mass.h_view(ihe4) = 0.99999998;
    X_mass.h_view(ic12) = 1.0e-8;  //Define enumerate at the top
    X_mass.h_view(io16) = 1.0e-8;

    // Sync X_mass
    X_mass.template modify<HostMemSpace>();
    X_mass.template sync<DevExeSpace>();

    Real X_initial[3];
    X_initial[ihe4] = X_mass.h_view(ihe4);
    X_initial[ic12] = X_mass.h_view(ic12);
    X_initial[io16] = X_mass.h_view(io16);

    // EOS object: constructed once, on the host; captured by value into the
    // per-cell init kernels below (it is not safe to default-construct an
    // EOS<> object on-device).
    Primitive::EOS<Primitive::IdealGas, Primitive::ResetFloor> eos;
    eos.SetNSpecies(NSPECIES);
    //eos.SetCodeUnitSystem(Make_CGS) !Very good idea

    // Compute mu, p_base, and s_base from the ideal gas EOS (Sackur-Tetrode
    // entropy) at the given (rho_base, temp_base).
    Thermo base;
    base.rho = rho_base;
    base.t = temp_base;
    for (int n = 0; n < NSPECIES; n++) base.xn[n] = X_initial[n];
    nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_rt, base, eos);
    Real s_base = base.s;
    Real p_base = base.p;

    //We define the vertical P and rho vectors along the y direction.
    HostArray1D<Atm> soln("soln", (NFACT)*npts);
    HostArray1D<Real> yloc("yloc", (NFACT)*npts);

    //We use RK4 to integrate an isentropic profile.
    Thermo atm;
    for (int n = 0; n < NSPECIES; n++) atm.xn[n] = X_initial[n];

    soln(0).pres = Rk4PressureIntegrationStep(p_base, s_base, g_grav, dy/2.0, X_initial, eos);
    atm.p = soln(0).pres; atm.s = s_base;
    nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_ps, atm, eos);
    soln(0).rho = atm.rho;
    soln(0).temp = atm.t;
    soln(0).yloc = ymin + 0.5*dy;
    for (int i = 0; i < ((NFACT)*npts - 1); i++) {
      soln(i+1).pres = Rk4PressureIntegrationStep(soln(i).pres, s_base, g_grav, dy, X_initial, eos);
      atm.p = soln(i+1).pres; atm.s = s_base;
      nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_ps, atm, eos);
      soln(i+1).rho = atm.rho;
      soln(i+1).temp = atm.t;
      soln(i+1).yloc = soln(i).yloc + dy;
    }

    // Binning: average P over the NFACT values of each grid cell, then get rho and
    // T from (P_avg, s_base) so every cell has exactly the base entropy.
    // Valid only at static refinement. (No AMR)
    DualArray1D<Atm> sol("sol", npts);

    for (int n=0; n<npts; n++) {
      Real pres_avg = 0.0;
      for (int m=0; m<(NFACT); m++) {
        pres_avg += soln(m + (NFACT)*n).pres;
      }
      pres_avg /= static_cast<Real>(NFACT);
      atm.p = pres_avg; atm.s = s_base;
      nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_ps, atm, eos);
      sol.h_view(n).pres = pres_avg;
      sol.h_view(n).rho  = atm.rho;
      sol.h_view(n).temp = atm.t;
      sol.h_view(n).yloc = ymin + (static_cast<Real>(n) + 0.5)*(NFACT)*dy;
    }

    // Sync sol to be used inside a Kokkos Kernel.
    sol.template modify<HostMemSpace>();
    sol.template sync<DevExeSpace>();

    // Ghost cells for HydrostaticBCs: continue the isentropic RK4 profile ng cells
    // below ymin and above ymax, and bin it the same way as the active cells.
    int ng = pm->mb_indcs.ng;
    if (ng > NGHOST_MAX) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "reactive_bubble supports at most " << NGHOST_MAX
                << " ghost cells" << std::endl;
      exit(EXIT_FAILURE);
    }
    Real pres_lo = p_base;                       // at ymin
    Real pres_hi = soln((NFACT)*npts - 1).pres;  // at ymax - dy/2
    for (int g=0; g<ng; g++) {  // g = 0 is the ghost layer next to the boundary
      Real pres_avg_lo = 0.0;
      Real pres_avg_hi = 0.0;
      for (int m=0; m<(NFACT); m++) {
        Real step_lo = (g == 0 && m == 0) ? 0.5*dy : dy;
        pres_lo = Rk4PressureIntegrationStep(pres_lo, s_base, g_grav, -step_lo, X_initial, eos);
        pres_hi = Rk4PressureIntegrationStep(pres_hi, s_base, g_grav, dy, X_initial, eos);
        pres_avg_lo += pres_lo;
        pres_avg_hi += pres_hi;
      }
      pres_avg_lo /= static_cast<Real>(NFACT);
      pres_avg_hi /= static_cast<Real>(NFACT);

      atm.p = pres_avg_lo; atm.s = s_base;
      nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_ps, atm, eos);
      ghost_atm.pres_lo[ng-1-g] = pres_avg_lo;
      ghost_atm.rho_lo[ng-1-g]  = atm.rho;

      atm.p = pres_avg_hi; atm.s = s_base;
      nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_ps, atm, eos);
      ghost_atm.pres_hi[g] = pres_avg_hi;
      ghost_atm.rho_hi[g]  = atm.rho;
    }

    // The ghost-cell table is needed by the BCs on restarts too
    if (restart) return;

    auto &w0 = pmbp->phydro->w0;
    auto &u0 = pmbp->phydro->u0;
    Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

    // capture variables for the kernel
    auto &indcs = pmy_mesh_->mb_indcs;
    int nmb = pmbp->nmb_thispack;
    int &is = indcs.is; int &ie = indcs.ie;
    int &js = indcs.js; int &je = indcs.je;
    int &ks = indcs.ks; int &ke = indcs.ke;
    auto &size = pmbp->pmb->mb_size;

    if (two_d) {
      // Define the center (vertical position from <problem>/y0)
      Real x0 = 0.5*(pm->mesh_size.x1min + pm->mesh_size.x1max);
      Real y0 = y_bubble;
  
      // Calculate index-offset of MeshBlocks in sol array
      DualArray1D<int> joff("offset2", pmbp->nmb_thispack);
      for (int m=0; m<(pmbp->nmb_thispack); m++) {
        int igid = pmbp->gids + m;
        LogicalLocation lloc=pm->lloc_eachmb[igid];
        joff.h_view(m) = lloc.lx2*(pm->mb_indcs.nx2);
      }
      // sync with device
      joff.template modify<HostMemSpace>();
      joff.template sync<DevExeSpace>();

      par_for("pgen_reactive_dir2", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m,int k, int j, int i) {
        int jo = joff.d_view(m);
        // Compute the pressure and temperature:
        Real pres = sol.d_view(jo+j-js).pres;
        Real temp = sol.d_view(jo+j-js).temp;
        Real rho = sol.d_view(jo+j-js).rho;
        
        // Apply the perturbation at the temperature
        //   Real &x1min = size.d_view(m).x1min;
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        Real rad = sqrt(SQR(x1v-x0)+SQR(x2v-y0));
        if (perturb) temp *= (1.0 + 0.6*(1.0 + tanh(4.0-rad/sigma)));

        // Compute the density from temp and pres. eos is captured by value
        // (KOKKOS_LAMBDA == capture-by-value), which makes it const inside
        // this lambda -- copy to a mutable local before passing to burn(),
        // which needs to call eos_local.SetBaryonMass() on it.
        Thermo cell;
        for (int n = 0; n < NSPECIES; n++) cell.xn[n] = X_mass.d_view(n);
        cell.p = pres;
        cell.t = temp;
        auto eos_local = eos;
        nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_pt, cell, eos_local);
        rho = cell.rho;

        // Define the primitive vector
        w0(m,IDN,k,j,i) = rho;
        w0(m,IVX,k,j,i) = 0.0;
        w0(m,IVY,k,j,i) = 0.0;
        w0(m,IVZ,k,j,i) = 0.0;
        w0(m,IEN,k,j,i) = pres/gm1;

        // Need to consider the mass initial mass fractions
        int nuc = 0;
        for (int n=nfluid; n<(nfluid+nscalars); n++) {
          w0(m,n,k,j,i) = X_mass.d_view(nuc);
          nuc++;
        }
      });
    } else if(three_d) {
      //Define the center
      Real x0 = 0.5*(pm->mesh_size.x1min + pm->mesh_size.x1max);
      Real y0 = 0.5*(pm->mesh_size.x2min + pm->mesh_size.x2max);
      Real z0 = y_bubble;  // vertical position from <problem>/y0

      DualArray1D<int> joff("offset3", pmbp->nmb_thispack);
      for (int m=0; m<(pmbp->nmb_thispack); m++) {
        int igid = pmbp->gids + m;
        LogicalLocation lloc=pm->lloc_eachmb[igid];
        joff.h_view(m) = lloc.lx3*(pm->mb_indcs.nx3);
      }
      // sync with device
      joff.template modify<HostMemSpace>();
      joff.template sync<DevExeSpace>();

      par_for("pgen_reactive_dir3", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m,int k, int j, int i) {
        int ko = joff.d_view(m);
        // Compute the pressure and temperature:
        Real pres = sol.d_view(ko+k-ks).pres;
        Real temp = sol.d_view(ko+k-ks).temp;

        // Apply the perturbation at the temperature
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
        
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

        Real rad = sqrt(SQR(x1v-x0)+SQR(x2v-y0)+SQR(x3v-z0));
        if (perturb) temp *= (1.0 + 0.6*(1.0 + tanh(4.0-rad/sigma)));
        // Compute the density from temp and pres. eos is captured by value
        // (KOKKOS_LAMBDA == capture-by-value), which makes it const inside
        // this lambda -- copy to a mutable local before passing to burn(),
        // which needs to call eos_local.SetBaryonMass() on it.
        Thermo cell;
        for (int n = 0; n < NSPECIES; n++) cell.xn[n] = X_mass.d_view(n);
        cell.p = pres;
        cell.t = temp;
        auto eos_local = eos;
        nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_pt, cell, eos_local);
        Real rho = cell.rho;

        // Define the primitive vector
        w0(m,IDN,k,j,i) = rho;
        w0(m,IVX,k,j,i) = 0.0;
        w0(m,IVY,k,j,i) = 0.0;
        w0(m,IVZ,k,j,i) = 0.0;
        w0(m,IEN,k,j,i) = sol.d_view(ko+k-ks).pres/gm1;

        int nuc = 0;
        // Need to define the initial mass fractions
        for (int n=nfluid; n<(nfluid+nscalars); n++) {
           w0(m,n,k,j,i) = X_mass.d_view(nuc);
           nuc++;
        }
      });
    }
    pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);
  }
}
//--------------- End initialization Hydro variables--------------------------------------


namespace {

//----------------------------------------------------------------------------------------
//! \fn void HydrostaticBCs(Mesh *pm)
//! \brief User BCs on the vertical faces (x2 in 2D, x3 in 3D). Ghost-cell density and
//! pressure continue the isentropic hydrostatic profile tabulated in UserProblem.
//! Velocities and mass fractions are reflected from the mirror-image active cell,
//! with the normal velocity negated, so there is no mass flux through the boundary.

  void HydrostaticBCs(Mesh *pm) {
    auto *phydro = pm->pmb_pack->phydro;
    if (phydro == nullptr) return;

    auto &indcs = pm->mb_indcs;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
    int &js = indcs.js;  int &je = indcs.je;
    int &ks = indcs.ks;  int &ke = indcs.ke;
    auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
    int nmb = pm->pmb_pack->nmb_thispack;
    auto &u0 = phydro->u0;
    int nhydro = phydro->nhydro;
    int nscalars = phydro->nscalars;
    Real gm1 = phydro->peos->eos_data.gamma - 1.0;
    GhostAtm ga = ghost_atm;  // copy, so it is captured by value into the kernels

    if (pm->two_d) {
      par_for("hsebc_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // inner x2: ghost j mirrors active cell 2*js-1-j
        if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
          int jm = 2*js - 1 - j;
          Real dm = u0(m,IDN,k,jm,i);
          Real vx =  u0(m,IM1,k,jm,i)/dm;
          Real vy = -u0(m,IM2,k,jm,i)/dm;
          Real vz =  u0(m,IM3,k,jm,i)/dm;
          Real rho = ga.rho_lo[j];
          u0(m,IDN,k,j,i) = rho;
          u0(m,IM1,k,j,i) = rho*vx;
          u0(m,IM2,k,j,i) = rho*vy;
          u0(m,IM3,k,j,i) = rho*vz;
          u0(m,IEN,k,j,i) = ga.pres_lo[j]/gm1 + 0.5*rho*(SQR(vx) + SQR(vy) + SQR(vz));
          for (int n=nhydro; n<(nhydro+nscalars); n++) {
            u0(m,n,k,j,i) = rho*u0(m,n,k,jm,i)/dm;
          }
        }
        // outer x2: ghost je+1+j mirrors active cell je-j
        if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
          int jg = je + 1 + j;
          int jm = je - j;
          Real dm = u0(m,IDN,k,jm,i);
          Real vx =  u0(m,IM1,k,jm,i)/dm;
          Real vy = -u0(m,IM2,k,jm,i)/dm;
          Real vz =  u0(m,IM3,k,jm,i)/dm;
          Real rho = ga.rho_hi[j];
          u0(m,IDN,k,jg,i) = rho;
          u0(m,IM1,k,jg,i) = rho*vx;
          u0(m,IM2,k,jg,i) = rho*vy;
          u0(m,IM3,k,jg,i) = rho*vz;
          u0(m,IEN,k,jg,i) = ga.pres_hi[j]/gm1 + 0.5*rho*(SQR(vx) + SQR(vy) + SQR(vz));
          for (int n=nhydro; n<(nhydro+nscalars); n++) {
            u0(m,n,k,jg,i) = rho*u0(m,n,k,jm,i)/dm;
          }
        }
      });
    } else if (pm->three_d) {
      par_for("hsebc_x3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // inner x3: ghost k mirrors active cell 2*ks-1-k
        if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
          int km = 2*ks - 1 - k;
          Real dm = u0(m,IDN,km,j,i);
          Real vx =  u0(m,IM1,km,j,i)/dm;
          Real vy =  u0(m,IM2,km,j,i)/dm;
          Real vz = -u0(m,IM3,km,j,i)/dm;
          Real rho = ga.rho_lo[k];
          u0(m,IDN,k,j,i) = rho;
          u0(m,IM1,k,j,i) = rho*vx;
          u0(m,IM2,k,j,i) = rho*vy;
          u0(m,IM3,k,j,i) = rho*vz;
          u0(m,IEN,k,j,i) = ga.pres_lo[k]/gm1 + 0.5*rho*(SQR(vx) + SQR(vy) + SQR(vz));
          for (int n=nhydro; n<(nhydro+nscalars); n++) {
            u0(m,n,k,j,i) = rho*u0(m,n,km,j,i)/dm;
          }
        }
        // outer x3: ghost ke+1+k mirrors active cell ke-k
        if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
          int kg = ke + 1 + k;
          int km = ke - k;
          Real dm = u0(m,IDN,km,j,i);
          Real vx =  u0(m,IM1,km,j,i)/dm;
          Real vy =  u0(m,IM2,km,j,i)/dm;
          Real vz = -u0(m,IM3,km,j,i)/dm;
          Real rho = ga.rho_hi[k];
          u0(m,IDN,kg,j,i) = rho;
          u0(m,IM1,kg,j,i) = rho*vx;
          u0(m,IM2,kg,j,i) = rho*vy;
          u0(m,IM3,kg,j,i) = rho*vz;
          u0(m,IEN,kg,j,i) = ga.pres_hi[k]/gm1 + 0.5*rho*(SQR(vx) + SQR(vy) + SQR(vz));
          for (int n=nhydro; n<(nhydro+nscalars); n++) {
            u0(m,n,kg,j,i) = rho*u0(m,n,km,j,i)/dm;
          }
        }
      });
    }
  }

  void SetReactiveSources(Mesh* pm, const Real bdt) {
    // Under this wrapper we want to integrate the source terms for the nuclear reactions.
    auto &pmbp = pm->pmb_pack;
    auto &indcs = pm->mb_indcs;
    int is=indcs.is, ie=indcs.ie, js=indcs.js, je=indcs.je, ks=indcs.ks, ke=indcs.ke;
    auto &size = pmbp->pmb->mb_size;
    int nmb1 = pmbp->nmb_thispack - 1;
    bool two_d   = pm->two_d;
    bool three_d = pm->three_d;

    // Let us extract the species primitive variables:
    auto &u0 = pmbp->phydro->u0;
    auto &w0 = pmbp->phydro->w0;
    int nfluid = pmbp->phydro->nhydro;
    int nscalars = pmbp->phydro->nscalars;

    const Real e_atol = e_atol_nuc;
    const Real e_rtol = e_rtol_nuc;
    const Real y_atol = y_atol_nuc;
    const Real y_rtol = y_rtol_nuc;
    const int max_iter = max_iter_nuc;
    const int n_refactor = n_refactor_nuc;
    const Real dt = bdt;

    // EOS object: constructed once, on the host; captured by value into the
    // per-cell Newton-solve kernel below.
    Primitive::EOS<Primitive::IdealGas, Primitive::ResetFloor> eos;
    eos.SetNSpecies(NSPECIES);

    // Burn every cell over dt, and at the same time find min(e/|edot|) at the burned
    // state, which sets the network substep size (dt_nuc) of the next hydro step.
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmkji = (nmb1 + 1)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    const Real factor = dt_nuc_factor;
    Real min_tburn = std::numeric_limits<float>::max();
    int nfail_total = 0;  // number of cells where the Newton solver failed

    Kokkos::parallel_reduce("integrate_jacobian",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_tb, int &nfail) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
     // The idea is to call the primitive variables necessary to evolve the conservative mass fraction
     // and the total energy density.
      Thermo state;
      Real ex[NSPECIES];

      // RHS
      Real F[NX];

      // Conservative (rho*X[n], rho*e) vector: UX is the state at the start of the
      // step, UX0 the current Newton iterate and DUX0 the Newton update.
      Real DUX0[NX];
      Real UX0[NX];
      Real UX[NX];

      // Jacobian and its LU factors (reused across iterations)
      Real J[NX][NX];
      Real LU[NX][NX];
      int piv[NX];

      // Energy and Species atol and rtol
      Real abs_res[2];
      Real rel_res[2];
      Real err[2];   // normalized errors of rho*e and rho*X (converged when <= 1)

      // Calling rho, e and {X}_n. Note that e is the internal energy density.
      Real rho = w0(m,IDN, k, j, i);
      Real energy_i = w0(m, IEN, k, j, i);
      state.rho = rho;

      // eos is captured by value (KOKKOS_LAMBDA == capture-by-value), which
      // makes it const inside this lambda -- copy to a mutable local before
      // passing to burn(), which needs to call eos_local.SetBaryonMass() on it.
      auto eos_local = eos;

      // Call the conservative UX = (rho*X[n], \rho*e vector)
      UX[IE] = energy_i;
      UX0[IE] = energy_i;
      for(int n=0; n<NSPECIES; n++) {
        UX[n] = u0(m, n+nfluid, k, j, i);
        UX0[n] = UX[n];
      }

      // Rebuild composition, mu, temp, cv and ex from the current iterate
      // UX0. burn() takes specific internal energy (erg/g), while UX0[IE] is
      // an energy DENSITY (erg/cm^3) -- divide by rho to convert. burn()
      // already computes state.cv/state.abar/state.dedA as part of this same
      // call (Thermo always carries them), so ex[n] is derived inline here
      // via the chain rule through abar, instead of a second, redundant
      // burn() call (which is what a separate ComputeEdX() would require).
      auto update_state = [&]() {
        for(int n=0; n<NSPECIES; n++) {
          state.xn[n] = UX0[n]/rho;
        }
        state.e = UX0[IE]/rho;
        nuclear::burn<Primitive::IdealGas>(nuclear::eos_input_re, state, eos_local);
        for(int n=0; n<NSPECIES; n++) {
          ex[n] = -state.abar*state.abar*aion_inv(n)*state.dedA;
        }
      };

      // Jacobian and LU factorization at the start of the step
      update_state();
      PopulateJac(rho, state.t, state.xn, state.cv, ex, J);
      bool factor_ok = FactorNewtonMatrix(dt, J, LU, piv);
      int iters_since_factor = 0;
      Real dnorm_old = 0.0;
      bool converged = false;

      // Begin Euler Iteration
      for(int iter=0; (iter<max_iter) && factor_ok; iter++) {
        ActualRHS(rho, state.t, state.xn, F);

        // Euler iteration, reusing the L and U factors:
        BackEulerIteration(rho, dt, UX, UX0, DUX0, F, LU, piv,
                           e_atol, e_rtol, y_atol, y_rtol, abs_res, rel_res, err);

        for(int n=0; n<NX; n++) {
          UX0[n] += DUX0[n];
        }
        update_state();
        iters_since_factor++;

        // Converged when |d(rho*e)| <= e_atol + e_rtol*|rho*e| and, for every species,
        // |d(rho*X)| <= y_atol + y_rtol*|rho*X|
        if ((err[0] <= 1.0) && (err[1] <= 1.0)) {
          converged = true;
          break;
        }

        // Refactor when convergence stalls or after n_refactor iterations
        Real dnorm = Kokkos::fmax(rel_res[0], rel_res[1]);
        bool stalled = (iters_since_factor > 1) && (dnorm > 0.5*dnorm_old);
        dnorm_old = dnorm;
        if (stalled || (iters_since_factor >= n_refactor)) {
          PopulateJac(rho, state.t, state.xn, state.cv, ex, J);
          factor_ok = FactorNewtonMatrix(dt, J, LU, piv);
          iters_since_factor = 0;
        }
      }

      // Count a Newton failure: singular matrix, no convergence within max_iter, or a
      // non-finite result. The run is stopped after the kernel if any cell failed.
      bool finite = true;
      for(int n=0; n<NX; n++) {
        finite = finite && Kokkos::isfinite(UX0[n]);
      }
      if (!(converged && finite)) nfail++;

      // Keep the last iterate
      for(int n=0; n<NX; n++) {
        UX[n] = UX0[n];
      }

      // Renormalize the mass fractions so that sum_n rho*X[n] = rho (corrects the drift
      // from the burn and from advecting the species separately)
      Real sum_rhox = 0.0;
      for(int n=0; n<NSPECIES; n++) {
        sum_rhox += UX[n];
      }
      for(int n=0; n<NSPECIES; n++) {
        UX[n] *= rho/sum_rhox;
      }

      // Internal energy update contribution
      u0(m, IEN, k, j, i) += (UX[IE] - energy_i);

      for(int n=0; n<NSPECIES; n++) {
        u0(m, n+nfluid, k, j, i) = UX[n];
      }

      // Burning timescale e/|edot| at the burned state (state is up to date with
      // UX0 after the last update_state() call)
      Real enuc, enuc_dt;
      ener_rate(rho, state.t, state.xn, enuc, enuc_dt);
      if (Kokkos::fabs(enuc) > 0.0) {
        min_tb = Kokkos::fmin((UX[IE]/rho)/Kokkos::fabs(enuc), min_tb);
      }
    }, Kokkos::Min<Real>(min_tburn), Kokkos::Sum<int>(nfail_total));

    // Stop the run if the Newton solver failed in any cell
    if (nfail_total > 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Nuclear Newton solver failed in " << nfail_total
                << " cells (no convergence in max_iter_nuc=" << max_iter
                << " iterations, singular matrix, or non-finite result)"
                << std::endl << "cycle=" << pm->ncycle << " time=" << pm->time
                << " substep h=" << dt << std::endl;
      exit(EXIT_FAILURE);
    }

    // Network substep size for the next hydro step: Nuclear::Integrate reads it once at
    // the start of the next hydro step, before subcycling. Without any burning
    // min_tburn stays huge, and the hydro step is covered by a single substep.
    pmbp->pnuc->dt_nuc = factor*min_tburn;

    pmbp->phydro->peos->ConsToPrim(u0, w0, false, is, ie, js, je, ks, ke);
    return;
  }
}
