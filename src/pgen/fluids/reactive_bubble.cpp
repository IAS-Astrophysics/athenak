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
#include <cfloat>
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
#include "pgen/fluids/reactive_bubble.hpp"
#include "nuclear/microphysics.hpp"
#include "nuclear/nuclear.hpp"

using nuclear::Thermo;
using EOSHelm = Primitive::EOS<Primitive::EOSHelmholtz, Primitive::ResetFloor>;

namespace {
  Real g_grav;         
  Real y_ref_cgs;      
  Real rindler_k;      
  Real rindler_yref;   
  int max_iter_nuc;
  int n_refactor_nuc;
  Real y_atol_nuc;
  Real y_rtol_nuc;
  Real dt_nuc_factor;  // dt_nuc = dt_nuc_factor * min(cv*T/|edot|)
                       
  // cgs -> code conversion factors (multiply a cgs value to get it in code units), set
  // in UserProblem from the unit systems of the DynGRMHD EOS and of its cgs copy. Host
  // only: the kernels take local copies.
  Real len_c2c, rho_c2c, pres_c2c, time_c2c;
  void SetReactiveSources(Mesh* pm, const Real bdt);
  void HydrostaticBCs(Mesh* pm);
  void SetADMVariablesToRindler(MeshBlockPack *pmbp);

  // Hydrostatic state of the vertical ghost cells, tabulated in UserProblem and used
  // by HydrostaticBCs, in code units. Index j is the ghost-cell index used in the BC
  // kernel (inner: cell j, outer: cell je+1+j).
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


// Right-hand side of the hydrostatic balance of a static fluid in the Rindler metric
// ds^2 = -alpha^2 c^2 dt^2 + dx^2 + dy^2 + dz^2, alpha = 1 + g*(y - y_ref)/c^2:
//   dp/dy = -(rho*(c^2 + eps) + p)/c^2 * g/alpha
// with eos_input_ps (fixed entropy s_base and composition xn[]). Device only in GPU builds: 
// the Helmholtz table lives in device memory.
KOKKOS_INLINE_FUNCTION
Real HydrostaticRHS(const Real &y, const Real &pres, const Real &s_base,
    const Real xn[NSPECIES], const EOSHelm &eos, const Real &g, const Real &y_ref) {
  Thermo stage;
  for (int n = 0; n < NSPECIES; n++) {
    stage.xn[n] = xn[n];
  }
  stage.p = pres; 
  stage.s = s_base;
  nuclear::burn(nuclear::eos_input_ps, stage, eos);
  const Real c2 = nuclear::c_light*nuclear::c_light;
  Real alpha = 1.0 + g*(y - y_ref)/c2;
  return -(stage.rho*(c2 + stage.e) + pres)/c2 * g/alpha;
}


// Advances the hydrostatic pressure profile by one RK4 step of size dy from height y
// (both in cm; dy < 0 integrates downward).
KOKKOS_INLINE_FUNCTION
Real Rk4PressureIntegrationStep(const Real &pres, const Real &s_base, const Real &y,
    const Real &dy, const Real xn[NSPECIES], const EOSHelm &eos, const Real &g,
    const Real &y_ref) {
  Real k1 = HydrostaticRHS(y, pres, s_base, xn, eos, g, y_ref);
  Real k2 = HydrostaticRHS(y + dy/2.0, pres + k1*dy/2.0, s_base, xn, eos, g, y_ref);
  Real k3 = HydrostaticRHS(y + dy/2.0, pres + k2*dy/2.0, s_base, xn, eos, g, y_ref);
  Real k4 = HydrostaticRHS(y + dy, pres + k3*dy, s_base, xn, eos, g, y_ref);
  return pres + dy*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
}


//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for reactive bubble problem. DynGRMHD with the Helmholtz
//! EOS and zero magnetic field; gravity enters through a Rindler metric. The
//! microphysics (burn(), the network) runs in cgs on a copy of the EOS, and the pgen
//! converts to and from code units with the EOS's code unit system.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  Mesh *pm  = pmy_mesh_;
  MeshBlockPack *pmbp = pm->pmb_pack;
  bool two_d   = pm->two_d;
  bool three_d = pm->three_d;

  if (pmbp->pdyngr == nullptr || pmbp->pmhd == nullptr ||
      pin->GetString("mhd", "dyn_eos").compare("helmholtz") != 0 ||
      pin->GetString("mhd", "dyn_error").compare("reset_floor") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "reactive_bubble needs DynGRMHD with mhd/dyn_eos = helmholtz and "
              << "mhd/dyn_error = reset_floor" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!two_d && !three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "reactive_bubble runs in 2D or 3D only" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->nscalars != NSPECIES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "mhd/nscalars must be " << NSPECIES << " (He4, C12, O16)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Code units cm, g, c = 1 on the DynGRMHD EOS. <mhd> has no units entry, so DynGRMHD
  // was constructed with the default c=1 system, providing a freedom to choose the unit system
  // inside the pgen; the PrimitiveSolver holds the EOS by value and every kernel reads it from 
  // there, so setting it here applies to the whole run (restarts included, since UserProblem runs again). 
  // The floors were// set at construction in the old units: set them again from <mhd>/dfloor [g/cm^3]
  // and <mhd>/tfloor [K].
  auto &eos_dyngr = static_cast<dyngr::DynGRMHDPS<Primitive::EOSHelmholtz,
                   Primitive::ResetFloor>*>(pmbp->pdyngr)->eos.ps.GetEOSMutable();
  eos_dyngr.SetCodeUnitSystem(Primitive::MakeCmGC1());
  eos_dyngr.SetDensityFloor(pin->GetOrAddReal("mhd", "dfloor", (FLT_MIN))/
                            eos_dyngr.GetBaryonMass());
  eos_dyngr.SetTemperatureFloor(pin->GetOrAddReal("mhd", "tfloor", (FLT_MIN)));

  // Shallow cgs copy of the EOS (shares the table) for burn() and the hydrostatic
  // profile; constructed on the host and captured by value.
  EOSHelm eos = eos_dyngr;
  eos.SetCodeUnitSystem(Primitive::MakeCGS());
  const Primitive::UnitSystem &code = eos_dyngr.GetCodeUnitSystem();
  const Primitive::UnitSystem &cgs  = eos.GetCodeUnitSystem();
  len_c2c  = cgs.LengthConversion(code);
  rho_c2c  = cgs.MassDensityConversion(code);
  pres_c2c = cgs.PressureConversion(code);
  time_c2c = cgs.TimeConversion(code);

  max_iter_nuc     = pin->GetInteger("problem", "max_iter_nuc");
  n_refactor_nuc   = pin->GetOrAddInteger("problem", "nuc_refactor", 3);
  y_atol_nuc       = pin-> GetReal("problem", "y_atol");
  y_rtol_nuc       = pin-> GetReal("problem", "y_rtol");
  dt_nuc_factor    = pin->GetOrAddReal("problem", "dt_nuc_factor", 0.1);

  user_srcs_nuc_func = &SetReactiveSources;
  user_bcs_func = &HydrostaticBCs;

  // Vertical extent of the mesh (x2 in 2D, x3 in 3D), code units
  Real ymin = two_d ? pm->mesh_size.x2min : pm->mesh_size.x3min;
  Real ymax = two_d ? pm->mesh_size.x2max : pm->mesh_size.x3max;
  int npts  = two_d ? pm->mesh_indcs.nx2 : pm->mesh_indcs.nx3;
  int ng = pm->mb_indcs.ng;

  // Gravity through the Rindler lapse. g_grav [cm/s^2] > 0 points toward -y; y_ref
  // (code length units, default: base of the domain) is where alpha = 1.
  g_grav       = pin->GetReal("problem", "g_grav");
  rindler_yref = pin->GetOrAddReal("problem", "y_ref", ymin);
  y_ref_cgs    = rindler_yref/len_c2c;
  rindler_k    = g_grav*len_c2c/(time_c2c*time_c2c);
  Real dy_cell = (ymax - ymin)/static_cast<Real>(npts);
  Real alpha_min = 1.0 + rindler_k*(ymin - ng*dy_cell - rindler_yref);
  if (!(g_grav > 0.0) || !(alpha_min > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "need g_grav > 0 and a positive Rindler lapse in the whole domain "
              << "(alpha at the lowest ghost cell = " << alpha_min << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  pmbp->padm->SetADMVariables = &SetADMVariablesToRindler;

  // Initial guess of the network substep size [code time], used until the first burn
  // sets it from cv*T/|edot| in SetReactiveSources (also needed on restarts)
  if (pmbp->pnuc != nullptr) {
    auto &mesh_indcs = pm->mesh_indcs;
    int npts_nuc = three_d
                 ? std::min({mesh_indcs.nx1, mesh_indcs.nx2, mesh_indcs.nx3})
                 : std::min(mesh_indcs.nx1, mesh_indcs.nx2);
    pmbp->pnuc->dt_nuc = (1.5e-4)*(64.0/static_cast<Real>(npts_nuc))*time_c2c;
  }

  // Fine grid of the hydrostatic integration: NFACT points per cell, similar to
  // cshock.cpp
  const int NFACT = 10;
  Real dy  = (ymax - ymin)/static_cast<Real>(NFACT*npts);  // code units
  Real dy_cgs = dy/len_c2c;
  Real ymin_cgs = ymin/len_c2c;

  //Inputs (cgs)
  Real rho_base    = pin->GetReal("problem", "rho_base");
  Real temp_base   = pin->GetReal("problem", "temp_base");
  bool perturb     = pin->GetOrAddBoolean("problem", "perturb", true);
  // Length scale of the perturbation [code length]: T = T0*(1 + 0.6*(1 + tanh(4 - r/sigma)))
  // (Zingale et al. 2019, Eq. 89, Table 9)
  Real sigma       = pin->GetOrAddReal("problem", "sigma", 2.56e5);
  // Vertical position of the bubble centre [code length] (x2 in 2D, x3 in 3D):
  Real y_bubble    = pin->GetOrAddReal("problem", "y0", 0.5*(ymin + ymax));

  // --------------------Initialize MHD variables -----------------
  int nmhd = pmbp->pmhd->nmhd;
  int nscalars = pmbp->pmhd->nscalars;

  // Define the initial mass fractions in the entire domain:
  DualArray1D<Real> X_mass("species", nscalars);
  X_mass.h_view(ihe4) = 0.99999998;
  X_mass.h_view(ic12) = 1.0e-8;  //Define enumerate at the top
  X_mass.h_view(io16) = 1.0e-8;

  // Sync X_mass
  X_mass.template modify<HostMemSpace>();
  X_mass.template sync<DevExeSpace>();


  // Hydrostatic profile (cgs), from the base state (rho_base, temp_base) at ymin:
  //   - RK4 on a fine grid of NFACT points per cell (soln),
  //   - binned into the active cells: average P over the NFACT values of each cell,
  //     then rho and T from (P_avg, s_base), so every cell has exactly the base
  //     entropy (sol; valid only at static refinement, no AMR),
  //   - continued ng cells below ymin and above ymax and binned the same way, for
  //     HydrostaticBCs (ghost, code units).
  // The integration is sequential and the Helmholtz table lives in device memory, so
  // it runs in a single-thread device kernel. NaN results (the profile leaving the
  // table) are counted and reported on the host.
  if (ng > NGHOST_MAX) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "reactive_bubble supports at most " << NGHOST_MAX
              << " ghost cells" << std::endl;
    exit(EXIT_FAILURE);
  }
  DvceArray1D<Atm> soln("soln", (NFACT)*npts);
  DualArray1D<Atm> sol("sol", npts);

  // ghost(0..ng-1): rho_lo, (ng..2ng-1): pres_lo, then rho_hi, pres_hi
  DualArray1D<Real> ghost("ghost", 4*ng);

  const Real g_ = g_grav;      // local copies: capturable by device lambda
  const Real yref_ = y_ref_cgs;
  const Real rho_c2c_ = rho_c2c, pres_c2c_ = pres_c2c;
  int nbad = 0;

  Kokkos::parallel_reduce("pgen_reactive_hse", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
  KOKKOS_LAMBDA(const int &, int &bad) {
    Real X[NSPECIES];
    for (int n = 0; n < NSPECIES; n++) X[n] = X_mass.d_view(n);

    // p_base and s_base from the Helmholtz EOS at the given (rho_base, temp_base)
    Thermo atm;
    atm.rho = rho_base;
    atm.t = temp_base;
    for (int n = 0; n < NSPECIES; n++) {
      atm.xn[n] = X[n];
    }
    nuclear::burn(nuclear::eos_input_rt, atm, eos);
    const Real s_base = atm.s;
    const Real p_base = atm.p;

    //We use RK4 to integrate an isentropic profile, starting at ymin.
    Real y_cgs = ymin_cgs;
    soln(0).pres = Rk4PressureIntegrationStep(p_base, s_base, y_cgs, dy_cgs/2.0, X, eos,
                                              g_, yref_);
    y_cgs += dy_cgs/2.0;
    soln(0).yloc = ymin + 0.5*dy;
    for (int i = 0; i < ((NFACT)*npts - 1); i++) {
      soln(i+1).pres = Rk4PressureIntegrationStep(soln(i).pres, s_base, y_cgs, dy_cgs,
                                                  X, eos, g_, yref_);
      y_cgs += dy_cgs;
      soln(i+1).yloc = soln(i).yloc + dy;
    }

    // Binning into the active cells
    for (int n=0; n<npts; n++) {
      Real pres_avg = 0.0;
      for (int m=0; m<(NFACT); m++) {
        pres_avg += soln(m + (NFACT)*n).pres;
      }
      pres_avg /= static_cast<Real>(NFACT);
      atm.p = pres_avg; 
      atm.s = s_base;
      nuclear::burn(nuclear::eos_input_ps, atm, eos);
      sol.d_view(n).pres = pres_avg;
      sol.d_view(n).rho  = atm.rho;
      sol.d_view(n).temp = atm.t;
      sol.d_view(n).yloc = ymin + (static_cast<Real>(n) + 0.5)*(NFACT)*dy;
      if (!(Kokkos::isfinite(atm.rho) && Kokkos::isfinite(atm.t))) bad++;
    }

    // Ghost cells: g = 0 is the ghost layer next to the boundary
    Real pres_lo = p_base;                       // at ymin
    Real pres_hi = soln((NFACT)*npts - 1).pres;  // at ymax - dy/2
    Real y_lo = ymin_cgs;
    Real y_hi = ymin_cgs + ((NFACT)*npts - 0.5)*dy_cgs;
    for (int g=0; g<ng; g++) {
      Real pres_avg_lo = 0.0;
      Real pres_avg_hi = 0.0;
      for (int m=0; m<(NFACT); m++) {
        Real step_lo = (g == 0 && m == 0) ? 0.5*dy_cgs : dy_cgs;
        pres_lo = Rk4PressureIntegrationStep(pres_lo, s_base, y_lo, -step_lo, X, eos,
                                             g_, yref_);
        pres_hi = Rk4PressureIntegrationStep(pres_hi, s_base, y_hi, dy_cgs, X, eos,
                                             g_, yref_);
        y_lo -= step_lo;
        y_hi += dy_cgs;
        pres_avg_lo += pres_lo;
        pres_avg_hi += pres_hi;
      }
      pres_avg_lo /= static_cast<Real>(NFACT);
      pres_avg_hi /= static_cast<Real>(NFACT);

      atm.p = pres_avg_lo; atm.s = s_base;
      nuclear::burn(nuclear::eos_input_ps, atm, eos);
      ghost.d_view(ng-1-g)        = atm.rho*rho_c2c_;
      ghost.d_view(ng + ng-1-g)   = pres_avg_lo*pres_c2c_;
      if (!Kokkos::isfinite(atm.rho)) bad++;

      atm.p = pres_avg_hi; atm.s = s_base;
      nuclear::burn(nuclear::eos_input_ps, atm, eos);
      ghost.d_view(2*ng + g)      = atm.rho*rho_c2c_;
      ghost.d_view(3*ng + g)      = pres_avg_hi*pres_c2c_;
      if (!Kokkos::isfinite(atm.rho)) bad++;
    }
  }, Kokkos::Sum<int>(nbad));

  if (nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "hydrostatic profile left the EOS table in " << nbad
              << " cells (active or ghost)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // sol is used on the device (initial data); the ghost table on the host
  sol.template modify<DevExeSpace>();
  sol.template sync<HostMemSpace>();

  ghost.template modify<DevExeSpace>();
  ghost.template sync<HostMemSpace>();

  for (int g=0; g<ng; g++) {
    ghost_atm.rho_lo[g]  = ghost.h_view(g);
    ghost_atm.pres_lo[g] = ghost.h_view(ng + g);
    ghost_atm.rho_hi[g]  = ghost.h_view(2*ng + g);
    ghost_atm.pres_hi[g] = ghost.h_view(3*ng + g);
  }

  // The ghost-cell table and the Rindler metric are needed on restarts too
  if (restart) return;

  auto &w0 = pmbp->pmhd->w0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // Bubble centre (code units); the vertical coordinate is x2 in 2D, x3 in 3D
  Real x0 = 0.5*(pm->mesh_size.x1min + pm->mesh_size.x1max);
  Real y0 = two_d ? y_bubble : 0.5*(pm->mesh_size.x2min + pm->mesh_size.x2max);
  Real z0 = y_bubble;

  // Index-offset of the MeshBlocks in the sol array (vertical logical location)
  DualArray1D<int> voff("voffset", pmbp->nmb_thispack);
  for (int m=0; m<(pmbp->nmb_thispack); m++) {
    int igid = pmbp->gids + m;
    LogicalLocation lloc=pm->lloc_eachmb[igid];
    voff.h_view(m) = two_d ? lloc.lx2*(pm->mb_indcs.nx2) : lloc.lx3*(pm->mb_indcs.nx3);
  }
  // sync with device
  voff.template modify<HostMemSpace>();
  voff.template sync<DevExeSpace>();

  par_for("pgen_reactive", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    int iv = voff.d_view(m) + (two_d ? j-js : k-ks);
    // Pressure and temperature of the atmosphere (cgs)
    Real pres = sol.d_view(iv).pres;
    Real temp = sol.d_view(iv).temp;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real rad;
    if (two_d) {
      rad = sqrt(SQR(x1v-x0) + SQR(x2v-y0));
    } else {
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      rad = sqrt(SQR(x1v-x0) + SQR(x2v-y0) + SQR(x3v-z0));
    }

    // Temperature perturbation at fixed pressure; the density follows from (p, T)
    if (perturb) temp *= (1.0 + 0.6*(1.0 + tanh(4.0-rad/sigma)));
    Thermo cell;
    for (int n = 0; n < NSPECIES; n++) cell.xn[n] = X_mass.d_view(n);
    cell.p = pres;
    cell.t = temp;
    nuclear::burn(nuclear::eos_input_pt, cell, eos);

    // Primitive vector in code units. DynGRMHD stores the pressure in IPR.
    w0(m,IDN,k,j,i) = cell.rho*rho_c2c_;
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IPR,k,j,i) = pres*pres_c2c_;
    for (int n=0; n<nscalars; n++) {
      w0(m,nmhd+n,k,j,i) = X_mass.d_view(n);
    }

    // Zero magnetic field
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = 0.0;
    if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
    if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
    if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}
    bcc0(m,IBX,k,j,i) = 0.0;
    bcc0(m,IBY,k,j,i) = 0.0;
    bcc0(m,IBZ,k,j,i) = 0.0;
  });

  // ADM variables (Rindler) and the conserved variables
  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
}
//--------------- End initialization MHD variables--------------------------------------


namespace {

//----------------------------------------------------------------------------------------
//! \fn void SetADMVariablesToRindler(MeshBlockPack *pmbp)
//! \brief Static Rindler metric: flat spatial metric, zero shift and extrinsic
//! curvature, and lapse alpha = 1 + rindler_k*(y - rindler_yref), linear in the
//! vertical coordinate (x2 in 2D, x3 in 3D). A static fluid then feels a uniform
//! acceleration g = rindler_k*c^2 (per unit lapse) toward -y.

void SetADMVariablesToRindler(MeshBlockPack *pmbp) {
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int js = indcs.js, ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  bool two_d = pmbp->pmesh->two_d;
  Real k_ = rindler_k;      // local copies: capturable by device lambda
  Real yref_ = rindler_yref;

  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real y;
    if (two_d) {
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      y = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    } else {
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      y = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    }

    adm.g_dd(m,0,0,k,j,i) = 1.0;
    adm.g_dd(m,0,1,k,j,i) = 0.0;
    adm.g_dd(m,0,2,k,j,i) = 0.0;
    adm.g_dd(m,1,1,k,j,i) = 1.0;
    adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.g_dd(m,2,2,k,j,i) = 1.0;

    adm.vK_dd(m,0,0,k,j,i) = 0.0;
    adm.vK_dd(m,0,1,k,j,i) = 0.0;
    adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = 0.0;
    adm.vK_dd(m,1,2,k,j,i) = 0.0;
    adm.vK_dd(m,2,2,k,j,i) = 0.0;

    adm.psi4(m,k,j,i) = 1.0;
    adm.alpha(m,k,j,i) = 1.0 + k_*(y - yref_);
    adm.beta_u(m,0,k,j,i) = 0.0;
    adm.beta_u(m,1,k,j,i) = 0.0;
    adm.beta_u(m,2,k,j,i) = 0.0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void HydrostaticBCs(Mesh *pm)
//! \brief User BCs on the vertical faces (x2 in 2D, x3 in 3D). MHD::ApplyPhysicalBCs
//! calls this with only the conserved variables up to date, and DynGRMHD's ConToPrim
//! then runs over the whole array, ghost cells included, so the BC must set u0.
//! The active rows next to the boundary are first converted to primitives (current
//! velocities and mass fractions). Each ghost cell then gets density and pressure from
//! the hydrostatic profile tabulated in UserProblem (code units), and velocities and
//! mass fractions reflected from the mirror-image active cell, with the normal velocity
//! negated, so there is no mass flux through the boundary. These primitives are turned
//! into the densitized conserved variables cell by cell, as in PrimToCons, with zero
//! magnetic field.

  void HydrostaticBCs(Mesh *pm) {
    auto *pmbp = pm->pmb_pack;
    auto *pmhd = pmbp->pmhd;
    if (pmhd == nullptr || pmbp->pdyngr == nullptr) return;

    auto &indcs = pm->mb_indcs;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
    int &js = indcs.js;  int &je = indcs.je;
    int &ks = indcs.ks;  int &ke = indcs.ke;
    auto &mb_bcs = pmbp->pmb->mb_bcs;
    int nmb = pmbp->nmb_thispack;
    auto &u0 = pmhd->u0;
    auto &w0 = pmhd->w0;
    auto &adm = pmbp->padm->adm;
    int nmhd = pmhd->nmhd;
    int nscalars = pmhd->nscalars;
    GhostAtm ga = ghost_atm;  // copy, so it is captured by value into the kernels

    auto *pdyngr = static_cast<dyngr::DynGRMHDPS<Primitive::EOSHelmholtz,
                   Primitive::ResetFloor>*>(pmbp->pdyngr);
    auto &ps = pdyngr->eos.ps;
    const Real mb = ps.GetEOS().GetBaryonMass();

    // Current primitives of the ng active rows next to each vertical boundary (active
    // cells in every MeshBlock, so this is safe on internal faces too)
    if (pm->two_d) {
      pdyngr->ConToPrimBC(0, n1-1, js, js+ng-1, 0, n3-1);
      pdyngr->ConToPrimBC(0, n1-1, je-ng+1, je, 0, n3-1);
    } else {
      pdyngr->ConToPrimBC(0, n1-1, 0, n2-1, ks, ks+ng-1);
      pdyngr->ConToPrimBC(0, n1-1, 0, n2-1, ke-ng+1, ke);
    }

    // Sets ghost cell (kg,jg,i) from the table (rho, p) and the mirror cell (km,jm,i)
    // (velocity, with component idir negated, and mass fractions), in w0 and u0
    auto set_ghost = KOKKOS_LAMBDA(int m, int kg, int jg, int km, int jm, int i,
                                   int idir, Real rho, Real pres) {
      w0(m,IDN,kg,jg,i) = rho;
      w0(m,IVX,kg,jg,i) = w0(m,IVX,km,jm,i);
      w0(m,IVY,kg,jg,i) = w0(m,IVY,km,jm,i);
      w0(m,IVZ,kg,jg,i) = w0(m,IVZ,km,jm,i);
      w0(m,idir,kg,jg,i) = -w0(m,idir,km,jm,i);
      w0(m,IPR,kg,jg,i) = pres;
      for (int n=nmhd; n<(nmhd+nscalars); n++) {
        w0(m,n,kg,jg,i) = w0(m,n,km,jm,i);
      }

      Real g3d[NSPMETRIC];
      g3d[S11] = adm.g_dd(m,0,0,kg,jg,i);
      g3d[S12] = adm.g_dd(m,0,1,kg,jg,i);
      g3d[S13] = adm.g_dd(m,0,2,kg,jg,i);
      g3d[S22] = adm.g_dd(m,1,1,kg,jg,i);
      g3d[S23] = adm.g_dd(m,1,2,kg,jg,i);
      g3d[S33] = adm.g_dd(m,2,2,kg,jg,i);
      Real sdetg = sqrt(Primitive::GetDeterminant(g3d));

      Real prim_pt[NPRIM], cons_pt[NCONS];
      Real b[NMAG] = {0.0, 0.0, 0.0};
      prim_pt[PRH] = rho/mb;
      prim_pt[PVX] = w0(m,IVX,kg,jg,i);
      prim_pt[PVY] = w0(m,IVY,kg,jg,i);
      prim_pt[PVZ] = w0(m,IVZ,kg,jg,i);
      for (int n=0; n<nscalars; n++) {
        prim_pt[PYF + n] = w0(m,nmhd+n,kg,jg,i);
      }
      prim_pt[PPR] = pres;
      prim_pt[PTM] = ps.GetEOS().GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR],
                                                     &prim_pt[PYF]);
      ps.PrimToCon(prim_pt, cons_pt, b, g3d);

      u0(m,IDN,kg,jg,i) = cons_pt[CDN]*sdetg;
      u0(m,IM1,kg,jg,i) = cons_pt[CSX]*sdetg;
      u0(m,IM2,kg,jg,i) = cons_pt[CSY]*sdetg;
      u0(m,IM3,kg,jg,i) = cons_pt[CSZ]*sdetg;
      u0(m,IEN,kg,jg,i) = cons_pt[CTA]*sdetg;
      for (int n=0; n<nscalars; n++) {
        u0(m,nmhd+n,kg,jg,i) = cons_pt[CYD + n]*sdetg;
      }
    };

    if (pm->two_d) {
      par_for("hsebc_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // inner x2: ghost j mirrors active cell 2*js-1-j
        if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
          set_ghost(m, k, j, k, 2*js-1-j, i, IVY, ga.rho_lo[j], ga.pres_lo[j]);
        }
        // outer x2: ghost je+1+j mirrors active cell je-j
        if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
          set_ghost(m, k, je+1+j, k, je-j, i, IVY, ga.rho_hi[j], ga.pres_hi[j]);
        }
      });
    } else if (pm->three_d) {
      par_for("hsebc_x3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // inner x3: ghost k mirrors active cell 2*ks-1-k
        if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
          set_ghost(m, k, j, 2*ks-1-k, j, i, IVZ, ga.rho_lo[k], ga.pres_lo[k]);
        }
        // outer x3: ghost ke+1+k mirrors active cell ke-k
        if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
          set_ghost(m, ke+1+k, j, ke-k, j, i, IVZ, ga.rho_hi[k], ga.pres_hi[k]);
        }
      });
    }
  }

  void SetReactiveSources(Mesh* pm, const Real bdt) {
    // Operator-split burn of the mass fractions over bdt [code time]. Only the partial
    // densities D*X[n] change: with the Helmholtz EOS, eps includes the nuclear binding
    // energy, so the release keeps tau (and eps, at these low velocities, where
    // tau ~ rho*eps) fixed and shows up as a higher T and p after ConToPrim.
    // The network runs in cgs on a copy of the EOS.
    auto &pmbp = pm->pmb_pack;
    auto &indcs = pm->mb_indcs;
    int is=indcs.is, ie=indcs.ie, js=indcs.js, je=indcs.je, ks=indcs.ks, ke=indcs.ke;
    int nmb1 = pmbp->nmb_thispack - 1;

    auto &u0 = pmbp->pmhd->u0;
    auto &w0 = pmbp->pmhd->w0;
    int nmhd = pmbp->pmhd->nmhd;

    const Real y_atol = y_atol_nuc;
    const Real y_rtol = y_rtol_nuc;
    const int max_iter = max_iter_nuc;
    const int n_refactor = n_refactor_nuc;
    const Real dt = bdt/time_c2c;     // [s]

    // cgs copy of the DynGRMHD EOS (shares the table), captured by value into the
    // per-cell Newton-solve kernel below.
    EOSHelm eos = static_cast<dyngr::DynGRMHDPS<Primitive::EOSHelmholtz,
                  Primitive::ResetFloor>*>(pmbp->pdyngr)->eos.ps.GetEOS();
    eos.SetCodeUnitSystem(Primitive::MakeCGS());

    // Burn every cell over dt, and at the same time find min(cv*T/|edot|) at the burned
    // state, which sets the network substep size (dt_nuc) of the next hydro step.
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int nmkji = (nmb1 + 1)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    const Real factor = dt_nuc_factor;
    const Real rho_c2c_ = rho_c2c, pres_c2c_ = pres_c2c;
    Real min_theat = std::numeric_limits<float>::max();
    int nfail_total = 0;  // number of cells where the Newton solver failed

    Kokkos::parallel_reduce("integrate_jacobian",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_th, int &nfail) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Thermo state;

      // RHS, d(eps)/dX[n], and the conserved rho*X[n] vector: UX is the state at the
      // start of the step, UX0 the current Newton iterate and DUX0 the Newton update.
      Real F[NSPECIES];
      Real ex[NSPECIES];
      Real DUX0[NSPECIES];
      Real UX0[NSPECIES];
      Real UX[NSPECIES];

      // Jacobian and its LU factors (reused across iterations)
      Real J[NSPECIES][NSPECIES];
      Real LU[NSPECIES][NSPECIES];
      int piv[NSPECIES];

      Real abs_res, rel_res, err;
      Real enuc_dot;

      // Primitives in cgs: rho, X[n], and T from the pressure (DynGRMHD stores p in
      // IPR); eps at that T is held fixed during the burn.
      const Real rho = w0(m,IDN,k,j,i)/rho_c2c_;  // fixed during the burn
      state.rho = rho;
      state.p = w0(m,IPR,k,j,i)/pres_c2c_;
      for(int n=0; n<NSPECIES; n++) {
        state.xn[n] = w0(m,nmhd+n,k,j,i);
        UX[n] = rho*state.xn[n];
        UX0[n] = UX[n];
      }

      nuclear::burn(nuclear::eos_input_rp, state, eos);
      const Real eps = state.e;  // held fixed during the burn (tau is unchanged)

      // Composition, T (at fixed eps), cv and d(e)/dX[n] of the current iterate UX0
      auto update_state = [&]() {
        for(int n=0; n<NSPECIES; n++) {
          state.xn[n] = UX0[n]/rho;
        }
        state.e = eps;
        nuclear::burn(nuclear::eos_input_re, state, eos);
        for(int n=0; n<NSPECIES; n++) {
          ex[n] = state.dedX[n];
        }
      };

      // Jacobian and LU factorization at the start of the step
      update_state();
      PopulateJac(rho, state.t, state.xn, state.cv, ex, J);
      bool factor_ok = FactorNewtonMatrix(dt, J, LU, piv);
      int iters_since_factor = 0;
      Real rel_res_old = 0.0;
      bool converged = false;

      // Begin Euler Iteration
      for(int iter=0; (iter<max_iter) && factor_ok; iter++) {
        ActualRHS(rho, state.t, state.xn, ex, F, enuc_dot);

        // Euler iteration, reusing the L and U factors:
        BackEulerIteration(rho, dt, UX, UX0, DUX0, F, LU, piv, y_atol, y_rtol,
                           abs_res, rel_res, err);

        for(int n=0; n<NSPECIES; n++) {
          UX0[n] += DUX0[n];
        }
        update_state();
        iters_since_factor++;

        // Converged when, for every species, |d(rho*X)| <= y_atol + y_rtol*|rho*X|
        if (err <= 1.0) {
          converged = true;
          break;
        }

        // Refactor when convergence stalls or after n_refactor iterations
        bool stalled = (iters_since_factor > 1) && (rel_res > 0.5*rel_res_old);
        rel_res_old = rel_res;
        if (stalled || (iters_since_factor >= n_refactor)) {
          PopulateJac(rho, state.t, state.xn, state.cv, ex, J);
          factor_ok = FactorNewtonMatrix(dt, J, LU, piv);
          iters_since_factor = 0;
        }
      }

      // Count a Newton failure: singular matrix, no convergence within max_iter, or a
      // non-finite result (also T, e.g. eps outside the table). The run is stopped after
      // the kernel if any cell failed.
      bool finite = Kokkos::isfinite(state.t);
      for(int n=0; n<NSPECIES; n++) {
        finite = finite && Kokkos::isfinite(UX0[n]);
      }
      if (!(converged && finite)) nfail++;

      // Mass fractions of the last iterate, renormalized so that sum_n X[n] = 1
      // (corrects the drift from the burn and from advecting the species separately)
      Real sum_rhox = 0.0;
      for(int n=0; n<NSPECIES; n++) {
        sum_rhox += UX0[n];
      }
      // Conserved D*X[n]; D and tau are unchanged by the burn
      for(int n=0; n<NSPECIES; n++) {
        u0(m, nmhd+n, k, j, i) = u0(m,IDN,k,j,i)*UX0[n]/sum_rhox;
      }

      // Heating timescale cv*T/|edot| [s] at the burned state (state is up to date with
      // UX0 after the last update_state() call)
      min_th = Kokkos::fmin(heating_timescale(rho, state.t, state.xn, ex, state.cv),
                            min_th);
    }, Kokkos::Min<Real>(min_theat), Kokkos::Sum<int>(nfail_total));

    // Stop the run if the Newton solver failed in any cell
    if (nfail_total > 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Nuclear Newton solver failed in " << nfail_total
                << " cells (no convergence in max_iter_nuc=" << max_iter
                << " iterations, singular matrix, or non-finite result)"
                << std::endl << "cycle=" << pm->ncycle << " time=" << pm->time
                << " substep h=" << bdt << std::endl;
      exit(EXIT_FAILURE);
    }

    // Network substep size for the next hydro step [code time]: Nuclear::Integrate
    // reads it once at the start of the next hydro step, before subcycling. Without any
    // burning min_theat stays huge, and the hydro step is covered by a single substep.
    pmbp->pnuc->dt_nuc = factor*min_theat*time_c2c;

    // New primitives of the active cells (T and p rise at fixed eps); the ghost zones
    // are refreshed by the Nuclear tasks afterwards
    pmbp->pdyngr->ConToPrimBC(is, ie, js, je, ks, ke);
    return;
  }
}
