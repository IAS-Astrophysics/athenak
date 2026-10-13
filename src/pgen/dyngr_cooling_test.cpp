//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cooling_test.cpp
//! \brief Minimal DynGRMHD + Valencia cooling verification problem.

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

// --- AthenaXXX Headers ---
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "units/units.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// Cooling curve
#include "srcterms/ismcooling.hpp"

//----------------------------------------------------------------------------------------
// Prototypes
void CoolingDiagHistory(HistoryData *pdata, Mesh *pm);
void AddValenciaGRCooling(Mesh *pm, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Uniform ICs for cooling-only verification.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr || pmbp->padm == nullptr || 
      pmbp->pmhd == nullptr || pmbp->punit == nullptr) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR: cooling_test requires DynGRMHD, ADM, MHD, and Units.\n";
    }
    exit(EXIT_FAILURE);
  }

  // Enroll diagnostics and source terms
  user_hist_func = CoolingDiagHistory;
  user_srcs_func = AddValenciaGRCooling; // Enroll the new source term

  if (restart) return;

  // Indices
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;

  // Arrays
  auto &w0_ = pmbp->pmhd->w0;
  auto &bcc0_ = pmbp->pmhd->bcc0;
  auto &b0_ = pmbp->pmhd->b0;

  // EOS floors
  auto &eos_data = pmbp->pmhd->peos->eos_data;
  const Real rho_floor = eos_data.dfloor;
  const Real p_floor   = eos_data.pfloor;

  // User IC parameters
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real p0   = pin->GetOrAddReal("problem", "p0",   1.0e-2);

  const Real vx = pin->GetOrAddReal("problem", "vx", 0.0);
  const Real vy = pin->GetOrAddReal("problem", "vy", 0.0);
  const Real vz = pin->GetOrAddReal("problem", "vz", 0.0);
  
  const Real vsq = vx*vx + vy*vy + vz*vz;
  if (vsq >= 1.0) {
    if (global_variable::my_rank == 0) std::cout << "### FATAL ERROR: v < 1 required\n";
    exit(EXIT_FAILURE);
  }
  const Real gamma = 1.0 / std::sqrt(1.0 - vsq);
  const Real u1p0 = gamma * vx;
  const Real u2p0 = gamma * vy;
  const Real u3p0 = gamma * vz;

  const Real rho_ic = fmax(rho0, rho_floor * 10.0);
  const Real p_ic   = fmax(p0  , p_floor   * 10.0);

  // Initialize Primitives
  par_for("pgen_cooling_uniform_prims", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    w0_(m, IDN, k, j, i) = rho_ic;
    w0_(m, IPR, k, j, i) = p_ic;
    w0_(m, IVX, k, j, i) = u1p0;
    w0_(m, IVY, k, j, i) = u2p0;
    w0_(m, IVZ, k, j, i) = u3p0;
    
    bcc0_(m, IBX, k, j, i) = 0.0;
    bcc0_(m, IBY, k, j, i) = 0.0;
    bcc0_(m, IBZ, k, j, i) = 0.0;
  });

  // Zero Face Fields
  par_for("pgen_cooling_zero_bf", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    b0_.x1f(m,k,j,i) = 0.0;
    b0_.x2f(m,k,j,i) = 0.0;
    b0_.x3f(m,k,j,i) = 0.0;
    if (i == ie) b0_.x1f(m,k,j,i+1) = 0.0;
    if (j == je) b0_.x2f(m,k,j+1,i) = 0.0;
    if (k == ke) b0_.x3f(m,k+1,j,i) = 0.0;
  });

  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);

  if (global_variable::my_rank == 0) {
    std::cout << "cooling_test ICs: rho=" << rho_ic << "  p=" << p_ic << "\n";
  }
}

//----------------------------------------------------------------------------------------
//! \fn void AddValenciaGRCooling(Mesh *pm, const Real bdt)
//! \brief Valencia GR cooling source term with subcycling.

void AddValenciaGRCooling(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // ---- Units ----
  Real temp_unit     = pmbp->punit->temperature_cgs();
  Real density_unit  = pmbp->punit->density_cgs();
  Real time_unit     = pmbp->punit->time_cgs();
  Real pressure_unit = pmbp->punit->pressure_cgs();

  Real mu  = pmbp->punit->mu();
  Real amu = pmbp->punit->atomic_mass_unit_cgs; 

  Real n_unit = density_unit / (mu * amu); 
  Real cooling_unit = pressure_unit / time_unit / (n_unit * n_unit);

  // ---- Indices ----
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  // ---- Accessors ----
  auto &adm = pmbp->padm->adm;
  auto &w0  = pmbp->pmhd->w0;  // primitives
  auto &u0  = pmbp->pmhd->u0;  // conserved 

  // ---- EOS ----
  auto &eos_data = pmbp->pmhd->peos->eos_data;
  Real gamma_adi = eos_data.gamma;
  Real gm1       = gamma_adi - 1.0;
  Real rho_floor = eos_data.dfloor; 
  Real p_floor   = eos_data.pfloor; 

  // ---- Stability Control ----
  // Use the simulation's global CFL number for consistency
  // instead of a hardcoded "by-hand" value.
  Real cfl_limit = pm->cfl_no; 

  constexpr int  max_sub  = 64;
  constexpr Real tiny     = 1.0e-30;

  par_for("Valencia_IsotropicCooling", DevExeSpace(),
          0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {

    // --- metric gamma_ij and sqrt(gamma) ---
    Real gxx = adm.g_dd(m,0,0,k,j,i);
    Real gxy = adm.g_dd(m,0,1,k,j,i);
    Real gxz = adm.g_dd(m,0,2,k,j,i);
    Real gyy = adm.g_dd(m,1,1,k,j,i);
    Real gyz = adm.g_dd(m,1,2,k,j,i);
    Real gzz = adm.g_dd(m,2,2,k,j,i);

    Real detg = adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
    detg = fmax(detg, tiny);
    Real sqrt_gamma = sqrt(detg);

    Real alpha = adm.alpha(m,k,j,i);

    // --- primitive state ---
    Real rho  = w0(m,IDN,k,j,i);
    if (rho <= rho_floor) return;

    Real pres = w0(m,IPR,k,j,i);

    // primitive stores u^{i'}
    Real u1p = w0(m,IVX,k,j,i);
    Real u2p = w0(m,IVY,k,j,i);
    Real u3p = w0(m,IVZ,k,j,i);

    // W = sqrt(1 + gamma_ij u^{i'} u^{j'})
    Real u_sq = gxx*u1p*u1p + 2.0*gxy*u1p*u2p + 2.0*gxz*u1p*u3p
              + gyy*u2p*u2p + 2.0*gyz*u2p*u3p + gzz*u3p*u3p;
    Real W = sqrt(1.0 + u_sq);

    // covariant spatial components u_i
    Real u1_cov = gxx*u1p + gxy*u2p + gxz*u3p;
    Real u2_cov = gxy*u1p + gyy*u2p + gyz*u3p;
    Real u3_cov = gxz*u1p + gyz*u2p + gzz*u3p;

    // comoving internal energy density
    Real e_int = pres / gm1;
    Real e_floor = p_floor / gm1;

    // Subcycling in coordinate time
    Real dt_rem = bdt;

    // Accumulated conserved decrements
    Real dTau_total = 0.0;
    Real dS1_total  = 0.0;
    Real dS2_total  = 0.0;
    Real dS3_total  = 0.0;

    for (int n = 0; n < max_sub && dt_rem > 0.0; ++n) {
      // Temperature proxy 
      Real T_cgs = ( (e_int * gm1) / rho ) * temp_unit;

      // Determine Cooling Rate
      Real Lambda_cgs = 0.0;
      if (T_cgs >= eos_data.tfloor * temp_unit) {
         Lambda_cgs = ISMCoolFn(T_cgs);
      }

      // q = n^2 Lambda
      Real q = (rho * rho) * (Lambda_cgs / cooling_unit); // code units

      if (q <= 0.0) break;

      // Coordinate-time cooling rate for e_int: de_int/dt = -(alpha/W) q
      Real rate_e_dt = (alpha / W) * q;

      // === DYNAMIC CLAMP (Based on CFL) ===
      // Max allowed rate is one that removes 'cfl_limit' fraction of e_int 
      // over the full timestep 'bdt'.
      // This ensures operator splitting doesn't shock the hydro solver.
      Real rate_max = (cfl_limit * e_int) / (bdt + tiny);
      
      rate_e_dt = fmin(rate_e_dt, rate_max);
      // ====================================

      // Choose substep using the same CFL limit
      Real dt_sub = cfl_limit * e_int / (rate_e_dt + tiny);
      dt_sub = fmin(dt_sub, dt_rem);
      dt_sub = fmax(dt_sub, tiny * bdt);

      // Proposed decrement 
      Real de = rate_e_dt * dt_sub; 

      // Enforce floor on e_int
      Real de_applied = de;
      if (e_int - de_applied < e_floor) {
        de_applied = e_int - e_floor;
        e_int = e_floor;
      } else {
        e_int -= de_applied;
      }

      if (de_applied <= 0.0) break;

      // Convert applied decrement back to q*dt (source term magnitude)
      Real q_dt = de_applied * (W / alpha);

      // Valencia isotropic cooling sources:
      Real dTau = sqrt_gamma * alpha * W * q_dt;
      Real dS1  = sqrt_gamma * alpha * u1_cov * q_dt;
      Real dS2  = sqrt_gamma * alpha * u2_cov * q_dt;
      Real dS3  = sqrt_gamma * alpha * u3_cov * q_dt;

      dTau_total += dTau;
      dS1_total  += dS1;
      dS2_total  += dS2;
      dS3_total  += dS3;

      dt_rem -= dt_sub;
    }

    // Apply to conserved variables 
    u0(m,IEN,k,j,i) -= dTau_total;
    u0(m,IM1,k,j,i) -= dS1_total;
    u0(m,IM2,k,j,i) -= dS2_total;
    u0(m,IM3,k,j,i) -= dS3_total;
  });
}

//----------------------------------------------------------------------------------------
// Diagnostics history (unchanged logic, just re-included for completeness)
void CoolingDiagHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->padm == nullptr || pmbp->punit == nullptr) {
    pdata->nhist = 0; return;
  }

  auto &indcs = pm->mb_indcs;
  const int nmb = pmbp->nmb_thispack;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nkji = nx3*nx2*nx1, nmkji = nmb*nkji, nji = nx2*nx1;

  auto &adm = pmbp->padm->adm;
  auto &w0  = pmbp->pmhd->w0;
  auto &u0  = pmbp->pmhd->u0;
  auto &eos_data = pmbp->pmhd->peos->eos_data;

  const Real temp_unit     = pmbp->punit->temperature_cgs();
  const Real density_unit  = pmbp->punit->density_cgs();
  const Real time_unit     = pmbp->punit->time_cgs();
  const Real pressure_unit = pmbp->punit->pressure_cgs();
  const Real mu  = pmbp->punit->mu();
  const Real amu = pmbp->punit->atomic_mass_unit_cgs;
  const Real n_unit = density_unit / (mu * amu);
  const Real cooling_unit = pressure_unit / time_unit / (n_unit * n_unit);
  const Real tfloor_code = eos_data.tfloor;
  const Real tiny = 1.0e-30;

  Real sum_rho = 0, sum_p = 0, sum_tau = 0, sum_s1 = 0, sum_s2 = 0, sum_s3 = 0;
  Real edot_p = 0, p1_p = 0, p2_p = 0, p3_p = 0;
  Real Tmin = 1e30, Tmax = 0, n_active = 0;

  Kokkos::parallel_reduce("cooling_diag", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int idx, Real &lrho, Real &lp, Real &ltau, Real &ls1, Real &ls2, Real &ls3,
                  Real &lE, Real &lp1, Real &lp2, Real &lp3, Real &lTmin, Real &lTmax, Real &lNa) {
      int m=idx/nkji, rem=idx%nkji;
      int k=(rem/nji)+ks, rem2=rem%nji;
      int j=(rem2/nx1)+js, i=(rem2%nx1)+is;

      const Real rho = w0(m, IDN, k, j, i);
      const Real p   = w0(m, IPR, k, j, i);
      lrho += rho; lp += p;
      ltau += u0(m, IEN, k, j, i);
      ls1  += u0(m, IM1, k, j, i);
      ls2  += u0(m, IM2, k, j, i);
      ls3  += u0(m, IM3, k, j, i);

      const Real gxx=adm.g_dd(m,0,0,k,j,i), gxy=adm.g_dd(m,0,1,k,j,i), gxz=adm.g_dd(m,0,2,k,j,i);
      const Real gyy=adm.g_dd(m,1,1,k,j,i), gyz=adm.g_dd(m,1,2,k,j,i), gzz=adm.g_dd(m,2,2,k,j,i);
      const Real vol = sqrt(fmax(adm::SpatialDet(gxx,gxy,gxz,gyy,gyz,gzz), tiny));
      const Real alpha = adm.alpha(m,k,j,i);

      const Real u1p=w0(m,IVX,k,j,i), u2p=w0(m,IVY,k,j,i), u3p=w0(m,IVZ,k,j,i);
      const Real W = sqrt(1.0 + gxx*u1p*u1p + 2*gxy*u1p*u2p + 2*gxz*u1p*u3p + gyy*u2p*u2p + 2*gyz*u2p*u3p + gzz*u3p*u3p);

      Real T_cgs = (rho>tiny) ? (p/rho)*temp_unit : 0.0;
      lTmin = fmin(lTmin, T_cgs); lTmax = fmax(lTmax, T_cgs);

      if (T_cgs >= tfloor_code*temp_unit) {
        Real L = ISMCoolFn(T_cgs);
        if (L > 0 && rho > tiny) {
          lNa += 1.0;
          const Real q = (rho*rho)*(L/cooling_unit);
          lE  += vol * alpha * q * W;
          lp1 += vol * alpha * q * (gxx*u1p+gxy*u2p+gxz*u3p);
          lp2 += vol * alpha * q * (gxy*u1p+gyy*u2p+gyz*u3p);
          lp3 += vol * alpha * q * (gxz*u1p+gyz*u2p+gzz*u3p);
        }
      }
    },
    Kokkos::Sum<Real>(sum_rho), Kokkos::Sum<Real>(sum_p),
    Kokkos::Sum<Real>(sum_tau), Kokkos::Sum<Real>(sum_s1), Kokkos::Sum<Real>(sum_s2), Kokkos::Sum<Real>(sum_s3),
    Kokkos::Sum<Real>(edot_p), Kokkos::Sum<Real>(p1_p), Kokkos::Sum<Real>(p2_p), Kokkos::Sum<Real>(p3_p),
    Kokkos::Min<Real>(Tmin), Kokkos::Max<Real>(Tmax), Kokkos::Sum<Real>(n_active)
  );

#if MPI_PARALLEL_ENABLED
  double buf[13] = {sum_rho, sum_p, sum_tau, sum_s1, sum_s2, sum_s3, edot_p, p1_p, p2_p, p3_p, Tmin, Tmax, n_active};
  MPI_Allreduce(MPI_IN_PLACE, buf, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &buf[10], 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &buf[11], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &buf[12], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  sum_rho=buf[0]; sum_p=buf[1]; sum_tau=buf[2]; sum_s1=buf[3]; sum_s2=buf[4]; sum_s3=buf[5];
  edot_p=buf[6]; p1_p=buf[7]; p2_p=buf[8]; p3_p=buf[9]; Tmin=buf[10]; Tmax=buf[11]; n_active=buf[12];
#endif

  const Real ncell = static_cast<Real>(pm->nmb_total) * nx1 * nx2 * nx3;
  const Real invN = 1.0/fmax(ncell, 1.0);

  static bool first = true;
  static Real t_prev = 0, tau_p = 0, s1_p = 0, s2_p = 0, s3_p = 0;
  Real relE = 0, relP1 = 0, relP2 = 0, relP3 = 0;
  Real Edot_fd = 0, P1_fd = 0, P2_fd = 0, P3_fd = 0;

  if (!first && (pm->time > t_prev)) {
    Real dt = pm->time - t_prev;
    Edot_fd = -(sum_tau - tau_p) / dt;
    P1_fd   = -(sum_s1 - s1_p) / dt;
    P2_fd   = -(sum_s2 - s2_p) / dt;
    P3_fd   = -(sum_s3 - s3_p) / dt;
    relE    = (Edot_fd - edot_p) / fmax(fabs(edot_p), 1e-60);
    relP1   = (P1_fd - p1_p) / fmax(fabs(p1_p), 1e-60);
    relP2   = (P2_fd - p2_p) / fmax(fabs(p2_p), 1e-60);
    relP3   = (P3_fd - p3_p) / fmax(fabs(p3_p), 1e-60);
  }
  first = false; t_prev = pm->time; tau_p = sum_tau; s1_p = sum_s1; s2_p = sum_s2; s3_p = sum_s3;

  pdata->nhist = 20;
  pdata->label[0] = "rho_avg";    pdata->hdata[0] = sum_rho * invN;
  pdata->label[1] = "p_avg";      pdata->hdata[1] = sum_p * invN;
  pdata->label[2] = "tau_sum";    pdata->hdata[2] = sum_tau;
  pdata->label[3] = "S1_sum";     pdata->hdata[3] = sum_s1;
  pdata->label[4] = "S2_sum";     pdata->hdata[4] = sum_s2;
  pdata->label[5] = "S3_sum";     pdata->hdata[5] = sum_s3;
  pdata->label[6] = "Edot_pred";  pdata->hdata[6] = edot_p;
  pdata->label[7] = "P1dot_pred"; pdata->hdata[7] = p1_p;
  pdata->label[8] = "P2dot_pred"; pdata->hdata[8] = p2_p;
  pdata->label[9] = "P3dot_pred"; pdata->hdata[9] = p3_p;
  pdata->label[10]= "Edot_fd";    pdata->hdata[10] = Edot_fd;
  pdata->label[11]= "P1dot_fd";   pdata->hdata[11] = P1_fd;
  pdata->label[12]= "P2dot_fd";   pdata->hdata[12] = P2_fd;
  pdata->label[13]= "P3dot_fd";   pdata->hdata[13] = P3_fd;
  pdata->label[14]= "relerr_E";   pdata->hdata[14] = relE;
  pdata->label[15]= "relerr_P1";  pdata->hdata[15] = relP1;
  pdata->label[16]= "Tmin_cgs";   pdata->hdata[16] = Tmin;
  pdata->label[17]= "frac_cool";  pdata->hdata[17] = n_active * invN;
}