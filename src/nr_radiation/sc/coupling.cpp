//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coupling.cpp
//! \brief Apply radiative heating/cooling to the fluid energy equation each RK stage:
//!   u0(IEN) += beta_dt * Q_rad   (Davis 2012 Eq. 26 with Q from Eq. 27 or Eq. 28),
//! and impose the Q_rad energy-depletion timestep limit that bounds that update.

#include <float.h>

#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SC::AddQrad
//! \brief Inserted after hydro/mhd RKUpdate, before HydroSrcTerms / MHDSrcTerms.

TaskStatus SC::AddQrad(Driver *pdrive, int stage) {
  if (!affect_fluid) return TaskStatus::complete;

  Real beta_dt = (pdrive->beta[stage-1]) * (pmy_pack->pmesh->dt);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto qrad_ = qrad;
  DvceArray5D<Real> u0;   // exactly one fluid exists when this task is registered
  if (pmy_pack->phydro != nullptr) {
    u0 = pmy_pack->phydro->u0;
  } else {
    u0 = pmy_pack->pmhd->u0;
  }
  if (cfl_qrad <= 0.0) {
    par_for("sc_addqrad", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IEN,k,j,i) += beta_dt * qrad_(m,k,j,i);
    });
    return TaskStatus::complete;
  }

  // Armed: the same update, plus a count of the cells whose internal energy it would
  // drive non-positive. LimitDtByQrad bounds this with the Q of THIS cycle, but Q is
  // frozen across the RK stages while the gas state moves, so a non-zero count says the
  // margin was too thin. Reported through Mesh::ecounter, as the EOS floors are.
  // w0(IEN) is the internal energy; u0(IEN) is the total, and a cell can keep u0 > 0
  // while its internal energy has already gone negative -- that conflation is exactly
  // what hides this failure in Athena-C.
  DvceArray5D<Real> w0;
  if (pmy_pack->phydro != nullptr) {
    w0 = pmy_pack->phydro->w0;
  } else {
    w0 = pmy_pack->pmhd->w0;
  }
  const int ni = ie - is + 1;
  const int nji = (je - js + 1)*ni;
  const int nkji = (ke - ks + 1)*nji;
  const int nmkji = (nmb1 + 1)*nkji;
  int nneg_ = 0;
  Kokkos::parallel_reduce("sc_addqrad", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, int &nneg) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + is;
    j += js;
    k += ks;
    Real dq = beta_dt * qrad_(m,k,j,i);
    if (w0(m,IEN,k,j,i) + dq <= 0.0) { nneg++; }
    u0(m,IEN,k,j,i) += dq;
  }, Kokkos::Sum<int>(nneg_));
  pmy_pack->pmesh->ecounter.nqrad_neg += nneg_;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void SC::LimitDtByQrad
//! \brief Reduce Mesh::dt so that Q_rad removes at most cfl_qrad of any cell's internal
//! energy this step:  dt <= cfl_qrad * e_int / |Q|.
//!
//! Called at the end of SolveTransfer, i.e. in "before_timeintegrator", before any RK
//! stage has run. That placement is Athena-C's (radtrans_dt is called between
//! formal_solution and rad_to_hydro, main.c:687-689) and it matters: put this in
//! NewTimeStep instead, which runs in "stagen", and it would bound the NEXT cycle with
//! THIS cycle's Q -- leaving it inert on cycle 0, where qrad is still zero, after every
//! restart, and after every remesh, where qrad is not carried with the blocks.
//!
//! This is a second, independent upper bound, min-combined with the relaxation limit of
//! sc/newdt.cpp -- never multiplied by it. The relaxation limit is a linearised *rate*
//! and routes through sigma_a; it therefore bounds Eq. 27 (in the thin cooling limit
//! 1/nu_rad = (1/4) e_int/|Q^int| identically) but not Eq. 28, whose magnitude is set by
//! the radiation field.
//!
//! Only ever lowers dt, so the tlim clamp and the 2x growth cap in Mesh::NewTimeStep
//! cannot be violated, and the next cycle's cap keys off the reduced value.

void SC::LimitDtByQrad() {
  if (cfl_qrad <= 0.0 || !affect_fluid) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.ie - is + 1;
  int nx2 = indcs.je - js + 1;
  int nx3 = indcs.ke - ks + 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmkji = (nmb1 + 1)*nkji;

  DvceArray5D<Real> w0;
  if (pmy_pack->phydro != nullptr) {
    w0 = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    w0 = pmy_pack->pmhd->w0;
  } else {
    return;   // no fluid: Q is never applied
  }
  auto qrad_ = qrad;
  const Real cfl_q = cfl_qrad;
  Real dt_q = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("sc_qrad_dt",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int idx, Real &ldt) {
      int m = idx/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      j += js;
      k += ks;
      Real eint = w0(m,IEN,k,j,i);
      if (eint <= 0.0) return;   // already broken; leave it to the EOS floor, and do not
                                 // feed a non-positive dt into the reduction
      // srcterms/srcterms_newdt.cpp idiom: a FLT_MIN floor on the rate, so a cell with
      // Q == 0 contributes eint/FLT_MIN and never wins the minimum
      Real qmag = FLT_MIN + fabs(qrad_(m,k,j,i));
      ldt = fmin(ldt, cfl_q * eint / qmag);
    }, Kokkos::Min<Real>(dt_q));

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &dt_q, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  // cfl_no scales every module's timescale in Mesh::NewTimeStep; apply it here too so
  // this bound carries the same single safety factor as 1/nu_rad does
  Real dt_lim = (pmy_pack->pmesh->cfl_no)*dt_q;
  if (dt_lim < pmy_pack->pmesh->dt) { pmy_pack->pmesh->dt = dt_lim; }
}

}  // namespace nr_radiation
