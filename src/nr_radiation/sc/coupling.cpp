//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coupling.cpp
//! \brief The gas-radiation coupling Q_rad: the face flux the differential forms need,
//! Q itself in its integral, differential and blended forms, the depletion timestep
//! limit it imposes, and its application to the fluid energy each RK stage,
//!   u0(IEN) += beta_dt * Q_rad   (Davis 2012 Eq. 26).

#include <float.h>

#include <cmath>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real FaceIntensity
//! \brief Upwind intensity of one ray at a physical boundary face, from its values in the
//! two cells either side (lo = lower index, hi = higher index) and its direction cosine
//! along the normal. Necessary because the outbound half of a ghost is not a physically
//! meaningful intensity. Used to form face H at physical boundaries. Interior faces are
//! formed from H in FillHflx.

KOKKOS_INLINE_FUNCTION
Real FaceIntensity(Real i_lo, Real i_hi, Real mu_n) {
  return (mu_n > 0.0) ? i_lo : i_hi;
}

//----------------------------------------------------------------------------------------
//! \fn bool IsPhysicalFace
//! \brief True when this meshblock face is a physical boundary of the domain, i.e. no
//! meshblock lies across it. mb_bcs holds block for a face with a neighbour and the mesh
//! flag at the domain edge -- including periodic, where the ghost is nonetheless a
//! neighbour's interior, so periodic is excluded along with block.

KOKKOS_INLINE_FUNCTION
bool IsPhysicalFace(BoundaryFlag f) {
  return (f != BoundaryFlag::block && f != BoundaryFlag::periodic);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void SC::FillHflx
//! \brief Face-centred first moment: x1f(i) = 0.5*(H1(i-1) + H1(i)), likewise x2f/x3f.

void SC::FillHflx() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int &ng = indcs.ng;
  // allocated on first use, so the integral form never pays for it
  int nc1 = indcs.nx1 + 2*ng;
  int nc2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int nc3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  if (hflx.x1f.extent_int(0) != (nmb1+1) || hflx.x1f.extent_int(4) != nc1) {
    Kokkos::realloc(hflx.x1f, (nmb1+1), 1, nc3, nc2, nc1);
    Kokkos::realloc(hflx.x2f, (nmb1+1), 1, nc3, nc2, nc1);
    Kokkos::realloc(hflx.x3f, (nmb1+1), 1, nc3, nc2, nc1);
    Kokkos::deep_copy(hflx.x1f, 0.0);
    Kokkos::deep_copy(hflx.x2f, 0.0);
    Kokkos::deep_copy(hflx.x3f, 0.0);
  }

  auto mom_ = moments;
  auto x1f_ = hflx.x1f;

  // slot 1 = H1; faces is..ie+1 bound the active cells
  par_for("sc_hflx_x1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    x1f_(m,0,k,j,i) = 0.5*(mom_(m,1,k,j,i-1) + mom_(m,1,k,j,i));
  });

  if (pmy_pack->pmesh->multi_d) {
    auto x2f_ = hflx.x2f;
    par_for("sc_hflx_x2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      x2f_(m,0,k,j,i) = 0.5*(mom_(m,2,k,j-1,i) + mom_(m,2,k,j,i));
    });
  }
  if (pmy_pack->pmesh->three_d) {
    auto x3f_ = hflx.x3f;
    par_for("sc_hflx_x3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      x3f_(m,0,k,j,i) = 0.5*(mom_(m,3,k-1,j,i) + mom_(m,3,k,j,i));
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::BoundaryHflx
//! \brief Overwrite the face flux on PHYSICAL faces with the per-ray upwind rule of
//! FaceIntensity: the average FillHflx wrote there folds in the invalid outbound half of
//! the boundary ghost.
//!
//! Launched over the face plane, not the volume: only the two bounding faces per
//! direction can be physical. Must follow FillHflx, whose values it overwrites.

void SC::BoundaryHflx() {
  auto &pm = pmy_pack->pmesh;
  if (pm->strictly_periodic) return;        // IsPhysicalFace is false everywhere; skip

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &mu = pang->mu;
  auto &wmu = pang->wmu;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  auto ir_ = ir;

  auto x1f_ = hflx.x1f;
  par_for("sc_hflx_bdry_x1", DevExeSpace(), 0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(int m, int k, int j) {
    for (int side = 0; side < 2; ++side) {
      const int face = (side == 0) ? BoundaryFace::inner_x1 : BoundaryFace::outer_x1;
      if (!IsPhysicalFace(mb_bcs.d_view(m, face))) continue;
      const int iface = (side == 0) ? is : ie + 1;
      Real flx = 0.0;
      for (int angg = 0; angg < nang_tot_; ++angg) {
        const int oct = angg / nang;
        const int a = angg - oct * nang;
        const Real mu_n = mu.d_view(oct, a, 0);
        flx += wmu.d_view(a) * mu_n
             * FaceIntensity(ir_(m, angg, k, j, iface - 1),
                             ir_(m, angg, k, j, iface), mu_n);
      }
      x1f_(m, 0, k, j, iface) = flx;
    }
  });

  if (pm->multi_d) {
    auto x2f_ = hflx.x2f;
    par_for("sc_hflx_bdry_x2", DevExeSpace(), 0, nmb1, ks, ke, is, ie,
    KOKKOS_LAMBDA(int m, int k, int i) {
      for (int side = 0; side < 2; ++side) {
        const int face = (side == 0) ? BoundaryFace::inner_x2 : BoundaryFace::outer_x2;
        if (!IsPhysicalFace(mb_bcs.d_view(m, face))) continue;
        const int jface = (side == 0) ? js : je + 1;
        Real flx = 0.0;
        for (int angg = 0; angg < nang_tot_; ++angg) {
          const int oct = angg / nang;
          const int a = angg - oct * nang;
          const Real mu_n = mu.d_view(oct, a, 1);
          flx += wmu.d_view(a) * mu_n
               * FaceIntensity(ir_(m, angg, k, jface - 1, i),
                               ir_(m, angg, k, jface, i), mu_n);
        }
        x2f_(m, 0, k, jface, i) = flx;
      }
    });
  }

  if (pm->three_d) {
    auto x3f_ = hflx.x3f;
    par_for("sc_hflx_bdry_x3", DevExeSpace(), 0, nmb1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
      for (int side = 0; side < 2; ++side) {
        const int face = (side == 0) ? BoundaryFace::inner_x3 : BoundaryFace::outer_x3;
        if (!IsPhysicalFace(mb_bcs.d_view(m, face))) continue;
        const int kface = (side == 0) ? ks : ke + 1;
        Real flx = 0.0;
        for (int angg = 0; angg < nang_tot_; ++angg) {
          const int oct = angg / nang;
          const int a = angg - oct * nang;
          const Real mu_n = mu.d_view(oct, a, 2);
          flx += wmu.d_view(a) * mu_n
               * FaceIntensity(ir_(m, angg, kface - 1, j, i),
                               ir_(m, angg, kface, j, i), mu_n);
        }
        x3f_(m, 0, kface, j, i) = flx;
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::BuildFaceFlux
//! \brief The face flux the differential forms difference: FillHflx, BoundaryHflx, and
//! on a multilevel mesh the coarse-fine correction task list, which is why this stage
//! takes the Driver.

void SC::BuildFaceFlux(Driver *pdrive) {
  FillHflx();
  BoundaryHflx();
  if (pmy_pack->pmesh->multilevel) {
    pdrive->ExecuteTaskList(pmy_pack->pmesh, "sc_flxcor", 0);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real DivH
//! \brief Divergence of the face flux over cell (m,k,j,i) in the dimensions the run has.

KOKKOS_INLINE_FUNCTION
Real DivH(const DvceArray5D<Real> &x1f, const DvceArray5D<Real> &x2f,
          const DvceArray5D<Real> &x3f, const RegionSize &sz,
          int m, int k, int j, int i, bool multi_d, bool three_d) {
  Real d = (x1f(m,0,k,j,i+1) - x1f(m,0,k,j,i)) / sz.dx1;
  if (multi_d) d += (x2f(m,0,k,j+1,i) - x2f(m,0,k,j,i)) / sz.dx2;
  if (three_d) d += (x3f(m,0,k+1,j,i) - x3f(m,0,k,j,i)) / sz.dx3;
  return d;
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeQrad
//! \brief Gas-radiation coupling in every active cell,
//!   Q = (1 - w) Q_int + w Q_dif,   Q_int =  C sigma_a (J - brad),   Q_dif = -C div(H),
//! with C = crat * prat and the weight fixed by qrad_form: w = 0 (integral), w = 1
//! (divh), or a smoothstep in log tau over [kTauLo, kTauHi] with tau = chi min(dx)
//! (hybrid).
//! Q_int and Q_dif agree in exact arithmetic; Q_int loses precision where the zone is
//! thick and near equilibrium, Q_dif is noisy where sigma_a -> 0. Q_dif is built from the
//! face flux so Q is conservative where w = 1.

void SC::ComputeQrad() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto sigma_a_ = sigma_a;
  auto chi_ = chi;
  auto brad_ = brad;
  auto mom_ = moments;
  auto qrad_ = qrad;
  auto x1f_ = hflx.x1f;
  auto x2f_ = hflx.x2f;
  auto x3f_ = hflx.x3f;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const Real crat_prat = crat * prat;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const Real tau_lo = kTauLo, tau_hi = kTauHi;
  const Real inv_decade = 1.0/(std::log10(kTauHi) - std::log10(kTauLo));
  const Real log_tau_lo = std::log10(kTauLo);
  const bool w_zero = (qrad_form == QradForm::integral);   // hflx is not built
  const bool w_one = (qrad_form == QradForm::divh);

  par_for("sc_compute_qrad", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real qint = crat_prat * sigma_a_(m,0,k,j,i)
                      * (mom_(m,0,k,j,i) - brad_(m,k,j,i));
    if (w_zero) {
      qrad_(m,k,j,i) = qint;
      return;
    }
    const Real qdif = -crat_prat * DivH(x1f_, x2f_, x3f_, mbsize.d_view(m),
                                        m, k, j, i, multi_d, three_d);
    if (w_one) {
      qrad_(m,k,j,i) = qdif;
      return;
    }
    Real dxmin = mbsize.d_view(m).dx1;
    if (multi_d) dxmin = fmin(dxmin, mbsize.d_view(m).dx2);
    if (three_d) dxmin = fmin(dxmin, mbsize.d_view(m).dx3);
    const Real tau = chi_(m,k,j,i) * dxmin;
    if (tau <= tau_lo) {
      qrad_(m,k,j,i) = qint;
    } else if (tau >= tau_hi) {
      qrad_(m,k,j,i) = qdif;
    } else {
      const Real t = (log10(tau) - log_tau_lo) * inv_decade;   // in [0,1]
      const Real w = t*t*(3.0 - 2.0*t);                        // smoothstep
      qrad_(m,k,j,i) = (1.0 - w)*qint + w*qdif;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::LimitDtByQrad
//! \brief Reduce Mesh::dt so that Q_rad removes at most cfl_qrad of any cell's internal
//! energy this step:  dt <= cfl_qrad * e_int / |Q|.
//!
//! Called at the end of SolveTransfer (before_timeintegrator) so it bounds the step that
//! applies THIS cycle's Q; in NewTimeStep (stagen) it would act one cycle late and be
//! inert on cycle 0, after a restart and after a remesh. Independent of the relaxation
//! limit in sc/newdt.cpp, which bounds Eq. 27 but not Eq. 28. Only ever lowers dt.

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

  // Determine if the update would drive internal energy negative; count such cells in
  // ecounter.
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

}  // namespace nr_radiation
