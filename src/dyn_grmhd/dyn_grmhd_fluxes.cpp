//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyn_grmhd_fluxes.cpp
//! \brief Calculate 3D fluxes (and area-averaged EMFs) for dynamical GRMHD.
//!
//! Uses the split-kernel approach: per direction, (1) a per-cell reconstruction kernel
//! materializes the L/R primitive states (w0) and L/R cell-centered magnetic field (bcc0)
//! into the global pmhd->wl3d/wr3d and bl3d/br3d buffers, followed by
//! (2) a per-face Riemann solve that reads those buffers and writes both the interface
//! conserved flux and the two area-averaged EMF components.  The fluid arrays live in the
//! MHD module, so the same split buffers allocated there are reused here.

#include <iostream>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd.hpp"
#include "dyn_grmhd_util.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "reconstruct/recon.hpp"
#include "dyn_grmhd/rsolvers/llf_dyn_grmhd.hpp"
#include "dyn_grmhd/rsolvers/hlle_dyn_grmhd.hpp"
// include PrimitiveSolver stuff
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn SolveFaceDynGR<rsolver,ivx>()
//! \brief Dispatch the (compile-time) dynamical-GRMHD Riemann solver for a single face,
//! writing the conserved flux and the two transverse EMF components.  Keeping the
//! constexpr-if branch inside a function (not the device lambda) avoids the CUDA 11.6+
//! "first-capture in constexpr-if context" restriction.
template <DynGRMHD_RSolver rsolver_method_, int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void SolveFaceDynGR(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
                    const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
                    const CoordData &coord,
                    const int m, const int k, const int j, const int i,
                    const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
                    const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
                    const DvceArray4D<Real> &bx,
                    const int nhyd, const int nscal,
                    const adm::ADM::ADM_vars& adm,
                    const DvceArray5D<Real> &flx, const DvceArray4D<Real> &ey,
                    const DvceArray4D<Real> &ez) {
  if constexpr (rsolver_method_ == DynGRMHD_RSolver::llf_dyngr) {
    LLF_DYNGR<ivx>(eos, indcs, size, coord, m, k, j, i, wl, wr, bl, br, bx,
                   nhyd, nscal, adm, flx, ey, ez);
  } else if constexpr (rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr) {
    HLLE_DYNGR<ivx>(eos, indcs, size, coord, m, k, j, i, wl, wr, bl, br, bx,
                    nhyd, nscal, adm, flx, ey, ez);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void DynGRMHDPS::CalcFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute dynamical GRMHD
//! fluxes.  Templated over the Riemann solver for better GPU performance.

template<class EOSPolicy, class ErrorPolicy> template <DynGRMHD_RSolver rsolver_method_>
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::CalcFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;

  int nhyd  = pmy_pack->pmhd->nmhd;
  int nvars = pmy_pack->pmhd->nmhd + pmy_pack->pmhd->nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = pmy_pack->pmhd->recon_method;

  auto &size_  = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_    = pmy_pack->pmhd->w0;
  auto &bcc0_  = pmy_pack->pmhd->bcc0;
  auto &eos_   = pmy_pack->pmhd->peos->eos_data;   // EOS_Data (recon floors; unused here)
  auto &dyn_eos_ = eos;                            // PrimitiveSolverHydro
  auto &adm_   = pmy_pack->padm->adm;
  bool use_fofc = pmy_pack->pmhd->use_fofc;

  auto wl_ = pmy_pack->pmhd->wl3d;
  auto wr_ = pmy_pack->pmhd->wr3d;
  auto bl_ = pmy_pack->pmhd->bl3d;
  auto br_ = pmy_pack->pmhd->br3d;

  // Short-circuit the flux calculation if everything is to be fixed.
  if (fixed_evolution) {
    return TaskStatus::complete;
  }

  //------------------------------------------------------------------------------------
  // x1 direction
  {
    auto &flx1 = pmy_pack->pmhd->uflx.x1f;
    auto &bx_  = pmy_pack->pmhd->b0.x1f;
    auto &e31  = pmy_pack->pmhd->e3x1;
    auto &e21  = pmy_pack->pmhd->e2x1;

    // CT-extended transverse range
    int jl = js, ju = je, kl = ks, ku = ke;
    if (pmy_pack->pmesh->multi_d) { jl = js-1; ju = je+1; }
    if (pmy_pack->pmesh->three_d) { kl = ks-1; ku = ke+1; }

    // face-normal range (extended by one cell each side when FOFC is enabled)
    int il = is, iu = ie+1;
    if (use_fofc) { il = is-1; iu = ie+2; }

    // Reconstruct W over cells i in [il-1, iu], variables n in [0, nvars-1]
    ReconDispatch<IVX>(recon_method_, "dyngrflux_x1_recon_w", nmb1,
        kl, ku, jl, ju, il-1, iu, eos_, false, nvars, w0_, wl_, wr_);
    // Reconstruct Bcc over cells i in [il-1, iu], components n in [0, 2]
    ReconDispatch<IVX>(recon_method_, "dyngrflux_x1_recon_b", nmb1,
        kl, ku, jl, ju, il-1, iu, eos_, false, 3,     bcc0_, bl_, br_);

    // Riemann solve over faces i in [il, iu]
    par_for("dyngrflux_x1_rsolve", DevExeSpace(),
      0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto dyn_eos = dyn_eos_;
        auto indcs = indcs_;  auto size = size_;  auto coord = coord_;  auto adm = adm_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = bx_;  auto flx = flx1;  auto eyl = e31;  auto ezl = e21;
        const int nh = nhyd, ns = nvars - nhyd;
        SolveFaceDynGR<rsolver_method_, IVX>(dyn_eos, indcs, size, coord, m, k, j, i,
                                             wl, wr, bl, br, bx, nh, ns, adm,
                                             flx, eyl, ezl);
      });

    // Scalar fluxes (upwind from sign of mass flux)
    if (nvars > nhyd) {
      par_for("dyngrflux_x1_scalars", DevExeSpace(),
        0, nmb1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nhyd; n < nvars; ++n) {
            if (flx1(m, IDN, k, j, i) >= 0.0) {
              flx1(m, n, k, j, i) = flx1(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx1(m, n, k, j, i) = flx1(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
        });
    }
  }

  //------------------------------------------------------------------------------------
  // x2 direction
  if (pmy_pack->pmesh->multi_d) {
    auto &flx2 = pmy_pack->pmhd->uflx.x2f;
    auto &by_  = pmy_pack->pmhd->b0.x2f;
    auto &e12  = pmy_pack->pmhd->e1x2;
    auto &e32  = pmy_pack->pmhd->e3x2;

    int kl = ks, ku = ke;
    if (pmy_pack->pmesh->three_d) { kl = ks-1; ku = ke+1; }

    // face-normal range (extended by one cell each side when FOFC is enabled)
    int jl = js, ju = je+1;
    if (use_fofc) { jl = js-1; ju = je+2; }

    // Reconstruct W over cells j in [jl-1, ju], i in [is-1, ie+1], n in [0, nvars-1]
    ReconDispatch<IVY>(recon_method_, "dyngrflux_x2_recon_w", nmb1,
        kl, ku, jl-1, ju, is-1, ie+1, eos_, false, nvars, w0_, wl_, wr_);
    // Reconstruct Bcc over cells j in [jl-1, ju], i in [is-1, ie+1], n in [0, 2]
    ReconDispatch<IVY>(recon_method_, "dyngrflux_x2_recon_b", nmb1,
        kl, ku, jl-1, ju, is-1, ie+1, eos_, false, 3,     bcc0_, bl_, br_);

    // Riemann solve over faces j in [jl, ju], i in [is-1, ie+1]
    par_for("dyngrflux_x2_rsolve", DevExeSpace(),
      0, nmb1, kl, ku, jl, ju, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto dyn_eos = dyn_eos_;
        auto indcs = indcs_;  auto size = size_;  auto coord = coord_;  auto adm = adm_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = by_;  auto flx = flx2;  auto eyl = e12;  auto ezl = e32;
        const int nh = nhyd, ns = nvars - nhyd;
        SolveFaceDynGR<rsolver_method_, IVY>(dyn_eos, indcs, size, coord, m, k, j, i,
                                             wl, wr, bl, br, bx, nh, ns, adm,
                                             flx, eyl, ezl);
      });

    if (nvars > nhyd) {
      par_for("dyngrflux_x2_scalars", DevExeSpace(),
        0, nmb1, kl, ku, jl, ju, is-1, ie+1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nhyd; n < nvars; ++n) {
            if (flx2(m, IDN, k, j, i) >= 0.0) {
              flx2(m, n, k, j, i) = flx2(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx2(m, n, k, j, i) = flx2(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
        });
    }
  }

  //------------------------------------------------------------------------------------
  // x3 direction
  if (pmy_pack->pmesh->three_d) {
    auto &flx3 = pmy_pack->pmhd->uflx.x3f;
    auto &bz_  = pmy_pack->pmhd->b0.x3f;
    auto &e23  = pmy_pack->pmhd->e2x3;
    auto &e13  = pmy_pack->pmhd->e1x3;

    // face-normal range (extended by one cell each side when FOFC is enabled)
    int kl = ks, ku = ke+1;
    if (use_fofc) { kl = ks-1; ku = ke+2; }

    // Reconstruct W over cells k in [kl-1, ku], j in [js-1, je+1], i in [is-1, ie+1],
    // variables n in [0, nvars-1]
    ReconDispatch<IVZ>(recon_method_, "dyngrflux_x3_recon_w", nmb1,
        kl-1, ku, js-1, je+1, is-1, ie+1, eos_, false, nvars, w0_, wl_, wr_);
    // Reconstruct Bcc over the same cells, components n in [0, 2]
    ReconDispatch<IVZ>(recon_method_, "dyngrflux_x3_recon_b", nmb1,
        kl-1, ku, js-1, je+1, is-1, ie+1, eos_, false, 3,     bcc0_, bl_, br_);

    // Riemann solve over faces k in [kl, ku], j in [js-1, je+1], i in [is-1, ie+1]
    par_for("dyngrflux_x3_rsolve", DevExeSpace(),
      0, nmb1, kl, ku, js-1, je+1, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto dyn_eos = dyn_eos_;
        auto indcs = indcs_;  auto size = size_;  auto coord = coord_;  auto adm = adm_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = bz_;  auto flx = flx3;  auto eyl = e23;  auto ezl = e13;
        const int nh = nhyd, ns = nvars - nhyd;
        SolveFaceDynGR<rsolver_method_, IVZ>(dyn_eos, indcs, size, coord, m, k, j, i,
                                             wl, wr, bl, br, bx, nh, ns, adm,
                                             flx, eyl, ezl);
      });

    if (nvars > nhyd) {
      par_for("dyngrflux_x3_scalars", DevExeSpace(),
        0, nmb1, kl, ku, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nhyd; n < nvars; ++n) {
            if (flx3(m, IDN, k, j, i) >= 0.0) {
              flx3(m, n, k, j, i) = flx3(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx3(m, n, k, j, i) = flx3(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
        });
    }
  }

  // Call FOFC if necessary
  if (pmy_pack->pmhd->use_fofc || pmy_pack->pcoord->coord_data.bh_excise) {
    FOFC<rsolver_method_>(pdriver, stage);
  }

  return TaskStatus::complete;
}

// function definitions for each template parameter
// Macro for instantiating every flux function for each Riemann solver
#define INSTANTIATE_CALC_FLUXES(EOSPolicy, ErrorPolicy) \
template \
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::\
            CalcFluxes<DynGRMHD_RSolver::llf_dyngr>(Driver *pdriver, int stage); \
template \
TaskStatus DynGRMHDPS<EOSPolicy, ErrorPolicy>::\
            CalcFluxes<DynGRMHD_RSolver::hlle_dyngr>(Driver *pdriver, int stage);

INSTANTIATE_CALC_FLUXES(Primitive::IdealGas, Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::PiecewisePolytrope, Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::EOSCompOSE<Primitive::NormalLogs>,
                        Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::EOSCompOSE<Primitive::NQTLogs>,
                        Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::EOSHybrid<Primitive::NormalLogs>,
                        Primitive::ResetFloor)
INSTANTIATE_CALC_FLUXES(Primitive::EOSHybrid<Primitive::NQTLogs>,
                        Primitive::ResetFloor)

} // namespace dyngr
