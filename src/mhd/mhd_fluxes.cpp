//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes.cpp
//! \brief Calculate 3D fluxes of the conserved variables, and area-averaged electric
//! fields E = - (v X B) on cell faces for mhd.
//!
//! Fluxes are computed with two 1D-RangePolicy kernels per direction: (1) a per-cell
//! reconstruction kernel that materializes the L/R primitive states (w0) in the global
//! wl_split/wr_split buffers and the L/R cell-centered magnetic field (bcc0) in the
//! bl_split/br_split buffers, followed by (2) a per-face Riemann solve that reads those
//! buffers and writes both the interface flux and the two area-averaged EMF components.
//! All reconstruction methods (DC/PLM/PPM4/PPMX/WENOZ) and Riemann solvers
//! (Advect/LLF/HLLE/HLLD and the SR/GR variants) are supported; the reconstruction method
//! is chosen at runtime, the solver at compile time via the rsolver template parameter.
//!
//! Fluxes are stored in face-centered vector 'uflx', while electric fields are stored in
//! individual arrays: e2x1,e3x1 on x1-faces; e1x2,e3x2 on x2-faces;
//! e1x3,e2x3 on x3-faces.
//! Because constrained transport needs EMFs at every transverse cell edge, the flux/EMF
//! kernels run over a transverse range extended by one cell beyond the active domain.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
#include "reconstruct/recon.hpp"
#include "mhd/rsolvers/advect_mhd.hpp"
#include "mhd/rsolvers/llf_mhd.hpp"
#include "mhd/rsolvers/hlle_mhd.hpp"
#include "mhd/rsolvers/hlld_mhd.hpp"
#include "mhd/rsolvers/llf_srmhd.hpp"
#include "mhd/rsolvers/hlle_srmhd.hpp"
#include "mhd/rsolvers/llf_grmhd.hpp"
#include "mhd/rsolvers/hlle_grmhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn SolveFaceMHD<rsolver,ivx>()
//! \brief Dispatch the (compile-time) MHD Riemann solver for a single face, writing the
//! conserved flux and the two transverse EMF components.  Capturing the solver inputs
//! into locals before the constexpr-if is required for CUDA 11.6+.
template <MHD_RSolver rsolver_method_, int ivx>
KOKKOS_INLINE_FUNCTION
void SolveFaceMHD(const EOS_Data &eos, const RegionIndcs &indcs,
                  const DualArray1D<RegionSize> &size, const CoordData &coord,
                  const int m, const int k, const int j, const int i,
                  const int is, const int js, const int ks,
                  const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
                  const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
                  const DvceArray4D<Real> &bx,
                  const DvceArray5D<Real> &flx,
                  const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez) {
  if constexpr (rsolver_method_ == MHD_RSolver::advect) {
    Advect<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::llf) {
    LLF<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlle) {
    HLLE<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlld) {
    HLLD<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::llf_sr) {
    LLF_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_sr) {
    HLLE_SR<ivx>(eos, m, k, j, i, is, js, ks, wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::llf_gr) {
    LLF_GR<ivx>(eos, indcs, size, coord, m, k, j, i, is, js, ks,
                wl, wr, bl, br, bx, flx, ey, ez);
  } else if constexpr (rsolver_method_ == MHD_RSolver::hlle_gr) {
    HLLE_GR<ivx>(eos, indcs, size, coord, m, k, j, i, is, js, ks,
                 wl, wr, bl, br, bx, flx, ey, ez);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::CalculateFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute MHD fluxes and
//! face-centered area-averaged EMFs.  Templated over the Riemann solver for GPU perf.

template <MHD_RSolver rsolver_method_>
void MHD::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;

  int &nmhd_ = nmhd;
  int nvars = nmhd + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;

  // Face-normal flux range. With FOFC enabled the first-order flux correction needs the
  // main fluxes/EMFs one cell beyond the active domain, so each direction's face-normal
  // range is extended by one cell on both sides (transverse ranges already cover the CT
  // edge, so they are unchanged).
  int il1 = is, iu1 = ie+1, jl2 = js, ju2 = je+1, kl3 = ks, ku3 = ke+1;
  if (use_fofc) {
    il1 = is-1; iu1 = ie+2;
    jl2 = js-1; ju2 = je+2;
    kl3 = ks-1; ku3 = ke+2;
  }

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;
  auto &bcc0_ = bcc0;
  auto wl_ = wl_split;
  auto wr_ = wr_split;
  auto bl_ = bl_split;
  auto br_ = br_split;

  //------------------------------------------------------------------------------------
  // x1 direction
  {
    auto &flx1 = uflx.x1f;
    auto &bx_ = b0.x1f;
    auto &e31 = e3x1;
    auto &e21 = e2x1;

    // CT-extended transverse range
    int jl = js, ju = je, kl = ks, ku = ke;
    if (pmy_pack->pmesh->multi_d) { jl = js-1; ju = je+1; }
    if (pmy_pack->pmesh->three_d) { kl = ks-1; ku = ke+1; }

    // Reconstruct W over cells i in [il1-1, iu1], variables n in [0, nvars-1]
    ReconDispatch<IVX>(recon_method_, "mflux_x1_recon_w", nmb1,
        kl, ku, jl, ju, il1-1, iu1, eos_, true,  nvars, w0_, wl_, wr_);
    // Reconstruct Bcc over cells i in [il1-1, iu1], components n in [0, 2]
    ReconDispatch<IVX>(recon_method_, "mflux_x1_recon_b", nmb1,
        kl, ku, jl, ju, il1-1, iu1, eos_, false, 3,     bcc0_, bl_, br_);

    // Riemann solve over faces i in [il1, iu1]
    par_for("mflux_x1_rsolve", DevExeSpace(),
      0, nmb1, kl, ku, jl, ju, il1, iu1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = bx_;  auto flx = flx1;  auto eyl = e31;  auto ezl = e21;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFaceMHD<rsolver_method_, IVX>(eos, indcs, size, coord,
                                           m, k, j, i, is_, js_, ks_,
                                           wl, wr, bl, br, bx, flx, eyl, ezl);
      });

    // Scalar fluxes (upwind from sign of mass flux), active transverse range
    if (nvars > nmhd_) {
      par_for("mflux_x1_scalars", DevExeSpace(),
        0, nmb1, ks, ke, js, je, is, ie+1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nmhd_; n < nvars; ++n) {
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
    auto &flx2 = uflx.x2f;
    auto &by_ = b0.x2f;
    auto &e12 = e1x2;
    auto &e32 = e3x2;

    int kl = ks, ku = ke;
    if (pmy_pack->pmesh->three_d) { kl = ks-1; ku = ke+1; }

    // Reconstruct W over cells j in [jl2-1, ju2], i in [is-1, ie+1], n in [0, nvars-1]
    ReconDispatch<IVY>(recon_method_, "mflux_x2_recon_w", nmb1,
        kl, ku, jl2-1, ju2, is-1, ie+1, eos_, true,  nvars, w0_, wl_, wr_);
    // Reconstruct Bcc over cells j in [jl2-1, ju2], i in [is-1, ie+1], n in [0, 2]
    ReconDispatch<IVY>(recon_method_, "mflux_x2_recon_b", nmb1,
        kl, ku, jl2-1, ju2, is-1, ie+1, eos_, false, 3,     bcc0_, bl_, br_);

    // Riemann solve over faces j in [jl2, ju2], i in [is-1, ie+1]
    par_for("mflux_x2_rsolve", DevExeSpace(),
      0, nmb1, kl, ku, jl2, ju2, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = by_;  auto flx = flx2;  auto eyl = e12;  auto ezl = e32;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFaceMHD<rsolver_method_, IVY>(eos, indcs, size, coord,
                                           m, k, j, i, is_, js_, ks_,
                                           wl, wr, bl, br, bx, flx, eyl, ezl);
      });

    if (nvars > nmhd_) {
      par_for("mflux_x2_scalars", DevExeSpace(),
        0, nmb1, ks, ke, js, je+1, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nmhd_; n < nvars; ++n) {
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
    auto &flx3 = uflx.x3f;
    auto &bz_ = b0.x3f;
    auto &e23 = e2x3;
    auto &e13 = e1x3;

    // Reconstruct W over cells k in [kl3-1, ku3], j in [js-1, je+1], i in [is-1, ie+1],
    // variables n in [0, nvars-1]
    ReconDispatch<IVZ>(recon_method_, "mflux_x3_recon_w", nmb1,
        kl3-1, ku3, js-1, je+1, is-1, ie+1, eos_, true,  nvars, w0_, wl_, wr_);
    // Reconstruct Bcc over the same cells, components n in [0, 2]
    ReconDispatch<IVZ>(recon_method_, "mflux_x3_recon_b", nmb1,
        kl3-1, ku3, js-1, je+1, is-1, ie+1, eos_, false, 3,     bcc0_, bl_, br_);

    // Riemann solve over faces k in [kl3, ku3], j in [js-1, je+1], i in [is-1, ie+1]
    par_for("mflux_x3_rsolve", DevExeSpace(),
      0, nmb1, kl3, ku3, js-1, je+1, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto wl = wl_;  auto wr = wr_;  auto bl = bl_;  auto br = br_;
        auto bx = bz_;  auto flx = flx3;  auto eyl = e23;  auto ezl = e13;
        const int is_ = is, js_ = js, ks_ = ks;
        SolveFaceMHD<rsolver_method_, IVZ>(eos, indcs, size, coord,
                                           m, k, j, i, is_, js_, ks_,
                                           wl, wr, bl, br, bx, flx, eyl, ezl);
      });

    if (nvars > nmhd_) {
      par_for("mflux_x3_scalars", DevExeSpace(),
        0, nmb1, ks, ke+1, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          for (int n = nmhd_; n < nvars; ++n) {
            if (flx3(m, IDN, k, j, i) >= 0.0) {
              flx3(m, n, k, j, i) = flx3(m, IDN, k, j, i) * wl_(m, n, k, j, i);
            } else {
              flx3(m, n, k, j, i) = flx3(m, IDN, k, j, i) * wr_(m, n, k, j, i);
            }
          }
        });
    }
  }

  return;
}

// function definitions for each template parameter
template void MHD::CalculateFluxes<MHD_RSolver::advect>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlld>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf_sr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::llf_gr>(Driver *pdriver, int stage);
template void MHD::CalculateFluxes<MHD_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace mhd
