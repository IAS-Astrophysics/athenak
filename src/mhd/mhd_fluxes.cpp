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
//! wl3d/wr3d buffers and the L/R cell-centered magnetic field (bcc0) in the
//! bl3d/br3d buffers, followed by (2) a per-face Riemann solve that reads those
//! buffers and writes both the interface flux and the two area-averaged EMF components.
//! All reconstruction methods (DC/PLM/PPM4/PPMX/WENOZ/TENO) and Riemann solvers
//! (Advect/LLF/HLLE/HLLD and the SR/GR variants) are supported; the reconstruction method
//! is chosen at runtime, the solver at compile time via the rsolver template parameter.
//!
//! Optional recon_region overwrite: after the global reconstruction kernels, faces whose
//! adjacent cells lie inside a user box/sphere have their L/R states replaced with an
//! alternate method (PLM or DC). This runs only on the reconstruction buffers; the
//! Riemann-solve stage is unchanged.
//!
//! Fluxes are stored in face-centered vector 'uflx', while electric fields are stored in
//! individual arrays: e2x1,e3x1 on x1-faces; e1x2,e3x2 on x2-faces;
//! e1x3,e2x3 on x3-faces.
//! Because constrained transport needs EMFs at every transverse cell edge, the flux/EMF
//! kernels run over a transverse range extended by one cell beyond the active domain.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
#include "reconstruct/recon.hpp"
#include "reconstruct/plm.hpp"
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
//! \fn InReconRegion
//! \brief Return true if cell (m,k,j,i) is inside the user-defined reconstruction region
//! (box or sphere in problem coordinates).

KOKKOS_INLINE_FUNCTION
bool InReconRegion(const int m, const int k, const int j, const int i,
                   const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
                   const bool use_box,
                   const Real rx1min, const Real rx1max, const Real rx2min,
                       const Real rx2max,
                   const Real rx3min, const Real rx3max,
                   const Real cx1, const Real cx2, const Real cx3, const Real r_radius) {
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  Real &x1min = size.d_view(m).x1min;
  Real &x1max = size.d_view(m).x1max;
  Real &x2min = size.d_view(m).x2min;
  Real &x2max = size.d_view(m).x2max;
  Real &x3min = size.d_view(m).x3min;
  Real &x3max = size.d_view(m).x3max;
  Real x1 = CellCenterX(i - is, nx1, x1min, x1max);
  Real x2 = CellCenterX(j - js, nx2, x2min, x2max);
  Real x3 = CellCenterX(k - ks, nx3, x3min, x3max);
  if (use_box) {
    return (x1 >= rx1min && x1 <= rx1max &&
            x2 >= rx2min && x2 <= rx2max &&
            x3 >= rx3min && x3 <= rx3max);
  } else {
    Real dx1 = x1 - cx1, dx2 = x2 - cx2, dx3 = x3 - cx3;
    return (dx1*dx1 + dx2*dx2 + dx3*dx3 <= r_radius*r_radius);
  }
}

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
  auto wl_ = wl3d;
  auto wr_ = wr3d;
  auto bl_ = bl3d;
  auto br_ = br3d;

  const bool use_recon_region_ = use_recon_region;
  const auto recon_region_method_ = recon_region_method;
  const bool recon_region_use_box_ = this->recon_region_use_box_;
  const Real recon_region_x1min_ = this->recon_region_x1min_;
  const Real recon_region_x1max_ = this->recon_region_x1max_;
  const Real recon_region_x2min_ = this->recon_region_x2min_;
  const Real recon_region_x2max_ = this->recon_region_x2max_;
  const Real recon_region_x3min_ = this->recon_region_x3min_;
  const Real recon_region_x3max_ = this->recon_region_x3max_;
  const Real recon_region_x1_center_ = this->recon_region_x1_center_;
  const Real recon_region_x2_center_ = this->recon_region_x2_center_;
  const Real recon_region_x3_center_ = this->recon_region_x3_center_;
  const Real recon_region_radius_ = this->recon_region_radius_;

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

    // Overwrite L/R states at faces inside recon_region (PLM or DC)
    if (use_recon_region_) {
      par_for("mflux_x1_recon_region", DevExeSpace(),
        0, nmb1, kl, ku, jl, ju, il1, iu1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          bool face_in_region = InReconRegion(m, k, j, i-1, indcs_, size_,
              recon_region_use_box_, recon_region_x1min_, recon_region_x1max_,
              recon_region_x2min_, recon_region_x2max_, recon_region_x3min_,
                  recon_region_x3max_,
              recon_region_x1_center_, recon_region_x2_center_, recon_region_x3_center_,
              recon_region_radius_) ||
              InReconRegion(m, k, j, i, indcs_, size_,
              recon_region_use_box_, recon_region_x1min_, recon_region_x1max_,
              recon_region_x2min_, recon_region_x2max_, recon_region_x3min_,
                  recon_region_x3max_,
              recon_region_x1_center_, recon_region_x2_center_, recon_region_x3_center_,
              recon_region_radius_);
          if (!face_in_region) return;
          if (recon_region_method_ == ReconstructionMethod::dc) {
            for (int n = 0; n < nvars; ++n) {
              wl_(m, n, k, j, i) = w0_(m, n, k, j, i-1);
              wr_(m, n, k, j, i) = w0_(m, n, k, j, i);
            }
            for (int b = 0; b < 3; ++b) {
              bl_(m, b, k, j, i) = bcc0_(m, b, k, j, i-1);
              br_(m, b, k, j, i) = bcc0_(m, b, k, j, i);
            }
          } else {
            Real ql_i, qr_im1, ql_ip1, qr_i;
            for (int n = 0; n < nvars; ++n) {
              PLM(w0_(m, n, k, j, i-2), w0_(m, n, k, j, i-1), w0_(m, n, k, j, i), ql_i,
                  qr_im1);
              wl_(m, n, k, j, i) = ql_i;
              PLM(w0_(m, n, k, j, i-1), w0_(m, n, k, j, i), w0_(m, n, k, j, i+1), ql_ip1,
                  qr_i);
              wr_(m, n, k, j, i) = qr_i;
            }
            for (int b = 0; b < 3; ++b) {
              PLM(bcc0_(m, b, k, j, i-2), bcc0_(m, b, k, j, i-1), bcc0_(m, b, k, j, i),
                  ql_i, qr_im1);
              bl_(m, b, k, j, i) = ql_i;
              PLM(bcc0_(m, b, k, j, i-1), bcc0_(m, b, k, j, i), bcc0_(m, b, k, j, i+1),
                  ql_ip1, qr_i);
              br_(m, b, k, j, i) = qr_i;
            }
          }
        });
    }

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

    if (use_recon_region_) {
      par_for("mflux_x2_recon_region", DevExeSpace(),
        0, nmb1, kl, ku, jl2, ju2, is-1, ie+1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          bool face_in_region = InReconRegion(m, k, j-1, i, indcs_, size_,
              recon_region_use_box_, recon_region_x1min_, recon_region_x1max_,
              recon_region_x2min_, recon_region_x2max_, recon_region_x3min_,
                  recon_region_x3max_,
              recon_region_x1_center_, recon_region_x2_center_,
                  recon_region_x3_center_,
              recon_region_radius_) ||
              InReconRegion(m, k, j, i, indcs_, size_,
              recon_region_use_box_, recon_region_x1min_, recon_region_x1max_,
              recon_region_x2min_, recon_region_x2max_, recon_region_x3min_,
                  recon_region_x3max_,
              recon_region_x1_center_, recon_region_x2_center_,
                  recon_region_x3_center_,
              recon_region_radius_);
          if (!face_in_region) return;
          if (recon_region_method_ == ReconstructionMethod::dc) {
            for (int n = 0; n < nvars; ++n) {
              wl_(m, n, k, j, i) = w0_(m, n, k, j-1, i);
              wr_(m, n, k, j, i) = w0_(m, n, k, j, i);
            }
            for (int b = 0; b < 3; ++b) {
              bl_(m, b, k, j, i) = bcc0_(m, b, k, j-1, i);
              br_(m, b, k, j, i) = bcc0_(m, b, k, j, i);
            }
          } else {
            Real ql_j, qr_jm1, ql_jp1, qr_j;
            for (int n = 0; n < nvars; ++n) {
              PLM(w0_(m, n, k, j-2, i), w0_(m, n, k, j-1, i), w0_(m, n, k, j, i),
                  ql_j, qr_jm1);
              wl_(m, n, k, j, i) = ql_j;
              PLM(w0_(m, n, k, j-1, i), w0_(m, n, k, j, i), w0_(m, n, k, j+1, i),
                  ql_jp1, qr_j);
              wr_(m, n, k, j, i) = qr_j;
            }
            for (int b = 0; b < 3; ++b) {
              PLM(bcc0_(m, b, k, j-2, i), bcc0_(m, b, k, j-1, i), bcc0_(m, b, k, j, i),
                  ql_j, qr_jm1);
              bl_(m, b, k, j, i) = ql_j;
              PLM(bcc0_(m, b, k, j-1, i), bcc0_(m, b, k, j, i), bcc0_(m, b, k, j+1, i),
                  ql_jp1, qr_j);
              br_(m, b, k, j, i) = qr_j;
            }
          }
        });
    }

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

    if (use_recon_region_) {
      par_for("mflux_x3_recon_region", DevExeSpace(),
        0, nmb1, kl3, ku3, js-1, je+1, is-1, ie+1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          bool face_in_region = InReconRegion(m, k-1, j, i, indcs_, size_,
              recon_region_use_box_, recon_region_x1min_, recon_region_x1max_,
              recon_region_x2min_, recon_region_x2max_, recon_region_x3min_,
                  recon_region_x3max_,
              recon_region_x1_center_, recon_region_x2_center_,
                  recon_region_x3_center_,
              recon_region_radius_) ||
              InReconRegion(m, k, j, i, indcs_, size_,
              recon_region_use_box_, recon_region_x1min_, recon_region_x1max_,
              recon_region_x2min_, recon_region_x2max_, recon_region_x3min_,
                  recon_region_x3max_,
              recon_region_x1_center_, recon_region_x2_center_,
                  recon_region_x3_center_,
              recon_region_radius_);
          if (!face_in_region) return;
          if (recon_region_method_ == ReconstructionMethod::dc) {
            for (int n = 0; n < nvars; ++n) {
              wl_(m, n, k, j, i) = w0_(m, n, k-1, j, i);
              wr_(m, n, k, j, i) = w0_(m, n, k, j, i);
            }
            for (int b = 0; b < 3; ++b) {
              bl_(m, b, k, j, i) = bcc0_(m, b, k-1, j, i);
              br_(m, b, k, j, i) = bcc0_(m, b, k, j, i);
            }
          } else {
            Real ql_k, qr_km1, ql_kp1, qr_k;
            for (int n = 0; n < nvars; ++n) {
              PLM(w0_(m, n, k-2, j, i), w0_(m, n, k-1, j, i), w0_(m, n, k, j, i),
                  ql_k, qr_km1);
              wl_(m, n, k, j, i) = ql_k;
              PLM(w0_(m, n, k-1, j, i), w0_(m, n, k, j, i), w0_(m, n, k+1, j, i),
                  ql_kp1, qr_k);
              wr_(m, n, k, j, i) = qr_k;
            }
            for (int b = 0; b < 3; ++b) {
              PLM(bcc0_(m, b, k-2, j, i), bcc0_(m, b, k-1, j, i), bcc0_(m, b, k, j, i),
                  ql_k, qr_km1);
              bl_(m, b, k, j, i) = ql_k;
              PLM(bcc0_(m, b, k-1, j, i), bcc0_(m, b, k, j, i), bcc0_(m, b, k+1, j, i),
                  ql_kp1, qr_k);
              br_(m, b, k, j, i) = qr_k;
            }
          }
        });
    }

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
