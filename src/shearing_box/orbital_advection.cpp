//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file orbital advection.cpp
//! \brief Functions to update cell-centered and face-centered quantities via orbital
//! advection

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro.hpp"
#include "eos/eos.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"

namespace shearing_box {
//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::CalculateFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute hydro fluxes
//! Note this function is templated over RS for better performance on GPUs.

void ShearingBox::OrbitalAdvectionCC(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;
  int ncells1 = indcs_.nx1 + 2*(indcs_.ng);

  int &nhyd_  = nhydro;
  int nvars = nhydro + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  bool extrema = false;
  if (recon_method == ReconstructionMethod::ppmx) {
    extrema = true;
  }

  auto &eos_ = peos->eos_data;
  auto &size_ = pmy_pack->pmb->mb_size;
  auto &coord_ = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;

  //--------------------------------------------------------------------------------------
  // j-direction

  scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
  auto &flx2_ = uflx.x2f;

  // set the loop limits for 1D/2D/3D problems
  il = is, iu = ie, jl = js-1, ju = je+1, kl = ks, ku = ke;
  if (use_fofc) {
    jl = js-2, ju = je+2;
    if (pmy_pack->pmesh->two_d) {
      il = is-1, iu = ie+1, kl = ks, ku = ke;
    } else {
      il = is-1, iu = ie+1, kl = ks-1, ku = ke+1;
    }
  }

  par_for_outer("hflux_x2",DevExeSpace(), scr_size, scr_level, 0, nmb1, kl, ku,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k) {
    ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);

    for (int j=jl; j<=ju; ++j) {
      // Permute scratch arrays.
      auto wl     = scr1;
      auto wl_jp1 = scr2;
      auto wr     = scr3;
      if ((j%2) == 0) {
        wl     = scr2;
        wl_jp1 = scr1;
      }

      // Compute x2-fluxes from fractional offset
      switch (recon_method_) {
        case ReconstructionMethod::dc:
          DonorCellOrbAdvFlx(member, m, k, j, il, iu, w0_, wl_jp1, wr);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinearOrbAdvFlx(member, m, k, j, il, iu, w0_, wl_jp1, wr);
          break;
        case ReconstructionMethod::ppm4:
        case ReconstructionMethod::ppmx:
          PiecewiseParabolicOrbAdvFlx(member,eos_,extrema,true,m,k,j,il,iu, w0_, wl_jp1, wr);
          break;
        default:
          break;
      }
      member.team_barrier();

      // compute fluxes over [js,je+1].  RS returns flux in input wr array
      if (j>jl) {
        // NOTE(@pdmullen): Capture variables prior to if constexpr.
        auto eos = eos_;
        auto indcs = indcs_;
        auto size = size_;
        auto coord = coord_;
        auto flx2 = flx2_;
        Advect(member, eos, indcs, size, coord, m, k, j, il, iu, IVY, wl, wr, flx2);
        member.team_barrier();
      }

      // calculate fluxes of scalars (if any)
      if (nvars > nhyd_) {
        for (int n=nhyd_; n<nvars; ++n) {
          par_for_inner(member, is, ie, [&](const int i) {
            if (flx2_(m,IDN,k,j,i) >= 0.0) {
              flx2_(m,n,k,j,i) = flx2_(m,IDN,k,j,i)*wl(n,i);
            } else {
              flx2_(m,n,k,j,i) = flx2_(m,IDN,k,j,i)*wr(n,i);
            }
          });
        }
      }
    } // end of loop over j
  // Update CC variables with orbital advection fluxes

  });


  return;
}

} // namespace shearing_box
