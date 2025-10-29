//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_fluxes.cpp
//! \brief Calculate fluxes of the conserved variables, and area-averaged electric fields
//! E = - (v X B) on cell faces for mhd.  Fluxes are stored in face-centered vector
//! 'uflx', while electric fields are stored in individual arrays: e2x1,e3x1 on x1-faces;
//! e1x2,e3x2 on x2-faces; e1x3,e2x3 on x3-faces.

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd.hpp"
#include "eos/eos.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "mhd/rsolvers/advect_mhd.hpp"
#include "mhd/rsolvers/llf_mhd.hpp"
#include "mhd/rsolvers/hlle_mhd.hpp"
#include "mhd/rsolvers/hlld_mhd.hpp"
#include "mhd/rsolvers/llf_srmhd.hpp"
#include "mhd/rsolvers/hlle_srmhd.hpp"
#include "mhd/rsolvers/llf_grmhd.hpp"
#include "mhd/rsolvers/hlle_grmhd.hpp"
// #include "mhd/rsolvers/roe_mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::CalculateFlux
//! \brief Calculate fluxes of conserved variables, and face-centered area-averaged EMFs
//! for evolution of magnetic field
//! Note this function is templated over RS for better performance on GPUs.

template <MHD_RSolver rsolver_method_>
void MHD::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs_ = pmy_pack->pmesh->mb_indcs;
  int is = indcs_.is, ie = indcs_.ie;
  int js = indcs_.js, je = indcs_.je;
  int ks = indcs_.ks, ke = indcs_.ke;
  int ncells1 = indcs_.nx1 + 2*(indcs_.ng);

  int &nmhd_ = nmhd;
  int nvars = nmhd + nscalars;
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
  auto &b0_ = bcc0;

  //--------------------------------------------------------------------------------------
  // i-direction

  // set the loop limits for 1D/2D/3D problems
  int jl,ju,kl,ku;
  if (pmy_pack->pmesh->one_d) {
    jl = js, ju = je, kl = ks, ku = ke;
  } else if (pmy_pack->pmesh->two_d) {
    jl = js-1, ju = je+1, kl = ks, ku = ke;
  } else {
    jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
  }
  int il = is, iu = ie+1;
  if (use_fofc) { il = is-1, iu = ie+2; }

  par_for("mhd_flux1",DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Reconstruct qR[i] and qL[i+1], for both W and Bcc
    PiecewiseLinear(m, k, j, i, w0_, wl, wr, 1);
    PiecewiseLinear(m, k, j, i, b0_, bl, br, 1);
    // compute fluxes over [is,ie+1].  MHD RS also computes electric fields, where
    // (IBY) component of flx = E_{z} = -(v x B)_{z} = -(v1*b2 - v2*b1)
    // (IBZ) component of flx = E_{y} = -(v x B)_{y} =  (v1*b3 - v3*b1)
    // NOTE(@pdmullen): Capture variables prior to if constexpr.  Required for cuda 11.6+.
    HLLD(eos_,indcs_,size_,coord_,m,k,j,i,IVX,wl,wr,bl,br,b0.x1f,uflx.x1f,e3x1,e2x1);
    });
  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmy_pack->pmesh->multi_d) {
    // set the loop limits for 2D/3D problems
    if (pmy_pack->pmesh->two_d) {
      kl = ks, ku = ke;
    } else { // 3D
      kl = ks-1, ku = ke+1;
    }
    jl = js-1, ju = je+1;
    if (use_fofc) { jl = js-2, ju = je+2; }

    
    par_for("mhd_flux2",DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Reconstruct qR[j] and qL[j+1], for both W and Bcc
        PiecewiseLinear(m, k, j, i, w0_, wl, wr, 2);
        PiecewiseLinear(m, k, j, i, b0_, bl, br, 2);
        // compute fluxes over [js,je+1].  MHD RS also computes electric fields, where
        // (IBY) component of flx = E_{x} = -(v x B)_{x} = -(v2*b3 - v3*b2)
        // (IBZ) component of flx = E_{z} = -(v x B)_{z} =  (v2*b1 - v1*b2)
        HLLD(eos_,indcs_,size_,coord_,m,k,j,i,IVY,wl,wr,bl,br,b0.x2f,uflx.x2f,e1x2,e3x2);
      });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    // set the loop limits
    kl = ks-1, ku = ke+1;
    if (use_fofc) { kl = ks-2, ku = ke+2; }

    par_for("mhd_flux3",DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {

        // Reconstruct qR[k] and qL[k+1], for both W and Bcc
        PiecewiseLinear(m, k, j, i, w0_, wl, wr, 3);
        PiecewiseLinear(m, k, j, i, b0_, bl, br, 3);
        HLLD(eos_,indcs_,size_,coord_,m,k,j,i,IVZ,wl,wr,bl,br,b0.x3f,uflx.x3f,e2x3,e1x3);
      });
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
