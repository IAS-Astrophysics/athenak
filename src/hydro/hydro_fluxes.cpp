//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes.cpp
//! \brief Calculate 3D fluxes for hydro

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
#include "hydro/rsolvers/advect_hyd.hpp"
#include "hydro/rsolvers/llf_hyd.hpp"
#include "hydro/rsolvers/hlle_hyd.hpp"
#include "hydro/rsolvers/hllc_hyd.hpp"
#include "hydro/rsolvers/roe_hyd.hpp"
#include "hydro/rsolvers/llf_srhyd.hpp"
#include "hydro/rsolvers/hlle_srhyd.hpp"
#include "hydro/rsolvers/hllc_srhyd.hpp"
#include "hydro/rsolvers/llf_grhyd.hpp"
#include "hydro/rsolvers/hlle_grhyd.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void Hydro::CalculateFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute hydro fluxes
//! Note this function is templated over RS for better performance on GPUs.

template <Hydro_RSolver rsolver_method_>
void Hydro::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);

  int nhyd  = nhydro;
  int nvars = nhydro + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  bool extrema = false;
  if (recon_method == ReconstructionMethod::ppmx) {
    extrema = true;
  }

  auto &eos = peos->eos_data;
  auto &size = pmy_pack->pmb->mb_size;
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &w0_ = w0;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 2;
  int scr_level = 0;
  auto flx1 = uflx.x1f;

  par_for_outer("hflux_x1",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
    ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);

    // Reconstruct qR[i] and qL[i+1]
    switch (recon_method_) {
      case ReconstructionMethod::dc:
        DonorCellX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
        break;
      case ReconstructionMethod::plm:
        PiecewiseLinearX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
        PiecewiseParabolicX1(member,eos,extrema,true, m, k, j, is-1, ie+1, w0_, wl, wr);
        break;
      case ReconstructionMethod::wenoz:
        WENOZX1(member, eos, true, m, k, j, is-1, ie+1, w0_, wl, wr);
        break;
      default:
        break;
    }
    // Sync all threads in the team so that scratch memory is consistent
    member.team_barrier();

    // compute fluxes over [is,ie+1]
    if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
      Advect(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
      LLF(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
      HLLE(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
      HLLC(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
      Roe(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
      LLF_SR(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
      HLLE_SR(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
      HLLC_SR(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
      LLF_GR(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
      HLLE_GR(member, eos, indcs, size, coord, m, k, j, is, ie+1, IVX, wl, wr, flx1);
    }
    member.team_barrier();

    // calculate fluxes of scalars (if any)
    if (nvars > nhyd) {
      for (int n=nhyd; n<nvars; ++n) {
        par_for_inner(member, is, ie+1, [&](const int i) {
          if (flx1(m,IDN,k,j,i) >= 0.0) {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wl(n,i);
          } else {
            flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wr(n,i);
          }
        });
      }
    }
  });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
    auto flx2 = uflx.x2f;

    par_for_outer("hflux_x2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k) {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);

      for (int j=js-1; j<=je+1; ++j) {
        // Permute scratch arrays.
        auto wl     = scr1;
        auto wl_jp1 = scr2;
        auto wr     = scr3;
        if ((j%2) == 0) {
          wl     = scr2;
          wl_jp1 = scr1;
        }

        // Reconstruct qR[j] and qL[j+1]
        switch (recon_method_) {
          case ReconstructionMethod::dc:
            DonorCellX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX2(member,eos,extrema,true,m,k,j,is,ie, w0_, wl_jp1, wr);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX2(member, eos, true, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [js,je+1].  RS returns flux in input wr array
        if (j>(js-1)) {
          if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
            Advect(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
            LLF(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
            HLLE(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
            HLLC(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
            Roe(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
            LLF_SR(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
            HLLE_SR(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
            HLLC_SR(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
            LLF_GR(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
            HLLE_GR(member, eos, indcs, size, coord, m, k, j, is, ie, IVY, wl, wr, flx2);
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nhyd) {
          for (int n=nhyd; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i) {
              if (flx2(m,IDN,k,j,i) >= 0.0) {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }
      } // end of loop over j
    });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
    auto flx3 = uflx.x3f;

    par_for_outer("hflux_x3",DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray2D<Real> scr1(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr2(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> scr3(member.team_scratch(scr_level), nvars, ncells1);

      for (int k=ks-1; k<=ke+1; ++k) {
        // Permute scratch arrays.
        auto wl     = scr1;
        auto wl_kp1 = scr2;
        auto wr     = scr3;
        if ((k%2) == 0) {
          wl     = scr2;
          wl_kp1 = scr1;
        }

        // Reconstruct qR[k] and qL[k+1]
        switch (recon_method_) {
          case ReconstructionMethod::dc:
            DonorCellX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
            break;
          case ReconstructionMethod::plm:
            PiecewiseLinearX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
            break;
          case ReconstructionMethod::ppm4:
          case ReconstructionMethod::ppmx:
            PiecewiseParabolicX3(member,eos,extrema,true,m,k,j,is,ie, w0_, wl_kp1, wr);
            break;
          case ReconstructionMethod::wenoz:
            WENOZX3(member, eos, true, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
            break;
          default:
            break;
        }
        member.team_barrier();

        // compute fluxes over [ks,ke+1].  RS returns flux in input wr array
        if (k>(ks-1)) {
          if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
            Advect(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
            LLF(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
            HLLE(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
            HLLC(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
            Roe(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
            LLF_SR(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
            HLLE_SR(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
            HLLC_SR(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_gr) {
            LLF_GR(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
            HLLE_GR(member, eos, indcs, size, coord, m, k, j, is, ie, IVZ, wl, wr, flx3);
          }
          member.team_barrier();
        }

        // calculate fluxes of scalars (if any)
        if (nvars > nhyd) {
          for (int n=nhyd; n<nvars; ++n) {
            par_for_inner(member, is, ie, [&](const int i) {
              if (flx3(m,IDN,k,j,i) >= 0.0) {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wl(n,i);
              } else {
                flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wr(n,i);
              }
            });
          }
        }
      } // end loop over k
    });
  }

  // handle excision masks
  if (pmy_pack->pcoord->is_general_relativistic) {
    if (coord.bh_excise) {
      auto &fc_mask_ = pmy_pack->pcoord->fc_mask;

      auto fcorr_x1  = uflx.x1f;
      auto fcorr_x2  = uflx.x2f;
      auto fcorr_x3  = uflx.x3f;
      par_for("excise_flux",DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;

        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
        Real x1f = LeftEdgeX  (i-is, indcs.nx1, x1min, x1max);
        Real x2f = LeftEdgeX  (j-js, indcs.nx2, x2min, x2max);
        Real x3f = LeftEdgeX  (k-ks, indcs.nx3, x3min, x3max);

        if (j<(je+1) && k<(ke+1)) {
          if (fc_mask_.x1f(m,k,j,i)) {
            HydPrim1D wim1;
            wim1.d  = w0_(m,IDN,k,j,i-1);
            wim1.vx = w0_(m,IVX,k,j,i-1);
            wim1.vy = w0_(m,IVY,k,j,i-1);
            wim1.vz = w0_(m,IVZ,k,j,i-1);
            wim1.e  = w0_(m,IEN,k,j,i-1);

            HydPrim1D wi;
            wi.d  = w0_(m,IDN,k,j,i);
            wi.vx = w0_(m,IVX,k,j,i);
            wi.vy = w0_(m,IVY,k,j,i);
            wi.vz = w0_(m,IVZ,k,j,i);
            wi.e  = w0_(m,IEN,k,j,i);

            HydCons1D flux;
            SingleStateLLF_GRHyd(wim1, wi, x1f, x2v, x3v, IVX, coord, eos, flux);

            fcorr_x1(m,IDN,k,j,i) = flux.d;
            fcorr_x1(m,IM1,k,j,i) = flux.mx;
            fcorr_x1(m,IM2,k,j,i) = flux.my;
            fcorr_x1(m,IM3,k,j,i) = flux.mz;
            fcorr_x1(m,IEN,k,j,i) = flux.e;
          }
        }

        if (i<(ie+1) && k<(ke+1)) {
          if (fc_mask_.x2f(m,k,j,i)) {
            HydPrim1D wjm1;
            wjm1.d  = w0_(m,IDN,k,j-1,i);
            wjm1.vx = w0_(m,IVY,k,j-1,i);
            wjm1.vy = w0_(m,IVZ,k,j-1,i);
            wjm1.vz = w0_(m,IVX,k,j-1,i);
            wjm1.e  = w0_(m,IEN,k,j-1,i);

            HydPrim1D wj;
            wj.d  = w0_(m,IDN,k,j,i);
            wj.vx = w0_(m,IVY,k,j,i);
            wj.vy = w0_(m,IVZ,k,j,i);
            wj.vz = w0_(m,IVX,k,j,i);
            wj.e  = w0_(m,IEN,k,j,i);

            HydCons1D flux;
            SingleStateLLF_GRHyd(wjm1, wj, x1v, x2f, x3v, IVY, coord, eos, flux);

            fcorr_x2(m,IDN,k,j,i) = flux.d;
            fcorr_x2(m,IM2,k,j,i) = flux.mx;
            fcorr_x2(m,IM3,k,j,i) = flux.my;
            fcorr_x2(m,IM1,k,j,i) = flux.mz;
            fcorr_x2(m,IEN,k,j,i) = flux.e;
          }
        }

        if (i<(ie+1) && j<(je+1)) {
          if (fc_mask_.x3f(m,k,j,i)) {
            HydPrim1D wkm1;
            wkm1.d  = w0_(m,IDN,k-1,j,i);
            wkm1.vx = w0_(m,IVZ,k-1,j,i);
            wkm1.vy = w0_(m,IVX,k-1,j,i);
            wkm1.vz = w0_(m,IVY,k-1,j,i);
            wkm1.e  = w0_(m,IEN,k-1,j,i);

            HydPrim1D wk;
            wk.d  = w0_(m,IDN,k,j,i);
            wk.vx = w0_(m,IVZ,k,j,i);
            wk.vy = w0_(m,IVX,k,j,i);
            wk.vz = w0_(m,IVY,k,j,i);
            wk.e  = w0_(m,IEN,k,j,i);

            HydCons1D flux;
            SingleStateLLF_GRHyd(wkm1, wk, x1v, x2v, x3f, IVZ, coord, eos, flux);

            fcorr_x3(m,IDN,k,j,i) = flux.d;
            fcorr_x3(m,IM3,k,j,i) = flux.mx;
            fcorr_x3(m,IM1,k,j,i) = flux.my;
            fcorr_x3(m,IM2,k,j,i) = flux.mz;
            fcorr_x3(m,IEN,k,j,i) = flux.e;
          }
        }
      });
    }
  }

  return;
}

// function definitions for each template parameter
template void Hydro::CalculateFluxes<Hydro_RSolver::advect>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hllc>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::roe>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hllc_sr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::llf_gr>(Driver *pdriver, int stage);
template void Hydro::CalculateFluxes<Hydro_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace hydro
