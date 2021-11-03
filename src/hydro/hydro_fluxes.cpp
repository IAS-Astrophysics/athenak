//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_fluxes.cpp
//  \brief Calculate 3D fluxes for hydro

#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
// include inlined reconstruction methods (yuck...)
#include "reconstruct/dc.cpp"
#include "reconstruct/plm.cpp"
#include "reconstruct/ppm.cpp"
#include "reconstruct/wenoz.cpp"
// include inlined Riemann solvers (double yuck...)
#include "hydro/rsolvers/advect_hyd.cpp"
#include "hydro/rsolvers/llf_hyd.cpp"
#include "hydro/rsolvers/hlle_hyd.cpp"
#include "hydro/rsolvers/hllc_hyd.cpp"
#include "hydro/rsolvers/roe_hyd.cpp"
#include "hydro/rsolvers/llf_srhyd.cpp"
#include "hydro/rsolvers/hlle_srhyd.cpp"
#include "hydro/rsolvers/hllc_srhyd.cpp"
#include "hydro/rsolvers/hlle_grhyd.cpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CalcFluxes
//! \brief Calls reconstruction and Riemann solver functions to compute hydro fluxes
//! Note this function is templated over RS for better performance on GPUs.

template <Hydro_RSolver rsolver_method_>
TaskStatus Hydro::CalcFluxes(Driver *pdriver, int stage)
{
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  
  int nhyd  = nhydro;
  int nvars = nhydro + nscalars;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const auto recon_method_ = recon_method;
  auto &eos = peos->eos_data;
  auto &mbd = pmy_pack->pcoord->mbdata;
  auto &w0_ = w0;

  //--------------------------------------------------------------------------------------
  // i-direction

  size_t scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 2;
  int scr_level = 0;
  auto flx1 = uflx.x1f;

  par_for_outer("hflux_x1",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
    {
      ScrArray2D<Real> wl(member.team_scratch(scr_level), nvars, ncells1);
      ScrArray2D<Real> wr(member.team_scratch(scr_level), nvars, ncells1);

      // Reconstruct qR[i] and qL[i+1]
      switch (recon_method_)
      {
        case ReconstructionMethod::dc:
          DonorCellX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          break;
        case ReconstructionMethod::plm:
          PiecewiseLinearX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          break;
        case ReconstructionMethod::ppm:
          PiecewiseParabolicX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          break;
        case ReconstructionMethod::wenoz:
          WENOZX1(member, m, k, j, is-1, ie+1, w0_, wl, wr);
          break;
        default:
          break;
      }
      // Sync all threads in the team so that scratch memory is consistent
      member.team_barrier();

      // compute fluxes over [is,ie+1]
      if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
        Advect(member, eos, mbd, m, k, j, is, ie+1, IVX, wl, wr, flx1);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
        LLF(member, eos, mbd, m, k, j, is, ie+1, IVX, wl, wr, flx1);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
        HLLE(member, eos, mbd, m, k, j, is, ie+1, IVX, wl, wr, flx1);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
        HLLC(member, eos, mbd, m, k, j, is, ie+1, IVX, wl, wr, flx1);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
        Roe(member, eos, mbd, m, k, j, is, ie+1, IVX, wl, wr, flx1);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
        LLF_SR(member, eos, mbd, m, k, j, is, ie+1, IVX, wl, wr, flx1);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
        HLLE_SR(member, eos, mbd, m, k, j, is, ie+1, IVX, wl, wr, flx1);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
        HLLC_SR(member, eos, mbd, m, k, j, is, ie+1, IVX, wl, wr, flx1);
      } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
        HLLE_GR(member, eos, mbd, m, k, j, is, ie+1, IVX, wl, wr, flx1);
      }
      member.team_barrier();

      // calculate fluxes of scalars (if any)
      if (nvars > nhyd) {
        for (int n=nhyd; n<nvars; ++n) {
          par_for_inner(member, is, ie+1, [&](const int i)
          {
            if (flx1(m,IDN,k,j,i) >= 0.0) {
              flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wl(n,i);
            } else {
              flx1(m,n,k,j,i) = flx1(m,IDN,k,j,i)*wr(n,i);
            }
          });
        }
      }

    }
  );

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
    auto flx2 = uflx.x2f;

    par_for_outer("hflux_x2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k)
      {
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
          switch (recon_method_)
          {
            case ReconstructionMethod::dc:
              DonorCellX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
              break;
            case ReconstructionMethod::plm:
              PiecewiseLinearX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
              break;
            case ReconstructionMethod::ppm:
              PiecewiseParabolicX2(member, m, k, j, is, ie, w0_, wl_jp1, wr);
              break;
            case ReconstructionMethod::wenoz:
              WENOZX2(member, m, k, j, is-1, ie+1, w0_, wl_jp1, wr);
              break;
            default:
              break;
          }
          member.team_barrier();

          // compute fluxes over [js,je+1].  RS returns flux in input wr array
          if (j>(js-1)) {
            if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
              Advect(member, eos, mbd, m, k, j, is, ie, IVY, wl, wr, flx2);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
              LLF(member, eos, mbd, m, k, j, is, ie, IVY, wl, wr, flx2);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
              HLLE(member, eos, mbd, m, k, j, is, ie, IVY, wl, wr, flx2);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
              HLLC(member, eos, mbd, m, k, j, is, ie, IVY, wl, wr, flx2);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
              Roe(member, eos, mbd, m, k, j, is, ie, IVY, wl, wr, flx2);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
              LLF_SR(member, eos, mbd, m, k, j, is, ie, IVY, wl, wr, flx2);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
              HLLE_SR(member, eos, mbd, m, k, j, is, ie, IVY, wl, wr, flx2);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
              HLLC_SR(member, eos, mbd, m, k, j, is, ie, IVY, wl, wr, flx2);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
              HLLE_GR(member, eos, mbd, m, k, j, is, ie, IVY, wl, wr, flx2);
            }
            member.team_barrier();
          }

          // calculate fluxes of scalars (if any)
          if (nvars > nhyd) {
            for (int n=nhyd; n<nvars; ++n) {
              par_for_inner(member, is, ie, [&](const int i)
              {
                if (flx2(m,IDN,k,j,i) >= 0.0) {
                  flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wl(n,i);
                } else {
                  flx2(m,n,k,j,i) = flx2(m,IDN,k,j,i)*wr(n,i);
                }
              });
            }
          }
  
        } // end of loop over j
      }
    );
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    scr_size = ScrArray2D<Real>::shmem_size(nvars, ncells1) * 3;
    auto flx3 = uflx.x3f;

    par_for_outer("hflux_x3",DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je,
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j)
      {
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
          switch (recon_method_)
          {
            case ReconstructionMethod::dc:
              DonorCellX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
              break;
            case ReconstructionMethod::plm:
              PiecewiseLinearX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
              break;
            case ReconstructionMethod::ppm:
              PiecewiseParabolicX3(member, m, k, j, is, ie, w0_, wl_kp1, wr);
              break;
            case ReconstructionMethod::wenoz:
              WENOZX3(member, m, k, j, is-1, ie+1, w0_, wl_kp1, wr);
              break;
            default:
              break;
          }
          member.team_barrier();

          // compute fluxes over [ks,ke+1].  RS returns flux in input wr array
          if (k>(ks-1)) {
            if constexpr (rsolver_method_ == Hydro_RSolver::advect) {
              Advect(member, eos, mbd, m, k, j, is, ie, IVZ, wl, wr, flx3);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf) {
              LLF(member, eos, mbd, m, k, j, is, ie, IVZ, wl, wr, flx3);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle) {
              HLLE(member, eos, mbd, m, k, j, is, ie, IVZ, wl, wr, flx3);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc) {
              HLLC(member, eos, mbd, m, k, j, is, ie, IVZ, wl, wr, flx3);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::roe) {
              Roe(member, eos, mbd, m, k, j, is, ie, IVZ, wl, wr, flx3);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::llf_sr) {
              LLF_SR(member, eos, mbd, m, k, j, is, ie, IVZ, wl, wr, flx3);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_sr) {
              HLLE_SR(member, eos, mbd, m, k, j, is, ie, IVZ, wl, wr, flx3);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hllc_sr) {
              HLLC_SR(member, eos, mbd, m, k, j, is, ie, IVZ, wl, wr, flx3);
            } else if constexpr (rsolver_method_ == Hydro_RSolver::hlle_gr) {
              HLLE_GR(member, eos, mbd, m, k, j, is, ie, IVZ, wl, wr, flx3);
            }
            member.team_barrier();
          }

          // calculate fluxes of scalars (if any)
          if (nvars > nhyd) {
            for (int n=nhyd; n<nvars; ++n) {
              par_for_inner(member, is, ie, [&](const int i)
              {
                if (flx3(m,IDN,k,j,i) >= 0.0) {
                  flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wl(n,i);
                } else {
                  flx3(m,n,k,j,i) = flx3(m,IDN,k,j,i)*wr(n,i);
                }
              });
            }
          }

        } // end loop over k
      }
    );
  }

  // Add viscous, resistive, heat-flux, etc fluxes
  if (pvisc != nullptr) {
    pvisc->IsotropicViscousFlux(u0, pvisc->nu, eos, uflx);
  }

  return TaskStatus::complete;
}

// function definitions for each template parameter
template TaskStatus Hydro::CalcFluxes<Hydro_RSolver::advect>(Driver *pdriver, int stage);
template TaskStatus Hydro::CalcFluxes<Hydro_RSolver::llf>(Driver *pdriver, int stage);
template TaskStatus Hydro::CalcFluxes<Hydro_RSolver::hlle>(Driver *pdriver, int stage);
template TaskStatus Hydro::CalcFluxes<Hydro_RSolver::hllc>(Driver *pdriver, int stage);
template TaskStatus Hydro::CalcFluxes<Hydro_RSolver::roe>(Driver *pdriver, int stage);
template TaskStatus Hydro::CalcFluxes<Hydro_RSolver::llf_sr>(Driver *pdriver, int stage);
template TaskStatus Hydro::CalcFluxes<Hydro_RSolver::hlle_sr>(Driver *pdriver, int stage);
template TaskStatus Hydro::CalcFluxes<Hydro_RSolver::hllc_sr>(Driver *pdriver, int stage);
template TaskStatus Hydro::CalcFluxes<Hydro_RSolver::hlle_gr>(Driver *pdriver, int stage);

} // namespace hydro
