//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_sts.cpp
//! \brief Hydro-owned helpers and task wrappers for super time stepping.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/viscosity.hpp"
#include "hydro.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool UpdateSTSHydroVariable(const int n, const bool update_momentum,
                            const bool update_energy) {
  return (update_momentum && (n == IVX || n == IVY || n == IVZ)) ||
         (update_energy && n == IEN);
}

} // namespace

namespace hydro {

//----------------------------------------------------------------------------------------
//! \fn void Hydro::AddSelectedDiffusionFluxes()
//! \brief Add only the requested subset of Hydro diffusion operators.

void Hydro::AddSelectedDiffusionFluxes(parabolic::DiffusionSelection selection) {
  const bool add_viscosity =
      selection == parabolic::DiffusionSelection::explicit_only
          ? has_explicit_viscosity : has_sts_viscosity;
  const bool add_conduction =
      selection == parabolic::DiffusionSelection::explicit_only
          ? has_explicit_conduction : has_sts_conduction;

  if (add_conduction) {
    pcond->AddHeatFluxes(w0, peos->eos_data, uflx);
  }
  if (add_viscosity) {
    pvisc->AddViscousFluxes(w0, peos->eos_data, uflx);
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::ClearSTSFlux()
//! \brief Zero the Hydro flux scratch before one STS stage.

TaskStatus Hydro::ClearSTSFlux(Driver *pdrive, int stage) {
  (void) pdrive;
  (void) stage;
  Kokkos::deep_copy(DevExeSpace(), uflx.x1f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), uflx.x2f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), uflx.x3f, 0.0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::STSFluxes()
//! \brief Accumulate only the STS-managed Hydro diffusion operators.

TaskStatus Hydro::STSFluxes(Driver *pdrive, int stage) {
  (void) pdrive;
  (void) stage;
  AddSelectedDiffusionFluxes(parabolic::DiffusionSelection::sts_only);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::STSUpdate()
//! \brief Apply one Hydro-owned RKL2 STS stage.

TaskStatus Hydro::STSUpdate(Driver *pdrive, int stage) {
  if (!pdrive->sts.enabled) {
    return TaskStatus::complete;
  }

  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u_sts0, u0);
    Kokkos::deep_copy(DevExeSpace(), u_sts1, u0);
    Kokkos::deep_copy(DevExeSpace(), u_sts2, u0);
  } else {
    Kokkos::deep_copy(DevExeSpace(), u_sts2, u_sts1);
    Kokkos::deep_copy(DevExeSpace(), u_sts1, u0);
  }

  const bool update_momentum = has_sts_viscosity;
  const bool update_energy =
      has_sts_conduction || (has_sts_viscosity && peos->eos_data.is_ideal);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real dt_sweep = pdrive->sts.dt_sweep;
  auto coeffs = pdrive->sts.coeffs;
  auto u0_ = u0;
  auto u_sts0_ = u_sts0;
  auto u_sts1_ = u_sts1;
  auto u_sts2_ = u_sts2;
  auto u_sts_rhs_ = u_sts_rhs;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &mbsize = pmy_pack->pmb->mb_size;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("hydro_sts_update", DevExeSpace(), scr_size, scr_level, 0, nmb1,
                0, nhydro - 1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
    if (!UpdateSTSHydroVariable(n, update_momentum, update_energy)) {
      return;
    }

    ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

    par_for_inner(member, is, ie, [&](const int i) {
      divf(i) = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
    });
    member.team_barrier();

    if (multi_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
      });
      member.team_barrier();
    }

    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
      });
      member.team_barrier();
    }

    par_for_inner(member, is, ie, [&](const int i) {
      const Real delta_u = -dt_sweep*divf(i);
      u0_(m,n,k,j,i) = coeffs.muj*u_sts1_(m,n,k,j,i)
                     + coeffs.nuj*u_sts2_(m,n,k,j,i)
                     + (1.0 - coeffs.muj - coeffs.nuj)*u_sts0_(m,n,k,j,i)
                     + coeffs.gammaj_tilde*u_sts_rhs_(m,n,k,j,i)
                     + coeffs.muj_tilde*delta_u;
      if (stage == 1) {
        u_sts_rhs_(m,n,k,j,i) = delta_u;
      }
    });
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::STSRefreshTimeStep()
//! \brief Refresh Hydro-local timestep estimates after the post-sweep.

TaskStatus Hydro::STSRefreshTimeStep(Driver *pdrive, int stage) {
  if (pdrive->sts.sweep == Driver::STSSweep::post && stage == pdrive->sts.nstages) {
    return NewTimeStep(pdrive, pdrive->nexp_stages);
  }
  return TaskStatus::complete;
}

} // namespace hydro
