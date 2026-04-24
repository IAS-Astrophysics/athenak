//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_sts.cpp
//! \brief MHD-owned helpers and task wrappers for super time stepping.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/viscosity.hpp"
#include "mhd.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool UpdateSTSMHDVariable(const int n, const bool update_momentum,
                          const bool update_energy) {
  if (update_momentum && (n == IVX || n == IVY || n == IVZ)) {
    return true;
  }
  if (update_energy && n == IEN) {
    return true;
  }
  return false;
}

KOKKOS_INLINE_FUNCTION
int CGLPressureVariable(const int n) {
  return (n == 0) ? IPR : IPP;
}

} // namespace

namespace mhd {

//! \fn void MHD::AddSelectedDiffusionFluxes()
//! \brief Add only the requested subset of MHD diffusion operators to the live flux
//! scratch array.

void MHD::AddSelectedDiffusionFluxes(DiffusionSelection selection) {
  const bool add_viscosity =
      (selection == DiffusionSelection::explicit_only) ? has_explicit_viscosity
                                                       : has_sts_viscosity;
  const bool add_conduction =
      (selection == DiffusionSelection::explicit_only) ? has_explicit_conduction
                                                       : has_sts_conduction;
  const bool add_resistivity =
      (selection == DiffusionSelection::explicit_only) ? has_explicit_resistivity
                                                       : has_sts_resistivity;

  if (add_viscosity && pvisc != nullptr) {
    pvisc->AddViscousFluxes(w0, peos->eos_data, uflx);
  }
  if (add_resistivity && presist != nullptr && peos->eos_data.is_ideal) {
    presist->AddResistiveFluxes(b0, uflx);
  }
  if (add_conduction && pcond != nullptr) {
    pcond->AddHeatFluxes(w0, peos->eos_data, uflx);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::AddSelectedDiffusionEMF()
//! \brief Add only the requested subset of MHD resistive EMFs to the live electric-field
//! scratch array.

void MHD::AddSelectedDiffusionEMF(DiffusionSelection selection) {
  const bool add_resistivity =
      (selection == DiffusionSelection::explicit_only) ? has_explicit_resistivity
                                                       : has_sts_resistivity;
  if (add_resistivity && presist != nullptr && presist->eta_ohm > 0.0) {
    presist->AddResistiveEMFs(b0, efld);
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearSTSFlux()
//! \brief Zero the MHD conserved-flux scratch before one STS stage.

TaskStatus MHD::ClearSTSFlux(Driver *pdrive, int stage) {
  (void) pdrive;
  (void) stage;
  Kokkos::deep_copy(DevExeSpace(), uflx.x1f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), uflx.x2f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), uflx.x3f, 0.0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearSTSEField()
//! \brief Zero the MHD electric-field scratch before one STS stage.

TaskStatus MHD::ClearSTSEField(Driver *pdrive, int stage) {
  (void) pdrive;
  (void) stage;
  Kokkos::deep_copy(DevExeSpace(), efld.x1e, 0.0);
  Kokkos::deep_copy(DevExeSpace(), efld.x2e, 0.0);
  Kokkos::deep_copy(DevExeSpace(), efld.x3e, 0.0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::STSFluxes()
//! \brief Accumulate only the STS-managed MHD diffusion operators that contribute to U.

TaskStatus MHD::STSFluxes(Driver *pdrive, int stage) {
  (void) pdrive;
  (void) stage;
  if (!has_any_sts_cell_update) {
    return TaskStatus::complete;
  }

  AddSelectedDiffusionFluxes(DiffusionSelection::sts_only);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::STSEField()
//! \brief Accumulate only the STS-managed resistive EMF contribution.

TaskStatus MHD::STSEField(Driver *pdrive, int stage) {
  (void) pdrive;
  (void) stage;
  if (!has_any_sts_field_update) {
    return TaskStatus::complete;
  }

  AddSelectedDiffusionEMF(DiffusionSelection::sts_only);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::STSUpdateU()
//! \brief Apply one MHD-owned RKL2 STS stage over the enrolled conserved variables.

TaskStatus MHD::STSUpdateU(Driver *pdrive, int stage) {
  if (!has_any_sts_cell_update || !(pdrive->sts.enabled)) {
    return TaskStatus::complete;
  }

  if (peos->eos_data.is_cgl && has_sts_conduction) {
    return STSUpdateCGLHeatFlux(pdrive, stage);
  }

  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u_sts0, u0);
  }
  Kokkos::deep_copy(DevExeSpace(), u_sts2, u_sts1);
  Kokkos::deep_copy(DevExeSpace(), u_sts1, u0);

  const bool update_momentum = has_sts_viscosity;
  const bool update_energy = (has_sts_conduction ||
                              ((has_sts_viscosity || has_sts_resistivity) &&
                               peos->eos_data.is_ideal));
  if (!(update_momentum || update_energy)) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nmhd_vars = nmhd;
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

  par_for_outer("mhd_sts_update_u", DevExeSpace(), scr_size, scr_level, 0, nmb1,
                0, nmhd_vars - 1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
    if (!UpdateSTSMHDVariable(n, update_momentum, update_energy)) {
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
//! \fn TaskStatus MHD::STSUpdateCGLHeatFlux()
//! \brief Apply one CGL heat-flux RKL2 stage to parallel/perpendicular pressures.

TaskStatus MHD::STSUpdateCGLHeatFlux(Driver *pdrive, int stage) {
  if (!has_sts_conduction || !(pdrive->sts.enabled)) {
    return TaskStatus::complete;
  }

  Kokkos::deep_copy(DevExeSpace(), cgl_p_sts2, cgl_p_sts1);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  Real dt_sweep = pdrive->sts.dt_sweep;
  Real pfloor = peos->eos_data.pfloor;
  auto coeffs = pdrive->sts.coeffs;
  auto w0_ = w0;
  auto p_sts0_ = cgl_p_sts0;
  auto p_sts1_ = cgl_p_sts1;
  auto p_sts2_ = cgl_p_sts2;
  auto p_sts_rhs_ = cgl_p_sts_rhs;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &mbsize = pmy_pack->pmb->mb_size;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("mhd_cgl_sts_update_p", DevExeSpace(), scr_size, scr_level, 0, nmb1,
                0, 1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
    ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);
    const int pressure_index = CGLPressureVariable(n);

    par_for_inner(member, is, ie, [&](const int i) {
      divf(i) = (flx1(m,pressure_index,k,j,i+1) -
                 flx1(m,pressure_index,k,j,i))/mbsize.d_view(m).dx1;
    });
    member.team_barrier();

    if (multi_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += (flx2(m,pressure_index,k,j+1,i) -
                    flx2(m,pressure_index,k,j,i))/mbsize.d_view(m).dx2;
      });
      member.team_barrier();
    }

    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += (flx3(m,pressure_index,k+1,j,i) -
                    flx3(m,pressure_index,k,j,i))/mbsize.d_view(m).dx3;
      });
      member.team_barrier();
    }

    par_for_inner(member, is, ie, [&](const int i) {
      const Real pressure_current = w0_(m,pressure_index,k,j,i);
      if (stage == 1) {
        p_sts0_(m,n,k,j,i) = pressure_current;
      }
      p_sts1_(m,n,k,j,i) = pressure_current;

      const Real delta_p = -dt_sweep*divf(i);
      Real pressure_next = coeffs.muj*pressure_current
                         + coeffs.nuj*p_sts2_(m,n,k,j,i)
                         + (1.0 - coeffs.muj - coeffs.nuj)*p_sts0_(m,n,k,j,i)
                         + coeffs.gammaj_tilde*p_sts_rhs_(m,n,k,j,i)
                         + coeffs.muj_tilde*delta_p;
      pressure_next = fmax(pressure_next, pfloor);
      w0_(m,pressure_index,k,j,i) = pressure_next;
      if (stage == 1) {
        p_sts_rhs_(m,n,k,j,i) = delta_p;
      }
    });
  });

  peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::STSUpdateB()
//! \brief Apply one CT-shaped RKL2 STS stage over the enrolled magnetic field.

TaskStatus MHD::STSUpdateB(Driver *pdrive, int stage) {
  if (!has_any_sts_field_update || !(pdrive->sts.enabled)) {
    return TaskStatus::complete;
  }

  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), b_sts0.x1f, b0.x1f);
    Kokkos::deep_copy(DevExeSpace(), b_sts0.x2f, b0.x2f);
    Kokkos::deep_copy(DevExeSpace(), b_sts0.x3f, b0.x3f);
  }
  Kokkos::deep_copy(DevExeSpace(), b_sts2.x1f, b_sts1.x1f);
  Kokkos::deep_copy(DevExeSpace(), b_sts2.x2f, b_sts1.x2f);
  Kokkos::deep_copy(DevExeSpace(), b_sts2.x3f, b_sts1.x3f);
  Kokkos::deep_copy(DevExeSpace(), b_sts1.x1f, b0.x1f);
  Kokkos::deep_copy(DevExeSpace(), b_sts1.x2f, b0.x2f);
  Kokkos::deep_copy(DevExeSpace(), b_sts1.x3f, b0.x3f);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  const bool &multi_d = pmy_pack->pmesh->multi_d;
  const bool &three_d = pmy_pack->pmesh->three_d;
  Real dt_sweep = pdrive->sts.dt_sweep;
  auto coeffs = pdrive->sts.coeffs;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto bx1f = b0.x1f;
  auto bx2f = b0.x2f;
  auto bx3f = b0.x3f;
  auto bx1f_sts0 = b_sts0.x1f;
  auto bx2f_sts0 = b_sts0.x2f;
  auto bx3f_sts0 = b_sts0.x3f;
  auto bx1f_sts1 = b_sts1.x1f;
  auto bx2f_sts1 = b_sts1.x2f;
  auto bx3f_sts1 = b_sts1.x3f;
  auto bx1f_sts2 = b_sts2.x1f;
  auto bx2f_sts2 = b_sts2.x2f;
  auto bx3f_sts2 = b_sts2.x3f;
  auto bx1f_rhs = b_sts_rhs.x1f;
  auto bx2f_rhs = b_sts_rhs.x2f;
  auto bx3f_rhs = b_sts_rhs.x3f;
  auto &mbsize = pmy_pack->pmb->mb_size;

  par_for("mhd_sts_update_b1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real delta_b = 0.0;
    if (multi_d) {
      delta_b -= dt_sweep*(e3(m,k,j+1,i) - e3(m,k,j,i))/mbsize.d_view(m).dx2;
      if (three_d) {
        delta_b += dt_sweep*(e2(m,k+1,j,i) - e2(m,k,j,i))/mbsize.d_view(m).dx3;
      }
    }
    bx1f(m,k,j,i) = coeffs.muj*bx1f_sts1(m,k,j,i)
                  + coeffs.nuj*bx1f_sts2(m,k,j,i)
                  + (1.0 - coeffs.muj - coeffs.nuj)*bx1f_sts0(m,k,j,i)
                  + coeffs.gammaj_tilde*bx1f_rhs(m,k,j,i)
                  + coeffs.muj_tilde*delta_b;
    if (stage == 1) {
      bx1f_rhs(m,k,j,i) = delta_b;
    }
  });

  par_for("mhd_sts_update_b2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real delta_b = dt_sweep*(e3(m,k,j,i+1) - e3(m,k,j,i))/mbsize.d_view(m).dx1;
    if (three_d) {
      delta_b -= dt_sweep*(e1(m,k+1,j,i) - e1(m,k,j,i))/mbsize.d_view(m).dx3;
    }
    bx2f(m,k,j,i) = coeffs.muj*bx2f_sts1(m,k,j,i)
                  + coeffs.nuj*bx2f_sts2(m,k,j,i)
                  + (1.0 - coeffs.muj - coeffs.nuj)*bx2f_sts0(m,k,j,i)
                  + coeffs.gammaj_tilde*bx2f_rhs(m,k,j,i)
                  + coeffs.muj_tilde*delta_b;
    if (stage == 1) {
      bx2f_rhs(m,k,j,i) = delta_b;
    }
  });

  par_for("mhd_sts_update_b3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real delta_b = -dt_sweep*(e2(m,k,j,i+1) - e2(m,k,j,i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      delta_b += dt_sweep*(e1(m,k,j+1,i) - e1(m,k,j,i))/mbsize.d_view(m).dx2;
    }
    bx3f(m,k,j,i) = coeffs.muj*bx3f_sts1(m,k,j,i)
                  + coeffs.nuj*bx3f_sts2(m,k,j,i)
                  + (1.0 - coeffs.muj - coeffs.nuj)*bx3f_sts0(m,k,j,i)
                  + coeffs.gammaj_tilde*bx3f_rhs(m,k,j,i)
                  + coeffs.muj_tilde*delta_b;
    if (stage == 1) {
      bx3f_rhs(m,k,j,i) = delta_b;
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::STSRefreshTimeStep()
//! \brief Refresh MHD-local timestep estimates after the final post sweep stage.

TaskStatus MHD::STSRefreshTimeStep(Driver *pdrive, int stage) {
  if (!has_any_sts_diffusion || !(pdrive->sts.enabled)) {
    return TaskStatus::complete;
  }
  if (pdrive->sts.sweep == Driver::STSSweep::post && stage == pdrive->sts.nstages) {
    RecomputeTimeStepFromCurrentState(pdrive);
  }
  return TaskStatus::complete;
}

} // namespace mhd
