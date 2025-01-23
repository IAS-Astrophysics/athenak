//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ion-neutral_tasks.cpp
//  \brief

#include <map>
#include <memory>
#include <string>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "hydro/hydro.hpp"
#include "ion-neutral.hpp"

namespace ion_neutral {
//----------------------------------------------------------------------------------------
//! \fn  void IonNeutral::AssembleIonNeutralTasks
//  \brief Adds tasks for ion-neutral (two-fluid) mhd to appropriate task lists
//  Called by MeshBlockPack::AddPhysics() function directly after MHD constrctr

void IonNeutral::AssembleIonNeutralTasks(
                                    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  using namespace hydro;  // NOLINT(build/namespaces)
  using namespace mhd;    // NOLINT(build/namespaces)
  MHD *pmhd = pmy_pack->pmhd;
  Hydro *phyd = pmy_pack->phydro;

  // assemble "before_stagen_tl" task list
  id.i_irecv = tl["before_stagen"]->AddTask(&MHD::InitRecv, pmhd, none);
  id.n_irecv = tl["before_stagen"]->AddTask(&Hydro::InitRecv, phyd, none);

  // assemble "stagen_tl" task list
  // FirstTwoImpRK task does CopyCons
  id.impl_2x = tl["stagen"]->AddTask(&IonNeutral::FirstTwoImpRK, this, none);

  id.i_flux   = tl["stagen"]->AddTask(&MHD::Fluxes, pmhd, id.impl_2x);
  id.i_sendf  = tl["stagen"]->AddTask(&MHD::SendFlux, pmhd, id.i_flux);
  id.i_recvf  = tl["stagen"]->AddTask(&MHD::RecvFlux, pmhd, id.i_sendf);
  id.i_rkupdt = tl["stagen"]->AddTask(&MHD::RKUpdate, pmhd, id.i_recvf);
  id.i_srctrms   = tl["stagen"]->AddTask(&MHD::MHDSrcTerms, pmhd, id.i_rkupdt);

  id.n_flux   = tl["stagen"]->AddTask(&Hydro::Fluxes, phyd, id.i_srctrms);
  id.n_sendf  = tl["stagen"]->AddTask(&Hydro::SendFlux, phyd, id.n_flux);
  id.n_recvf  = tl["stagen"]->AddTask(&Hydro::RecvFlux, phyd, id.n_sendf);
  id.n_rkupdt = tl["stagen"]->AddTask(&Hydro::RKUpdate, phyd, id.n_recvf);
  id.n_srctrms   = tl["stagen"]->AddTask(&Hydro::HydroSrcTerms, phyd, id.n_rkupdt);

  id.impl     = tl["stagen"]->AddTask(&IonNeutral::ImpRKUpdate, this, id.n_srctrms);
  id.i_restu  = tl["stagen"]->AddTask(&MHD::RestrictU, pmhd, id.impl);
  id.n_restu  = tl["stagen"]->AddTask(&Hydro::RestrictU, phyd, id.i_restu);

  id.i_sendu  = tl["stagen"]->AddTask(&MHD::SendU, pmhd, id.n_restu);
  id.n_sendu  = tl["stagen"]->AddTask(&Hydro::SendU, phyd, id.n_restu);
  id.i_recvu  = tl["stagen"]->AddTask(&MHD::RecvU, pmhd, id.i_sendu);
  id.n_recvu  = tl["stagen"]->AddTask(&Hydro::RecvU, phyd, id.n_sendu);

  id.efld     = tl["stagen"]->AddTask(&MHD::CornerE, pmhd, id.i_recvu);
  id.sende    = tl["stagen"]->AddTask(&MHD::SendE, pmhd, id.efld);
  id.recve    = tl["stagen"]->AddTask(&MHD::RecvE, pmhd, id.sende);
  id.ct       = tl["stagen"]->AddTask(&MHD::CT, pmhd, id.recve);
  id.restb    = tl["stagen"]->AddTask(&MHD::RestrictB, pmhd, id.ct);
  id.sendb    = tl["stagen"]->AddTask(&MHD::SendB, pmhd, id.restb);
  id.recvb    = tl["stagen"]->AddTask(&MHD::RecvB, pmhd, id.sendb);

  id.i_bcs    = tl["stagen"]->AddTask(&MHD::ApplyPhysicalBCs, pmhd, id.recvb);
  id.n_bcs    = tl["stagen"]->AddTask(&Hydro::ApplyPhysicalBCs, phyd, id.n_recvu);
  id.i_prol   = tl["stagen"]->AddTask(&MHD::Prolongate, pmhd, id.i_bcs);
  id.n_prol   = tl["stagen"]->AddTask(&Hydro::Prolongate, phyd, id.n_bcs);
  id.i_c2p    = tl["stagen"]->AddTask(&MHD::ConToPrim, pmhd, id.i_prol);
  id.n_c2p    = tl["stagen"]->AddTask(&Hydro::ConToPrim, phyd, id.n_prol);
  id.i_newdt  = tl["stagen"]->AddTask(&MHD::NewTimeStep, pmhd, id.i_c2p);
  id.n_newdt  = tl["stagen"]->AddTask(&Hydro::NewTimeStep, phyd, id.n_c2p);

  // assemble "after_stagen_tl" task list
  id.i_clear = tl["after_stagen"]->AddTask(&MHD::ClearSend, pmhd, none);
  id.n_clear = tl["after_stagen"]->AddTask(&Hydro::ClearSend, phyd, none);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn IonNeutral::FirstTwoImpRK
//  \brief Executes first two implicit stages of the ImEx integrator for ion-neutral
//  drag term.  Should be the first task called in TaskList.

TaskStatus IonNeutral::FirstTwoImpRK(Driver *pdrive, int stage) {
  if (stage != 1) {return TaskStatus::complete;}  // only execute on first stage

  mhd::MHD *pmhd = pmy_pack->pmhd;
  hydro::Hydro *phyd = pmy_pack->phydro;

  // copy conserved hydro and MHD variables
  Kokkos::deep_copy(DevExeSpace(), phyd->u1, phyd->u0);
  Kokkos::deep_copy(DevExeSpace(), pmhd->u1, pmhd->u0);
  Kokkos::deep_copy(DevExeSpace(), pmhd->b1.x1f, pmhd->b0.x1f);
  Kokkos::deep_copy(DevExeSpace(), pmhd->b1.x2f, pmhd->b0.x2f);
  Kokkos::deep_copy(DevExeSpace(), pmhd->b1.x3f, pmhd->b0.x3f);

  // Solve implicit equations first time (nexp_stage = -1)
  auto status = ImpRKUpdate(pdrive, -1);

  // Solve implicit equations second time (nexp_stage = 0)
  status = ImpRKUpdate(pdrive, 0);

  // update primitive variables for both hydro and MHD
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  phyd->peos->ConsToPrim(phyd->u0, phyd->w0, false, 0, n1m1, 0, n2m1, 0, n3m1);
  pmhd->peos->ConsToPrim(pmhd->u0, pmhd->b0, pmhd->w0, pmhd->bcc0,
                         false, 0, n1m1, 0, n2m1, 0, n3m1);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void IonNeutral::ImpRKUpdate
//  \brief Implicit RK update of ion-neutral drag term. Used as part of ImEx RK integrator
//  This function should be added AFTER the explicit updates in the task list, so that
//  source terms are evaluated using partially updated values (including explicit terms
//  such as flux divergence).  This means soure terms must only be evaluated using
//  conserved variables (u0), as primitives (w0) are not updated until end of TaskList.
//
//  Note indices of source term array correspond to:
//     ru(0) -> ui(IM1)     ru(3) -> un(IM1)
//     ru(1) -> ui(IM2)     ru(4) -> un(IM2)
//     ru(2) -> ui(IM3)     ru(5) -> un(IM3)
//     ru(6) -> ui(IDN)     ru(7) -> un(IDN)
//  where ui=pmhd->u0 and un=phydro->u0


TaskStatus IonNeutral::ImpRKUpdate(Driver *pdriver, int estage) {
  // # of implicit stage (1,2,3,4,[5]).  Note estage=(# of explicit stage)=(1,2,[3])
  // estage <= 0 corresponds to first two fully implicit stages
  int istage = estage + 2;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int n1 = indcs.nx1 + 2*indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*indcs.ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*indcs.ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // Add stiff source term (ion-neutral drag) evaluated with values from previous stages,
  // i.e. the R(U^1), R(U^2), etc. terms, to partially updated conserved variables.
  // Only required for istage = (2,3,4,[5])
  mhd::MHD *pmhd = pmy_pack->pmhd;
  hydro::Hydro *phyd = pmy_pack->phydro;
  if (istage > 1) {
    int scr_level = 0;
    size_t scr_size = 0;
    auto ui = pmhd->u0;
    auto un = phyd->u0;
    auto &a_twid = pdriver->a_twid;
    Real dt = pmy_pack->pmesh->dt;
    auto ru_ = pdriver->impl_src;
    par_for_outer("imex_exp",DevExeSpace(),scr_size,scr_level,0,nmb1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
      for (int s=0; s<=(istage-2); ++s) {
        Real adt = a_twid[istage-2][s]*dt;
        par_for_inner(member, 0, (n1-1), [&](const int i) {
          ui(m,IM1,k,j,i) += adt*ru_(s,m,0,k,j,i);
          ui(m,IM2,k,j,i) += adt*ru_(s,m,1,k,j,i);
          ui(m,IM3,k,j,i) += adt*ru_(s,m,2,k,j,i);
          un(m,IM1,k,j,i) += adt*ru_(s,m,3,k,j,i);
          un(m,IM2,k,j,i) += adt*ru_(s,m,4,k,j,i);
          un(m,IM3,k,j,i) += adt*ru_(s,m,5,k,j,i);
          ui(m,IDN,k,j,i) += adt*ru_(s,m,6,k,j,i);
          un(m,IDN,k,j,i) += adt*ru_(s,m,7,k,j,i);
        });
      }
    });
  }

  // Update ion/neutral momentum equations with analytic solution of implicit difference
  // equations for ion-neutral drag.
  // Only required for istage = (1,2,3,[4])
  if (estage < pdriver->nexp_stages) {
    Real gamma_adt;
    Real xi_adt;
    Real alpha_adt;

    // Condition to set gamma_adt, xi_adt, and alpha_adt to zero
    if (istage < 3 && pdriver->integrator == "imex2+") {
      gamma_adt = 0.0;
      xi_adt = 0.0;
      alpha_adt = 0.0;
    } else {
      gamma_adt = drag_coeff * (pdriver->a_impl) * (pmy_pack->pmesh->dt);
      xi_adt = ionization_coeff * (pdriver->a_impl) * (pmy_pack->pmesh->dt);
      alpha_adt = recombination_coeff * (pdriver->a_impl) * (pmy_pack->pmesh->dt);
    }

    //Real gamma_adt = drag_coeff*(pdriver->a_impl)*(pmy_pack->pmesh->dt);
    //Real xi_adt = ionization_coeff*(pdriver->a_impl)*(pmy_pack->pmesh->dt);
    //Real alpha_adt = recombination_coeff*(pdriver->a_impl)*(pmy_pack->pmesh->dt)
    auto ui = pmhd->u0;
    auto un = phyd->u0;
    par_for("imex_imp",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real rho_i = ui(m,IDN,k,j,i);
      if (alpha_adt > 0) { // to avoid division by zero
        Real d = 1./4./alpha_adt/alpha_adt + xi_adt/2./alpha_adt/alpha_adt
                 + xi_adt*xi_adt/4./alpha_adt/alpha_adt + ui(m,IDN,k,j,i)/alpha_adt +
                 xi_adt/alpha_adt * (ui(m,IDN,k,j,i)+un(m,IDN,k,j,i));
        rho_i = -1./2./alpha_adt - xi_adt/2./alpha_adt + sqrt(d);
      }
      Real rho_n = ui(m,IDN,k,j,i) + un(m,IDN,k,j,i) - rho_i;
      ui(m,IDN,k,j,i) = rho_i;
      un(m,IDN,k,j,i) = rho_n;

      Real denom = 1.0 + gamma_adt*(rho_i+rho_n) + xi_adt + alpha_adt*rho_i;
      // compute new ion/neutral momenta in x1
      Real sum = (ui(m,IM1,k,j,i) + un(m,IM1,k,j,i));
      Real u_i = (ui(m,IM1,k,j,i) +
                 (gamma_adt*rho_i + xi_adt)*sum)/denom;
      Real u_n = sum - u_i;
      ui(m,IM1,k,j,i) = u_i;
      un(m,IM1,k,j,i) = u_n;
      // compute new ion/neutral momenta in x2
      sum = (ui(m,IM2,k,j,i) + un(m,IM2,k,j,i));
      u_i = (ui(m,IM2,k,j,i) +
                 (gamma_adt*rho_i + xi_adt)*sum)/denom;
      u_n = sum - u_i;
      ui(m,IM2,k,j,i) = u_i;
      un(m,IM2,k,j,i) = u_n;
      // compute new ion/neutral momenta in x3
      sum = (ui(m,IM3,k,j,i) + un(m,IM3,k,j,i));
      u_i = (ui(m,IM3,k,j,i) +
                 (gamma_adt*rho_i + xi_adt)*sum)/denom;
      u_n = sum - u_i;
      ui(m,IM3,k,j,i) = u_i;
      un(m,IM3,k,j,i) = u_n;
    });
  }

  // Compute stiff source term (ion-neutral drag) using variables updated in this stage,
  // i.e R(U^n), for use in later stages.  Only required for istage = (1,2,3,[4])
  if (estage < pdriver->nexp_stages) {
    int s = istage-1;
    auto ui = pmhd->u0;
    auto un = phyd->u0;
    auto drag = drag_coeff;
    auto xi = ionization_coeff;
    auto alpha = recombination_coeff;
    auto ru_ = pdriver->impl_src;
    par_for("imex_rup",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      // drag term in IM1 component of ion momentum
      ru_(s,m,0,k,j,i) = drag*(ui(m,IDN,k,j,i)*un(m,IM1,k,j,i) -
                              un(m,IDN,k,j,i)*ui(m,IM1,k,j,i)) +
                              xi*un(m,IM1,k,j,i) - alpha*ui(m,IDN,k,j,i)*ui(m,IM1,k,j,i);
      // drag term in IM2 component of ion momentum
      ru_(s,m,1,k,j,i) = drag*(ui(m,IDN,k,j,i)*un(m,IM2,k,j,i) -
                              un(m,IDN,k,j,i)*ui(m,IM2,k,j,i)) +
                              xi*un(m,IM2,k,j,i) - alpha*ui(m,IDN,k,j,i)*ui(m,IM2,k,j,i);
      // drag term in IM3 component of ion momentum
      ru_(s,m,2,k,j,i) = drag*(ui(m,IDN,k,j,i)*un(m,IM3,k,j,i) -
                              un(m,IDN,k,j,i)*ui(m,IM3,k,j,i)) +
                              xi*un(m,IM3,k,j,i) - alpha*ui(m,IDN,k,j,i)*ui(m,IM3,k,j,i);
      // drag term in IM1 component of neutral momentum
      ru_(s,m,3,k,j,i) = drag*(un(m,IDN,k,j,i)*ui(m,IM1,k,j,i) -
                              ui(m,IDN,k,j,i)*un(m,IM1,k,j,i)) -
                              xi*un(m,IM1,k,j,i) + alpha*ui(m,IDN,k,j,i)*ui(m,IM1,k,j,i);
      // drag term in IM2 component of neutral momentum
      ru_(s,m,4,k,j,i) = drag*(un(m,IDN,k,j,i)*ui(m,IM2,k,j,i) -
                              ui(m,IDN,k,j,i)*un(m,IM2,k,j,i)) -
                              xi*un(m,IM2,k,j,i) + alpha*ui(m,IDN,k,j,i)*ui(m,IM2,k,j,i);
      // drag term in IM3 component of neutral momentum
      ru_(s,m,5,k,j,i) = drag*(un(m,IDN,k,j,i)*ui(m,IM3,k,j,i) -
                              ui(m,IDN,k,j,i)*un(m,IM3,k,j,i)) -
                              xi*un(m,IM3,k,j,i) + alpha*ui(m,IDN,k,j,i)*ui(m,IM3,k,j,i);
      // drag term in IDN component of ion momentum
      ru_(s,m,6,k,j,i) = xi*un(m,IDN,k,j,i) - alpha*ui(m,IDN,k,j,i)*ui(m,IDN,k,j,i);
      // drag term in IDN component of neutral momentum
      ru_(s,m,7,k,j,i) =-xi*un(m,IDN,k,j,i) + alpha*ui(m,IDN,k,j,i)*ui(m,IDN,k,j,i);
    });
  }

  return TaskStatus::complete;
}

} // namespace ion_neutral
