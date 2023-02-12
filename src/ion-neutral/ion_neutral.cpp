//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ion-neutral.cpp
//  \brief

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "hydro/hydro.hpp"
#include "ion_neutral.hpp"

namespace ion_neutral {
//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters

IonNeutral::IonNeutral(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp) {
  // Read various coefficients
  drag_coeff = pin->GetReal("ion-neutral","drag_coeff");
  ionization_coeff = pin->GetOrAddReal("ion-neutral","ionization_coeff",0.0);
  recombination_coeff = pin->GetOrAddReal("ion-neutral","recombination_coeff",0.0);
  mi_mn = pin->GetOrAddReal("ion-neutral","mi_mn",1.0);
}

//----------------------------------------------------------------------------------------
//! \fn  void IonNeutral::AssembleIonNeutralTasks
//  \brief Adds tasks for ion-neutral (two-fluid) mhd to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysics() function directly after MHD constrctr

void IonNeutral::AssembleIonNeutralTasks(TaskList &start, TaskList &run, TaskList &end) {
  TaskID none(0);
  using namespace hydro;  // NOLINT(build/namespaces)
  using namespace mhd;    // NOLINT(build/namespaces)
  MHD *pmhd = pmy_pack->pmhd;
  Hydro *phyd = pmy_pack->phydro;

  // assemble start task list
  id.i_irecv = start.AddTask(&MHD::InitRecv, pmhd, none);
  id.n_irecv = start.AddTask(&Hydro::InitRecv, phyd, none);

  // assemble run task list
  id.impl_2x = run.AddTask(&IonNeutral::FirstTwoImpRK, this, none);

  id.i_flux = run.AddTask(&MHD::Fluxes, pmhd, id.impl_2x);
  id.i_expl = run.AddTask(&MHD::ExpRKUpdate, pmhd, id.i_flux);

  id.n_flux = run.AddTask(&Hydro::Fluxes, phyd, id.i_expl);
  id.n_expl = run.AddTask(&Hydro::ExpRKUpdate, phyd, id.n_flux);

  id.impl = run.AddTask(&IonNeutral::ImpRKUpdate, this, id.n_expl);

  id.i_sendu = run.AddTask(&MHD::SendU, pmhd, id.impl);
  id.n_sendu = run.AddTask(&Hydro::SendU, phyd, id.impl);
  id.i_recvu = run.AddTask(&MHD::RecvU, pmhd, id.i_sendu);
  id.n_recvu = run.AddTask(&Hydro::RecvU, phyd, id.n_sendu);

  id.efld  = run.AddTask(&MHD::CornerE, pmhd, id.i_recvu);
  id.ct    = run.AddTask(&MHD::CT, pmhd, id.efld);
  id.sendb = run.AddTask(&MHD::SendB, pmhd, id.ct);
  id.recvb  = run.AddTask(&MHD::RecvB, pmhd, id.sendb);

  id.i_bcs   = run.AddTask(&MHD::ApplyPhysicalBCs, pmhd, id.recvb);
  id.n_bcs   = run.AddTask(&Hydro::ApplyPhysicalBCs, phyd, id.n_recvu);
  id.i_c2p   = run.AddTask(&MHD::ConToPrim, pmhd, id.i_bcs);
  id.n_c2p   = run.AddTask(&Hydro::ConToPrim, phyd, id.n_bcs);
  id.i_newdt = run.AddTask(&MHD::NewTimeStep, pmhd, id.i_c2p);
  id.n_newdt = run.AddTask(&Hydro::NewTimeStep, phyd, id.n_c2p);

  // assemble end task list
  id.i_clear = end.AddTask(&MHD::ClearSend, pmhd, none);
  id.n_clear = end.AddTask(&Hydro::ClearSend, phyd, none);

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
//     ru(8) -> ui(IEN)     ru(9) -> un(IEN)
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
    EOS_Data &eos_hyd = phyd->peos->eos_data;
    EOS_Data &eos_mhd = pmhd->peos->eos_data;
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
          if (eos_hyd.is_ideal) {
            ui(m,IEN,k,j,i) += adt*ru_(s,m,8,k,j,i);
            un(m,IEN,k,j,i) += adt*ru_(s,m,9,k,j,i);
          }
        });
      }
    });
  }

  // Update ion/neutral momentum equations with analytic solution of implicit difference
  // equations for ion-neutral drag.
  // Only required for istage = (1,2,3,[4])
  if (estage < pdriver->nexp_stages) {
    Real gamma_adt = drag_coeff*(pdriver->a_impl)*(pmy_pack->pmesh->dt);
    Real xi_adt = ionization_coeff*(pdriver->a_impl)*(pmy_pack->pmesh->dt);
    Real alpha_adt = recombination_coeff*(pdriver->a_impl)*(pmy_pack->pmesh->dt);
    auto ui = pmhd->u0;
    auto un = phyd->u0;
    EOS_Data &eos_hyd = phyd->peos->eos_data;
    EOS_Data &eos_mhd = pmhd->peos->eos_data;
    par_for("imex_imp",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real kin_prev_i = (ui(m,IM1,k,j,i)*ui(m,IM1,k,j,i)+ui(m,IM2,k,j,i)*ui(m,IM2,k,j,i)+
                        ui(m,IM3,k,j,i)*ui(m,IM3,k,j,i))/2./ui(m,IDN,k,j,i);
      Real kin_prev_n = (un(m,IM1,k,j,i)*un(m,IM1,k,j,i)+un(m,IM2,k,j,i)*un(m,IM2,k,j,i)+
                        un(m,IM3,k,j,i)*un(m,IM3,k,j,i))/2./un(m,IDN,k,j,i);
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

      if (eos_hyd.is_ideal) {
        Real kin_i = (ui(m,IM1,k,j,i)*ui(m,IM1,k,j,i)+ui(m,IM2,k,j,i)*ui(m,IM2,k,j,i)+
                        ui(m,IM3,k,j,i)*ui(m,IM3,k,j,i))/2./ui(m,IDN,k,j,i);
        Real kin_n = (un(m,IM1,k,j,i)*un(m,IM1,k,j,i)+un(m,IM2,k,j,i)*un(m,IM2,k,j,i)+
                        un(m,IM3,k,j,i)*un(m,IM3,k,j,i))/2./un(m,IDN,k,j,i);
        Real fmass = 1./(1.+mi_mn);
        Real gamma_i = eos_mhd.gamma, gamma_n = eos_hyd.gamma;
        Real a_i, a_n, b_i, b_n, c_i, c_n;
        Real u1_i = ui(m,IM1,k,j,i)/rho_i;
        Real u2_i = ui(m,IM2,k,j,i)/rho_i;
        Real u3_i = ui(m,IM3,k,j,i)/rho_i;
        Real u1_n = un(m,IM1,k,j,i)/rho_n;
        Real u2_n = un(m,IM2,k,j,i)/rho_n;
        Real u3_n = un(m,IM3,k,j,i)/rho_n;
        Real usq_i = u1_i*u1_i + u2_i*u2_i + u3_i*u3_i;
        Real usq_n = u1_n*u1_n + u2_n*u2_n + u3_n*u3_n;
        Real uav1 = u1_i*(1.-fmass) + u1_n*fmass;
        Real uav2 = u2_i*(1.-fmass) + u2_n*fmass;
        Real uav3 = u3_i*(1.-fmass) + u3_n*fmass;
        Real b1 = pmhd->bcc0(m,0,k,j,i);
        Real b2 = pmhd->bcc0(m,1,k,j,i);
        Real b3 = pmhd->bcc0(m,2,k,j,i);
        Real bsq = b1*b1+b2*b2+b3*b3;
        a_i = -alpha_adt*rho_i - 3.*gamma_adt*rho_n*(gamma_i-1.)*(1.-fmass);
        a_n = alpha_adt*rho_i*mi_mn + 3.*gamma_adt*rho_n*(gamma_i-1.)*(1.-fmass);
        b_i = xi_adt/mi_mn + 3.*gamma_adt*rho_i*(gamma_n-1.)*fmass;
        b_n = -xi_adt - 3.*gamma_adt*rho_i*(gamma_n-1.)*fmass;
        c_i = ui(m,IEN,k,j,i) - alpha_adt*rho_i*rho_i*usq_i +
                                xi_adt*rho_n*(usq_i + usq_n)/2. +
                                gamma_adt*rho_n*rho_i*((u1_n-u1_i)*uav1+
                                      (u2_n-u2_i)*uav2+(u3_n-u3_i)*uav3)-
                                a_i*(rho_i*usq_i/2.+bsq/2.)-
                                b_i*(rho_n*usq_n/2.);
        c_n = un(m,IEN,k,j,i) + alpha_adt*rho_i*rho_i*(usq_i + usq_n)/2. -
                                xi_adt*rho_n*usq_n +
                                gamma_adt*rho_n*rho_i*((u1_i-u1_n)*uav1 +
                                      (u2_i-u2_n)*uav2+(u3_i-u3_n)*uav3) -
                                a_n*(rho_i*usq_i/2.+bsq/2.)-
                                b_n*(rho_n*usq_n/2.);
        ui(m,IEN,k,j,i) = (c_i+c_n*b_i-c_i*b_n)/(1.-a_i-b_n+a_i*b_n-a_n*b_i);
        un(m,IEN,k,j,i) = (c_n+c_i*a_n-c_n*a_i)/(1.-a_i-b_n+a_i*b_n-a_n*b_i);
      }
    });
  }

  // Compute stiff source term (ion-neutral drag) using variables updated in this stage,
  // i.e R(U^n), for use in later stages.  Only required for istage = (1,2,3,[4])
  if (estage < pdriver->nexp_stages) {
    int s = istage-1;
    auto ui = pmhd->u0;
    auto un = phyd->u0;
    EOS_Data &eos_hyd = phyd->peos->eos_data;
    EOS_Data &eos_mhd = pmhd->peos->eos_data;
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
      if (eos_hyd.is_ideal) {
        Real gamma_i = eos_mhd.gamma, gamma_n = eos_hyd.gamma;
        Real fmass = 1./(1.+mi_mn);
        Real b1 = pmhd->bcc0(m,0,k,j,i);
        Real b2 = pmhd->bcc0(m,1,k,j,i);
        Real b3 = pmhd->bcc0(m,2,k,j,i);
        Real bsq = b1*b1+b2*b2+b3*b3;
        Real kin_i = (ui(m,IM1,k,j,i)*ui(m,IM1,k,j,i)+ui(m,IM2,k,j,i)*ui(m,IM2,k,j,i)+
                        ui(m,IM3,k,j,i)*ui(m,IM3,k,j,i))/ui(m,IDN,k,j,i);
        Real kin_n = (un(m,IM1,k,j,i)*un(m,IM1,k,j,i)+un(m,IM2,k,j,i)*un(m,IM2,k,j,i)+
                        un(m,IM3,k,j,i)*un(m,IM3,k,j,i))/un(m,IDN,k,j,i);
        Real f1 = drag*(ui(m,IDN,k,j,i)*un(m,IM1,k,j,i) -
                        un(m,IDN,k,j,i)*ui(m,IM1,k,j,i)) +
                        xi*un(m,IM1,k,j,i) - alpha*ui(m,IDN,k,j,i)*ui(m,IM1,k,j,i);
        Real f2 = drag*(ui(m,IDN,k,j,i)*un(m,IM2,k,j,i) -
                        un(m,IDN,k,j,i)*ui(m,IM2,k,j,i)) +
                        xi*un(m,IM2,k,j,i) - alpha*ui(m,IDN,k,j,i)*ui(m,IM2,k,j,i);
        Real f3 = drag*(ui(m,IDN,k,j,i)*un(m,IM3,k,j,i) -
                        un(m,IDN,k,j,i)*ui(m,IM3,k,j,i)) +
                        xi*un(m,IM3,k,j,i) - alpha*ui(m,IDN,k,j,i)*ui(m,IM3,k,j,i);
        Real fi_ui = (f1*ui(m,IM1,k,j,i)+f2*ui(m,IM2,k,j,i)+
                      f3*ui(m,IM3,k,j,i))/ui(m,IDN,k,j,i);
        Real fn_un =-(f1*un(m,IM1,k,j,i)+f2*un(m,IM2,k,j,i)+
                      f3*un(m,IM3,k,j,i))/un(m,IDN,k,j,i);
        Real ti_mi = (ui(m,IEN,k,j,i)-kin_i-bsq/2.)*(gamma_i-1)/ui(m,IDN,k,j,i);
        Real tn_mn = (un(m,IEN,k,j,i)-kin_n)*(gamma_n-1)/un(m,IDN,k,j,i);
        Real si = xi*un(m,IDN,k,j,i);
        Real sn = alpha*ui(m,IDN,k,j,i)*ui(m,IDN,k,j,i);
        Real u1_i = ui(m,IM1,k,j,i)/ui(m,IDN,k,j,i);
        Real u2_i = ui(m,IM2,k,j,i)/ui(m,IDN,k,j,i);
        Real u3_i = ui(m,IM3,k,j,i)/ui(m,IDN,k,j,i);
        Real u1_n = un(m,IM1,k,j,i)/un(m,IDN,k,j,i);
        Real u2_n = un(m,IM2,k,j,i)/un(m,IDN,k,j,i);
        Real u3_n = un(m,IM3,k,j,i)/un(m,IDN,k,j,i);
        Real du_sq = (u1_i-u1_n)*(u1_i-u1_n)+(u2_i-u2_n)*(u2_i-u2_n)+
                     (u3_i-u3_n)*(u3_i-u3_n);
        Real gi = si*du_sq/2. + si*tn_mn/mi_mn/(gamma_n-1) - sn*ti_mi/(gamma_i-1) +
                  drag*ui(m,IDN,k,j,i)*un(m,IDN,k,j,i) *
                  (3.*(tn_mn*fmass-ti_mi*(1.-fmass)) + fmass*du_sq);
        Real gn = sn*du_sq/2. + sn*ti_mi*mi_mn/(gamma_i-1) - si*tn_mn/(gamma_n-1) +
                  drag*ui(m,IDN,k,j,i)*un(m,IDN,k,j,i) *
                  (-3.*(tn_mn*fmass-ti_mi*(1.-fmass)) + (1.-fmass)*du_sq);
        ru_(s,m,8,k,j,i) = fi_ui + gi;
        ru_(s,m,9,k,j,i) = fn_un + gn;
      }
    });
  }

  return TaskStatus::complete;
}

} // namespace ion_neutral
