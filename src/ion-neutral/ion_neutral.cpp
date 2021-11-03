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

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters

IonNeutral::IonNeutral(MeshBlockPack *pp, ParameterInput *pin, Driver *pdrive) :
  pmy_pack(pp),
  ru("ru",1,1,1,1,1,1)
{
  // Read drag coeff
  drag_coeff = pin->GetReal("ion-neutral","drag_coeff");

  // allocate memory for stiff source term R(U^n)
  int nimp_stages = pdrive->nimp_stages;
  if (nimp_stages == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "IonNetral MHD can only be run with ImEx integrators." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int nmb = pmy_pack->nmb_thispack;
  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(ru, nimp_stages, nmb, 6, ncells3, ncells2, ncells1);
}

//----------------------------------------------------------------------------------------
//! \fn  void IonNeutral::AssembleIonNeutralTasks
//  \brief Adds tasks for ion-neutral (two-fluid) mhd to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysicsModules() function directly after MHD constrctr
  
void IonNeutral::AssembleIonNeutralTasks(TaskList &start, TaskList &run, TaskList &end)
{ 
  TaskID none(0);
  using namespace hydro;
  using namespace mhd;
  MHD *pmhd = pmy_pack->pmhd;
  Hydro *phyd = pmy_pack->phydro;
  
  // start task list
  id.i_irecv = start.AddTask(&MHD::InitRecv, pmhd, none);
  id.n_irecv = start.AddTask(&Hydro::InitRecv, phyd, none);

  // run task list
  id.impl_2x = run.AddTask(&IonNeutral::FirstTwoImpRK, this, none);

  // select which calculate_flux function to add based on MHD rsolver_method
  if (pmhd->rsolver_method == MHD_RSolver::advect) {
    id.i_flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::advect>,pmhd,id.impl_2x);
  } else if (pmhd->rsolver_method == MHD_RSolver::llf) {
    id.i_flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::llf>,pmhd,id.impl_2x);
  } else if (pmhd->rsolver_method == MHD_RSolver::hlle) {
    id.i_flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::hlle>,pmhd,id.impl_2x);
  } else if (pmhd->rsolver_method == MHD_RSolver::hlld) {
    id.i_flux = run.AddTask(&MHD::CalcFluxes<MHD_RSolver::hlld>,pmhd,id.impl_2x);
  }
  id.i_expl = run.AddTask(&MHD::ExpRKUpdate, pmhd, id.i_flux);

  // select which calculate_flux function to add based on Hydro rsolver_method
  if (phyd->rsolver_method == Hydro_RSolver::advect) {
    id.n_flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::advect>,phyd,id.i_expl);
  } else if (phyd->rsolver_method == Hydro_RSolver::llf) {
    id.n_flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::llf>,phyd,id.i_expl);
  } else if (phyd->rsolver_method == Hydro_RSolver::hlle) {
    id.n_flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::hlle>,phyd,id.i_expl);
  } else if (phyd->rsolver_method == Hydro_RSolver::hllc) {
    id.n_flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::hllc>,phyd,id.i_expl);
  } else if (phyd->rsolver_method == Hydro_RSolver::roe) {
    id.n_flux = run.AddTask(&Hydro::CalcFluxes<Hydro_RSolver::roe>,phyd,id.i_expl);
  }
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

  // end task list
  id.i_clear = end.AddTask(&MHD::ClearSend, pmhd, none);
  id.n_clear = end.AddTask(&Hydro::ClearSend, phyd, none);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn IonNeutral::FirstTwoImpRK 
//  \brief Executes first two implicit stages of the ImEx integrator for ion-neutral
//  drag term.  Should be the first task called in TaskList.

TaskStatus IonNeutral::FirstTwoImpRK(Driver *pdrive, int stage)
{
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
  phyd->peos->ConsToPrim(phyd->u0, phyd->w0);
  pmhd->peos->ConsToPrim(pmhd->u0, pmhd->b0, pmhd->w0, pmhd->bcc0);

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
//  where ui=pmhd->u0 and un=phydro->u0


TaskStatus IonNeutral::ImpRKUpdate(Driver *pdriver, int estage)
{
  // # of implicit stage (1,2,3,4,[5]).  Note estage=(# of explicit stage)=(1,2,[3])
  // estage <= 0 corresponds to first two fully implicit stages
  int istage = estage + 2;

  auto &indcs = pmy_pack->pcoord->mbdata.indcs;
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
    auto ru_ = ru;
    par_for_outer("imex_exp",DevExeSpace(),scr_size,scr_level,0,nmb1,0,(n3-1),0,(n2-1),
      KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j)
      {
        for (int s=0; s<=(istage-2); ++s) {
          Real adt = a_twid[istage-2][s]*dt;
          par_for_inner(member, 0, (n1-1), [&](const int i)
          {
            ui(m,IM1,k,j,i) += adt*ru_(s,m,0,k,j,i);
            ui(m,IM2,k,j,i) += adt*ru_(s,m,1,k,j,i);
            ui(m,IM3,k,j,i) += adt*ru_(s,m,2,k,j,i);
            un(m,IM1,k,j,i) += adt*ru_(s,m,3,k,j,i);
            un(m,IM2,k,j,i) += adt*ru_(s,m,4,k,j,i);
            un(m,IM3,k,j,i) += adt*ru_(s,m,5,k,j,i);
          });
        }
      }
    );
  }

  // Update ion/neutral momentum equations with analytic solution of implicit difference
  // equations for ion-neutral drag.
  // Only required for istage = (1,2,3,[4])
  if (estage < pdriver->nexp_stages) {
    Real adt = drag_coeff*(pdriver->a_impl)*(pmy_pack->pmesh->dt);
    auto ui = pmhd->u0;
    auto un = phyd->u0;
    par_for("imex_imp",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
      {
        Real denom = 1.0 + adt*(ui(m,IDN,k,j,i) + un(m,IDN,k,j,i));
        // compute new ion/neutral momenta in x1
        Real sum = (ui(m,IM1,k,j,i) + un(m,IM1,k,j,i));
        Real u_i = (ui(m,IM1,k,j,i) + adt*ui(m,IDN,k,j,i)*sum)/denom;
        Real u_n = (un(m,IM1,k,j,i) + adt*un(m,IDN,k,j,i)*sum)/denom;
        ui(m,IM1,k,j,i) = u_i;
        un(m,IM1,k,j,i) = u_n;
        // compute new ion/neutral momenta in x2
        sum = (ui(m,IM2,k,j,i) + un(m,IM2,k,j,i));
        u_i = (ui(m,IM2,k,j,i) + adt*ui(m,IDN,k,j,i)*sum)/denom;
        u_n = (un(m,IM2,k,j,i) + adt*un(m,IDN,k,j,i)*sum)/denom;
        ui(m,IM2,k,j,i) = u_i;
        un(m,IM2,k,j,i) = u_n;
        // compute new ion/neutral momenta in x3
        sum = (ui(m,IM3,k,j,i) + un(m,IM3,k,j,i));
        u_i = (ui(m,IM3,k,j,i) + adt*ui(m,IDN,k,j,i)*sum)/denom;
        u_n = (un(m,IM3,k,j,i) + adt*un(m,IDN,k,j,i)*sum)/denom;
        ui(m,IM3,k,j,i) = u_i;
        un(m,IM3,k,j,i) = u_n;
      }
    );
  }

  // Compute stiff source term (ion-neutral drag) using variables updated in this stage,
  // i.e R(U^n), for use in later stages.  Only required for istage = (1,2,3,[4])
  if (estage < pdriver->nexp_stages) {
    int s = istage-1;
    auto ui = pmhd->u0;
    auto un = phyd->u0;
    auto drag = drag_coeff;
    auto ru_ = ru;
    par_for("imex_rup",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
      {
        // drag term in IM1 component of ion momentum
        ru_(s,m,0,k,j,i) = drag*(ui(m,IDN,k,j,i)*un(m,IM1,k,j,i) -
                                 un(m,IDN,k,j,i)*ui(m,IM1,k,j,i));
        // drag term in IM2 component of ion momentum
        ru_(s,m,1,k,j,i) = drag*(ui(m,IDN,k,j,i)*un(m,IM2,k,j,i) -
                                 un(m,IDN,k,j,i)*ui(m,IM2,k,j,i));
        // drag term in IM3 component of ion momentum
        ru_(s,m,2,k,j,i) = drag*(ui(m,IDN,k,j,i)*un(m,IM3,k,j,i) -
                                 un(m,IDN,k,j,i)*ui(m,IM3,k,j,i));
        // drag term in IM1 component of neutral momentum
        ru_(s,m,3,k,j,i) = drag*(un(m,IDN,k,j,i)*ui(m,IM1,k,j,i) -
                                 ui(m,IDN,k,j,i)*un(m,IM1,k,j,i));
        // drag term in IM2 component of neutral momentum
        ru_(s,m,4,k,j,i) = drag*(un(m,IDN,k,j,i)*ui(m,IM2,k,j,i) -
                                 ui(m,IDN,k,j,i)*un(m,IM2,k,j,i));
        // drag term in IM3 component of neutral momentum
        ru_(s,m,5,k,j,i) = drag*(un(m,IDN,k,j,i)*ui(m,IM3,k,j,i) -
                                 ui(m,IDN,k,j,i)*un(m,IM3,k,j,i));
      }
    );
  }

  return TaskStatus::complete;
}
