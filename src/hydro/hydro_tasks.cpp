//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_tasks.cpp
//! \brief functions that control Hydro tasks stored in tasklists in MeshBlockPack

#include <map>
#include <memory>
#include <string>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleHydroTasks
//! \brief Adds hydro tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after Hydro constructor.
//! Many of the functions in the task list are implemented in this file because they are
//! simple, or they are wrappers that call one or more other functions.
//!
//! "before_stagen" tasks are those that must be cmpleted over all MeshBlocks BEFORE each
//! stage can be run (such as posting MPI receives, setting BoundaryCommStatus flags, etc)
//!
//! "stagen" tasks are those performed DURING each stage
//!
//! "after_stagen" tasks are those that can only be completed AFTER all the "stagen" tasks
//! are completed over ALL MeshBlocks for each stage, such as clearing all MPI calls, etc.
//!
//! In addition there are "before_timeintegrator" and "after_timeintegrator" task lists
//! in the tl map, which are generally used for operator split tasks.

void Hydro::AssembleHydroTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // assemble "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&Hydro::InitRecv, this, none);

  // assemble "stagen" task list
  id.copyu     = tl["stagen"]->AddTask(&Hydro::CopyCons, this, none);
  id.flux      = tl["stagen"]->AddTask(&Hydro::Fluxes,this,id.copyu);
  id.sendf     = tl["stagen"]->AddTask(&Hydro::SendFlux, this, id.flux);
  id.recvf     = tl["stagen"]->AddTask(&Hydro::RecvFlux, this, id.sendf);
  id.rkupdt    = tl["stagen"]->AddTask(&Hydro::RKUpdate, this, id.recvf);
  id.srctrms   = tl["stagen"]->AddTask(&Hydro::HydroSrcTerms, this, id.rkupdt);
  id.sendu_oa  = tl["stagen"]->AddTask(&Hydro::SendU_OA, this, id.srctrms);
  id.recvu_oa  = tl["stagen"]->AddTask(&Hydro::RecvU_OA, this, id.sendu_oa);
  id.restu     = tl["stagen"]->AddTask(&Hydro::RestrictU, this, id.recvu_oa);
  id.sendu     = tl["stagen"]->AddTask(&Hydro::SendU, this, id.restu);
  id.recvu     = tl["stagen"]->AddTask(&Hydro::RecvU, this, id.sendu);
  id.sendu_shr = tl["stagen"]->AddTask(&Hydro::SendU_Shr, this, id.recvu);
  id.recvu_shr = tl["stagen"]->AddTask(&Hydro::RecvU_Shr, this, id.sendu_shr);
  id.bcs       = tl["stagen"]->AddTask(&Hydro::ApplyPhysicalBCs, this, id.recvu_shr);
  id.prol      = tl["stagen"]->AddTask(&Hydro::Prolongate, this, id.bcs);
  id.c2p       = tl["stagen"]->AddTask(&Hydro::ConToPrim, this, id.prol);
  id.newdt     = tl["stagen"]->AddTask(&Hydro::NewTimeStep, this, id.c2p);

  // assemble "after_stagen" task list
  id.csend = tl["after_stagen"]->AddTask(&Hydro::ClearSend, this, none);
  // although RecvFlux/U functions check that all recvs complete, add ClearRecv to
  // task list anyways to catch potential bugs in MPI communication logic
  id.crecv = tl["after_stagen"]->AddTask(&Hydro::ClearRecv, this, id.csend);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI), and
//! initialize all boundary receive status flags to waiting (with or without MPI).

TaskStatus Hydro::InitRecv(Driver *pdrive, int stage) {
  // post receives for U
  TaskStatus tstat = pbval_u->InitRecv(nhydro+nscalars);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of U
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->InitFluxRecv(nhydro+nscalars);
  }
  if (tstat != TaskStatus::complete) return tstat;

  // with orbital advection post receives for U
  // only execute for (last stage) AND (3D OR 2d_r_phi)
  if (porb_u != nullptr) {
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->InitRecv();
    }
  }
  if (tstat != TaskStatus::complete) return tstat;

  // with shearing box boundaries calculate x2-distance x1-boundarues have sheared and
  // with MPI post receives for U.
  // only execute if (3D OR 2d_r_phi)
  if (psbox_u != nullptr) {
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      Real time = pmy_pack->pmesh->time;
      if (stage == pdrive->nexp_stages) {
        time += pmy_pack->pmesh->dt;
      }
      tstat = psbox_u->InitRecv(time);
    }
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CopyCons
//! \brief Simple task list function that copies u0 --> u1 in first stage.  Extended to
//!  handle RK register logic at given stage

TaskStatus Hydro::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  } else {
    if (pdrive->integrator == "rk4") {
      // parallel loop to update u1 with u0 at later stages, only for rk4
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int is = indcs.is, ie = indcs.ie;
      int js = indcs.js, je = indcs.je;
      int ks = indcs.ks, ke = indcs.ke;
      int nmb1 = pmy_pack->nmb_thispack - 1;
      int nvar = nhydro + nscalars;
      auto &u0 = pmy_pack->phydro->u0;
      auto &u1 = pmy_pack->phydro->u1;
      Real &delta = pdrive->delta[stage-1];
      par_for("rk4_copy_cons", DevExeSpace(),0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        u1(m,n,k,j,i) += delta*u0(m,n,k,j,i);
      });
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::Fluxes
//! \brief Wrapper task list function that calls everything necessary to compute fluxes
//! of conserved variables

TaskStatus Hydro::Fluxes(Driver *pdrive, int stage) {
  // select which calculate_flux function to call based on rsolver_method
  if (rsolver_method == Hydro_RSolver::advect) {
    CalculateFluxes<Hydro_RSolver::advect>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::llf) {
    CalculateFluxes<Hydro_RSolver::llf>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hlle) {
    CalculateFluxes<Hydro_RSolver::hlle>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hllc) {
    CalculateFluxes<Hydro_RSolver::hllc>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::roe) {
    CalculateFluxes<Hydro_RSolver::roe>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::llf_sr) {
    CalculateFluxes<Hydro_RSolver::llf_sr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hlle_sr) {
    CalculateFluxes<Hydro_RSolver::hlle_sr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hllc_sr) {
    CalculateFluxes<Hydro_RSolver::hllc_sr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::llf_gr) {
    CalculateFluxes<Hydro_RSolver::llf_gr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hlle_gr) {
    CalculateFluxes<Hydro_RSolver::hlle_gr>(pdrive, stage);
  }

  // Add viscous, heat-flux, etc fluxes
  if (pvisc != nullptr) {
    pvisc->IsotropicViscousFlux(w0, pvisc->nu_iso, peos->eos_data, uflx);
  }
  if (pcond != nullptr) {
    pcond->AddHeatFlux(w0, peos->eos_data, uflx);
  }

  // call FOFC if necessary
  if (use_fofc) {
    FOFC(pdrive, stage);
  } else if (pmy_pack->pcoord->is_general_relativistic) {
    if (pmy_pack->pcoord->coord_data.bh_excise) {
      FOFC(pdrive, stage);
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus Hydro::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryVaLUES function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->PackAndSendFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus Hydro::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::HydroSrcTerms
//! \brief Wrapper task list function to apply source terms to conservative vars
//! Note source terms must be computed using only primitives (w0), as the conserved
//! variables (u0) have already been partially updated when this fn called.

TaskStatus Hydro::HydroSrcTerms(Driver *pdrive, int stage) {
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);

  // Add physics source terms (must be computed from primitives)
  if (psrc != nullptr) psrc->ApplySrcTerms(w0, peos->eos_data,  beta_dt, u0);

  // Add shearing box source terms for cell-centered hydro variables
  if (psbox_u != nullptr) psbox_u->SourceTermsCC(w0, peos->eos_data, beta_dt, u0);

  // Add coordinate source terms in GR.  Again, must be computed with only primitives.
  if (pmy_pack->pcoord->is_general_relativistic) {
    pmy_pack->pcoord->CoordSrcTerms(w0, peos->eos_data, beta_dt, u0);
  }

  // Add user source terms
  if (pmy_pack->pmesh->pgen->user_srcs) {
    (pmy_pack->pmesh->pgen->user_srcs_func)(pmy_pack->pmesh, beta_dt);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendU_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus Hydro::SendU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_u != nullptr) {
    // only execute if (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->PackAndSendCC(u0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvU_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection
//! Orbital remap is performed in this step.

TaskStatus Hydro::RecvU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_u != nullptr) {
    // only execute if (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->RecvAndUnpackCC(u0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RestrictU
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus Hydro::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved variables

TaskStatus Hydro::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved variables

TaskStatus Hydro::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendU_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus Hydro::SendU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_u != nullptr) {
    // only execute if (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      tstat = psbox_u->PackAndSendCC(u0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvU_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus Hydro::RecvU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_u != nullptr) {
    // only execute if (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      tstat = psbox_u->RecvAndUnpackCC(u0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ApplyPhysicalBCs
//! \brief Wrapper task list function to call funtions that set physical and user BCs,

TaskStatus Hydro::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs
  pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse boundaries with SMR/AMR

TaskStatus Hydro::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    if (pmy_pack->pmesh->pmr->prolong_prims) {
      pbval_u->ConsToPrimCoarseBndry(coarse_u0, coarse_w0);
      pbval_u->ProlongateCC(w0, coarse_w0);
      pbval_u->PrimToConsFineBndry(w0, u0);
    } else {
      pbval_u->ProlongateCC(u0, coarse_u0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ConToPrim
//! \brief Wrapper task list function to call ConsToPrim over entire mesh (including gz)

TaskStatus Hydro::ConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  peos->ConsToPrim(u0, w0, false, 0, n1m1, 0, n2m1, 0, n3m1);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears sends of U, Flx_U, U_OA, U_Shr
//! If (last stage)>stage>=(0): clears sends of U, Flx_U,       U_Shr
//! If stage=(-1):              clears sends of U
//! If stage=(-4):              clears sends of                 U_Shr

TaskStatus Hydro::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat;
  // check sends of U complete
  if ((stage >= 0) || (stage == -1)) {
    tstat = pbval_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check sends of restricted fluxes of U complete
  // do not check flux send for ICs (stage < 0)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection check sends of U complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if (porb_u != nullptr) {
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries check sends of U complete
  // only execute when (shearing box defined) AND (stage>=0 or -4) AND (3D OR 2d_r_phi)
  if (psbox_u != nullptr) {
    if (((stage >= 0) || (stage == -4)) &&
        (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi)) {
      tstat = psbox_u->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears recvs of U, Flx_U, U_OA, U_Shr
//! If (last stage)>stage>=(0): clears recvs of U, Flx_U,       U_Shr
//! If stage=(-1):              clears recvs of U
//! If stage=(-4):              clears recvs of                 U_Shr

TaskStatus Hydro::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat;
  // check receives of U complete
  if ((stage >= 0) || (stage == -1)) {
    tstat = pbval_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check receives of restricted fluxes of U complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection check receives of U complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if (porb_u != nullptr) {
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries check receives of U complete
  // only execute when (shearing box defined) AND (stage>=0 or -4) AND (3D OR 2d_r_phi)
  if (psbox_u != nullptr) {
    if (((stage >= 0) || (stage == -4)) &&
        (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi)) {
      tstat = psbox_u->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return tstat;
}

} // namespace hydro
