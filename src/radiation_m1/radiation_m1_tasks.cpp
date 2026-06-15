//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_tasks.cpp
//  \brief functions that control M1 tasks stored in tasklists in MeshBlockPack

#include <iostream>
#include <map>
#include <memory>
#include <mhd/mhd.hpp>
#include <string>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "tasklist/task_list.hpp"

namespace radiationm1 {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::AssembleRadiationM1Tasks
//! \brief Adds radiatoin tasks to appropriate task lists used by time
//! integrators. Called by MeshBlockPack::AddPhysics() function directly after
//! Hydro constructor. Many of the functions in the task list are implemented in
//! this file because they are simple, or they are wrappers that call one or
//! more other functions.
//!
//! "before_stagen" tasks are those that must be cmpleted over all MeshBlocks
//! BEFORE each stage can be run (such as posting MPI receives, setting
//! BoundaryCommStatus flags, etc)
//!
//! "stagen" tasks are those performed DURING each stage
//!
//! "after_stagen" tasks are those that can only be completed AFTER all the
//! "stagen" tasks are completed over ALL MeshBlocks for each stage, such as
//! clearing all MPI calls, etc.
//!
//! In addition there are "before_timeintegrator" and "after_timeintegrator"
//! task lists in the tl map, which are generally used for operator split tasks.

void RadiationM1::AssembleRadiationM1Tasks(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // assemble "before_stagen" task list
  id.M1_irecv = tl["opsplit_before_stagen"]->AddTask(&RadiationM1::InitRecv, this, none, "RadiationM1::InitRecv");

  // assemble "stagen" task list
  id.M1_copyu = tl["opsplit_stagen"]->AddTask(&RadiationM1::CopyCons, this, none, "RadiationM1::CopyU");
  id.M1_setmask =
      tl["opsplit_stagen"]->AddTask(&RadiationM1::SetMask, this, id.M1_copyu, "RadiationM1::SetMask");
  id.M1_closure =
      tl["opsplit_stagen"]->AddTask(&RadiationM1::FloorAndCalcClosure, this, id.M1_setmask, "RadiationM1::FloorAndCalcClosure");

  // decide what type of opacities to compute
  if (!params.matter_sources) {
    id.M1_mattersrc = id.M1_closure;
#if ENABLE_NURATES
  } else if (params.opacity_type == BnsNurates) {
    id.M1_mattersrc = tl["opsplit_stagen"]->AddTask(&RadiationM1::CalcOpacityNurates,
                                                    this, id.M1_closure, "RadiationM1::CalcOpacityNurates");
#endif
  } else if (params.opacity_type == Photons) {
    id.M1_mattersrc = tl["opsplit_stagen"]->AddTask(&RadiationM1::CalcOpacityPhotons,
                                                    this, id.M1_closure, "RadiationM1::CalcOpacityPhotons");
  } else {
    id.M1_mattersrc =
        tl["opsplit_stagen"]->AddTask(&RadiationM1::CalcOpacityToy, this, id.M1_closure, "RadiationM1::CalcOpacityToy");
  }

  id.M1_flux =
      tl["opsplit_stagen"]->AddTask(&RadiationM1::CalculateFluxes, this, id.M1_mattersrc, "RadiationM1::CalculateFluxes");
  id.M1_sendf = tl["opsplit_stagen"]->AddTask(&RadiationM1::SendFlux, this, id.M1_flux, "RadiationM1::SendFlux");
  id.M1_recvf = tl["opsplit_stagen"]->AddTask(&RadiationM1::RecvFlux, this, id.M1_sendf, "RadiationM1::RecvFlux");
  id.M1_rkupdt =
      tl["opsplit_stagen"]->AddTask(&RadiationM1::TimeUpdate, this, id.M1_recvf, "RadiationM1::TimeUpdate");
  id.M1_restu =
      tl["opsplit_stagen"]->AddTask(&RadiationM1::RestrictU, this, id.M1_rkupdt, "RadiationM1::RestrictU");
  id.M1_sendu = tl["opsplit_stagen"]->AddTask(&RadiationM1::SendU, this, id.M1_restu, "RadiationM1::SendU");
  id.M1_recvu = tl["opsplit_stagen"]->AddTask(&RadiationM1::RecvU, this, id.M1_sendu, "RadiationM1::RecvU");
  id.M1_prol = tl["opsplit_stagen"]->AddTask(&RadiationM1::Prolongate, this, id.M1_recvu, "RadiationM1::Prolongate");
  id.M1_bcs =
      tl["opsplit_stagen"]->AddTask(&RadiationM1::ApplyPhysicalBCs, this, id.M1_prol, "RadiationM1::ApplyPhysicalBCs");
  id.M1_newdt =
      tl["opsplit_stagen"]->AddTask(&RadiationM1::NewTimeStep, this, id.M1_bcs, "RadiationM1::NewTimeStep");

  // assemble "after_stagen" task list
  id.M1_csend = tl["opsplit_after_stagen"]->AddTask(&RadiationM1::ClearSend, this, none, "RadiationM1::ClearSend");
  // although RecvFlux/U functions check that all recvs complete, add ClearRecv
  // to task list anyway to catch potential bugs in MPI communication logic
  id.M1_crecv =
      tl["opsplit_after_stagen"]->AddTask(&RadiationM1::ClearRecv, this, id.M1_csend, "RadiationM1::ClearRecv");

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI),
//! and initialize all boundary receive status flags to waiting (with or without
//! MPI).

TaskStatus RadiationM1::InitRecv(Driver *pdrive, int stage) {
  // post receives for U
  TaskStatus tstat = pbval_u->InitRecv(nvarstot);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of U
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->InitFluxRecv(nvarstot);
  }
  if (tstat != TaskStatus::complete) return tstat;

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void RadiationM1::CopyCons
//! \brief Simple task list function that copies u0 --> u1 in first stage.
TaskStatus RadiationM1::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void RadiationM1::SetMask
//! \brief Sets the radiation excision mask from the coordinate (horizon/lapse/fixed)
//! excision mask and zeroes the radiation fields in masked cells. Mirrors THC's
//! THC_M1_SetMask, which derives thc_m1_mask from the matter (hydro) excision mask.
//! Runs each stage before the closure so that closure/opacities/Tmunu/update (all of
//! which already honor radiation_mask) see an up-to-date mask, and so that fluxes
//! across the excision boundary are reconstructed from zeroed states.
TaskStatus RadiationM1::SetMask(Driver *pdrive, int stage) {
  // If BH excision is disabled, the coordinate masks are not allocated and
  // radiation_mask remains all-false (set at construction); nothing to do.
  if (!pmy_pack->pcoord->coord_data.bh_excise) {
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // index limits including ghosts (match closure coverage so reconstruction near
  // the excision boundary uses zeroed states)
  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = (indcs.nx2 > 1) ? js - indcs.ng : js;
  int jeg = (indcs.nx2 > 1) ? je + indcs.ng : je;
  int ksg = (indcs.nx3 > 1) ? ks - indcs.ng : ks;
  int keg = (indcs.nx3 > 1) ? ke + indcs.ng : ke;

  auto &u0_ = u0;
  auto &mask_ = radiation_mask;
  auto &excision_ = pmy_pack->pcoord->excision_floor;
  auto &nvars_ = nvars;
  auto &nspecies_ = nspecies;

  par_for(
      "radiation_m1_setmask", DevExeSpace(), 0, nmb1, ksg, keg, jsg, jeg, isg, ieg,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        bool excise = excision_(m, k, j, i);
        mask_(m, k, j, i) = excise;
        if (excise) {
          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) = 0;
            if (nspecies_ > 1) {
              u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) = 0;
            }
          }
        }
      });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes
//! of conserved variables at fine/coarse boundaries
TaskStatus RadiationM1::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryVaLUES function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->PackAndSendFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes
//! of conserved variables at fine/coarse boundaries
TaskStatus RadiationM1::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::RestrictU
//! \brief Wrapper task list function to restrict conserved vars
TaskStatus RadiationM1::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved
//! variables
TaskStatus RadiationM1::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved
//! variables
TaskStatus RadiationM1::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::ApplyPhysicalBCs
//! \brief Wrapper task list function to call funtions that set physical and
//! user BCs,
TaskStatus RadiationM1::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs
  pbval_u->RadiationM1BCs((pmy_pack), (pbval_u->u_in), u0);

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive)
//! variables at fine/coarse boundaries with SMR/AMR
TaskStatus RadiationM1::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);

    // Step 1: apply physical BCs to the coarse array, so the prolongation stencil
    //         reads valid data in coarse ghost zones that sit at a physical boundary.
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      pbval_u->RadiationM1BCsCoarse(pmy_pack, pbval_u->u_in, coarse_u0);
    }

    // Step 2: prolongate fine ghost zones from the coarse array.
    pbval_u->ProlongateCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed.
//! Used in TaskList and in Driver::InitBoundaryValuesAndPrimitives() If
//! stage=(last stage):      clears sends of U, Flx_U, U_OA, U_Shr If (last
//! stage)>stage>=(0): clears sends of U, Flx_U,       U_Shr If stage=(-1):
//! clears sends of U If stage=(-4):              clears sends of U_Shr
TaskStatus RadiationM1::ClearSend(Driver *pdrive, int stage) {
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

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList RadiationM1::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have
//! completed. Used in TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears recvs of U, Flx_U, U_OA, U_Shr
//! If (last stage)>stage>=(0): clears recvs of U, Flx_U,       U_Shr
//! If stage=(-1):              clears recvs of U
//! If stage=(-4):              clears recvs of                 U_Shr
TaskStatus RadiationM1::ClearRecv(Driver *pdrive, int stage) {
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

  return tstat;
}

}  // namespace radiationm1
