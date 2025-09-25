//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_tasks.cpp
//! \brief functions that control z4c tasks in the appropriate task list

#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <cstdio>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/horizon_dump.hpp"
#include "z4c/z4c.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "z4c/cce/cce.hpp"

namespace z4c {

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::QueueZ4cTasks
//! \brief queue Z4c tasks into NumericalRelativity
void Z4c::QueueZ4cTasks() {
  printf("AssembleZ4cTasks\n");
  using namespace mhd;     // NOLINT(build/namespaces)
  using namespace numrel;  // NOLINT(build/namespaces)
  NumericalRelativity *pnr = pmy_pack->pnr;
  auto &indcs = pmy_pack->pmesh->mb_indcs;

  // Start task list
  pnr->QueueTask(&Z4c::InitRecv, this, Z4c_Recv, "Z4c_Recv", Task_Start);
  pnr->QueueTask(&Z4c::InitRecvWeyl, this, Z4c_IRecvW, "Z4c_IRecvW", Task_Start);

  // Run task list
  pnr->QueueTask(&Z4c::CopyU, this, Z4c_CopyU, "Z4c_CopyU", Task_Run);
  switch (indcs.ng) {
    case 2:
      pnr->QueueTask(&Z4c::CalcRHS<2>, this, Z4c_CalcRHS, "Z4c_CalcRHS",
                     Task_Run, {Z4c_CopyU}, {MHD_SetTmunu});
      break;
    case 3:
      pnr->QueueTask(&Z4c::CalcRHS<3>, this, Z4c_CalcRHS, "Z4c_CalcRHS",
                     Task_Run, {Z4c_CopyU}, {MHD_SetTmunu});
      break;
    case 4:
      pnr->QueueTask(&Z4c::CalcRHS<4>, this, Z4c_CalcRHS, "Z4c_CalcRHS",
                     Task_Run, {Z4c_CopyU}, {MHD_SetTmunu});
      break;
  }
  pnr->QueueTask(&Z4c::Z4cBoundaryRHS, this, Z4c_SomBC, "Z4c_SomBC", Task_Run,
                 {Z4c_CalcRHS});
  pnr->QueueTask(&Z4c::ExpRKUpdate, this, Z4c_ExplRK, "Z4c_ExplRK", Task_Run,
                 {Z4c_SomBC},{MHD_EField});
  pnr->QueueTask(&Z4c::RestrictU, this, Z4c_RestU, "Z4c_RestU", Task_Run, {Z4c_ExplRK});
  pnr->QueueTask(&Z4c::SendU, this, Z4c_SendU, "Z4c_SendU", Task_Run, {Z4c_RestU});
  pnr->QueueTask(&Z4c::RecvU, this, Z4c_RecvU, "Z4c_RecvU", Task_Run, {Z4c_SendU});
  pnr->QueueTask(&Z4c::ApplyPhysicalBCs, this, Z4c_BCS, "Z4c_BCS", Task_Run, {Z4c_RecvU});
  pnr->QueueTask(&Z4c::Prolongate, this, Z4c_Prolong, "Z4c_Prolong", Task_Run, {Z4c_BCS});
  pnr->QueueTask(&Z4c::EnforceAlgConstr, this, Z4c_AlgC, "Z4c_AlgC", Task_Run,
                 {Z4c_Prolong});
  pnr->QueueTask(&Z4c::ConvertZ4cToADM, this, Z4c_Z4c2ADM, "Z4c_Z4c2ADM",
                 Task_Run, {Z4c_AlgC});
  if (pmy_pack->pdyngr != nullptr) {
    pnr->QueueTask(&Z4c::UpdateExcisionMasks, this, Z4c_Excise, "Z4c_Excise", Task_Run,
                   {Z4c_Z4c2ADM});
  }
  pnr->QueueTask(&Z4c::NewTimeStep, this, Z4c_Newdt, "Z4c_Newdt", Task_Run,
                 {Z4c_Z4c2ADM});

  // End task list
  pnr->QueueTask(&Z4c::ClearSend, this, Z4c_ClearS, "Z4c_ClearS", Task_End);
  pnr->QueueTask(&Z4c::ClearRecv, this, Z4c_ClearR, "Z4c_ClearR", Task_End, {Z4c_ClearS});
  /*pnr->QueueTask(&Z4c::Z4cToADM, this, Z4c_Z4c2ADM, "Z4c_Z4c2ADM", Task_End,
                 {Z4c_ClearR});*/
  pnr->QueueTask(&Z4c::ADMConstraints_, this, Z4c_ADMC, "Z4c_ADMC", Task_End,
  //               {Z4c_Z4c2ADM});
                 {Z4c_ClearR});
  pnr->QueueTask(&Z4c::CalcWeylScalar, this, Z4c_Weyl, "Z4c_Weyl", Task_End, {Z4c_ADMC});
  pnr->QueueTask(&Z4c::RestrictWeyl, this, Z4c_RestW, "Z4c_RestW", Task_End, {Z4c_Weyl});
  pnr->QueueTask(&Z4c::SendWeyl, this, Z4c_SendW, "Z4c_SendW", Task_End, {Z4c_RestW});
  pnr->QueueTask(&Z4c::RecvWeyl, this, Z4c_RecvW, "Z4c_RecvW", Task_End, {Z4c_SendW});
  pnr->QueueTask(&Z4c::ProlongateWeyl, this, Z4c_ProlW, "Z4c_ProlW", Task_End,
                 {Z4c_RecvW});
  pnr->QueueTask(&Z4c::ClearSendWeyl, this, Z4c_ClearSW, "Z4c_ClearS2", Task_End,
                 {Z4c_ProlW});
  pnr->QueueTask(&Z4c::ClearRecvWeyl, this, Z4c_ClearRW, "Z4c_ClearR2", Task_End,
                 {Z4c_ClearSW});
  pnr->QueueTask(&Z4c::CalcWaveForm, this, Z4c_Wave, "Z4c_Wave", Task_End,
                 {Z4c_ClearRW});
  pnr->QueueTask(&Z4c::TrackCompactObjects, this, Z4c_PT, "Z4c_PT", Task_End, {Z4c_Wave});
  pnr->QueueTask(&Z4c::CCEDump, this, Z4c_CCE, "CCEDump", Task_End, {Z4c_PT});
  pnr->QueueTask(&Z4c::DumpHorizons, this, Z4c_DumpHorizon, "Z4c_DumpHorizon",
                Task_End, {Z4c_CCE});
}
//----------------------------------------------------------------------------------------
//! \fn  void Wave::InitRecv
//! \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Wave variables.

TaskStatus Z4c::InitRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->InitRecv(nz4c);
  if (tstat != TaskStatus::complete) return tstat;
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::ClearRecv
//! \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Z4c::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ClearSend
//! \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Z4c::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::CopyU
//! \brief  copy u0 --> u1 in first stage

TaskStatus Z4c::CopyU(Driver *pdrive, int stage) {
  auto integrator = pdrive->integrator;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nz4c;
  auto &u0 = pmy_pack->pz4c->u0;
  auto &u1 = pmy_pack->pz4c->u1;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator.
  // Important to use vector inner loop for good performance on cpus
  if (integrator == "rk4") {
    Real &delta = pdrive->delta[stage-1];
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      par_for("CopyCons", DevExeSpace(),0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i){
        u1(m,n,k,j,i) += delta*u0(m,n,k,j,i);
      });
    }
  } else {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::SendU
//! \brief sends cell-centered conserved variables

TaskStatus Z4c::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RecvU
//! \brief receives cell-centered conserved variables

TaskStatus Z4c::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::EnforceAlgConstr
//! \brief

TaskStatus Z4c::EnforceAlgConstr(Driver *pdrive, int stage) {
  if (pmy_pack->pdyngr != nullptr || stage == pdrive->nexp_stages) {
    AlgConstr(pmy_pack);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ADMToZ4c_
//! \brief

TaskStatus Z4c::ConvertZ4cToADM(Driver *pdrive, int stage) {
  if (pmy_pack->pdyngr != nullptr || stage == pdrive->nexp_stages) {
    Z4cToADM(pmy_pack);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::UpdateExcisionMasks
//! \brief

TaskStatus Z4c::UpdateExcisionMasks(Driver *pdrive, int stage) {
  if (pmy_pack->pcoord->coord_data.bh_excise && stage == pdrive->nexp_stages) {
    pmy_pack->pcoord->UpdateExcisionMasks();
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ADM_Constraints_
//! \brief

TaskStatus Z4c::ADMConstraints_(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (stage == pdrive->nexp_stages) {
    switch (indcs.ng) {
      case 2: ADMConstraints<2>(pmy_pack);
              break;
      case 3: ADMConstraints<3>(pmy_pack);
              break;
      case 4: ADMConstraints<4>(pmy_pack);
              break;
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RestrictU
//! \brief

TaskStatus Z4c::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Z4c::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse boundaries with SMR/AMR

TaskStatus Z4c::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
//    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    pbval_u->ProlongateCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ApplyPhysicalBCs
//! \brief

TaskStatus Z4c::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // only apply BCs if domain is not strictly periodic
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    // physical BCs
    pbval_u->Z4cBCs((pmy_pack), (pbval_u->u_in), u0, coarse_u0);

    // user BCs
    if (pmy_pack->pmesh->pgen->user_bcs) {
      (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
    }
  }
  return TaskStatus::complete;
}

TaskStatus Z4c::TrackCompactObjects(Driver *pdrive, int stage) {
  if (stage == pdrive->nexp_stages) {
    for (auto & pt : ptracker) {
      pt->InterpolateVelocity(pmy_pack);
      pt->EvolveTracker(pmy_pack);
      pt->WriteTracker();
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// ! \fn TaskList CCEDump
// ! \brief CCE initial data for Pittnull code (cce dumps for Pittnull).

TaskStatus Z4c::CCEDump(Driver *pdrive, int stage) {
  float time_32 = static_cast<float>(pmy_pack->pmesh->time);
  float next_32 = static_cast<float>(cce_dump_last_output_time+cce_dump_dt);
  if ((time_32 >= next_32)) {
    if (stage == pdrive->nexp_stages) {
      //printf("%s:(ctime,dt)=(%f,%f)",__func__,pmy_pack->pmesh->time,cce_dump_dt);
      for (auto cce : pmy_pack->pz4c_cce) {
        cce->InterpolateAndDecompose(pmy_pack);
      }
      cce_dump_last_output_time = time_32;
    }
  }
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void Z4c::CalcWeylScalar_
//! \brief

TaskStatus Z4c::CalcWeylScalar(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if (last_output_time==time_32 && stage == pdrive->nexp_stages) {
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      switch (indcs.ng) {
        case 2: Z4cWeyl<2>(pmy_pack);
                break;
        case 3: Z4cWeyl<3>(pmy_pack);
                break;
        case 4: Z4cWeyl<4>(pmy_pack);
                break;
      }
    }
    return TaskStatus::complete;
  }
}

TaskStatus Z4c::CalcWaveForm(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      WaveExtr(pmy_pack);
    }
    return TaskStatus::complete;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::SendWeyl
//! \brief sends cell-centered conserved variables

TaskStatus Z4c::SendWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      TaskStatus tstat = pbval_weyl->PackAndSendCC(u_weyl, coarse_u_weyl);
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RecvWeyl
//! \brief receives cell-centered conserved variables

TaskStatus Z4c::RecvWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      TaskStatus tstat = pbval_weyl->RecvAndUnpackCC(u_weyl, coarse_u_weyl);
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::RestrictU
//! \brief

TaskStatus Z4c::RestrictWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      if (pmy_pack->pmesh->multilevel) {
        pmy_pack->pmesh->pmr->RestrictCC(u_weyl, coarse_u_weyl, true);
      }
    }
    return TaskStatus::complete;
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Z4c::ProlongateWeyl
//! \brief Wrapper task list function to prolongate weyl scalar
//! at fine/coarse boundaries with SMR/AMR

TaskStatus Z4c::ProlongateWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      if (pmy_pack->pmesh->multilevel) {
        pbval_weyl->ProlongateCC(u_weyl, coarse_u_weyl);
      }
    }
    return TaskStatus::complete;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::InitRecvWeyl
//! \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Wave variables.

TaskStatus Z4c::InitRecvWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    float next_32 = static_cast<float>(last_output_time+waveform_dt);
    if (((time_32 >= next_32) || (time_32 == 0)) && stage == pdrive->nexp_stages) {
      last_output_time = time_32;
      TaskStatus tstat = pbval_weyl->InitRecv(2);
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Wave::ClearRecvWeyl
//! \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Z4c::ClearRecvWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      TaskStatus tstat = pbval_weyl->ClearRecv();
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Z4c::ClearSendWeyl
//! \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Z4c::ClearSendWeyl(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->nrad == 0) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    if ((last_output_time==time_32) && (stage == pdrive->nexp_stages)) {
      TaskStatus tstat = pbval_weyl->ClearSend();
      return tstat;
    } else {
      return TaskStatus::complete;
    }
  }
}

TaskStatus Z4c::DumpHorizons(Driver *pdrive, int stage) {
  if (pmy_pack->pz4c->phorizon_dump.size() == 0 || stage != pdrive->nexp_stages) {
    return TaskStatus::complete;
  } else {
    float time_32 = static_cast<float>(pmy_pack->pmesh->time);
    float next_32 = static_cast<float>(pmy_pack->pz4c->phorizon_dump[0]
                                    ->horizon_last_output_time
                                    +pmy_pack->pz4c->phorizon_dump[0]->horizon_dt);
    if (((time_32 >= next_32) || (time_32 == 0))) {
      int i = 0;
      for (auto & hd : phorizon_dump) {
        hd->horizon_last_output_time = time_32;
        hd->SetGridAndInterpolate(pmy_pack->pz4c->ptracker[i]->GetPos());
        i++;
      }
    }
    return TaskStatus::complete;
  }

  return TaskStatus::complete;
}

} // namespace z4c
