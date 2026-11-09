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
#include "tasklist/task_list.hpp"
#include "hydro/hydro.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "nuclear/nuclear.hpp"

namespace nuclear {
void Nuclear::AssembleNuclearTasks(
  std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  // The burn only updates the active cells, so refresh the hydro ghost zones (and the
  // primitives in them) before the hydro stages read them, following
  // Driver::InitBoundaryValuesAndPrimitives().
  id.irecv     = tl["opsplit_before_stagen"]->AddTask(&Nuclear::InitRecvU, this, none, "Nuclear::InitRecvU");
  id.integrate = tl["opsplit_stagen"]->AddTask(&Nuclear::Integrate, this, none, "Nuclear::Integrate");
  id.sendu     = tl["opsplit_stagen"]->AddTask(&Nuclear::SendU, this, id.integrate, "Nuclear::SendU");
  id.recvu     = tl["opsplit_stagen"]->AddTask(&Nuclear::RecvU, this, id.sendu, "Nuclear::RecvU");
  id.setgz     = tl["opsplit_stagen"]->AddTask(&Nuclear::SetGhostZones, this, id.recvu, "Nuclear::SetGhostZones");
  id.csend     = tl["opsplit_after_stagen"]->AddTask(&Nuclear::ClearSendU, this, none, "Nuclear::ClearSendU");
  id.crecv     = tl["opsplit_after_stagen"]->AddTask(&Nuclear::ClearRecvU, this, id.csend, "Nuclear::ClearRecvU");
}

// The functions below wrap the Hydro or MHD tasks, looking up phydro/pmhd (built before
// Nuclear in MeshBlockPack::AddPhysics()) when the tasks run. They use stage = -1 where
// it matters: the burn computes no fluxes, so no
// flux-correction messages are posted or cleared. With MHD, B is exchanged along with
// U (the burn does not change it), because MHD::InitRecv/ClearRecv always post/clear
// the receives of both, as in Driver::InitBoundaryValuesAndPrimitives().

TaskStatus Nuclear::InitRecvU(Driver *pdrive, int stage) {
  if (pmy_pack->pmhd != nullptr) {
    return pmy_pack->pmhd->InitRecv(pdrive, -1);  // stage < 0 suppresses InitFluxRecv
  }
  hydro::Hydro *phyd = pmy_pack->phydro;
  if (phyd == nullptr) return TaskStatus::complete;
  return phyd->InitRecv(pdrive, -1);  // stage < 0 suppresses InitFluxRecv
}

TaskStatus Nuclear::SendU(Driver *pdrive, int stage) {
  mhd::MHD *pmhd = pmy_pack->pmhd;
  if (pmhd != nullptr) {
    (void) pmhd->RestrictU(pdrive, stage);
    (void) pmhd->RestrictB(pdrive, stage);
    TaskStatus tstat = pmhd->SendU(pdrive, stage);
    if (tstat != TaskStatus::complete) return tstat;
    return pmhd->SendB(pdrive, stage);
  }
  hydro::Hydro *phyd = pmy_pack->phydro;
  if (phyd == nullptr) return TaskStatus::complete;
  (void) phyd->RestrictU(pdrive, stage);
  return phyd->SendU(pdrive, stage);
}

TaskStatus Nuclear::RecvU(Driver *pdrive, int stage) {
  mhd::MHD *pmhd = pmy_pack->pmhd;
  if (pmhd != nullptr) {
    TaskStatus tstat = pmhd->RecvU(pdrive, stage);
    if (tstat != TaskStatus::complete) return tstat;
    return pmhd->RecvB(pdrive, stage);
  }
  hydro::Hydro *phyd = pmy_pack->phydro;
  if (phyd == nullptr) return TaskStatus::complete;
  return phyd->RecvU(pdrive, stage);
}

// Prolongation at fine/coarse boundaries, physical BCs, and primitives everywhere
TaskStatus Nuclear::SetGhostZones(Driver *pdrive, int stage) {
  mhd::MHD *pmhd = pmy_pack->pmhd;
  if (pmhd != nullptr) {
    (void) pmhd->Prolongate(pdrive, stage);
    (void) pmhd->ApplyPhysicalBCs(pdrive, stage);
    if (pmy_pack->pdyngr != nullptr) {
      return pmy_pack->pdyngr->ConToPrim(pdrive, stage);
    }
    return pmhd->ConToPrim(pdrive, stage);
  }
  hydro::Hydro *phyd = pmy_pack->phydro;
  if (phyd == nullptr) return TaskStatus::complete;
  (void) phyd->Prolongate(pdrive, stage);
  (void) phyd->ApplyPhysicalBCs(pdrive, stage);
  return phyd->ConToPrim(pdrive, stage);
}

TaskStatus Nuclear::ClearSendU(Driver *pdrive, int stage) {
  if (pmy_pack->pmhd != nullptr) {
    return pmy_pack->pmhd->ClearSend(pdrive, -1);  // stage = -1 only clears U, B
  }
  hydro::Hydro *phyd = pmy_pack->phydro;
  if (phyd == nullptr) return TaskStatus::complete;
  return phyd->ClearSend(pdrive, -1);  // stage = -1 only clears SendU
}

TaskStatus Nuclear::ClearRecvU(Driver *pdrive, int stage) {
  if (pmy_pack->pmhd != nullptr) {
    return pmy_pack->pmhd->ClearRecv(pdrive, -1);  // stage = -1 only clears U, B
  }
  hydro::Hydro *phyd = pmy_pack->phydro;
  if (phyd == nullptr) return TaskStatus::complete;
  return phyd->ClearRecv(pdrive, -1);  // stage = -1 only clears RecvU
}
}  // namespace nuclear
