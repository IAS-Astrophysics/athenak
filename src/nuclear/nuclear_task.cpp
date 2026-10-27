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
#include "nuclear/nuclear.hpp"

namespace nuclear {
	void Nuclear::AssembleNuclearTasks(
	  std::map<std::string, std::shared_ptr<TaskList>> tl) {
	  TaskID none(0);
	  // The burn only updates the active cells, so refresh the hydro ghost zones (and the
	  // primitives in them) before the hydro stages read them, following
	  // Driver::InitBoundaryValuesAndPrimitives().
	  TaskID irecv = tl["opsplit_before_stagen"]->AddTask(&Nuclear::InitRecvU, this, none, "Nuclear::InitRecvU");

	  id.Nuclear_integrate = tl["opsplit_stagen"]->AddTask(&Nuclear::Integrate, this, none, "Nuclear::Integrate");
	  TaskID sendu = tl["opsplit_stagen"]->AddTask(&Nuclear::SendU, this, id.Nuclear_integrate, "Nuclear::SendU");
	  TaskID recvu = tl["opsplit_stagen"]->AddTask(&Nuclear::RecvU, this, sendu, "Nuclear::RecvU");
	  TaskID setgz = tl["opsplit_stagen"]->AddTask(&Nuclear::SetGhostZones, this, recvu, "Nuclear::SetGhostZones");

	  TaskID csend = tl["opsplit_after_stagen"]->AddTask(&Nuclear::ClearSendU, this, none, "Nuclear::ClearSendU");
	  TaskID crecv = tl["opsplit_after_stagen"]->AddTask(&Nuclear::ClearRecvU, this, csend, "Nuclear::ClearRecvU");
	  (void) irecv; (void) setgz; (void) crecv;
	}

	// The functions below wrap the Hydro tasks. Hydro is constructed after Nuclear in
	// MeshBlockPack::AddPhysics(), so phydro is looked up when the tasks run. They use
	// stage = -1 where it matters: the burn computes no fluxes, so no flux-correction
	// messages are posted or cleared.

	TaskStatus Nuclear::InitRecvU(Driver *pdrive, int stage) {
	  hydro::Hydro *phyd = pmy_pack->phydro;
	  if (phyd == nullptr) return TaskStatus::complete;
	  return phyd->InitRecv(pdrive, -1);  // stage < 0 suppresses InitFluxRecv
	}

	TaskStatus Nuclear::SendU(Driver *pdrive, int stage) {
	  hydro::Hydro *phyd = pmy_pack->phydro;
	  if (phyd == nullptr) return TaskStatus::complete;
	  (void) phyd->RestrictU(pdrive, stage);
	  return phyd->SendU(pdrive, stage);
	}

	TaskStatus Nuclear::RecvU(Driver *pdrive, int stage) {
	  hydro::Hydro *phyd = pmy_pack->phydro;
	  if (phyd == nullptr) return TaskStatus::complete;
	  return phyd->RecvU(pdrive, stage);
	}

	// Prolongation at fine/coarse boundaries, physical BCs, and primitives everywhere
	TaskStatus Nuclear::SetGhostZones(Driver *pdrive, int stage) {
	  hydro::Hydro *phyd = pmy_pack->phydro;
	  if (phyd == nullptr) return TaskStatus::complete;
	  (void) phyd->Prolongate(pdrive, stage);
	  (void) phyd->ApplyPhysicalBCs(pdrive, stage);
	  return phyd->ConToPrim(pdrive, stage);
	}

	TaskStatus Nuclear::ClearSendU(Driver *pdrive, int stage) {
	  hydro::Hydro *phyd = pmy_pack->phydro;
	  if (phyd == nullptr) return TaskStatus::complete;
	  return phyd->ClearSend(pdrive, -1);  // stage = -1 only clears SendU
	}

	TaskStatus Nuclear::ClearRecvU(Driver *pdrive, int stage) {
	  hydro::Hydro *phyd = pmy_pack->phydro;
	  if (phyd == nullptr) return TaskStatus::complete;
	  return phyd->ClearRecv(pdrive, -1);  // stage = -1 only clears RecvU
	}
}
