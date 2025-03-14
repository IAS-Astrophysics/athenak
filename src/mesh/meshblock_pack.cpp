//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file meshblock_pack.cpp
//  \brief implementation of constructor and functions in MeshBlockPack class

#include <cstdlib>
#include <iostream>
#include <utility>
#include <memory>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "coordinates/adm.hpp"
#include "z4c/tmunu.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "z4c/cce/cce.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/turb_driver.hpp"
#include "particles/particles.hpp"
#include "units/units.hpp"
#include "meshblock_pack.hpp"

//----------------------------------------------------------------------------------------
// MeshBlockPack constructor:

MeshBlockPack::MeshBlockPack(Mesh *pm, int igids, int igide) :
  pmesh(pm),
  gids(igids),
  gide(igide),
  nmb_thispack(igide - igids + 1) {
  // create map for task lists
  tl_map.insert(std::make_pair("before_timeintegrator",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("after_timeintegrator",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("before_stagen",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("stagen",std::make_shared<TaskList>()));
  tl_map.insert(std::make_pair("after_stagen",std::make_shared<TaskList>()));
}

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlockPack::~MeshBlockPack() {
  delete pcoord;
  if (phydro != nullptr) {delete phydro;}
  if (pmhd   != nullptr) {delete pmhd;}
  if (padm   != nullptr) {delete padm;}
  if (ptmunu != nullptr) {delete ptmunu;}
  if (prad   != nullptr) {delete prad;}
  if (pdyngr != nullptr) {delete pdyngr;}
  if (pnr    != nullptr) {delete pnr;}
  if (pturb  != nullptr) {delete pturb;}
  if (punit  != nullptr) {delete punit;}
  if (pz4c   != nullptr) {
    delete pz4c;
    // cce dump
    for (auto cce : pz4c_cce) {
      delete cce;
    }
    pz4c_cce.resize(0);
  }
  if (ppart  != nullptr) {delete ppart;}
  // must be last, since it calls ~BoundaryValues() which (MPI) uses pmy_pack->pmb->nnghbr
  delete pmb;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockPack::AddMeshBlocks(ParameterInput *pin)
//! \brief Wrapper function for calling MeshBlock constructor inside MeshBlockPack.
//! Allows for passing of pointer to 'this' pack.

void MeshBlockPack::AddMeshBlocks(ParameterInput *pin) {
  pmb = new MeshBlock(this, gids, nmb_thispack);
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockPack::AddCoordinates(ParameterInput *pin)
//! \brief Wrapper function for calling Coordinates constructor inside MeshBlockPack.
//! Allows for passing of pointer to 'this' pack. Must be called BEFORE AddPhysics()
//! function, since latter uses data inside Coordinates class.

void MeshBlockPack::AddCoordinates(ParameterInput *pin) {
  pcoord = new Coordinates(pin, this);
}

//----------------------------------------------------------------------------------------
// \fn MeshBlockPack::AddPhysics()
// \brief construct physics modules and tasks lists in this MeshBlockPack, based on which
// <blocks> are present in the input file.  Called from main().

void MeshBlockPack::AddPhysics(ParameterInput *pin) {
  int nphysics = 0;
  TaskID none(0);

  // (1) Units.  Create first so that they can be used in other physics constructors
  // Default units are simply code units
  if (pin->DoesBlockExist("units")) {
    punit = new units::Units(pin);
  } else {
    punit = nullptr;
  }

  // (2) HYDRODYNAMICS
  // Create Hydro physics module.  Create TaskLists only for single-fluid hydro
  // (Note TaskLists stored in MeshBlockPack)
  if (pin->DoesBlockExist("hydro")) {
    phydro = new hydro::Hydro(this, pin);
    nphysics++;
    if (!(pin->DoesBlockExist("mhd")) && !(pin->DoesBlockExist("radiation")) &&
        !(pin->DoesBlockExist("adm")) && !(pin->DoesBlockExist("z4c")) ) {
      phydro->AssembleHydroTasks(tl_map);
    }
  } else {
    phydro = nullptr;
  }

  // (3) MHD
  // Create MHD physics module.  Create TaskLists only for single-fluid MHD
  if (pin->DoesBlockExist("mhd")) {
    pmhd = new mhd::MHD(this, pin);
    nphysics++;
    if (!(pin->DoesBlockExist("hydro")) && !(pin->DoesBlockExist("radiation")) &&
        !(pin->DoesBlockExist("adm")) && !(pin->DoesBlockExist("z4c")) ) {
      pmhd->AssembleMHDTasks(tl_map);
    }
  } else {
    pmhd = nullptr;
  }

  // (4) ION_NEUTRAL (two-fluid) MHD
  // Create Ion-Neutral physics module and TaskLists. Error if <hydro> and <mhd> are not
  // both defined as well.
  if (pin->DoesBlockExist("ion-neutral")) {
    pionn = new ion_neutral::IonNeutral(this, pin);   // construct new MHD object
    if (pin->DoesBlockExist("hydro") && pin->DoesBlockExist("mhd") &&
        !(pin->DoesBlockExist("adm")) && !(pin->DoesBlockExist("z4c")) ) {
      pionn->AssembleIonNeutralTasks(tl_map);
      nphysics++;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<ion-neutral> block detected in input file, but either"
                << " <hydro> or <mhd> block missing" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    // Error if both <hydro> and <mhd> defined, but not <ion-neutral>
    if (pin->DoesBlockExist("hydro") && pin->DoesBlockExist("mhd")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Both <hydro> and <mhd> blocks detected in input file, "
                << "but <ion-neutral> block missing" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pionn = nullptr;
  }

  // (5) RADIATION
  // Create radiation physics module.  Create tasklist.
  if (pin->DoesBlockExist("radiation")) {
    prad = new radiation::Radiation(this, pin);
    nphysics++;
    prad->AssembleRadTasks(tl_map);
  } else {
    prad = nullptr;
  }

  // (6) TURBULENCE DRIVER
  // This is a special module to drive turbulence in hydro, MHD, or both. Cannot be
  // included as a source term since it requires evolving force array via O-U process.
  // Instead, TurbulenceDriver object is stored in MeshBlockPack and tasks for evolving
  // force and adding force to fluid are included in operator_split and stage_run
  // task lists respectively.
  if (pin->DoesBlockExist("turb_driving")) {
    pturb = new TurbulenceDriver(this, pin);
    pturb->IncludeInitializeModesTask(tl_map["before_timeintegrator"], none);
    pturb->IncludeAddForcingTask(tl_map["stagen"], none);
  } else {
    pturb = nullptr;
  }

  // (7) Z4c and ADM
  // Create Z4c and ADM physics module.
  if (pin->DoesBlockExist("z4c")) {
    pz4c = new z4c::Z4c(this, pin);
    padm = new adm::ADM(this, pin);
    ptmunu = nullptr;
    // init cce dump
    pz4c_cce.reserve(0);
    int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
    pz4c_cce.reserve(ncce);// 10 different components for each radius
    for(int n = 0; n < ncce; ++n) {
      // NOTE: these names are used for pittnull code, so DON'T change the convention
      pz4c_cce.push_back(new z4c::CCE(pmesh, pin,n));
    }
    nphysics++;
  } else {
    pz4c = nullptr;
    if (pin->DoesBlockExist("adm")) {
      padm = new adm::ADM(this, pin);
    } else {
      padm = nullptr;
    }
  }

  // (8) Dynamical Spacetime and Matter (MHD TODO)
  if ((pin->DoesBlockExist("z4c") || pin->DoesBlockExist("adm")) &&
      (pin->DoesBlockExist("hydro")) ) {
    std::cout << "Dynamical metric and hydro not compatible; use MHD instead  "
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((pin->DoesBlockExist("z4c") || pin->DoesBlockExist("adm")) &&
      (pin->DoesBlockExist("mhd")) ) {
    pdyngr = dyngr::BuildDynGRMHD(this, pin);
    ptmunu = new Tmunu(this, pin);
  }

  if (pz4c != nullptr || padm != nullptr) {
    pnr = new numrel::NumericalRelativity(this, pin);
    pnr->AssembleNumericalRelativityTasks(tl_map);
  }

  // (8) PARTICLES
  // Create particles module.  Create tasklist.
  if (pin->DoesBlockExist("particles")) {
    ppart = new particles::Particles(this, pin);
    ppart->AssembleTasks(tl_map);
    nphysics++;
  } else {
    ppart = nullptr;
  }

  // Check that at least ONE is requested and initialized.
  // Error if there are no physics blocks in the input file.
  if (nphysics == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "At least one physics module must be specified in input file." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return;
}
