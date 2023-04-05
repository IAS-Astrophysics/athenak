//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file meshblock_pack.cpp
//  \brief implementation of constructor and functions in MeshBlockPack class

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "ion-neutral/ion_neutral.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/turb_driver.hpp"
#include "units/units.hpp"
#include "meshblock_pack.hpp"

//----------------------------------------------------------------------------------------
// MeshBlockPack constructor:

MeshBlockPack::MeshBlockPack(Mesh *pm, int igids, int igide) :
  pmesh(pm),
  gids(igids),
  gide(igide),
  nmb_thispack(igide - igids + 1) {
}

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlockPack::~MeshBlockPack() {
  delete pcoord;
  if (phydro != nullptr) {delete phydro;}
  if (pmhd   != nullptr) {delete pmhd;}
  if (prad   != nullptr) {delete prad;}
  if (pturb  != nullptr) {delete pturb;}
  if (punit  != nullptr) {delete punit;}
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

  // (1) Units
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
    if (!(pin->DoesBlockExist("mhd")) && !(pin->DoesBlockExist("radiation"))) {
      phydro->AssembleHydroTasks(start_tl, run_tl, end_tl);
    }
  } else {
    phydro = nullptr;
  }

  // (3) MHD
  // Create MHD physics module.  Create TaskLists only for single-fluid MHD
  if (pin->DoesBlockExist("mhd")) {
    pmhd = new mhd::MHD(this, pin);
    nphysics++;
    if (!(pin->DoesBlockExist("hydro")) && !(pin->DoesBlockExist("radiation"))) {
      pmhd->AssembleMHDTasks(start_tl, run_tl, end_tl);
    }
  } else {
    pmhd = nullptr;
  }

  // (4) ION_NEUTRAL (two-fluid) MHD
  // Create Ion-Neutral physics module and TaskLists. Error if <hydro> and <mhd> are not
  // both defined as well.
  if (pin->DoesBlockExist("ion-neutral")) {
    pionn = new ion_neutral::IonNeutral(this, pin);   // construct new MHD object
    if (pin->DoesBlockExist("hydro") && pin->DoesBlockExist("mhd")) {
      pionn->AssembleIonNeutralTasks(start_tl, run_tl, end_tl);
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
    prad->AssembleRadiationTasks(start_tl, run_tl, end_tl);
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
    pturb->IncludeInitializeModesTask(operator_split_tl, none);
    pturb->IncludeAddForcingTask(run_tl, none);
  } else {
    pturb = nullptr;
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
