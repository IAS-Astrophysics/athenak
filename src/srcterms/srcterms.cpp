//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  \brief Implementation of functions for source terms in equations of motion.  Functions
//  implememnted in this file are the SourceTerms constructor/destructor, and functions
//  that insert tasks for various source terms in the time-integrator or operator-split
//  TaskLists. Implementation of the source terms themselves are in other files, generally
//  in this directory.

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "turb_driver.hpp"
#include "srcterms.hpp"

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters

SourceTerms::SourceTerms(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp),
  random_forcing(false),
  const_accel(false),
  shearing_box(false),
  twofluid_mhd(false)
{
  // Parse input file to see if any source terms are specified

  // (1) Random forcing to drive turbulence
  if (pin->DoesBlockExist("forcing")) {
    pturb = new TurbulenceDriver(pp, pin);
    pturb->InitializeModes();
    random_forcing = true;
  } else {
    pturb = nullptr;
  }

  // (2) gravitational accelerations
  if (pin->DoesBlockExist("gravity")) {
    const_acc1 = pin->GetOrAddReal("gravity","const_acc1",0.0);
    const_acc2 = pin->GetOrAddReal("gravity","const_acc2",0.0);
    const_acc3 = pin->GetOrAddReal("gravity","const_acc3",0.0);
    if (const_acc1 != 0.0 or const_acc2 != 0.0 or const_acc3 != 0.0) {
      const_accel = true;
    }
  }

  // (3) shearing box
  if (pin->DoesBlockExist("shearing_box")) {
    qshear = pin->GetReal("shearing_box","qshear");
    omega0 = pin->GetReal("shearing_box","omega0");
    shearing_box = true;
  }

}

//----------------------------------------------------------------------------------------
// destructor
  
SourceTerms::~SourceTerms()
{
  if (pturb != nullptr) delete pturb;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeSplitSrcTermTasks
//  \brief includes tasks for source terms into operator split task list
//  Called by MeshBlockPack::AddPhysicsModules() function directly after SourceTerms cons

void SourceTerms::IncludeSplitSrcTermTasks(TaskList &tl, TaskID start)
{
  if (random_forcing) {
    auto id = tl.AddTask(&SourceTerms::ApplyRandomForcing, this, start);
    split_tasks.emplace(SplitSrcTermTaskName::hydro_forcing, id);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeUnsplitSrcTermTasks
//  \brief includes tasks for source terms into time integrator task list
//  Called by MeshBlockPack::AddPhysicsModules() function directly after SourceTerms cons

void SourceTerms::IncludeUnsplitSrcTermTasks(TaskList &tl, TaskID start)
{   
  // Source terms for Hydro fluid.
  // These must be inserted after update task, but before send_u for hydro
  if (pmy_pack->phydro != nullptr) {

    // add constant (gravitational?) acceleration to Hydro fluid
    if (const_accel) {
      auto id = tl.InsertTask(&SourceTerms::AddConstantAccelHydro, this, 
                         pmy_pack->phydro->hydro_tasks[HydroTaskName::calc_flux],
                         pmy_pack->phydro->hydro_tasks[HydroTaskName::update]);
      unsplit_tasks.emplace(UnsplitSrcTermTaskName::hydro_acc, id);
    }
 
    // add shearing box source terms to Hydro fluid
    if (shearing_box) {
      auto id = tl.InsertTask(&SourceTerms::AddSBoxMomentumHydro, this,
                         pmy_pack->phydro->hydro_tasks[HydroTaskName::calc_flux],
                         pmy_pack->phydro->hydro_tasks[HydroTaskName::update]);
      unsplit_tasks.emplace(UnsplitSrcTermTaskName::hydro_sbox, id);
    }
  }

  // Source terms for MHD fluid.
  // These must be inserted after update task, but before send_u for mhd 
  if (pmy_pack->pmhd != nullptr) {
    // add constant (gravitational?) acceleration to MHD fluid
    if (const_accel) {
      auto id = tl.InsertTask(&SourceTerms::AddConstantAccelMHD, this, 
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::calc_flux],
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::update]);
      unsplit_tasks.emplace(UnsplitSrcTermTaskName::mhd_acc, id);
    }

    // add shearing box source terms to MHD fluid AND shearing box EMF
    if (shearing_box) {
      auto id = tl.InsertTask(&SourceTerms::AddSBoxMomentumMHD, this,
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::calc_flux],
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::update]);
      unsplit_tasks.emplace(UnsplitSrcTermTaskName::mhd_sbox, id);
      id = tl.InsertTask(&SourceTerms::AddSBoxEMF, this,
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::corner_e],
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::ct]);
      unsplit_tasks.emplace(UnsplitSrcTermTaskName::mhd_sbox_emf, id);
    }
  }

  // Source terms for coupled Hydro and MHD fluid.
  // Must be inserted after respective update tasks, before sends of conserved variables,
  // and BEFORE either cons2prim call. 
  if (pmy_pack->phydro != nullptr and pmy_pack->pmhd != nullptr) {
    if (twofluid_mhd) {
      auto id = tl.InsertTask(&SourceTerms::AddTwoFluidDragHydro, this,
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::calc_flux],
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::update]);
      unsplit_tasks.emplace(UnsplitSrcTermTaskName::hydro_drag, id);
      id = tl.InsertTask(&SourceTerms::AddTwoFluidDragMHD, this,
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::calc_flux],
                         pmy_pack->pmhd->mhd_tasks[MHDTaskName::update]);
      unsplit_tasks.emplace(UnsplitSrcTermTaskName::mhd_drag, id);
    }
  }

  return; 
}

//----------------------------------------------------------------------------------------
//! \fn

TaskStatus SourceTerms::AddTwoFluidDragHydro(Driver *pdrive, int stage)
{
  auto &u = pmy_pack->phydro->u0;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn

TaskStatus SourceTerms::AddTwoFluidDragMHD(Driver *pdrive, int stage)
{
  return TaskStatus::complete;
}
