//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  \brief Implementation of functions for source terms in equations of motion

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "turb_driver.hpp"
#include "srcterms.hpp"

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters

SourceTerms::SourceTerms(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp), random_forcing(false)
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
    split_tasks.emplace(SplitSrcTermTaskName::random_forcing, id);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void IncludeUnsplitSrcTermTasks
//  \brief includes tasks for source terms into time integrator task list
//  Called by MeshBlockPack::AddPhysicsModules() function directly after SourceTerms cons

void SourceTerms::IncludeUnsplitSrcTermTasks(TaskList &tl, TaskID start)
{   
  return; 
}
