//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.cpp
//  \brief implementation of source terms in equations of motion

#include <iostream>

#include "athena.hpp"
#include "turb_driver.hpp"
#include "srcterms.hpp"

//----------------------------------------------------------------------------------------
// constructor, parses input file and initializes data structures and parameters

SourceTerms::SourceTerms(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp), no_unsplit_terms(true), no_split_terms(true)
{
  if (pp->pturb_driver != nullptr) {
    no_unsplit_terms = false;
  }
}

//----------------------------------------------------------------------------------------
// destructor
  
SourceTerms::~SourceTerms()
{
}

//----------------------------------------------------------------------------------------
//! \fn ApplySrcTermsStageRunTL()
// apply unsplit source terms added in EACH stage of the stage run task list

void SourceTerms::ApplySrcTermsStageRunTL(DvceArray5D<Real> &u)
{
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ApplySrcTermsStageRunTL()
// apply operator split source terms added in the operator split task list

void SourceTerms::ApplySrcTermsOperatorSplitTL(DvceArray5D<Real> &u)
{
  if (pmy_pack->pturb_driver != nullptr) {
    pmy_pack->pturb_driver->ApplyForcing(u);
  }
  return;
}
