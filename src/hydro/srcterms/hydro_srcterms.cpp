//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_srcterms.cpp
//  \brief implementation of source terms in class Hydro

#include <iostream>

#include "athena.hpp"
#include "hydro_srcterms.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

HydroSourceTerms::HydroSourceTerms(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp)
{
}

//----------------------------------------------------------------------------------------
// destructor
  
HydroSourceTerms::~HydroSourceTerms()
{
}

//----------------------------------------------------------------------------------------
//! \fn ApplySrcTermsStageRunTL()
// apply unsplit source terms added in EACH stage of the stage run task list

void HydroSourceTerms::ApplySrcTermsStageRunTL()
{
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ApplySrcTermsStageRunTL()
// apply operator split source terms added in the operator split task list

void HydroSourceTerms::ApplySrcTermsOperatorSplitTL()
{
  return;
}

} // namespace hydro
