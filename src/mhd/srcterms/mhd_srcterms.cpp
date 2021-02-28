//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_srcterms.cpp
//  \brief implementation of source terms in class MHD

#include <iostream>

#include "athena.hpp"
#include "mhd_srcterms.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

MHDSourceTerms::MHDSourceTerms(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp)
{
}

//----------------------------------------------------------------------------------------
// destructor
  
MHDSourceTerms::~MHDSourceTerms()
{
}

//----------------------------------------------------------------------------------------
//! \fn ApplySrcTermsStageRunTL()
// apply unsplit source terms added in EACH stage of the stage run task list

void MHDSourceTerms::ApplySrcTermsStageRunTL()
{
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ApplySrcTermsStageRunTL()
// apply operator split source terms added in the operator split task list

void MHDSourceTerms::ApplySrcTermsOperatorSplitTL()
{
  return;
}

} // namespace mhd
