#ifndef SRCTERMS_SRCTERMS_HPP_
#define SRCTERMS_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.hpp
//! \brief defines class SourceTerms
//!
//! Contains data and functions that implement various physical source terms in equations
//  of motion.  Can be used in both Hydro and MHD classes to avoid duplicating code.


#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class SourceTerms
//! \brief data and functions for physical source terms

class SourceTerms
{
 public:
  SourceTerms(MeshBlockPack *pp, ParameterInput *pin);
  ~SourceTerms();

  // accessors

  // data
  bool operatorsplit_terms, stagerun_terms;   // flags as to whether source terms exist

  // functions
  void ApplySrcTermsStageRunTL(DvceArray5D<Real> &u);
  void ApplySrcTermsOperatorSplitTL(DvceArray5D<Real> &u);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // SRCTERMS_SRCTERMS_HPP_
