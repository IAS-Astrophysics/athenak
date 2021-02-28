#ifndef MHD_SRCTERMS_MHD_SRCTERMS_HPP_
#define MHD_SRCTERMS_MHD_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_srcterms.hpp
//! \brief defines class MHDSourceTerms
//!
//! Contains data and functions that implement various physical source terms in MHD
//! equations of motion.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHDSourceTerms
//! \brief data and functions for physical source terms in mhd

class MHDSourceTerms
{
 public:
  MHDSourceTerms(MeshBlockPack *pp, ParameterInput *pin);
  ~MHDSourceTerms();

  // accessors

  // data

  // functions
  void ApplySrcTermsStageRunTL();
  void ApplySrcTermsOperatorSplitTL();

 private:
  MeshBlockPack* pmy_pack;
};

} // namespace mhd
#endif // MHD_SRCTERMS_MHD_SRCTERMS_HPP_
