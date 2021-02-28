#ifndef HYDRO_SRCTERMS_HYDRO_SRCTERMS_HPP_
#define HYDRO_SRCTERMS_HYDRO_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_srcterms.hpp
//! \brief defines class HydroSourceTerms
//!
//! Contains data and functions that implement various physical source terms in Hydro
//! equations of motion.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class HydroSourceTerms
//! \brief data and functions for physical source terms in the hydro

class HydroSourceTerms
{
 public:
  HydroSourceTerms(MeshBlockPack *pp, ParameterInput *pin);
  ~HydroSourceTerms();

  // accessors

  // data

  // functions
  void ApplySrcTermsStageRunTL();
  void ApplySrcTermsOperatorSplitTL();

 private:
  MeshBlockPack* pmy_pack;
};

} // namespace hydro
#endif // HYDRO_SRCTERMS_HYDRO_SRCTERMS_HPP_
