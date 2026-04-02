#ifndef DIFFUSION_CONDUCTION_HPP_
#define DIFFUSION_CONDUCTION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.hpp
//! \brief Contains data and functions that implement various formulations for conduction.
//  Currently only isotropic conduction implemented

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class Conduction
//! \brief data and functions that implement thermal conduction in Hydro and MHD

class Conduction {
 public:
  Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~Conduction();

  // data
  Real dtnew;
  std::string iso_cond_type; // "constant", "spitzer", or "spitzer_limited"
  Real kappa_iso;            // isotropic thermal conductivity
  Real kappa_iso_limit;      // limit to isotropic thermal conductivity

  // functions
  void AddHeatFluxes(const DvceArray5D<Real> &w, const EOS_Data &eos,
                     DvceFaceFld5D<Real> &f);
  void AddIsotropicHeatFluxConstCond(const DvceArray5D<Real> &w, const EOS_Data &eos,
                                     DvceFaceFld5D<Real> &f);
  void AddIsotropicHeatFluxSpitzerCond(const DvceArray5D<Real> &w, const EOS_Data &eos,
                                       DvceFaceFld5D<Real> &f);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);

 private:
  MeshBlockPack* pmy_pack;
};
#endif // DIFFUSION_CONDUCTION_HPP_
