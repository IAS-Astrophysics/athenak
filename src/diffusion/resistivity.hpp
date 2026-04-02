#ifndef DIFFUSION_RESISTIVITY_HPP_
#define DIFFUSION_RESISTIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.hpp
//  \brief Contains data and functions that implement various non-ideal MHD (resistive)
//  processes, such as Ohmic diffusion. TODO(@user): add ambipolar diffusion, Hall effect

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/meshblock.hpp"

//----------------------------------------------------------------------------------------
//! \class Resistivity
//  \brief data and functions that implement various resistive physics

class Resistivity {
 public:
  Resistivity(MeshBlockPack *pp, ParameterInput *pin);
  ~Resistivity();

  // data
  Real dtnew;
  std::string iso_resist_type;  // only "constant" implemented
  Real eta_ohm;

  // functions to add resistive E-Field and energy flux
  void AddResistiveEMFs(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddResistiveFluxes(const DvceFaceFld4D<Real> &b0, DvceFaceFld5D<Real> &flx);
  void AddEMFConstantResist(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddFluxConstantResist(const DvceFaceFld4D<Real> &b, DvceFaceFld5D<Real> &flx);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_RESISTIVITY_HPP_
