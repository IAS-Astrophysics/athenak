#ifndef DIFFUSION_RESISTIVITY_HPP_
#define DIFFUSION_RESISTIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.hpp
//  \brief Contains data and functions that implement various non-ideal MHD (resistive) 
//  processes, such as Ohmic diffusion.
//  TODO: add ambipolar diffusion, Hall effect
//
//  Design is based on EOS class

#include "athena.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class Resistivity
//  \brief Abstract base class for 

class Resistivity
{
 public:
  Resistivity(MeshBlockPack *pp, ParameterInput *pin);
  virtual ~Resistivity() = default;

  MeshBlockPack* pmy_pack;

  // pure virtual functions to compute resistive EMF and Poynting (energy) flux
  virtual TaskStatus AddResistiveEMF(const DvceFaceFld4D<Real> &b0,
                                     DvceEdgeFld4D<Real> &efld) = 0;

 private:
};

//----------------------------------------------------------------------------------------
//! \class AdibaticHydro
//  \brief Derived class for Hydro adiabatic EOS

class Ohmic : public Resistivity
{
 public:
  Ohmic(MeshBlockPack *pp, ParameterInput *pin);

  // data
  Real eta_ohm;

  // overrides of pure virtual functions in base class
  TaskStatus AddResistiveEMF(const DvceFaceFld4D<Real> &b0,
                             DvceEdgeFld4D<Real> &efld) override;
};

#endif // DIFFUSION_RESISTIVITY_HPP_
