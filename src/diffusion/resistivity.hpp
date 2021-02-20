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
//  \brief Abstract base class for resistive physics

class Resistivity
{
 public:
  Resistivity(MeshBlockPack *pp, ParameterInput *pin);
  virtual ~Resistivity() = default;

  // data
  Real dtnew;
  MeshBlockPack* pmy_pack;

  // pure virtual functions to compute resistive EMF and Poynting (energy) flux
  virtual void AddResistiveEMF(const DvceFaceFld4D<Real> &b0,
                               DvceEdgeFld4D<Real> &efld) = 0;

 private:
};

//----------------------------------------------------------------------------------------
//! \class Ohmic
//  \brief Derived class for Ohmic resistivity

class Ohmic : public Resistivity
{
 public:
  Ohmic(MeshBlockPack *pp, ParameterInput *pin, Real eta);

  // data
  Real eta_ohm;

  // overrides of pure virtual functions in base class
  void AddResistiveEMF(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld) override;
};

#endif // DIFFUSION_RESISTIVITY_HPP_
