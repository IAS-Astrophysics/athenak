#ifndef DIFFUSION_VISCOSITY_HPP_
#define DIFFUSION_VISCOSITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.hpp
//  \brief Contains data and functions that implement various formulations for viscosity.
//  Currently only Navier-Stokes (uniform, isotropic) viscosity implemented
//  TODO: add Braginskii viscosity
//
//  Design is based on EOS class

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class Viscosity
//  \brief Abstract base class for viscosity physics

class Viscosity
{
 public:
  Viscosity(MeshBlockPack *pp, ParameterInput *pin);
  virtual ~Viscosity() = default;

  // data
  Real dtnew;
  MeshBlockPack* pmy_pack;

  // pure virtual function to add viscous fluxes to 
  virtual void AddViscousFlux(const DvceArray5D<Real> &w, DvceFaceFld5D<Real> &f) = 0;

 private:
};

//----------------------------------------------------------------------------------------
//! \class IsoViscosity
//  \brief Derived class for isotropic viscosity for a Newtonian fluid

class IsoViscosity : public Viscosity
{
 public:
  IsoViscosity(MeshBlockPack *pp, ParameterInput *pin, Real nu);

  // data
  Real nu_iso;

  // overrides of pure virtual functions in base class
  void AddViscousFlux(const DvceArray5D<Real> &w, DvceFaceFld5D<Real> &f) override;
};

#endif // DIFFUSION_VISCOSITY_HPP_
