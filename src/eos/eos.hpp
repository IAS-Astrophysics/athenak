#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.hpp
//  \brief Contains data and functions that implement conserved->primitive variable
//  conversion for various EOS (e.g. adiabatic, isothermal, etc.), for various fluids
//  (Hydro, MHD, etc.), and for non-relativistic and relativistic flows.

#include <cmath> 

#include "athena.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \struct EOSData
//  \brief container for variables associated with EOS, and in-lined wave speed functions
//  Storing everything in a container makes it easier to capture EOS variables and
//  functions in kernels elsewhere in the code.

struct EOS_Data
{
  Real gamma;
  Real iso_cs;
  bool is_adiabatic;
  Real density_floor, pressure_floor;

  // sound speed function for adiabatic EOS 
  KOKKOS_INLINE_FUNCTION
  Real SoundSpeed(Real p, Real d) const {
    return std::sqrt(gamma*p/d);
  }

  // function to compute maximal wave speeds for special relativistic adiabatic EOS 
  // Inputs:
  //   h: enthalpy per unit volume
  //   p: gas pressure
  //   vx: 3-velocity component v^x
  //   lor_sq: Lorentz factor \gamma^2
  // Outputs:
  //   l_p/m: most positive/negative wavespeed
  // References:
  //   Del Zanna & Bucciantini, A&A 390, 1177 (2002)
  //   Mignone & Bodo 2005, MNRAS 364 126 (MB).
  //   Del Zanna et al, A&A 473, 11 (2007) (eq. 76)
  KOKKOS_INLINE_FUNCTION
  void WaveSpeeds_SR(Real h, Real p, Real vx, Real lor_sq, Real& l_p, Real& l_m) const {
    Real cs2 = gamma * p / h;  // (MB 4)
    Real v2 = 1.0 - 1.0/lor_sq;
    auto const p1 = vx * (1.0 - cs2);
    auto const tmp = sqrt(cs2 * ((1.0-v2*cs2) - p1*vx) / lor_sq);
    auto const invden = 1.0/(1.0 - v2*cs2);

    l_p = (p1 + tmp) * invden;
    l_m = (p1 - tmp) * invden;
  }

  // fast magnetosonic speed function for adiabatic EOS 
  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeed(Real d, Real p, Real bx, Real by, Real bz) const {
    Real asq = gamma*p;
    Real ct2 = by*by + bz*bz;
    Real qsq = bx*bx + ct2 + asq;
    Real tmp = bx*bx + ct2 - asq;
    return std::sqrt(0.5*(qsq + std::sqrt(tmp*tmp + 4.0*asq*ct2))/d);
  }

  // fast magnetosonic speed function for isothermal EOS 
  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeed(Real d, Real bx, Real by, Real bz) const {
    Real asq = (iso_cs*iso_cs)*d;
    Real ct2 = by*by + bz*bz;
    Real qsq = bx*bx + ct2 + asq;
    Real tmp = bx*bx + ct2 - asq;
    return std::sqrt(0.5*(qsq + std::sqrt(tmp*tmp + 4.0*asq*ct2))/d);
  }
};

//----------------------------------------------------------------------------------------
//! \class EquationOfState
//  \brief Abstract base class for Hydro EOS

class EquationOfState
{
 public:
  EquationOfState(MeshBlockPack *pp, ParameterInput *pin);
  virtual ~EquationOfState() = default;

  MeshBlockPack* pmy_pack;
  EOS_Data eos_data;

  // virtual functions to convert cons to prim, overwritten in derived eos classes
  virtual void ConsToPrimHydro(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim);
  virtual void ConsToPrimMHD(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                             DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc);

 private:
};

//----------------------------------------------------------------------------------------
//! \class AdiabaticHydro
//  \brief Derived class for Hydro adiabatic EOS

class AdiabaticHydro : public EquationOfState
{
 public:
  AdiabaticHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrimHydro(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
};

//----------------------------------------------------------------------------------------
//! \class AdiabaticHydroSR
//  \brief Derived class for special relativistic Hydro adiabatic EOS 

class AdiabaticHydroSR : public EquationOfState
{
 public:
  AdiabaticHydroSR(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrimHydro(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalHydro
//  \brief Derived class for Hydro isothermal EOS

class IsothermalHydro : public EquationOfState
{ 
 public:
  IsothermalHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrimHydro(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
};

//----------------------------------------------------------------------------------------
//! \class AdibaticMHD
//  \brief Derived class for MHD adiabatic EOS

class AdiabaticMHD : public EquationOfState
{
 public:
  AdiabaticMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrimMHD(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) override;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalMHD
//  \brief Derived class for MHD isothermal EOS

class IsothermalMHD : public EquationOfState
{
 public:
  IsothermalMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrimMHD(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) override;
};

#endif // EOS_EOS_HPP_
