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
//  \brief container for variables associated with EOS

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

  KOKKOS_INLINE_FUNCTION
  void SoundSpeed_SR(Real rho_h, Real pgas, Real vx, Real gamma_lorentz_sq, Real& plambda_plus, Real& plambda_minus)
    const {
      Real cs2 = gamma * pgas / rho_h;  // (MB 4)
      Real v2 = 1. - 1./gamma_lorentz_sq;
      auto const tmp = sqrt( 
	  cs2 * (1. - (vx + (v2 - vx*vx) * cs2))/gamma_lorentz_sq
	  );

      auto const p1 = vx * (1. - cs2);
      auto const invden =1./ (1. - v2 * cs2);

      plambda_plus = (p1 + tmp) * invden;
      plambda_minus = (p1 - tmp) * invden;
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

  // pure virtual functions to convert cons to prim, overwritten in derived eos classes
  virtual void ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) = 0;
  virtual void ConsToPrim(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) = 0;

 private:
};

//----------------------------------------------------------------------------------------
//! \class AdibaticHydro
//  \brief Derived class for Hydro adiabatic EOS

class AdiabaticHydro : public EquationOfState
{
 public:
  AdiabaticHydro(MeshBlockPack *pp, ParameterInput *pin);
  // prototype for Hydro conversion function
  void ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
  // prototype for MHD conversion function (never used)
  void ConsToPrim(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) override;
};

//----------------------------------------------------------------------------------------
//! \class AdibaticHydroRel
//  \brief Derived class for relativistic Hydro adiabatic EOS 

class AdiabaticHydroRel : public EquationOfState
{
 public:
  AdiabaticHydroRel(MeshBlockPack *pp, ParameterInput *pin);
  // prototype for Hydro conversion function
  void ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
  // prototype for MHD conversion function (never used)
  void ConsToPrim(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) override;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalHydro
//  \brief Derived class for Hydro isothermal EOS

class IsothermalHydro : public EquationOfState
{ 
 public:
  IsothermalHydro(MeshBlockPack *pp, ParameterInput *pin);
  // prototype for Hydro conversion function
  void ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
  // prototype for MHD conversion function (never used)
  void ConsToPrim(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) override;
};

//----------------------------------------------------------------------------------------
//! \class AdibaticMHD
//  \brief Derived class for MHD adiabatic EOS

class AdiabaticMHD : public EquationOfState
{
 public:
  AdiabaticMHD(MeshBlockPack *pp, ParameterInput *pin);
  // prototype for Hydro conversion function (never used)
  void ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
  // prototype for MHD conversion function
  void ConsToPrim(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) override;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalMHD
//  \brief Derived class for MHD isothermal EOS

class IsothermalMHD : public EquationOfState
{
 public:
  IsothermalMHD(MeshBlockPack *pp, ParameterInput *pin);
  // prototype for Hydro conversion function (never used)
  void ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
  // prototype for MHD conversion function
  void ConsToPrim(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) override;
};

#endif // EOS_EOS_HPP_
