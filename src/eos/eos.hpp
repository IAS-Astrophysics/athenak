#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.hpp
//  \brief Contains data and functions that implement conserved->primitive variable
//  conversion for various EOS (e.g. ideal gas, isothermal, etc.), for various fluids
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
  bool is_ideal;
  Real density_floor, pressure_floor;

  // inlined sound speed function for ideal gas EOS in nonrelativistic hydro 
  KOKKOS_INLINE_FUNCTION
  Real SoundSpeed(Real p, Real d)
  const {
    return std::sqrt(gamma*p/d);
  }

  // inlined fast magnetosonic speed function for ideal gas EOS in nonrelativistic mhd
  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeed(Real d, Real p, Real bx, Real by, Real bz)
  const {
    Real asq = gamma*p;
    Real ct2 = by*by + bz*bz;
    Real qsq = bx*bx + ct2 + asq;
    Real tmp = bx*bx + ct2 - asq;
    return std::sqrt(0.5*(qsq + std::sqrt(tmp*tmp + 4.0*asq*ct2))/d);
  }

  // inlined fast magnetosonic speed function for isothermal EOS in nonrelativistic mhd
  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeed(Real d, Real bx, Real by, Real bz)
  const {
    Real asq = (iso_cs*iso_cs)*d;
    Real ct2 = by*by + bz*bz;
    Real qsq = bx*bx + ct2 + asq;
    Real tmp = bx*bx + ct2 - asq;
    return std::sqrt(0.5*(qsq + std::sqrt(tmp*tmp + 4.0*asq*ct2))/d);
  }

  // inlined maximal wave speeds function for ideal gas in SR hydro
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
  void WaveSpeedsSR(Real h, Real p, Real vx, Real lor_sq, Real& l_p, Real& l_m)
  const {
    Real cs2 = gamma * p / h;  // (MB 4)
    Real v2 = 1.0 - 1.0/lor_sq;
    auto const p1 = vx * (1.0 - cs2);
    auto const tmp = sqrt(cs2 * ((1.0-v2*cs2) - p1*vx) / lor_sq);
    auto const invden = 1.0/(1.0 - v2*cs2);

    l_p = (p1 + tmp) * invden;
    l_m = (p1 - tmp) * invden;
  }

  // inlined maximal wave speeds function for ideal gas in GR hydro
  // Inputs:
  //  - h: enthalpy per unit volume
  //  - p: gas pressure
  //  - u0,u1: 4-velocity components u^0, u^1
  //  - g00,g01,g11: metric components g^00, g^01, g^11
  // Outputs:
  //  - l_p/l_m: most positive/negative wavespeed
  // Notes:
  //  - Follows same general procedure as vchar() in phys.c in Harm.
  //  - Variables are named as though 1 is normal direction.
  KOKKOS_INLINE_FUNCTION
  void WaveSpeedsGR(Real h, Real p, Real u0, Real u1, Real g00, Real g01, Real g11,
                     Real& l_p, Real& l_m)
  const {
    // Parameters and constants
    const Real discriminant_tol = -1.0e-10;  // values between this and 0 are considered 0

    // Calculate comoving sound speed
    Real cs_sq = gamma * p / h;

    // Set sound speeds in appropriate coordinates
    Real a = SQR(u0) - (g00 + SQR(u0)) * cs_sq;
    Real b = -2.0 * (u0*u1 - (g01 + u0*u1) * cs_sq);
    Real c = SQR(u1) - (g11 + SQR(u1)) * cs_sq;
    Real d = SQR(b) - 4.0*a*c;
    if (d < 0.0 && d > discriminant_tol) {
      d = 0.0;
    }
    Real d_sqrt = sqrt(d);
    Real root_1 = (-b + d_sqrt) / (2.0*a);
    Real root_2 = (-b - d_sqrt) / (2.0*a);
    if (root_1 > root_2) {
      l_p = root_1;
      l_m = root_2;
    } else {
      l_p = root_2;
      l_m = root_1;
    }
  }

  // inlined maximal fast magnetosonic wave speeds function for ideal gas in GRMHD
  // Inputs:
  //  - h: enthalpy per unit volume
  //  - p: gas pressure
  //  - u0, u1: contravariant components of 4-velocity
  //  - b_sq: b_\mu b^\mu
  //  - g00, g01, g11: contravariant components of metric (-1, 0, 1 in SR)
  // Outputs:
  //  - l_p/l_m: most positive/negative wavespeed
  // Notes:
  //  - Follows same general procedure as vchar() in phys.c in Harm.
  //  - Variables are named as though 1 is normal direction.
  KOKKOS_INLINE_FUNCTION
  void FastSpeedsGR(Real h, Real p, Real u0, Real u1, Real b_sq,
                    Real g00, Real g01, Real g11, Real& l_p, Real& l_m)
  const {
    // Calculate comoving fast magnetosonic speed
    Real cs_sq = gamma * p / h;
    Real va_sq = b_sq / (b_sq + h);
    Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;

    // Set fast magnetosonic speeds in appropriate coordinates
    Real a = SQR(u0) - (g00 + SQR(u0)) * cms_sq;
    Real b = -2.0 * (u0 * u1 - (g01 + u0 * u1) * cms_sq);
    Real c = SQR(u1) - (g11 + SQR(u1)) * cms_sq;
    Real a1 = b / a;
    Real a0 = c / a;
    Real s = fmax(SQR(a1) - 4.0 * a0, 0.0);
    s = sqrt(s);
    l_p = (a1 >= 0.0) ? -2.0 * a0 / (a1 + s) : (-a1 + s) / 2.0;
    l_m = (a1 >= 0.0) ? (-a1 - s) / 2.0 : -2.0 * a0 / (a1 - s);
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

  // virtual functions to convert cons to prim in either Hydro or MHD (depending on
  // arguments), overwritten in derived eos classes
  virtual void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim);
  virtual void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc);

  // virtual functions to convert prim to cons in either Hydro or MHD (depending on
  // arguments), overwritten in derived eos classes.  Used in SR/GR, and for prolongation
  // with SMR/AMR.
  virtual void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons);
  virtual void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                          DvceArray5D<Real> &cons);
};

//----------------------------------------------------------------------------------------
//! \class IsothermalHydro
//  \brief Derived class for Hydro isothermal EOS

class IsothermalHydro : public EquationOfState
{ 
public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IsothermalHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealHydro
//  \brief Derived class for ideal gas EOS in nonrelativistic hydro

class IdealHydro : public EquationOfState
{
public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealHydroSR
//  \brief Derived class for ideal gas EOS in special relativistic Hydro

class IdealSRHydro : public EquationOfState
{
public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealSRHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealHydroGR
//  \brief Derived class for ideal gas EOS in general relativistic Hydro

class IdealGRHydro : public EquationOfState
{
public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealGRHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons) override;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalMHD
//  \brief Derived class for isothermal EOS in nonrelativistic MHD

class IsothermalMHD : public EquationOfState
{
public:
  // Following suppress warnings that Hydro versions are not over-ridden
  using EquationOfState::ConsToPrim; 
  using EquationOfState::PrimToCons;

  IsothermalMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealMHD
//  \brief Derived class for ideal gas EOS in nonrelativistic MHD

class IdealMHD : public EquationOfState
{
public:
  // Following suppress warnings that Hydro versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons) override;
};

#endif // EOS_EOS_HPP_
