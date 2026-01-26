#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.hpp
//! \brief Contains data and functions that implement conserved->primitive variable
//! conversion for various EOS (e.g. ideal gas, isothermal, etc.), for various fluids
//! (Hydro, MHD, etc.), and for non-relativistic and relativistic flows.

//#include <cmath>
#include <math.h>
#include <string>

#include "athena.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \struct EOSData
//! \brief container for EOS variables and functions needed inside kernels. Storing
//! everything in a container makes them easier to capture, and pass to inline functions,
//! inside kernels.

struct EOS_Data {
  Real gamma;        // ratio of specific heats for ideal gas
  Real iso_cs;       // isothermal sound speed
  bool is_ideal;     // flag to denote ideal gas EOS
  bool use_e, use_t; // use internal energy density (e) or temperature (t) as primitive
  Real dfloor, pfloor, tfloor, sfloor;  // density, pressure, temperature, entropy floors
  Real sfloor1, sfloor2, rho1, rho2; // density-dependent entropy floor
  Real gamma_max;    // ceiling on Lorentz factor in SR/GR

  // IDEAL GAS PRESSURE: converts primitive variable (either internal energy density e
  // or temperature e/d) into pressure.
  KOKKOS_INLINE_FUNCTION
  Real IdealGasPressure(const Real eint) const {
    return ((gamma-1.0)*eint);
  }

  // NON-RELATIVISTIC IDEAL GAS HYDRO: inlined sound speed function
  KOKKOS_INLINE_FUNCTION
  Real IdealHydroSoundSpeed(const Real d, const Real p) const {
    return sqrt(gamma*p/d);
  }

  // NON-RELATIVISTIC IDEAL GAS MHD: inlined fast magnetosonic speed function
  KOKKOS_INLINE_FUNCTION
  Real IdealMHDFastSpeed(const Real d, const Real p,
                         const Real bx, const Real by, const Real bz) const {
    Real asq = gamma*p;
    Real ct2 = by*by + bz*bz;
    Real qsq = bx*bx + ct2 + asq;
    Real tmp = bx*bx + ct2 - asq;
    return sqrt(0.5*(qsq + sqrt(tmp*tmp + 4.0*asq*ct2))/d);
  }

  // NON-RELATIVISTIC ISOTHERMAL MHD: inlined fast magnetosonic speed function
  KOKKOS_INLINE_FUNCTION
  Real IdealMHDFastSpeed(const Real d,
                         const Real bx, const Real by, const Real bz) const {
    Real asq = (iso_cs*iso_cs)*d;
    Real ct2 = by*by + bz*bz;
    Real qsq = bx*bx + ct2 + asq;
    Real tmp = bx*bx + ct2 - asq;
    return sqrt(0.5*(qsq + sqrt(tmp*tmp + 4.0*asq*ct2))/d);
  }

  // SPECIAL RELATIVISTIC IDEAL GAS HYDRO: inlined maximal sound wave speeds function
  // Inputs:
  //   d: density in comoving frame
  //   p: gas pressure
  //   ux: x-component of 4-velocity u^x
  //   lor: Lorentz factor \gamma
  // Outputs:
  //   l_p/m: most positive/negative wavespeed
  // Reference:
  //   Del Zanna et al, A&A 473, 11 (2007) (eq. 76)
  KOKKOS_INLINE_FUNCTION
  void IdealSRHydroSoundSpeeds(const Real d, const Real p, const Real ux, const Real lor,
                               Real& l_p, Real& l_m) const {
    Real cs2 = gamma*p / (d + gamma*p/(gamma - 1.0));  // (DZB 73)
    Real v2 = 1.0 - 1.0/(lor*lor);
    auto const p1 = (ux/lor) * (1.0 - cs2);
    auto const tmp = sqrt(cs2 * ((1.0-v2*cs2) - p1*(ux/lor))) / lor;
    auto const invden = 1.0/(1.0 - v2*cs2);

    l_p = (p1 + tmp) * invden;
    l_m = (p1 - tmp) * invden;
  }

  // SPECIAL RELATIVISTIC IDEAL GAS MHD: inlined maximal fast magnetosonic wave speeds fn
  // arguments same or SR hydro version, with the addition of b_sq = b_\mu b_\mu
  // Reference:
  //   Del Zanna et al, A&A 473, 11 (2007) (eq. 76)
  KOKKOS_INLINE_FUNCTION
  void IdealSRMHDFastSpeeds(const Real d, const Real p, const Real ux, const Real lor,
                            const Real b_sq, Real& l_p, Real& l_m) const {
    // Calculate comoving fast magnetosonic speed
    Real w = d + gamma*p/(gamma - 1.0);
    Real cs_sq = gamma*p/w;                            // (DZB 73)
    Real va_sq = b_sq / (b_sq + w);                    // (DZB 73)
    Real cms_sq = cs_sq + va_sq - cs_sq * va_sq;       // (DZB 72)

    Real v2 = 1.0 - 1.0/(lor*lor);
    auto const p1 = (ux/lor) * (1.0 - cms_sq);
    auto const tmp = sqrt(cms_sq * ((1.0-v2*cms_sq) - p1*(ux/lor))) / lor;
    auto const invden = 1.0/(1.0 - v2*cms_sq);

    l_p = (p1 + tmp) * invden;
    l_m = (p1 - tmp) * invden;
  }

  // GENERAL RELATIVISTIC IDEAL GAS HYDRO: inlined maximal sound wave speeds fn
  // Inputs:
  //  - d: density in comoving frame
  //  - p: gas pressure
  //  - u0,u1: 4-velocity components u^0, u^1
  //  - g00,g01,g11: metric components g^00, g^01, g^11
  // Outputs:
  //  - l_p/l_m: most positive/negative wavespeed
  // Notes:
  //  - Follows same general procedure as vchar() in phys.c in Harm.
  //  - Variables are named as though 1 is normal direction.
  KOKKOS_INLINE_FUNCTION
  void IdealGRHydroSoundSpeeds(const Real d, const Real p, const Real u0, const Real u1,
                               const Real g00, const Real g01, const Real g11,
                               Real& l_p, Real& l_m) const {
    // Parameters and constants
    const Real discriminant_tol = -1.0e-10;  // values between this and 0 are considered 0

    // Calculate comoving sound speed
    Real cs_sq = gamma * p / (d + gamma*p/(gamma - 1.0));

    // Set sound speeds in appropriate coordinates
    Real a = SQR(u0) - (g00 + SQR(u0)) * cs_sq;
    Real b = -2.0 * (u0*u1 - (g01 + u0*u1) * cs_sq);
    Real c = SQR(u1) - (g11 + SQR(u1)) * cs_sq;
    Real dis = SQR(b) - 4.0*a*c;
    if (dis < 0.0 && dis > discriminant_tol) {
      dis = 0.0;
    }
    // TODO(@pdmullen): fmax(dis, 0.0) prevents NaNs (see Issue #7), but this should be
    // eliminated after enforcing positivity on recon L/R densities and pressures
    Real dis_sqrt = sqrt(fmax(dis, 0.0));
    Real root_1 = (-b + dis_sqrt) / (2.0*a);
    Real root_2 = (-b - dis_sqrt) / (2.0*a);
    if (root_1 > root_2) {
      l_p = root_1;
      l_m = root_2;
    } else {
      l_p = root_2;
      l_m = root_1;
    }
  }

  // GENERAL RELATIVISTIC IDEAL GAS MHD: inlined maximal fast magnetosonic wave speeds fn
  // Inputs:
  //  - d: density in comoving frame
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
  void IdealGRMHDFastSpeeds(const Real d, const Real p, const Real u0, const Real u1,
                            const Real b_sq, const Real g00, const Real g01,
                            const Real g11, Real& l_p, Real& l_m) const {
    // Calculate comoving fast magnetosonic speed
    Real w = d + gamma*p/(gamma - 1.0);
    Real cs_sq = gamma * p / w;
    Real va_sq = b_sq / (b_sq + w);
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
//! \brief Abstract base class for EOS.

class EquationOfState {
 public:
  EquationOfState(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  virtual ~EquationOfState() = default;

  MeshBlockPack* pmy_pack;
  EOS_Data eos_data;

  // virtual functions to convert cons to prim in either Hydro or MHD (depending on
  // arguments), overwritten in derived eos classes
  virtual void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                          const bool only_testfloors,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku);
  virtual void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                          DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                          const bool only_testfloors,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku);

  // virtual functions to convert prim to cons in either Hydro or MHD (depending on
  // arguments), overwritten in derived eos classes.
  virtual void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                          const int il, const int iu, const int jl, const int ju,
                          const int kl, const int ku);
  virtual void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                          DvceArray5D<Real> &cons, const int il, const int iu,
                          const int jl, const int ju, const int kl, const int ku);
};

//----------------------------------------------------------------------------------------
//! \class IsothermalHydro
//! \brief Derived class for isothermal EOS in nonrelativistic Hydro

class IsothermalHydro : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IsothermalHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealHydro
//! \brief Derived class for ideal gas EOS in nonrelativistic hydro

class IdealHydro : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealSRHydro
//! \brief Derived class for ideal gas EOS in special relativistic Hydro

class IdealSRHydro : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealSRHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealGRHydro
//! \brief Derived class for ideal gas EOS in general relativistic Hydro

class IdealGRHydro : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealGRHydro(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IsothermalMHD
//! \brief Derived class for isothermal EOS in nonrelativistic MHD

class IsothermalMHD : public EquationOfState {
 public:
  // Following suppress warnings that Hydro versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IsothermalMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealMHD
//! \brief Derived class for ideal gas EOS in nonrelativistic MHD

class IdealMHD : public EquationOfState {
 public:
  // Following suppress warnings that Hydro versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealSRMHD
//! \brief Derived class for ideal gas EOS in special relativistic MHD

class IdealSRMHD : public EquationOfState {
 public:
  // Following suppress warnings that hydro versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealSRMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
};

//----------------------------------------------------------------------------------------
//! \class IdealGRMHD
//! \brief Derived class for ideal gas EOS in general relativistic MHD

class IdealGRMHD : public EquationOfState {
 public:
  // Following suppress warnings that MHD versions are not over-ridden
  using EquationOfState::ConsToPrim;
  using EquationOfState::PrimToCons;

  IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin);
  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                  DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  const bool only_testfloors,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) override;
  void PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons, const int il, const int iu,
                  const int jl, const int ju, const int kl, const int ku) override;
};

#endif // EOS_EOS_HPP_
