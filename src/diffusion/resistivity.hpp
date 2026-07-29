#ifndef DIFFUSION_RESISTIVITY_HPP_
#define DIFFUSION_RESISTIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.hpp
//  \brief Contains data and functions that implement various non-ideal MHD (resistive)
//  processes, such as Ohmic diffusion and ambipolar diffusion. TODO(@user): Hall effect.
//
//  Non-ideal terms are enabled via the <mhd> input block: a non-zero "eta_ohm" enables
//  Ohmic resistivity, a non-zero "eta_ad" enables ambipolar diffusion. The presence of
//  either parameter triggers construction of this class in the MHD constructor.

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/meshblock.hpp"

//----------------------------------------------------------------------------------------
//! \class Resistivity
//  \brief data and functions that implement various non-ideal MHD physics (Ohmic
//  resistivity and ambipolar diffusion)

class Resistivity {
 public:
  Resistivity(MeshBlockPack *pp, ParameterInput *pin);
  ~Resistivity();

  // data
  Real dtnew;
  Real eta_ohm;   // Ohmic resistivity coefficient (0 => Ohmic term off)
  Real eta_ad;    // ambipolar diffusion coefficient (0 => ambipolar term off)

  // wrapper functions: add non-ideal E-Field and energy (Poynting) flux. Each dispatches
  // to the Ohmic and/or ambipolar implementations depending on which coefficients are on.
  void AddResistiveEMFs(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddResistiveFluxes(const DvceFaceFld4D<Real> &b0, DvceFaceFld5D<Real> &flx);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);

  // Implementation kernels. These MUST be public: each launches Kokkos par_for kernels,
  // nvcc forbids an extended __host__ __device__ lambda inside a private/protected member
  // function (the CUDA build fails otherwise; the CPU/Serial build does not catch this).
  // Matches main, where the Ohmic implementations are also public.
  // Ohmic resistivity (constant coefficient): E = eta_ohm * J
  void AddEMFConstantResist(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddFluxConstantResist(const DvceFaceFld4D<Real> &b, DvceFaceFld5D<Real> &flx);

  // Ambipolar diffusion (constant coefficient): E = eta_ad * [B^2 J - (J.B) B].
  void AddEMFConstantAmbipolar(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddFluxConstantAmbipolar(const DvceFaceFld4D<Real> &b0, DvceFaceFld5D<Real> &flx);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_RESISTIVITY_HPP_
