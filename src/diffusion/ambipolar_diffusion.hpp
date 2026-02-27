#ifndef DIFFUSION_AMBIPOLAR_DIFFUSION_HPP_
#define DIFFUSION_AMBIPOLAR_DIFFUSION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ambipolar_diffusion.hpp
//  \brief Contains data and functions that implement ambipolar diffusion in MHD.
//  The ambipolar EMF is: E_amb = eta_ad * [B^2 * J - (J.B) * B]
//  where J = curl(B) is the current density.

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/meshblock.hpp"

//----------------------------------------------------------------------------------------
//! \class AmbipolarDiffusion
//  \brief data and functions for ambipolar diffusion in MHD

class AmbipolarDiffusion {
 public:
  AmbipolarDiffusion(MeshBlockPack *pp, ParameterInput *pin);
  ~AmbipolarDiffusion();

  // data
  Real dtnew;
  std::string amb_type;  // only "constant" implemented
  Real eta_ad;           // ambipolar diffusion coefficient

  // wrapper functions (dispatch to type-specific implementations)
  void AddAmbipolarEMFs(const DvceFaceFld4D<Real> &b0,
                        const DvceArray5D<Real> &bcc0,
                        DvceEdgeFld4D<Real> &efld);
  void AddAmbipolarFluxes(const DvceFaceFld4D<Real> &b0,
                          const DvceArray5D<Real> &bcc0,
                          DvceFaceFld5D<Real> &flx);

  // type-specific implementations (constant coefficient)
  void AddEMFConstantAmbipolar(const DvceFaceFld4D<Real> &b0,
                               const DvceArray5D<Real> &bcc0,
                               DvceEdgeFld4D<Real> &efld);
  void AddFluxConstantAmbipolar(const DvceFaceFld4D<Real> &b0,
                                const DvceArray5D<Real> &bcc0,
                                DvceFaceFld5D<Real> &flx);

  // dynamic timestep (recomputed each cycle from B field)
  void NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_AMBIPOLAR_DIFFUSION_HPP_
