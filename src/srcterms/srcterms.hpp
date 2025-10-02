#ifndef SRCTERMS_SRCTERMS_HPP_
#define SRCTERMS_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.hpp
//! \brief Data, functions, and classes to implement various source terms in the hydro
//! and/or MHD equations of motion.  Currently implemented:
//!  (1) constant (gravitational) acceleration - for RTI
//!  (2) shearing box in 2D (x-z), for both hydro and MHD
//!  (3) random forcing to drive turbulence - implemented in TurbulenceDriver class

#include <map>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class SourceTerms
//! \brief data and functions for physical source terms

class SourceTerms {
 public:
  SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~SourceTerms();

  // data
  // flags for various source terms
  bool const_accel;
  bool ism_cooling;
  bool rel_cooling;
  bool rad_beam;

  // new timestep
  Real dtnew;

  // data for constant accel
  Real const_accel_val;   // magnitude of accn
  int const_accel_dir;    // direction of accn

  // data for ISM cooling
  Real hrate;

  // data for relativistic cooling
  Real crate_rel;
  Real cpower_rel;

  // data for radiation beam source
  Real dii_dt;            // injection rate
  Real pos1, pos2, pos3;  // position of source
  Real dir1, dir2, dir3;  // direction of source
  Real width, spread;     // spatial width of source region, spread in angles

  // functions
  void ApplySrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, DvceArray5D<Real> &u0);
  void ApplySrcTerms(DvceArray5D<Real> &i0, const Real bdt);
  void ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, DvceArray5D<Real> &u0);
  void ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void BeamSource(DvceArray5D<Real> &i0, const Real bdt);
  void NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos);

 private:
  MeshBlockPack *pmy_pack;
};

#endif  // SRCTERMS_SRCTERMS_HPP_
