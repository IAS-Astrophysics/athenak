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

// forward declarations
class TurbulenceDriver;
class Driver;

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
  bool beam;
  bool shearing_box, shearing_box_r_phi;

  // new timestep
  Real dtnew;

  // magnitude and direction of constant accel
  Real const_accel_val;
  int const_accel_dir;

  // heating rate used with ISM cooling
  Real hrate;

  // cooling rate used with relativistic cooling
  Real crate_rel;
  Real cpower_rel;

  // beam source
  Real dii_dt;

  // shearing box
  Real qshear, omega0;

  // functions
  void ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real dt, DvceArray5D<Real> &u0);
  void ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real dt, DvceArray5D<Real> &u0);
  void RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real dt, DvceArray5D<Real> &u0);
  void BeamSource(DvceArray5D<Real> &i0, const Real dt);
  void ShearingBox(const DvceArray5D<Real> &w0, const EOS_Data &eos_data, const Real bdt,
                   DvceArray5D<Real> &u0);
  void ShearingBox(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                   const EOS_Data &eos_data, const Real bdt, DvceArray5D<Real> &u0);
  // in 2D shearing box there is a source term for Ex and Ey
  void SBoxEField(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);

  void NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos);

 private:
  MeshBlockPack *pmy_pack;
};

#endif  // SRCTERMS_SRCTERMS_HPP_
