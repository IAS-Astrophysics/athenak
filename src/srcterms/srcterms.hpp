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
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"

// constants that enumerate operator split and unsplit source terms
enum class SplitSrcTermTaskName {undef=0, hydro_forcing, mhd_forcing};
enum class UnsplitSrcTermTaskName {undef=0, hydro_acc, hydro_sbox, hydro_drag, mhd_acc,
  mhd_sbox, mhd_sbox_emf, mhd_drag};

// forward declarations
class TurbulenceDriver;
class Driver;

//----------------------------------------------------------------------------------------
//! \class SourceTerms
//! \brief data and functions for physical source terms

class SourceTerms
{
 public:
  SourceTerms(MeshBlockPack *pp, ParameterInput *pin);
  ~SourceTerms();

  // data
  // flags for various source terms
  bool random_forcing;
  bool const_accel;
  bool shearing_box;
  bool twofluid_mhd;

  // constants/coefficients for various terms
  Real const_acc1, const_acc2, const_acc3;
  Real omega0, qshear;
  Real twofluid_drag;
  TurbulenceDriver *pturb=nullptr;   // class which implements random forcing

  // map for associating source term TaskName with TaskID
  std::map<SplitSrcTermTaskName, TaskID> split_tasks;
  std::map<UnsplitSrcTermTaskName, TaskID> unsplit_tasks;

  // functions
  void IncludeSplitSrcTermTasks(TaskList &tl, TaskID start);
  void IncludeUnsplitSrcTermTasks(TaskList &tl, TaskID start);
  TaskStatus ApplyRandomForcing(Driver *pdrive, int stage);
  TaskStatus AddConstantAccelHydro(Driver *pdrive, int stage);
  TaskStatus AddConstantAccelMHD(Driver *pdrive, int stage);
  TaskStatus AddSBoxMomentumHydro(Driver *pdrive, int stage);
  TaskStatus AddSBoxMomentumMHD(Driver *pdrive, int stage);
  TaskStatus AddSBoxEMF(Driver *pdrive, int stage);
  TaskStatus AddTwoFluidDragHydro(Driver *pdrive, int stage);
  TaskStatus AddTwoFluidDragMHD(Driver *pdrive, int stage);
  void ConstantAccel(DvceArray5D<Real> &u, DvceArray5D<Real> &w,
                     const EOS_Data &eos, Real bdt);
  // shearing box terms for hydro
  void SBoxMomentumTerms(DvceArray5D<Real> &u, DvceArray5D<Real> &w, 
                         const EOS_Data &eos, Real bdt);
  // shearing box terms for MHD (face-centered B included in arguments)
  void SBoxMomentumTerms(DvceArray5D<Real> &u, DvceArray5D<Real> &w, 
                         DvceFaceFld4D<Real> &b, const EOS_Data &eos, Real bdt);
  void SBoxEMF(const DvceFaceFld4D<Real> &b, DvceEdgeFld4D<Real> &efld);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // SRCTERMS_SRCTERMS_HPP_
