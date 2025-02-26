#ifndef RADIATION_M1_PARAMS_HPP
#define RADIATION_M1_PARAMS_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_params.hpp
//  \brief enums/structs for various params of Grey M1

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \enum RadiationM1Closure
//  \brief choice of M1 closure
enum RadiationM1Closure {
  Minerbo,
  Eddington,
  Thin,
};

//----------------------------------------------------------------------------------------
//! \enum RadiationM1OpacityType
//  \brief choice of neutrino opacity library
enum RadiationM1OpacityType {
  None,
  Toy,
  BnsNurates,
};

//----------------------------------------------------------------------------------------
//! \enum RadiationM1SrcUpdate
//  \brief method to treat radiation sources
enum RadiationM1SrcUpdate {
  Explicit,
  Implicit,
};

//----------------------------------------------------------------------------------------
//! \struct RadiationM1Params
//  \brief parameters for the Grey M1 class
struct RadiationM1Params {
  RadiationM1Closure closure_fun;       // choice of closure
  RadiationM1OpacityType opacity_type;  // choice of opacity library
  RadiationM1SrcUpdate src_update;      // choice of source update

  bool gr_sources;      // include GR sources
  bool matter_sources;  // include matter sources
  bool theta_limiter;   // activate theta limiter

  int nspecies;              // number of neutrino species
  Real closure_epsilon;      // precision with which to find closure
  int closure_maxiter;       // maximum number of iterations in closure root finder
  Real inv_closure_epsilon;  // precision with which to find inverse closure
  int inv_closure_maxiter;  // maximum number of iterations in inverse closure root finder

  Real minmod_theta;    // value of theta for minmod limiter
  Real rad_E_floor;     // radiation energy density floor
  Real rad_N_floor;     // radiation number density floor
  Real rad_eps;         // Impose F_a F^a < (1 - rad_E_eps) E2
  Real source_Ye_max;   // maximum allowed Ye for matter
  Real source_Ye_min;   // minimum allowed Ye for matter
  Real source_limiter;  // limiter for matter source (0: sources disabled, 1: sources
                        // limited to avoid negative energies)
  Real source_epsabs;   // target absolute precision for non-linear solver
  Real source_epsrel;   // target relative precision for non-linear solver
  int source_maxiter;   // maximum number of iterations for non-linear solver
};

}  // namespace radiationm1
#endif  // RADIATION_M1_PARAMS_HPP
