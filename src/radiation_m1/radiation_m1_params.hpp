#ifndef RADIATION_M1_PARAMS_HPP
#define RADIATION_M1_PARAMS_HPP

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
  Toy,
  Weakrates,
  BnsNurates,
};

//----------------------------------------------------------------------------------------
//! \enum RadiationM1SrcUpdate
//  \brief method to treat radiation sources
enum RadiationM1SrcUpdate {
  Explicit,
  Implicit,
  Boost,
};

//----------------------------------------------------------------------------------------
//! \struct RadiationM1Params
//  \brief parameters for the Grey M1 class
struct RadiationM1Params {
  Real rad_E_floor;
  Real rad_N_floor;
  Real rad_eps;
  Real minmod_theta;
  RadiationM1Closure closure_fun;
  RadiationM1OpacityType opacity_type;
  RadiationM1SrcUpdate src_update;
};
} // namespace radiationm1
#endif // RADIATION_M1_PARAMS_HPP
