#ifndef RADIATION_M1_MACRO_HPP
#define RADIATION_M1_MACRO_HPP

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_macro.hpp
//  \brief macros for Grey M1 radiation class

#define M1_E_IDX 0
#define M1_FX_IDX 1
#define M1_FY_IDX 2
#define M1_FZ_IDX 3
#define M1_N_IDX 4

#define M1_MULTIROOTS_DIM 4

// The (EOS policy, error policy) pairs M1 supports. Every dispatch site expands
// this same list, so a new EOS is registered here once rather than at each site.
// Requires "dyn_grmhd/dyn_grmhd.hpp" and the EOS headers to be included first.
#define M1_FOREACH_EOS(DO)                                     \
  DO(Primitive::EOSCompOSE<Primitive::NQTLogs>,                \
     Primitive::ResetFloor)                                    \
  DO(Primitive::EOSCompOSE<Primitive::NormalLogs>,             \
     Primitive::ResetFloor)                                    \
  DO(Primitive::EOSTransition<Primitive::NQTLogs>,             \
     Primitive::ResetFloorTransition)                          \
  DO(Primitive::EOSTransition<Primitive::NormalLogs>,          \
     Primitive::ResetFloorTransition)

#endif // RADIATION_M1_MACRO_HPP
