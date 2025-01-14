#ifndef RADIATION_M1_MACRO_HPP
#define RADIATION_M1_MACRO_HPP

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_macro.hpp
//  \brief macros for Grey M1 radiation class

//#ifndef WARN_FOR_SRC_FIX
//#define WARN_FOR_SRC_FIX
//#endif

#define RADIATION_M1_SRC_EXPL         1       // explicit RHS
#define RADIATION_M1_SRC_IMPL         2       // implicit RHS (default)
#define RADIATION_M1_SRC_BOOST        3       // boost to fluid frame (approximate!)
#ifndef RADIATION_M1_SRC_METHOD
#define RADIATION_M1_SRC_METHOD       RADIATION_M1_SRC_IMPL
#endif

#define RADIATION_M1_NGHOST           2

#define RADIATION_M1_CLS_SHIBATA      1       // optically thin limit from Shibata 2011
#define RADIATION_M1_CLS_SIMPLE       2       // simplified closure (needed for inv closure to work)
#ifndef RADIATION_M1_CLS_METHOD
#define RADIATION_M1_CLS_METHOD       RADIATION_M1_CLS_SIMPLE
#endif

#define CGS_GCC (1.619100425158886e-18) // Multiply to convert density from CGS to Cactus

#define SQ(X) ((X)*(X))

#endif //RADIATION_M1_MACRO_HPP
