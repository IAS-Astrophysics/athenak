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

#define M1_TOTAL_NUM_SPECIES 1
#define M1_MULTIROOTS_DIM 4

#define HC_MEVCM 1.23984172e-10 // hc in units of MeV*cm
#define MEV_TO_ERG 1.60217733e-6
#define NORMFACT 1e50

#define SQ(X) ((X) * (X))

#endif // RADIATION_M1_MACRO_HPP
