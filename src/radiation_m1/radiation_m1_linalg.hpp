#ifndef RADIATION_M1_LINALG_HPP
#define RADIATION_M1_LINALG_HPP

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_linalg.hpp
//  \brief Linear algebra routines for M1

#include "radiation_m1_macro.hpp"

#include <athena.hpp>

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::compute_diag
//  \brief computes the columnwise L2 norm of a matrix J abd store in diag
KOKKOS_INLINE_FUNCTION
void qr_factorize(const Real (&J)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                  Real (&Q)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
                  Real (&R)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM]) {}

KOKKOS_INLINE_FUNCTION
void qr_update(Real (&Q)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],
               Real (&R)[M1_MULTIROOTS_DIM][M1_MULTIROOTS_DIM],  Real (&W)[M1_MULTIROOTS_DIM],
                Real (&V)[M1_MULTIROOTS_DIM]) {

}
} // namespace radiationm1
#endif // RADIATION_M1_LINALG_HPP
