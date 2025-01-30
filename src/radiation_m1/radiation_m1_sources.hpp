#ifndef RADIATION_M1_SOURCES_HPP
#define RADIATION_M1_SOURCES_HPP

#include "athena.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include <athena_tensor.hpp>

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_macro.hpp
//  \brief macros for Grey M1 radiation class

// Solves the implicit problem
// .  q^new = q^star + dt S[q^new]
// The source term is S^a = (eta - ka J) u^a - (ka + ks) H^a and includes
// also emission.
int source_update(
    const Real cdt, const Real alp,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> gamma_ud,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud, const Real W,
    const Real Eold,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fold_d,
    const Real Estar,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fstar_d,
    const Real eta, const Real kabs, const Real kscat, Real chi, Real Enew,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fnew_d) {
    return 0;
}

#endif // RADIATION_M1_SOURCES_HPP
