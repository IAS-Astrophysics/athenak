#ifndef ADM_ADM_HPP_
#define ADM_ADM_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm.hpp
//! \brief definitions for ADM class
//! WARNING: The ADM object needs to be allocated after Z4c

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "parameter_input.hpp"

// forward declarations
class MeshBlockPack;

namespace adm {

//! \class ADM
class ADM {
  public:
    ADM(MeshBlockPack *ppack, ParameterInput *pin);
    ~ADM();

    // Indices of ADM variables
    enum {
      I_ADM_gxx, I_ADM_gxy, I_ADM_gxz, I_ADM_gyy, I_ADM_gyz, I_ADM_gzz,
      I_ADM_Kxx, I_ADM_Kxy, I_ADM_Kxz, I_ADM_Kyy, I_ADM_Kyz, I_ADM_Kzz,
      I_ADM_psi4,
      I_ADM_alpha, I_ADM_betax, I_ADM_betay, I_ADM_betaz,
      N_ADM
    };
    // Names of ADM variables
    static char const * const ADM_names[N_ADM];

    struct ADM_vars {
      AthenaTensorField<Real, TensorSymm::NONE, 3, 0> alpha;     // lapse
      AthenaTensorField<Real, TensorSymm::NONE, 3, 1> beta_u;    // shift vector
      AthenaTensorField<Real, TensorSymm::NONE, 3, 0> psi4;      // conformal factor 
      AthenaTensorField<Real, TensorSymm::SYM2, 3, 2> g_dd;      // spatial metric
      AthenaTensorField<Real, TensorSymm::SYM2, 3, 2> K_dd;      // extrinsic curvature
    }; 
    ADM_vars adm;

    DvceArray5D<Real> u_adm;                                     // adm variables 

  private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Z4c
};

}

#endif
