//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm_z4c.cpp
//  \brief implementation of functions in the Z4c class related to ADM decomposition


// C++ standard headers
#include <cmath> // pow
#include <iostream>
#include <fstream>

// Athena++ headers
#include "parameter_input.hpp" 
#include "athena.hpp" 
#include "mesh/mesh.hpp" 
#include "adm/adm.hpp"
#include "eos/eos.hpp"
#include "z4c/z4c.hpp" 
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp" 

namespace z4c {

//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMMatterSource(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat)
// \brief compute matter source terms
//

void Z4c::ADMMatterSource(MeshBlockPack *pmbp) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;


// don't need this in GZs?
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;

  auto &eos = pmbp->phydro->peos->eos_data; 
  Real const gamma_prime = eos.gamma / (eos.gamma - 1.0);


//  auto &z4c = pmbp->pz4c->z4c;
  auto &adm = pmbp->padm->adm;
  auto &mat = pmbp->pz4c->mat;
  auto &w0_ = pmbp->phydro->w0;

  int scr_level = 0;
  // 2 1D scratch array and 1 2D scratch array
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*4     // 0 tensors
                  + ScrArray2D<Real>::shmem_size(3,ncells1)*1;  // vectors


  par_for_outer("ADM Matter Sources",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {  
    AthenaScratchTensor<Real, TensorSymm::NONE, 1, 0> pgas, wgas, w_lor, w_lor2;
    AthenaScratchTensor<Real, TensorSymm::NONE, 1, 1> u_d;
    
    pgas.NewAthenaScratchTensor(member, scr_level, ncells1);
    wgas.NewAthenaScratchTensor(member, scr_level, ncells1);
    w_lor.NewAthenaScratchTensor(member, scr_level, ncells1);
    w_lor2.NewAthenaScratchTensor(member, scr_level, ncells1);
    u_d.NewAthenaScratchTensor(member, scr_level, ncells1);

    w_lor2.ZeroClear();
    u_d.ZeroClear();

    par_for_inner(member, is, ie, [&](int const i) {
      pgas(i) = eos.IdealGasPressure(w0_(m,IEN,k,j,i));
      wgas(i) =  w0_(m,IDN,k,j,i) + gamma_prime *pgas(i);
    });

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](int const i) {
          w_lor2(i) += adm.g_dd(m,a,b,k,j,i)*w0_(m,IVX+a,k,j,i)*w0_(m,IVX+b,k,j,i);
        });
      }
    }

    par_for_inner(member, is, ie, [&](int const i) {
      w_lor2(i) = w_lor2(i) + 1.0;
      w_lor(i) = std::sqrt(w_lor2(i));
    });
  
  
    for (int a = 0; a < 3; ++a) {
     for (int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](int const i) {
        u_d(a,i) += adm.g_dd(m,a,b,k,j,i)*w0_(m,IVX+b,k,j,i);
      });
    }
  
    par_for_inner(member, is, ie, [&](int const i) {
      mat.rho(m,k,j,i) = wgas(i) * w_lor2(i) - pgas(i);
    });
    
    for (int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](int const i) {
        mat.S_d(m,a,k,j,i) = wgas(i) * w_lor(i) * u_d(a,i);
      });
      for (int b = 0; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](int const i) {
          mat.S_dd(m,a,b,k,j,i) = wgas(i)  * u_d(a,i) * u_d(b,i) + pgas(i)*adm.g_dd(m,a,b,k,j,i);
        });
      }
    }
  }
  
});
}
}
