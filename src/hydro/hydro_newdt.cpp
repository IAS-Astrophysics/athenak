//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_newdt.cpp
//  \brief functions to computes timestep on given MeshBlock using CFL condition

#include <algorithm>  // min()
#include <cmath>      // fabs(), sqrt()
#include <limits>
#include <iostream>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// \!fn void Hydro::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlock for hydrodynamic problems

TaskStatus Hydro::NewTimeStep(Driver *pdrive, int stage) {

  MeshBlock *pmb = pmy_mblock;
  int is = pmb->indx.is; int js = pmb->indx.js; int ks = pmb->indx.ks;
  int ie = pmb->indx.ie; int je = pmb->indx.je; int ke = pmb->indx.ke;
  Real wi[5];

  Real dt1 = std::numeric_limits<float>::min();
  Real dt2 = std::numeric_limits<float>::min();
  Real dt3 = std::numeric_limits<float>::min();

  if (hydro_evol == HydroEvolution::kinematic) {

    // find largest (v) in each dirn for advection problems
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          wi[IVX] = u0(IVX,k,j,i)/u0(IDN,k,j,i);
          wi[IVY] = u0(IVY,k,j,i)/u0(IDN,k,j,i);
          wi[IVZ] = u0(IVZ,k,j,i)/u0(IDN,k,j,i);
          dt1 = std::max((std::abs(wi[IVX])), dt1);
          dt2 = std::max((std::abs(wi[IVY])), dt2);
          dt3 = std::max((std::abs(wi[IVZ])), dt3);
        }
      }
    }

  } else {
    // find largest (v +/- C) in each dirn for hydrodynamic problems
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        peos->ConservedToPrimitive(k, j, is, ie, u0, w_);
        for (int i=is; i<=ie; ++i) {
          wi[IDN] = w_(IDN,i);
          wi[IVX] = w_(IVX,i);
          wi[IVY] = w_(IVY,i);
          wi[IVZ] = w_(IVZ,i);
          wi[IPR] = w_(IPR,i);  // this value never used in isothermal EOS
          Real cs = peos->SoundSpeed(wi);
          dt1 = std::max((std::abs(wi[IVX]) + cs), dt1);
          dt2 = std::max((std::abs(wi[IVY]) + cs), dt2);
          dt3 = std::max((std::abs(wi[IVZ]) + cs), dt3);
        }
      }
    }

  }

  // compute minimum of dx1/(max_speed)
  dtnew = std::numeric_limits<float>::max();
  dtnew = std::min(dtnew, (pmb->mblock_size.dx1/dt1));

  // if grid is 2D/3D, compute minimum of dx2/(max_speed)
  if (pmb->pmy_mesh->nx2gt1) {
    dtnew = std::min(dtnew, (pmb->mblock_size.dx2/dt2));
  }

  // if grid is 3D, compute minimum of dx3/(max_speed)
  if (pmb->pmy_mesh->nx3gt1) {
    dtnew = std::min(dtnew, (pmb->mblock_size.dx3/dt3));
  }

  return TaskStatus::complete;
}

} // namespace hydro
