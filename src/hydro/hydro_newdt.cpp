//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_newdt.cpp
//  \brief functions to computes timestep on given MeshBlock using CFL condition

#include <algorithm>  // min()
#include <limits>
#include <math.h>
#include <iostream>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// \!fn void Hydro::NewTimeStep()
// \brief calculate the minimum timestep within a MeshBlock for hydrodynamic problems

TaskStatus Hydro::NewTimeStep(Driver *pdrive, int stage) {

  if (stage != pdrive->nstages) return TaskStatus::complete; // only execute on last stage
  
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int is = pmb->mb_cells.is; int ie = pmb->mb_cells.ie;
  int js = pmb->mb_cells.js; int je = pmb->mb_cells.je;
  int ks = pmb->mb_cells.ks; int ke = pmb->mb_cells.ke;

  Real dv1 = std::numeric_limits<float>::min();
  Real dv2 = std::numeric_limits<float>::min();
  Real dv3 = std::numeric_limits<float>::min();

  if (hydro_evol == HydroEvolution::kinematic) {

    // find largest (v) in each dirn for advection problems
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          dv1 = std::max(fabs(w0(IVX,k,j,i)), dv1);
          dv2 = std::max(fabs(w0(IVY,k,j,i)), dv2);
          dv3 = std::max(fabs(w0(IVZ,k,j,i)), dv3);
        }
      }
    }

  } else {
    // find largest (v +/- C) in each dirn for hydrodynamic problems
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real wi[5];
          wi[IDN] = w0(IDN,k,j,i);
          wi[IVX] = w0(IVX,k,j,i);
          wi[IVY] = w0(IVY,k,j,i);
          wi[IVZ] = w0(IVZ,k,j,i);
          wi[IPR] = w0(IPR,k,j,i);  // this value never used in isothermal EOS
          Real cs = peos->SoundSpeed(wi);
          dv1 = std::max((std::abs(wi[IVX]) + cs), dv1);
          dv2 = std::max((std::abs(wi[IVY]) + cs), dv2);
          dv3 = std::max((std::abs(wi[IVZ]) + cs), dv3);
        }
      }
    }

  }

  // compute minimum of dx1/(max_speed)
  dtnew = std::numeric_limits<float>::max();
  dtnew = std::min(dtnew, (pmb->mb_cells.dx1/dv1));

  // if grid is 2D/3D, compute minimum of dx2/(max_speed)
  if (pmesh_->nx2gt1) {
    dtnew = std::min(dtnew, (pmb->mb_cells.dx2/dv2));
  }

  // if grid is 3D, compute minimum of dx3/(max_speed)
  if (pmesh_->nx3gt1) {
    dtnew = std::min(dtnew, (pmb->mb_cells.dx3/dv3));
  }

  return TaskStatus::complete;
}

} // namespace hydro
