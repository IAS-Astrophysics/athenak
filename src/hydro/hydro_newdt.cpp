//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_newdt.cpp
//  \brief functions to computes timestep on given MeshBlock using CFL condition

//#include <algorithm>  // min()
#include <limits>
#include <math.h>
#include <iostream>

#include "athena.hpp"
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
  int is = pmb->mb_cells.is; int nx1 = pmb->mb_cells.nx1;
  int js = pmb->mb_cells.js; int nx2 = pmb->mb_cells.nx2;
  int ks = pmb->mb_cells.ks; int nx3 = pmb->mb_cells.nx3;
//  int ie = pmb->mb_cells.ie;
//  int je = pmb->mb_cells.je;
//  int ke = pmb->mb_cells.ke;

  Real dv1 = std::numeric_limits<float>::min();
  Real dv2 = std::numeric_limits<float>::min();
  Real dv3 = std::numeric_limits<float>::min();

std::cout << "in nudt" << std::endl;

  if (hydro_evol == HydroEvolution::kinematic) {

std::cout << "start loop in nudt" << std::endl;
    // find largest (v) in each direction for advection problems
    // Kokkos::parallel_reduce uses 1D range policy

    auto &w0_ = w0;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;

    Kokkos::parallel_reduce("HydroNudt1",Kokkos::RangePolicy<>(pmb->exe_space, 0, nkji),
      KOKKOS_LAMBDA(const int &idx, Real &max_dv1, Real &max_dv2, Real &max_dv3)
      {
      // compute n,k,j,i indices of thread and call function
      int k = (idx)/nji;
      int j = (idx - k*nji)/nx1;
      int i = (idx - k*nji - j*nx1);
      k += ks;
      j += js;
      i += is;
      max_dv1 = fmax(fabs(w0_(IVX,k,j,i)), max_dv1);
      max_dv2 = fmax(fabs(w0_(IVY,k,j,i)), max_dv2);
      max_dv3 = fmax(fabs(w0_(IVZ,k,j,i)), max_dv3);
    }, Kokkos::Max<Real>(dv1), Kokkos::Max<Real>(dv2),Kokkos::Max<Real>(dv3));
 
std::cout << "end loop in nudt" << std::endl;

  } else {
    // find largest (v +/- C) in each dirn for hydrodynamic problems
    // Kokkos::parallel_reduce uses 1D range policy
    auto &w0_ = w0;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    Kokkos::parallel_reduce("HydroNudt2",Kokkos::RangePolicy<>(pmb->exe_space, 0, nkji),
      KOKKOS_LAMBDA(const int &idx, Real &max_dv1, Real &max_dv2, Real &max_dv3)
    { 
      // compute n,k,j,i indices of thread and call function
      int k = (idx)/nji;
      int j = (idx - k*nji)/nx1;
      int i = (idx - k*nji - j*nx1);
      k += ks;
      j += js;
      i += is;

      Real cs = peos->SoundSpeed(w0_(IPR,k,j,i),w0_(IDN,k,j,i));
      max_dv1 = fmax((fabs(w0_(IVX,k,j,i)) + cs), max_dv1);
      max_dv2 = fmax((fabs(w0_(IVY,k,j,i)) + cs), max_dv2);
      max_dv3 = fmax((fabs(w0_(IVZ,k,j,i)) + cs), max_dv3);
    }, Kokkos::Max<Real>(dv1), Kokkos::Max<Real>(dv2),Kokkos::Max<Real>(dv3));

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

std::cout << "done nudt" << std::endl;
  return TaskStatus::complete;
}

} // namespace hydro
