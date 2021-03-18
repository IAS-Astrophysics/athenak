//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_newdt.cpp
//  \brief function to compute hydro timestep across all MeshBlock(s) in a MeshBlockPack

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

TaskStatus Hydro::NewTimeStep(Driver *pdriver, int stage)
{
  if (stage != pdriver->nstages) return TaskStatus::complete; // only execute last stage
  
  int is = pmy_pack->mb_cells.is; int nx1 = pmy_pack->mb_cells.nx1;
  int js = pmy_pack->mb_cells.js; int nx2 = pmy_pack->mb_cells.nx2;
  int ks = pmy_pack->mb_cells.ks; int nx3 = pmy_pack->mb_cells.nx3;
  auto &eos = pmy_pack->phydro->peos->eos_data;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  auto &w0_ = w0;
  auto &mbsize = pmy_pack->pmb->mbsize;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  if (pdriver->time_evolution == TimeEvolution::kinematic) {
    // find smallest (dx/v) in each direction for advection problems
    Kokkos::parallel_reduce("HydroNudt1",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3)
      {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      min_dt1 = fmin((mbsize.dx1.d_view(m)/fabs(w0_(m,IVX,k,j,i))), min_dt1);
      min_dt2 = fmin((mbsize.dx2.d_view(m)/fabs(w0_(m,IVY,k,j,i))), min_dt2);
      min_dt3 = fmin((mbsize.dx3.d_view(m)/fabs(w0_(m,IVZ,k,j,i))), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
 
  } else {

    if (!relativistic) {
      // find smallest dx/(v +/- C) in each direction for hydrodynamic problems
      Kokkos::parallel_reduce("HydroNudt2",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
	KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3)
      { 
	// compute m,k,j,i indices of thread and call function
	int m = (idx)/nkji;
	int k = (idx - m*nkji)/nji;
	int j = (idx - m*nkji - k*nji)/nx1;
	int i = (idx - m*nkji - k*nji - j*nx1) + is;
	k += ks;
	j += js;

	Real cs;
	if (eos.is_adiabatic) {
	  cs = eos.SoundSpeed(w0_(m,IPR,k,j,i),w0_(m,IDN,k,j,i));
	} else {
	  cs = eos.iso_cs;
	}
	min_dt1 = fmin((mbsize.dx1.d_view(m)/(fabs(w0_(m,IVX,k,j,i)) + cs)), min_dt1);
	min_dt2 = fmin((mbsize.dx2.d_view(m)/(fabs(w0_(m,IVY,k,j,i)) + cs)), min_dt2);
	min_dt3 = fmin((mbsize.dx3.d_view(m)/(fabs(w0_(m,IVZ,k,j,i)) + cs)), min_dt3);
      }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));


    }else{ // end non-relativistic

      // find largest (v +/- C) in each dirn for hydrodynamic problems
      Kokkos::parallel_reduce("RelHydroNudt2",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
	KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3)
      { 
	// compute m,k,j,i indices of thread and call function
	int m = (idx)/nkji;
	int k = (idx - m*nkji)/nji;
	int j = (idx - m*nkji - k*nji)/nx1;
	int i = (idx - m*nkji - k*nji - j*nx1) + is;
	k += ks;
	j += js;

	Real u2 = SQR(w0_(m,IVX,k,j,i)) + SQR(w0_(m,IVY,k,j,i)) + SQR(w0_(m,IVZ,k,j,i));
	
	Real u0  = sqrt(1. + u2);

	// FIXME ERM: Ideal fluid for now
	Real wgas = w0_(m,IDN,k,j,i) + (eos.gamma/(eos.gamma-1.)) * w0_(m,IPR,k,j,i);

	Real lm,lp;
	Real max_dv1 =0.;
	eos.SoundSpeed_SR(wgas, w0_(m,IPR,k,j,i), w0_(m,IVX,k,j,i)/u0, u0*u0, lp, lm);
	lm = fmax(-lm, 0.);
	lp = fmax( lp, lm);
	max_dv1 = fmax(lp, max_dv1);

	Real max_dv2 =0.;
	eos.SoundSpeed_SR(wgas, w0_(m,IPR,k,j,i), w0_(m,IVY,k,j,i)/u0, u0*u0, lp, lm);
	lm = fmax(-lm, 0.);
	lp = fmax( lp, lm);
	max_dv2 = fmax(lp, max_dv2);

	Real max_dv3 =0.;
	eos.SoundSpeed_SR(wgas, w0_(m,IPR,k,j,i), w0_(m,IVZ,k,j,i)/u0, u0*u0, lp, lm);
	lm = fmax(-lm, 0.);
	lp = fmax( lp, lm);
	max_dv3 = fmax(lp, max_dv3);

	min_dt1 = fmin((mbsize.dx1.d_view(m)/max_dv1), min_dt1);
	min_dt2 = fmin((mbsize.dx2.d_view(m)/max_dv2), min_dt2);
	min_dt3 = fmin((mbsize.dx3.d_view(m)/max_dv3), min_dt3);
      }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
 
   }

  }

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  dtnew = dt1;
  if (pmy_pack->pmesh->nx2gt1) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->nx3gt1) { dtnew = std::min(dtnew, dt3); }

  return TaskStatus::complete;
}
} // namespace hydro
