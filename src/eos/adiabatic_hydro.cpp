//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements EOS functions in derived class for nonrelativistic adiabatic hydro

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief Converts conserved into primitive variables in nonrelativistic adiabatic hydro

void EquationOfState::HydroConToPrimAdi(AthenaArray4D<Real> &cons,AthenaArray4D<Real> &prim)
{
  MeshBlock* pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  int &nhydro = pmb->phydro->nhydro;
  int &nscalars = pmb->phydro->nscalars;
  Real gm1 = eos_data.gamma - 1.0;

  Real &dfloor_ = eos_data.density_floor;
  Real &pfloor_ = eos_data.pressure_floor;

  par_for("hydro_update", pmb->exe_space, 0, (ncells3-1), 0, (ncells2-1), 0, (ncells1-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real& u_d  = cons(hydro::IDN,k,j,i);
      Real& u_m1 = cons(hydro::IM1,k,j,i);
      Real& u_m2 = cons(hydro::IM2,k,j,i);
      Real& u_m3 = cons(hydro::IM3,k,j,i);
      Real& u_e  = cons(hydro::IEN,k,j,i);

      Real& w_d  = prim(hydro::IDN,k,j,i);
      Real& w_vx = prim(hydro::IVX,k,j,i);
      Real& w_vy = prim(hydro::IVY,k,j,i);
      Real& w_vz = prim(hydro::IVZ,k,j,i);
      Real& w_p  = prim(hydro::IPR,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > dfloor_) ?  u_d : dfloor_;
      w_d = u_d;

      Real di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      Real e_k = 0.5*di*(u_m1*u_m1 + u_m2*u_m2 + u_m3*u_m3);
      w_p = gm1*(u_e - e_k);

      // apply pressure floor, correct total energy
      u_e = (w_p > pfloor_) ?  u_e : ((pfloor_/gm1) + e_k);
      w_p = (w_p > pfloor_) ?  w_p : pfloor_;

      // convert scalars (if any)
      for (int n=nhydro; n<(nhydro+nscalars); ++n) {
        prim(n,k,j,i) = cons(n,k,j,i)/u_d;
      }
    }
  );

  return;
}
