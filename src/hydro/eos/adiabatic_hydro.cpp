//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements EOS functions in derived class for nonrelativistic adiabatic hydro

#include <iostream>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// AdiabaticHydro constructor
    
AdiabaticHydro::AdiabaticHydro(Mesh* pm, ParameterInput *pin, int igid)
  : EquationOfState(pm, pin, igid, "adiabatic")
{
  adiabatic_eos = true;
  gamma_ = pin->GetReal("eos", "gamma");
}

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief Converts conserved into primitive variables in nonrelativistic adiabatic hydro

void AdiabaticHydro::ConservedToPrimitive(AthenaArray<Real> &cons, AthenaArray<Real> &prim)
{
  MeshBlock* pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;
  Real gm1 = GetGamma() - 1.0;

  for (int k=0; k<ncells3; ++k) {
    for (int j=0; j<ncells2; ++j) {
      for (int i=0; i<ncells1; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        Real& w_d  = prim(IDN,k,j,i);
        Real& w_vx = prim(IVX,k,j,i);
        Real& w_vy = prim(IVY,k,j,i);
        Real& w_vz = prim(IVZ,k,j,i);
        Real& w_p  = prim(IPR,k,j,i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
        w_d = u_d;

        Real di = 1.0/u_d;
        w_vx = u_m1*di;
        w_vy = u_m2*di;
        w_vz = u_m3*di;

        Real e_k = 0.5*di*(u_m1*u_m1 + u_m2*u_m2 + u_m3*u_m3);
        w_p = gm1*(u_e - e_k);

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1) + e_k);
        w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;
      }
    }
  }

  return;
}

} // namespace hydro
