//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file isothermal_hydro.cpp
//  \brief implements EOS functions in derived class for nonrelativistic isothermal hydro

// Athena++ headers
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"

namespace hydro {

//----------------------------------------------------------------------------------------
// IsothermalHydro constructor
    
IsothermalHydro::IsothermalHydro(Mesh* pm, ParameterInput *pin, int igid)
  : EquationOfState(pm, pin, igid, "isothermal")
{
  adiabatic_eos = false;
  iso_cs_ = pin->GetReal("eos", "iso_sound_speed");
}

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief Converts conserved into primitive variables in nonrelativistic isothermal hydro

void IsothermalHydro::ConservedToPrimitive(AthenaArray<Real> &cons, AthenaArray<Real> &prim)
{
  MeshBlock* pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ng = pmb->mb_cells.ng;
  int ncells1 = pmb->mb_cells.nx1 + 2*ng;
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*ng) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*ng) : 1;

  for (int k=0; k<ncells3; ++k) {
    for (int j=0; j<ncells2; ++j) {
      for (int i=0; i<ncells1; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);

        Real& w_d  = prim(IDN,i);
        Real& w_vx = prim(IVX,i);
        Real& w_vy = prim(IVY,i);
        Real& w_vz = prim(IVZ,i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
        w_d = u_d;

        Real di = 1.0/u_d;
        w_vx = u_m1*di;
        w_vy = u_m2*di;
        w_vz = u_m3*di;
      }
    }
  }

  return;
}

} // namespace hydro
