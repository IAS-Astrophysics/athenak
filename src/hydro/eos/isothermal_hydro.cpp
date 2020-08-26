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
  : EquationOfState(pm, pin, igid)
{
  iso_cs_ = pin->GetReal("eos", "iso_sound_speed");
}

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief Converts conserved into primitive variables in nonrelativistic isothermal hydro

void IsothermalHydro::ConservedToPrimitive(const int k, const int j, const int il,
    const int iu, AthenaArray<Real> &cons, AthenaArray<Real> &prim)
{
  for (int i=il; i<=iu; ++i) {
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

  return;
}

} // namespace hydro
