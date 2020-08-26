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
  : EquationOfState(pm, pin, igid)
{
  gamma_ = pin->GetReal("eos", "gamma");
}

//----------------------------------------------------------------------------------------
// \!fn void ConservedToPrimitive()
// \brief Converts conserved into primitive variables in nonrelativistic adiabatic hydro

void AdiabaticHydro::ConservedToPrimitive(const int k, const int j, const int il,
    const int iu, AthenaArray<Real> &cons, AthenaArray<Real> &prim)
{
  Real gm1 = GetGamma() - 1.0;

  for (int i=il; i<=iu; ++i) {
    Real& u_d  = cons(IDN,k,j,i);
    Real& u_m1 = cons(IM1,k,j,i);
    Real& u_m2 = cons(IM2,k,j,i);
    Real& u_m3 = cons(IM3,k,j,i);
    Real& u_e  = cons(IEN,k,j,i);

    Real& w_d  = prim(IDN,i);
    Real& w_vx = prim(IVX,i);
    Real& w_vy = prim(IVY,i);
    Real& w_vz = prim(IVZ,i);
    Real& w_p  = prim(IPR,i);

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

  return;
}

} // namespace hydro
