//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file isothermal_hydro.cpp
//  \brief defines derived class that implements EOS functions for nonrelativistic
//   isothermal hydro

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IsothermalHydro::IsothermalHydro(MeshBlockPack *pp, ParameterInput *pin)
  : EquationOfState(pp, pin)
{
    eos_data.is_adiabatic = false;
    eos_data.iso_cs = pin->GetReal("eos","iso_sound_speed");
    eos_data.gamma = 0.0;
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief Converts conserved into primitive variables in nonrelativistic isothermal hydro

void IsothermalHydro::ConsToPrim(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim)
{
  auto ncells = pmy_pack->mb_cells;
  int ng = ncells.ng;
  int n1 = ncells.nx1 + 2*ng;
  int n2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*ng) : 1;
  int n3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*ng) : 1;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;

  Real &dfloor_ = eos_data.density_floor;

  par_for("hyd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real& u_d  = cons(m,IDN,k,j,i);
      Real& u_m1 = cons(m,IM1,k,j,i);
      Real& u_m2 = cons(m,IM2,k,j,i);
      Real& u_m3 = cons(m,IM3,k,j,i);

      Real& w_d  = prim(m,IDN,k,j,i);
      Real& w_vx = prim(m,IVX,k,j,i);
      Real& w_vy = prim(m,IVY,k,j,i);
      Real& w_vz = prim(m,IVZ,k,j,i);

      // apply density floor, without changing momentum or energy
      u_d = (u_d > dfloor_) ?  u_d : dfloor_;
      w_d = u_d;

      Real di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
      }
    }
  );

  return;
}
