//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_hyd.cpp
//  \brief derived class that implements ideal gas EOS in nonrelativistic hydro

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealHydro::IdealHydro(MeshBlockPack *pp, ParameterInput *pin) :
  EquationOfState(pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;

  // Read flags specifying which variable to use in primitives
  // if nothing set in input file, use e as default
  if (!(pin->DoesParameterExist("hydro","use_e")) &&
      !(pin->DoesParameterExist("hydro","use_t")) ) {
    eos_data.use_e = true;
    eos_data.use_t = false;
  } else {
    eos_data.use_e = pin->GetOrAddBoolean("hydro","use_e",false);
    eos_data.use_t = pin->GetOrAddBoolean("hydro","use_t",false);
  }
  if (!(eos_data.use_e) && !(eos_data.use_t)) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Both use_e and use_t set to false" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eos_data.use_e && eos_data.use_t) {
    std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Both use_e and use_t set to true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief Converts conserved into primitive variables. Operates over entire MeshBlock,
//  including ghost cells.

void IdealHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = (eos_data.gamma - 1.0);
  Real igm1 = 1.0/(gm1);

  Real &dfloor_ = eos_data.density_floor;
  Real &pfloor_ = eos_data.pressure_floor;
  Real &tfloor_ = eos_data.temperature_floor;
  bool &use_e = eos_data.use_e;

  par_for("hyd_con2prim", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real& u_d  = cons(m,IDN,k,j,i);
    Real& u_e  = cons(m,IEN,k,j,i);
    const Real& u_m1 = cons(m,IM1,k,j,i);
    const Real& u_m2 = cons(m,IM2,k,j,i);
    const Real& u_m3 = cons(m,IM3,k,j,i);

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

    Real e_k = 0.5*di*(u_m1*u_m1 + u_m2*u_m2 + u_m3*u_m3);
    if (use_e) {  // internal energy density is primitive
      Real& w_e  = prim(m,IEN,k,j,i);
      w_e = (u_e - e_k);
      // apply pressure floor, correct total energy
      u_e = (w_e > (pfloor_*igm1)) ?  u_e : ((pfloor_*igm1) + e_k);
      w_e = (w_e > (pfloor_*igm1)) ?  w_e : (pfloor_*igm1);
    } else {  // temperature is primitive
      Real& w_t  = prim(m,ITM,k,j,i);
      w_t = (gm1/u_d)*(u_e - e_k);
      // apply temperature floor, correct total energy
      u_e = (w_t > tfloor_) ?  u_e : ((u_d*igm1*tfloor_) + e_k);
      w_t = (w_t > tfloor_) ?  w_t : tfloor_;
    }

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void PrimToCons()
// \brief Converts conserved into primitive variables. Operates over only active cells.

void IdealHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  bool &use_e = eos_data.use_e;
  Real igm1 = 1.0/(eos_data.gamma - 1.0);

  par_for("hyd_prim2con", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real& u_d  = cons(m,IDN,k,j,i);
    Real& u_e  = cons(m,IEN,k,j,i);
    Real& u_m1 = cons(m,IM1,k,j,i);
    Real& u_m2 = cons(m,IM2,k,j,i);
    Real& u_m3 = cons(m,IM3,k,j,i);

    const Real& w_d  = prim(m,IDN,k,j,i);
    const Real& w_vx = prim(m,IVX,k,j,i);
    const Real& w_vy = prim(m,IVY,k,j,i);
    const Real& w_vz = prim(m,IVZ,k,j,i);

    u_d  = w_d;
    u_m1 = w_vx*w_d;
    u_m2 = w_vy*w_d;
    u_m3 = w_vz*w_d;
    if (use_e) {  // internal energy density is primitive
      const Real& w_e  = prim(m,IEN,k,j,i);
      u_e = w_e + 0.5*w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
    } else {  // temperature is primitive
      const Real& w_t  = prim(m,ITM,k,j,i);
      u_e = w_t*u_d*igm1 + 0.5*w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
    }

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      cons(m,n,k,j,i) = prim(m,n,k,j,i)*w_d;
    }
  });

  return;
}
