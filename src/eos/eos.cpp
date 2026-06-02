//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.cpp
//! \brief implements constructor and some fns for EquationOfState abstract base class

#include <float.h>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
// EquationOfState constructor

EquationOfState::EquationOfState(std::string bk, MeshBlockPack* pp, ParameterInput *pin) :
    pmy_pack(pp) {
  eos_data.dfloor = pin->GetOrAddReal(bk,"dfloor",(FLT_MIN));
  eos_data.pfloor = pin->GetOrAddReal(bk,"pfloor",(FLT_MIN));
  eos_data.tfloor = pin->GetOrAddReal(bk,"tfloor",(FLT_MIN));
  eos_data.sfloor = pin->GetOrAddReal(bk,"sfloor",(FLT_MIN));
  eos_data.sfloor1 = pin->GetOrAddReal(bk,"sfloor1",eos_data.sfloor);
  eos_data.sfloor2 = pin->GetOrAddReal(bk,"sfloor2",eos_data.sfloor);
  eos_data.rho1 = pin->GetOrAddReal(bk,"rho1",eos_data.dfloor);
  eos_data.rho2 = pin->GetOrAddReal(bk,"rho2",2.0*eos_data.dfloor);

  if (!(eos_data.dfloor > 0.0) || !(eos_data.sfloor > 0.0) ||
      !(eos_data.sfloor1 > 0.0) || !(eos_data.sfloor2 > 0.0) ||
      !(eos_data.rho1 > 0.0) || !(eos_data.rho2 > 0.0) ||
      eos_data.rho1 == eos_data.rho2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<" << bk << "> EOS floors require dfloor, sfloor, "
              << "sfloor1, sfloor2, rho1, and rho2 > 0 with rho1 != rho2"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief No-Op versions of hydro and MHD conservative to primitive functions.
//! Required because each derived class overrides only one.

void EquationOfState::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                 const bool only_testfloors,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}

void EquationOfState::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                                 DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                                 const bool only_testfloors,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCon()
//! \brief No-Op versions of hydro and MHD primitive to conservative functions.
//! Required because each derived class overrides only one.

void EquationOfState::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}
void EquationOfState::PrimToCons(const DvceArray5D<Real> &prim,
                                 const DvceArray5D<Real> &bcc, DvceArray5D<Real> &cons,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}
