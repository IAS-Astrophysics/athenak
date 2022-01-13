//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.cpp
//  \brief implements constructor and some fns for EquationOfState abstract base class

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlockPack* pp, ParameterInput *pin) :
  pmy_pack(pp) {
  eos_data.density_floor = pin->GetOrAddReal("eos","density_floor",(FLT_MIN));
  eos_data.pressure_floor = pin->GetOrAddReal("eos","pressure_floor",(FLT_MIN));
  eos_data.temperature_floor = pin->GetOrAddReal("eos","temperature_floor",(FLT_MIN));
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief No-Op versions of hydro and MHD cons to prim functions.

void EquationOfState::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim) {
}

void EquationOfState::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                                 DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc) {
}

//----------------------------------------------------------------------------------------
// \!fn void PrimToCon()
// \brief No-Op versions of hydro and MHD prim to cons functions.

void EquationOfState::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons) {
}
void EquationOfState::PrimToCons(const DvceArray5D<Real> &prim,
                                 const DvceArray5D<Real> &bcc, DvceArray5D<Real> &cons) {
}
