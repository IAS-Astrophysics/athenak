//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.cpp
//  \brief implements ctor and fns for EquationOfState abstract base class

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlockPack* pp, ParameterInput *pin)
   : pmy_pack(pp)
{
  eos_data.density_floor = pin->GetOrAddReal("eos","density_floor",(FLT_MIN));
  eos_data.pressure_floor = pin->GetOrAddReal("eos","pressure_floor",(FLT_MIN));
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief No-Op version of hydro cons to prim function. This version is never used in MHD,
// and is overwritten in Hydro

void EquationOfState::ConsToPrimHydro(const DvceArray5D<Real> &c, DvceArray5D<Real> &p)
{        
}

//----------------------------------------------------------------------------------------
// \!fn void ConsToPrim()
// \brief No-Op version of MHD cons to prim function. This version is never used in Hydro,
// and is overwritten in MHD.

void EquationOfState::ConsToPrimMHD(const DvceArray5D<Real> &c,
      const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &p, DvceArray5D<Real> &bcc)
{
}
