//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_output.cpp
//! \brief Declarative catalog of Hydro output variables.
//!
//! Each entry maps an input-file "variable=" choice to the device-array components it
//! emits. Component indices use the VariableIndex enums from athena.hpp. The energy
//! component is tagged FieldRule::IfIdealEos (emitted only for an ideal EOS). Passive
//! scalars, the eint->press rename, and dynamical-GR temperature are added by post-steps
//! in output_vars.cpp (they are dynamic / conditional), not listed here. See outputs.hpp
//! for the FieldSpec/GroupSpec/VarSpec definitions and the table format.

#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "hydro/hydro.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
//! \fn HydroOutputVars
//! \brief return the Hydro output catalog with live device-array pointers filled in.
//! Only called when phydro != nullptr (guaranteed by the registry in output_vars.cpp).

std::vector<VarSpec> HydroOutputVars(MeshBlockPack *p) {
  DvceArray5D<Real> *u0 = &(p->phydro->u0);   // conserved
  DvceArray5D<Real> *w0 = &(p->phydro->w0);   // primitive
  return {
    // --- conserved (u0) ---
    {"hydro_u_d",  {{u0, {{"dens", IDN}}}}},
    {"hydro_u_m1", {{u0, {{"mom1", IM1}}}}},
    {"hydro_u_m2", {{u0, {{"mom2", IM2}}}}},
    {"hydro_u_m3", {{u0, {{"mom3", IM3}}}}},
    {"hydro_u_e",  {{u0, {{"ener", IEN, FieldRule::IfIdealEos}}}}},
    {"hydro_u",    {{u0, {{"dens", IDN}, {"mom1", IM1}, {"mom2", IM2}, {"mom3", IM3},
                         {"ener", IEN, FieldRule::IfIdealEos}}}}},  // + r_NN scalars

    // --- primitive (w0); "eint" may be renamed to "press" by ApplyDynGR ---
    {"hydro_w_d",  {{w0, {{"dens", IDN}}}}},
    {"hydro_w_vx", {{w0, {{"velx", IVX}}}}},
    {"hydro_w_vy", {{w0, {{"vely", IVY}}}}},
    {"hydro_w_vz", {{w0, {{"velz", IVZ}}}}},
    {"hydro_w_e",  {{w0, {{"eint", IEN, FieldRule::IfIdealEos}}}}},
    {"hydro_w",    {{w0, {{"dens", IDN}, {"velx", IVX}, {"vely", IVY}, {"velz", IVZ},
                         {"eint", IEN, FieldRule::IfIdealEos}}}}},  // + s_NN scalars

    // --- passive scalars only (fields come entirely from AppendScalars) ---
    {"hydro_u_s", {}},
    {"hydro_w_s", {}},
  };
}
