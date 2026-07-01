//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_output.cpp
//! \brief Declarative catalog of MHD output variables. See hydro_output.cpp and
//! outputs.hpp for the table format. Cell-centered magnetic field components use the
//! BFieldIndex enums (IBX, IBY, IBZ). Passive scalars, the eint->press rename, and the
//! dynamical-GR temperature field (mhd_w, mhd_w_bcc, mhd_t) are added by post-steps in
//! output_vars.cpp, not listed here.

#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
//! \fn MhdOutputVars
//! \brief return the MHD output catalog with live device-array pointers filled in.
//! Only called when pmhd != nullptr (guaranteed by the registry in output_vars.cpp).

std::vector<VarSpec> MhdOutputVars(MeshBlockPack *p) {
  DvceArray5D<Real> *u0   = &(p->pmhd->u0);     // conserved
  DvceArray5D<Real> *w0   = &(p->pmhd->w0);     // primitive
  DvceArray5D<Real> *bcc0 = &(p->pmhd->bcc0);   // cell-centered magnetic field
  return {
    // --- conserved (u0) ---
    {"mhd_u_d",  {{u0, {{"dens", IDN}}}}},
    {"mhd_u_m1", {{u0, {{"mom1", IM1}}}}},
    {"mhd_u_m2", {{u0, {{"mom2", IM2}}}}},
    {"mhd_u_m3", {{u0, {{"mom3", IM3}}}}},
    {"mhd_u_e",  {{u0, {{"ener", IEN, FieldRule::IfIdealEos}}}}},
    {"mhd_u",    {{u0, {{"dens", IDN}, {"mom1", IM1}, {"mom2", IM2}, {"mom3", IM3},
                       {"ener", IEN, FieldRule::IfIdealEos}}}}},  // + r_NN scalars

    // --- primitive (w0); "eint" may be renamed to "press" by ApplyDynGR ---
    {"mhd_w_d",  {{w0, {{"dens", IDN}}}}},
    {"mhd_w_vx", {{w0, {{"velx", IVX}}}}},
    {"mhd_w_vy", {{w0, {{"vely", IVY}}}}},
    {"mhd_w_vz", {{w0, {{"velz", IVZ}}}}},
    {"mhd_w_e",  {{w0, {{"eint", IEN, FieldRule::IfIdealEos}}}}},
    {"mhd_w",    {{w0, {{"dens", IDN}, {"velx", IVX}, {"vely", IVY}, {"velz", IVZ},
                       {"eint", IEN, FieldRule::IfIdealEos}}}}},  // + s_NN scalars, + temp

    // --- passive scalars only ---
    {"mhd_u_s", {}},
    {"mhd_w_s", {}},

    // --- cell-centered magnetic field (bcc0) ---
    {"mhd_bcc1", {{bcc0, {{"bcc1", IBX}}}}},
    {"mhd_bcc2", {{bcc0, {{"bcc2", IBY}}}}},
    {"mhd_bcc3", {{bcc0, {{"bcc3", IBZ}}}}},
    {"mhd_bcc",  {{bcc0, {{"bcc1", IBX}, {"bcc2", IBY}, {"bcc3", IBZ}}}}},

    // --- combined conserved + B ---
    {"mhd_u_bcc", {{u0, {{"dens", IDN}, {"mom1", IM1}, {"mom2", IM2}, {"mom3", IM3},
                         {"ener", IEN, FieldRule::IfIdealEos}}},
                   {bcc0, {{"bcc1", IBX}, {"bcc2", IBY}, {"bcc3", IBZ}}}}},  // + r_NN

    // --- combined primitive + B (+ temperature under dyngr) ---
    {"mhd_w_bcc", {{w0, {{"dens", IDN}, {"velx", IVX}, {"vely", IVY}, {"velz", IVZ},
                         {"eint", IEN, FieldRule::IfIdealEos}}},
                   {bcc0, {{"bcc1", IBX}, {"bcc2", IBY}, {"bcc3", IBZ}}}}},  // + s_NN, temp

    // --- dynamical-GR temperature only (temperature added by ApplyDynGR; needs <dyngr>) ---
    {"mhd_t", {}},
  };
}
