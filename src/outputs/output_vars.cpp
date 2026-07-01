//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file output_vars.cpp
//! \brief Machinery for the declarative, per-module output-variable tables.
//!
//! Holds the module registry (AllModuleTables), the name lookup (FindVarSpec), the
//! VarSpec -> outvars expansion (BaseTypeOutput::BuildOutvars), and the post-steps that
//! handle the dynamic/conditional pieces kept out of the tables: passive scalars
//! (AppendScalars) and dynamical-GR adjustments (ApplyDynGR: eint->press + temperature).
//! The per-module catalogs themselves live in outputs/<module>_output.cpp.

#include <cstdio>    // snprintf
#include <string>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
//! \fn IsIdeal
//! \brief return whether the EOS gating a component's array is ideal. Raw pointers do not
//! carry module identity, so we identify the owning module by comparing against its arrays.
//! Arrays without an EOS gate (e.g. bcc0) never carry an IfIdealEos field, so the default
//! (true) is harmless for them.

static bool IsIdeal(DvceArray5D<Real> *a, MeshBlockPack *p) {
  if (p->phydro != nullptr && (a == &p->phydro->u0 || a == &p->phydro->w0)) {
    return p->phydro->peos->eos_data.is_ideal;
  }
  if (p->pmhd != nullptr && (a == &p->pmhd->u0 || a == &p->pmhd->w0)) {
    return p->pmhd->peos->eos_data.is_ideal;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn AppendScalarsImpl
//! \brief append passive-scalar components with the given prefix ("r_"/"s_") and count.

static void AppendScalarsImpl(std::vector<OutputVariableInfo> &outvars, int nvar,
                              int nscalars, DvceArray5D<Real> *arr, const char *prefix) {
  for (int n = nvar; n < nvar + nscalars; ++n) {
    char number[3];
    std::snprintf(number, sizeof(number), "%02d", (n - nvar) % 100);
    outvars.emplace_back(std::string(prefix) + number, n, arr);
  }
}

//----------------------------------------------------------------------------------------
//! \struct ScalarInfo
//! \brief which passive scalars (if any) a given output variable carries. Conserved-type
//! outputs get r_NN from u0; primitive-type outputs get s_NN from w0. Membership is keyed
//! on the variable name to mirror the historical behavior. The scalars are emitted right
//! after their owning (u0/w0) group in BuildOutvars, matching the legacy component order.

struct ScalarInfo {
  bool active = false;
  DvceArray5D<Real> *src = nullptr;  // array the scalars live in (u0 or w0)
  const char *prefix = "";           // "r_" for conserved, "s_" for primitive
  int nvar = 0;                      // index of first scalar in the array
  int nscalars = 0;                  // number of passive scalars
};

//! \fn ScalarsFor
//! \brief return the passive-scalar descriptor for a variable name (inactive if none).

static ScalarInfo ScalarsFor(const std::string &var, MeshBlockPack *p) {
  if (var == "hydro_u" || var == "hydro_u_s") {
    return {true, &p->phydro->u0, "r_", p->phydro->nhydro, p->phydro->nscalars};
  } else if (var == "hydro_w" || var == "hydro_w_s") {
    return {true, &p->phydro->w0, "s_", p->phydro->nhydro, p->phydro->nscalars};
  } else if (var == "mhd_u" || var == "mhd_u_s" || var == "mhd_u_bcc") {
    return {true, &p->pmhd->u0, "r_", p->pmhd->nmhd, p->pmhd->nscalars};
  } else if (var == "mhd_w" || var == "mhd_w_s" || var == "mhd_w_bcc") {
    return {true, &p->pmhd->w0, "s_", p->pmhd->nmhd, p->pmhd->nscalars};
  }
  return {};
}

//----------------------------------------------------------------------------------------
//! \fn ApplyDynGR
//! \brief dynamical-GR adjustments applied OUTSIDE the table: rename "eint" -> "press",
//! and append the temperature field for primitive-based MHD outputs.

static void ApplyDynGR(std::vector<OutputVariableInfo> &outvars,
                       const std::string &var, MeshBlockPack *p) {
  if (p->pdyngr == nullptr) { return; }
  for (auto &ov : outvars) {
    if (ov.label == "eint") { ov.label = "press"; }
  }
  if (var == "mhd_w" || var == "mhd_w_bcc" || var == "mhd_t") {
    outvars.emplace_back("temperature", 0, &(p->pdyngr->temperature));
  }
}

//----------------------------------------------------------------------------------------
//! \fn BaseTypeOutput::BuildOutvars
//! \brief expand one VarSpec's fixed fields into outvars, register any derived-quantity
//! kernels, then apply the passive-scalar and dynamical-GR post-steps. Array pointers are
//! already resolved in the table; this only applies FieldRule and derived-slot bookkeeping.

void BaseTypeOutput::BuildOutvars(const VarSpec &spec, Mesh *pm) {
  MeshBlockPack *p = pm->pmb_pack;
  ScalarInfo si = ScalarsFor(spec.name, p);
  bool scalars_done = false;
  int derived_count = 0;
  for (const auto &g : spec.groups) {
    DvceArray5D<Real> *arr = (g.array != nullptr) ? g.array : &derived_var;
    if (g.on_select != nullptr) { g.on_select(pm); }   // one-time setup (e.g. SetSaveWBcc)
    for (const auto &f : g.fields) {
      if (f.rule == FieldRule::IfIdealEos && !IsIdeal(g.array, p)) { continue; }
      int idx = (g.compute != nullptr) ? (derived_count + f.index) : f.index;
      outvars.emplace_back(f.label, idx, arr);
    }
    if (g.compute != nullptr) {
      derived_kernels.push_back({g.compute, derived_count});
      derived_count += static_cast<int>(g.fields.size());
    }
    // passive scalars immediately follow the conserved/primitive group they belong to,
    // BEFORE any later group (e.g. bcc), matching the legacy component ordering
    if (si.active && !scalars_done && g.array == si.src) {
      AppendScalarsImpl(outvars, si.nvar, si.nscalars, si.src, si.prefix);
      scalars_done = true;
    }
  }
  // variables with no explicit groups (e.g. mhd_u_s / mhd_w_s) get their scalars here
  if (si.active && !scalars_done) {
    AppendScalarsImpl(outvars, si.nvar, si.nscalars, si.src, si.prefix);
  }
  out_params.n_derived = derived_count;
  out_params.contains_derived = !derived_kernels.empty();

  // dynamical-GR eint->press rename + temperature field (kept out of the tables)
  ApplyDynGR(outvars, spec.name, p);
}

//----------------------------------------------------------------------------------------
// Registry of per-module output catalogs ("the lists of the modules being used").

static bool HydroActive(MeshBlockPack *p) { return p->phydro != nullptr; }
static bool MhdActive(MeshBlockPack *p)   { return p->pmhd   != nullptr; }

const std::vector<ModuleOutputTable> &AllModuleTables() {
  static const std::vector<ModuleOutputTable> tables = {
    {HydroDynamics,        HydroActive, HydroOutputVars},
    {MagnetoHydroDynamics, MhdActive,   MhdOutputVars},
    // future modules add a row here + a src/outputs/<module>_output.cpp
  };
  return tables;
}

//----------------------------------------------------------------------------------------
//! \fn FindVarSpec
//! \brief search `name` among the ACTIVE modules only (inactive modules are not built,
//! since building would dereference their null array pointers). Returns true and fills
//! `out` on success.

bool FindVarSpec(const std::string &name, MeshBlockPack *p, VarSpec &out) {
  for (const auto &m : AllModuleTables()) {
    if (!m.is_active(p)) { continue; }
    for (const auto &v : m.get_vars(p)) {
      if (name == v.name) { out = v; return true; }
    }
  }
  return false;
}
