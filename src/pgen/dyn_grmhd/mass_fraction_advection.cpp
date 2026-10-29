//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mass_fraction_advection.cpp
//! \brief Uniform rho, P and v^x in a periodic box with an independent random composition
//! of the EOS mass-fraction block in every cell (all scalars if the EOS declares none).
//! Neighbouring fractions are not affinely related, so limited reconstructions of the
//! species do not sum to 1 at the faces. The test checks that sum_s DY_s = D over the
//! block is kept by the fluxes alone (Plewa & Mueller 1999, A&A 342, 179). The other
//! scalars are uniform, from <problem> s<n> (default 0). tie_from/tie_to copy one random
//! fraction onto another before normalizing (e.g. Xp = Xn keeps rhine's heavy-nucleus
//! charge at Ah/2 for Ye = 0.5).
//! Build with -DPROBLEM=dyn_grmhd/mass_fraction_advection.

#include <cstdint>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "pgen/pgen.hpp"

namespace {
// splitmix64 hash of the global cell index -> uniform [0,1), decomposition independent
KOKKOS_INLINE_FUNCTION Real Uniform01(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30))*0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27))*0x94d049bb133111ebULL;
  x ^= (x >> 31);
  return static_cast<Real>(x >> 11)*(1.0/9007199254740992.0);
}
constexpr int kMaxScal = MAX_SPECIES;
struct ScalarConst { Real v[kMaxScal]; };  // uniform scalars, captured by value
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const int nscal = (pmbp->pmhd != nullptr) ? pmbp->pmhd->nscalars : 0;
  if (pmbp->pdyngr == nullptr || nscal < 1 || nscal > kMaxScal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "needs dyn_grmhd with 1.." << kMaxScal << " scalars"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  auto &msize = pmy_mesh_->mesh_size;
  const Real x1min = msize.x1min, x2min = msize.x2min, x3min = msize.x3min;
  const std::uint64_t ng1 = pmy_mesh_->mesh_indcs.nx1, ng2 = pmy_mesh_->mesh_indcs.nx2;
  const Real rho = pin->GetReal("problem", "rho");
  const Real P   = pin->GetReal("problem", "press");
  const Real v   = pin->GetReal("problem", "vx");
  const Real xfl = pin->GetOrAddReal("problem", "xfloor", 0.05);  // keeps X_s > 0
  const std::uint64_t seed = pin->GetOrAddInteger("problem", "seed", 1);
  const Real W = 1.0/sqrt(1.0 - v*v);
  const int imf = (pmbp->pdyngr->nmfrac > 0) ? pmbp->pdyngr->imfrac : 0;
  const int nmf = (pmbp->pdyngr->nmfrac > 0) ? pmbp->pdyngr->nmfrac : nscal;
  const int tie_from = pin->GetOrAddInteger("problem", "tie_from", -1);
  const int tie_to   = pin->GetOrAddInteger("problem", "tie_to", -1);
  const bool tie_off = (tie_from < 0 && tie_to < 0);
  const bool tie_ok = (tie_from >= imf && tie_from < imf + nmf &&
                       tie_to >= imf && tie_to < imf + nmf);
  if (!tie_off && !tie_ok) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "tie_from/tie_to must lie in [" << imf << ","
              << imf + nmf << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  ScalarConst sc;
  for (int s = 0; s < nscal; ++s) {
    sc.v[s] = pin->GetOrAddReal("problem", "s" + std::to_string(s), 0.0);
  }
  auto &w0 = pmbp->pmhd->w0;
  auto &b0 = pmbp->pmhd->b0;
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_mfadv", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // global cell index, so every cell gets its own composition in 1D/2D/3D
    Real x1 = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2 = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real x3 = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    std::uint64_t g1 = static_cast<std::uint64_t>((x1 - x1min)/size.d_view(m).dx1);
    std::uint64_t g2 = static_cast<std::uint64_t>((x2 - x2min)/size.d_view(m).dx2);
    std::uint64_t g3 = static_cast<std::uint64_t>((x3 - x3min)/size.d_view(m).dx3);
    std::uint64_t g = g1 + ng1*(g2 + ng2*g3);
    w0(m,IDN,k,j,i) = rho;
    w0(m,IVX,k,j,i) = W*v;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = P;
    Real r[kMaxScal], sum = 0.0;
    for (int s = imf; s < imf + nmf; ++s) {
      r[s] = xfl + Uniform01(seed*1000003ULL + g*kMaxScal + s);
    }
    if (!tie_off) { r[tie_to] = r[tie_from]; }
    for (int s = imf; s < imf + nmf; ++s) { sum += r[s]; }
    for (int s = 0; s < nscal; ++s) {
      w0(m,IYF+s,k,j,i) = (s >= imf && s < imf + nmf) ? r[s]/sum : sc.v[s];
    }
    b0.x1f(m,k,j,i) = 0.0; b0.x2f(m,k,j,i) = 0.0; b0.x3f(m,k,j,i) = 0.0;
    if (i == ie) b0.x1f(m,k,j,i+1) = 0.0;
    if (j == je) b0.x2f(m,k,j+1,i) = 0.0;
    if (k == ke) b0.x3f(m,k+1,j,i) = 0.0;
    bcc0(m,IBX,k,j,i) = 0.0; bcc0(m,IBY,k,j,i) = 0.0; bcc0(m,IBZ,k,j,i) = 0.0;
  });
  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
}
