//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file divb_amr.cpp
//! \brief High-sensitivity AMR test for preservation of the face-centered div(B)=0
//! constraint in 1D/2D/3D.

// C/C++ headers
#include <algorithm>  // min, max
#include <cmath>      // fabs(), sin(), cos()
#include <cstdlib>    // exit()
#include <iostream>   // endl

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

void DivBAMRHistory(HistoryData *pdata, Mesh *pm);
void DivBAMRRefinementCondition(MeshBlockPack *pmbp);

namespace {

struct DivBAMRConfig {
  Real rho0 = 1.0;
  Real pgas0 = 10.0;
  Real vx0 = 0.08;
  Real vy0 = 0.05;
  Real vz0 = 0.03;
  Real guide_b1 = 0.7;
  Real guide_b2 = 0.2;
  Real guide_b3 = -0.15;
  Real field_amp = 0.25;
  Real field_k = 2.0;
  Real divb_bnorm = 1.0;
  Real x1min = 0.0;
  Real x1max = 1.0;
  Real x2min = 0.0;
  Real x2max = 1.0;
  Real x3min = 0.0;
  Real x3max = 1.0;
  int target_level = 0;
};

DivBAMRConfig divb_amr;

KOKKOS_INLINE_FUNCTION
Real Phase(const Real x, const Real xmin, const Real xmax, const Real k) {
  const Real width = xmax - xmin;
  if (width <= 0.0) return 0.0;
  return 2.0*M_PI*k*(x - xmin)/width;
}

KOKKOS_INLINE_FUNCTION
Real A1(const Real x1, const Real x2, const Real x3, const DivBAMRConfig cfg) {
  const Real x = Phase(x1, cfg.x1min, cfg.x1max, cfg.field_k);
  const Real y = Phase(x2, cfg.x2min, cfg.x2max, cfg.field_k);
  const Real z = Phase(x3, cfg.x3min, cfg.x3max, cfg.field_k);
  const Real scale = cfg.field_amp/(2.0*M_PI*cfg.field_k);
  return scale*(0.50*std::sin(y + 2.0*z) + 0.33*std::cos(2.0*x - z)
              + 0.21*std::sin(3.0*y - x + z));
}

KOKKOS_INLINE_FUNCTION
Real A2(const Real x1, const Real x2, const Real x3, const DivBAMRConfig cfg) {
  const Real x = Phase(x1, cfg.x1min, cfg.x1max, cfg.field_k);
  const Real y = Phase(x2, cfg.x2min, cfg.x2max, cfg.field_k);
  const Real z = Phase(x3, cfg.x3min, cfg.x3max, cfg.field_k);
  const Real scale = cfg.field_amp/(2.0*M_PI*cfg.field_k);
  return scale*(0.47*std::sin(z + 2.0*x) + 0.29*std::cos(2.0*y - x)
              + 0.19*std::sin(3.0*z - y + x));
}

KOKKOS_INLINE_FUNCTION
Real A3(const Real x1, const Real x2, const Real x3, const DivBAMRConfig cfg) {
  const Real x = Phase(x1, cfg.x1min, cfg.x1max, cfg.field_k);
  const Real y = Phase(x2, cfg.x2min, cfg.x2max, cfg.field_k);
  const Real z = Phase(x3, cfg.x3min, cfg.x3max, cfg.field_k);
  const Real scale = cfg.field_amp/(2.0*M_PI*cfg.field_k);
  return scale*(0.53*std::sin(x + 2.0*y) + 0.31*std::cos(2.0*z - y)
              + 0.23*std::sin(3.0*x - z + y));
}

KOKKOS_INLINE_FUNCTION
Real WrapUnit(Real x) {
  while (x < 0.0) x += 1.0;
  while (x >= 1.0) x -= 1.0;
  return x;
}

KOKKOS_INLINE_FUNCTION
Real PeriodicDistance(const Real a, const Real b) {
  const Real d = fabs(a - b);
  return fmin(d, 1.0 - d);
}

KOKKOS_INLINE_FUNCTION
Real CenterUnit(const Real xmin, const Real xmax, const Real mesh_min,
                const Real mesh_max) {
  const Real width = mesh_max - mesh_min;
  if (width <= 0.0) return 0.0;
  return WrapUnit((0.5*(xmin + xmax) - mesh_min)/width);
}

KOKKOS_INLINE_FUNCTION
Real HalfWidthUnit(const Real xmin, const Real xmax, const Real mesh_min,
                   const Real mesh_max) {
  const Real width = mesh_max - mesh_min;
  if (width <= 0.0) return 0.5;
  return 0.5*(xmax - xmin)/width;
}

KOKKOS_INLINE_FUNCTION
bool OverlapsPeriodicBox(const RegionSize &sz, const RegionSize &mesh_size,
                         const Real c1, const Real c2, const Real c3,
                         const Real h1, const Real h2, const Real h3,
                         const bool multi_d, const bool three_d) {
  const Real x1c = CenterUnit(sz.x1min, sz.x1max, mesh_size.x1min, mesh_size.x1max);
  const Real x1h = HalfWidthUnit(sz.x1min, sz.x1max, mesh_size.x1min, mesh_size.x1max);
  bool overlaps = PeriodicDistance(x1c, c1) <= (x1h + h1);

  if (multi_d) {
    const Real x2c = CenterUnit(sz.x2min, sz.x2max, mesh_size.x2min, mesh_size.x2max);
    const Real x2h = HalfWidthUnit(sz.x2min, sz.x2max, mesh_size.x2min, mesh_size.x2max);
    overlaps = overlaps && (PeriodicDistance(x2c, c2) <= (x2h + h2));
  }
  if (three_d) {
    const Real x3c = CenterUnit(sz.x3min, sz.x3max, mesh_size.x3min, mesh_size.x3max);
    const Real x3h = HalfWidthUnit(sz.x3min, sz.x3max, mesh_size.x3min, mesh_size.x3max);
    overlaps = overlaps && (PeriodicDistance(x3c, c3) <= (x3h + h3));
  }
  return overlaps;
}

KOKKOS_INLINE_FUNCTION
bool InRefinementPattern(const RegionSize &sz, const RegionSize &mesh_size,
                         const Real phase, const bool multi_d, const bool three_d) {
  const bool box1 = OverlapsPeriodicBox(
      sz, mesh_size, WrapUnit(0.19 + 0.113*phase), WrapUnit(0.31 + 0.071*phase),
      WrapUnit(0.47 + 0.053*phase), 0.135, 0.110, 0.090, multi_d, three_d);
  const bool box2 = OverlapsPeriodicBox(
      sz, mesh_size, WrapUnit(0.63 - 0.097*phase), WrapUnit(0.72 + 0.127*phase),
      WrapUnit(0.24 - 0.061*phase), 0.115, 0.155, 0.105, multi_d, three_d);
  const bool box3 = OverlapsPeriodicBox(
      sz, mesh_size, WrapUnit(0.82 + 0.049*phase), WrapUnit(0.18 - 0.109*phase),
      WrapUnit(0.76 + 0.083*phase), 0.075, 0.095, 0.145, multi_d, three_d);
  return box1 || box2 || box3;
}

} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::DivBAMR()
//! \brief Initialize a high-gradient, discretely divergence-free magnetic field and
//! moving AMR pattern for div(B) preservation tests.

void ProblemGenerator::DivBAMR(ParameterInput *pin, const bool restart) {
  user_ref_func = DivBAMRRefinementCondition;
  user_hist_func = DivBAMRHistory;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "divb_amr can only be run in MHD, but no <mhd> block is present"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  divb_amr.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  divb_amr.pgas0 = pin->GetOrAddReal("problem", "pgas0", 10.0);
  divb_amr.vx0 = pin->GetOrAddReal("problem", "vx0", 0.08);
  divb_amr.vy0 = pin->GetOrAddReal("problem", "vy0", 0.05);
  divb_amr.vz0 = pin->GetOrAddReal("problem", "vz0", 0.03);
  divb_amr.guide_b1 = pin->GetOrAddReal("problem", "guide_b1", 0.7);
  divb_amr.guide_b2 = pin->GetOrAddReal("problem", "guide_b2", 0.2);
  divb_amr.guide_b3 = pin->GetOrAddReal("problem", "guide_b3", -0.15);
  divb_amr.field_amp = pin->GetOrAddReal("problem", "field_amp", 0.25);
  divb_amr.field_k = pin->GetOrAddReal("problem", "field_k", 2.0);
  divb_amr.x1min = pmy_mesh_->mesh_size.x1min;
  divb_amr.x1max = pmy_mesh_->mesh_size.x1max;
  divb_amr.x2min = pmy_mesh_->mesh_size.x2min;
  divb_amr.x2max = pmy_mesh_->mesh_size.x2max;
  divb_amr.x3min = pmy_mesh_->mesh_size.x3min;
  divb_amr.x3max = pmy_mesh_->mesh_size.x3max;
  const int refine_levels = pin->GetOrAddInteger("problem", "refine_levels", 2);
  divb_amr.target_level = std::min(pmy_mesh_->root_level + refine_levels,
                                   pmy_mesh_->max_level);
  const Real default_bnorm = std::max(static_cast<Real>(1.0),
      std::abs(divb_amr.guide_b1) + std::abs(divb_amr.guide_b2)
    + std::abs(divb_amr.guide_b3) + 6.0*std::abs(divb_amr.field_amp));
  divb_amr.divb_bnorm = pin->GetOrAddReal("problem", "divb_bnorm", default_bnorm);

  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  const bool multi_d = pmy_mesh_->multi_d;
  const bool three_d = pmy_mesh_->three_d;

  int nmb = pmbp->nmb_thispack;
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  const Real gm1 = eos.gamma - 1.0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  auto &size = pmbp->pmb->mb_size;
  const auto cfg = divb_amr;

  int ncells1 = nx1 + 2*indcs.ng;
  int ncells2 = (nx2 > 1) ? (nx2 + 2*indcs.ng) : 2;
  int ncells3 = (nx3 > 1) ? (nx3 + 2*indcs.ng) : 2;
  DvceArray4D<Real> a1("divb-amr-a1", nmb, ncells3, ncells2, ncells1);
  DvceArray4D<Real> a2("divb-amr-a2", nmb, ncells3, ncells2, ncells1);
  DvceArray4D<Real> a3("divb-amr-a3", nmb, ncells3, ncells2, ncells1);

  par_for("divb_amr_vector_potential", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je+1,
  is, ie+1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1min = size.d_view(m).x1min;
    const Real x1max = size.d_view(m).x1max;
    const Real x2min = size.d_view(m).x2min;
    const Real x2max = size.d_view(m).x2max;
    const Real x3min = size.d_view(m).x3min;
    const Real x3max = size.d_view(m).x3max;

    const Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
    const Real x1f = LeftEdgeX(i - is, nx1, x1min, x1max);
    const Real x2v = CellCenterX(j - js, nx2, x2min, x2max);
    const Real x2f = LeftEdgeX(j - js, nx2, x2min, x2max);
    const Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);
    const Real x3f = LeftEdgeX(k - ks, nx3, x3min, x3max);

    a1(m,k,j,i) = A1(x1v, x2f, x3f, cfg);
    a2(m,k,j,i) = A2(x1f, x2v, x3f, cfg);
    a3(m,k,j,i) = A3(x1f, x2f, x3v, cfg);
  });

  par_for("divb_amr_b1", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real b1 = cfg.guide_b1;
    if (multi_d) {
      b1 += (a3(m,k,j+1,i) - a3(m,k,j,i))/size.d_view(m).dx2;
    }
    if (three_d) {
      b1 -= (a2(m,k+1,j,i) - a2(m,k,j,i))/size.d_view(m).dx3;
    }
    b0.x1f(m,k,j,i) = b1;
  });

  par_for("divb_amr_b2", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real b2 = cfg.guide_b2 - (a3(m,k,j,i+1) - a3(m,k,j,i))/size.d_view(m).dx1;
    if (three_d) {
      b2 += (a1(m,k+1,j,i) - a1(m,k,j,i))/size.d_view(m).dx3;
    }
    b0.x2f(m,k,j,i) = b2;
  });

  par_for("divb_amr_b3", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real b3 = cfg.guide_b3 + (a2(m,k,j,i+1) - a2(m,k,j,i))/size.d_view(m).dx1;
    if (multi_d) {
      b3 -= (a1(m,k,j+1,i) - a1(m,k,j,i))/size.d_view(m).dx2;
    }
    b0.x3f(m,k,j,i) = b3;
  });

  par_for("divb_amr_cons", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real rho = cfg.rho0;
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho*cfg.vx0;
    u0(m,IM2,k,j,i) = rho*(multi_d ? cfg.vy0 : 0.0);
    u0(m,IM3,k,j,i) = rho*(three_d ? cfg.vz0 : 0.0);
    const Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    const Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    const Real bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    const Real v2 = SQR(cfg.vx0) + (multi_d ? SQR(cfg.vy0) : 0.0)
                  + (three_d ? SQR(cfg.vz0) : 0.0);
    u0(m,IEN,k,j,i) = cfg.pgas0/gm1 + 0.5*rho*v2
                    + 0.5*(SQR(bx) + SQR(by) + SQR(bz));
  });
}

//----------------------------------------------------------------------------------------
//! \fn void DivBAMRRefinementCondition()
//! \brief Moving periodic box pattern that creates many coarse/fine edges and corners.

void DivBAMRRefinementCondition(MeshBlockPack *pmbp) {
  Mesh *pmesh = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &mblev = pmbp->pmb->mb_lev;
  auto &mb_size = pmbp->pmb->mb_size;
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  const int root_level = pmesh->root_level;
  const bool multi_d = pmesh->multi_d;
  const bool three_d = pmesh->three_d;
  const RegionSize mesh_size = pmesh->mesh_size;
  const auto cfg = divb_amr;
  const Real phase = pmesh->time + static_cast<Real>(pmesh->ncycle);

  par_for("divb_amr_refinement", DevExeSpace(), 0, nmb-1, KOKKOS_LAMBDA(int m) {
    const bool refine_region = InRefinementPattern(
        mb_size.d_view(m), mesh_size, phase, multi_d, three_d);
    const int level = mblev.d_view(m);
    if (refine_region && (level < cfg.target_level)) {
      refine_flag.d_view(m + mbs) = 1;
    } else if ((!refine_region) && (level > root_level)) {
      refine_flag.d_view(m + mbs) = -1;
    }
  });

  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void DivBAMRHistory()
//! \brief Direct discrete divergence diagnostics over active cells.

void DivBAMRHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 8;
  pdata->label[0] = "max_divb";
  pdata->label[1] = "max_ndiv";
  pdata->label[2] = "sum_divb";
  pdata->label[3] = "sum_d2";
  pdata->label[4] = "sum_ndiv";
  pdata->label[5] = "sum_n2";
  pdata->label[6] = "vol";
  pdata->label[7] = "ncell";

  auto &b = pm->pmb_pack->pmhd->b0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const bool multi_d = pm->multi_d;
  const bool three_d = pm->three_d;
  const auto cfg = divb_amr;

  array_sum::GlobalSum sum_this_mb;
  Real max_abs_divb = 0.0;
  Real max_norm_divb = 0.0;
  Kokkos::parallel_reduce(
      "divb_amr_history", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_abs,
                Real &max_norm) {
    const int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    const int i = (idx - m*nkji - k*nji - j*nx1) + is;
    j += js;
    k += ks;

    Real divb = (b.x1f(m,k,j,i+1) - b.x1f(m,k,j,i))/size.d_view(m).dx1;
    Real dx_min = size.d_view(m).dx1;
    Real vol = size.d_view(m).dx1;
    if (multi_d) {
      divb += (b.x2f(m,k,j+1,i) - b.x2f(m,k,j,i))/size.d_view(m).dx2;
      dx_min = fmin(dx_min, size.d_view(m).dx2);
      vol *= size.d_view(m).dx2;
    }
    if (three_d) {
      divb += (b.x3f(m,k+1,j,i) - b.x3f(m,k,j,i))/size.d_view(m).dx3;
      dx_min = fmin(dx_min, size.d_view(m).dx3);
      vol *= size.d_view(m).dx3;
    }

    const Real abs_divb = fabs(divb);
    const Real norm_divb = abs_divb*dx_min/cfg.divb_bnorm;

    array_sum::GlobalSum hvars;
    hvars.the_array[2] = abs_divb*vol;
    hvars.the_array[3] = SQR(divb)*vol;
    hvars.the_array[4] = norm_divb*vol;
    hvars.the_array[5] = SQR(norm_divb)*vol;
    hvars.the_array[6] = vol;
    hvars.the_array[7] = 1.0;
    for (int n=8; n<NREDUCTION_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }
    max_abs = fmax(max_abs, abs_divb);
    max_norm = fmax(max_norm, norm_divb);
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb),
     Kokkos::Max<Real>(max_abs_divb), Kokkos::Max<Real>(max_norm_divb));

  pdata->hdata[0] = max_abs_divb;
  pdata->hdata[1] = max_norm_divb;
  for (int n=2; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }
}
