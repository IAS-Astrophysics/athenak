//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.cpp
//! \brief create multigrid solver for gravity

// C headers

// C++ headers
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iomanip>

// Athena++ headers
#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mhd/mhd.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "gravity.hpp"
#include "mg_gravity.hpp"
#include "../driver/driver.hpp"

class MeshBlockPack;

namespace {

PoissonCompositeMaskCounts ReducePoissonMaskCounts(
    const PoissonCompositeMaskCounts &local) {
  PoissonCompositeMaskCounts global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local.solve, &global.solve, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.resid, &global.resid, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.reset, &global.reset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.stencil, &global.stencil, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.covered, &global.covered, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  return global;
}

void PrintPoissonMaskCounts(const std::string &prefix,
                            const PoissonCompositeMaskCounts &counts) {
  if (global_variable::my_rank != 0) return;
  std::cout << prefix
            << " solve=" << counts.solve
            << " resid=" << counts.resid
            << " reset=" << counts.reset
            << " stencil=" << counts.stencil
            << " covered=" << counts.covered << std::endl;
}

PoissonCompositeMaskCounts SubtractPoissonMaskCounts(
    const PoissonCompositeMaskCounts &a, const PoissonCompositeMaskCounts &b) {
  PoissonCompositeMaskCounts c;
  c.solve = a.solve - b.solve;
  c.resid = a.resid - b.resid;
  c.reset = a.reset - b.reset;
  c.stencil = a.stencil - b.stencil;
  c.covered = a.covered - b.covered;
  return c;
}

struct PoissonResidualCategory {
  long long count = 0;
  Real sum2 = 0.0;
  Real maxabs = 0.0;
};

struct PoissonResidualSplit {
  PoissonResidualCategory solve;
  PoissonResidualCategory reset;
  PoissonResidualCategory stencil;
  PoissonResidualCategory covered;
  PoissonResidualCategory accepted;
};

PoissonResidualSplit ReducePoissonResidualSplit(const PoissonResidualSplit &local) {
  PoissonResidualSplit global = local;
  auto reduce_category = [](const PoissonResidualCategory &l,
                            PoissonResidualCategory &g) {
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&l.count, &g.count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&l.sum2, &g.sum2, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&l.maxabs, &g.maxabs, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  };
  reduce_category(local.solve, global.solve);
  reduce_category(local.reset, global.reset);
  reduce_category(local.stencil, global.stencil);
  reduce_category(local.covered, global.covered);
  reduce_category(local.accepted, global.accepted);
  return global;
}

void PrintPoissonResidualCategory(const char *name,
                                  const PoissonResidualCategory &category) {
  const Real rms = category.count > 0
      ? std::sqrt(category.sum2/static_cast<Real>(category.count)) : 0.0;
  std::cout << " " << name << "_count=" << category.count
            << " " << name << "_rms=" << rms
            << " " << name << "_max=" << category.maxabs;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::MGGravityDriver(Mesh *pm, ParameterInput *pin)
//! \brief MGGravityDriver constructor

MGGravityDriver::MGGravityDriver(MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, 1) {
    four_pi_G_ = pin->GetOrAddReal("gravity", "four_pi_G", -1.0);
    omega_ = pin->GetOrAddReal("gravity", "omega", 1.15);
    poisson_test_enabled_ =
        pin->GetOrAddBoolean("poisson_test", "enabled", false);
    poisson_test_composite_fas_ =
        pin->GetOrAddBoolean("poisson_test", "composite_fas", false);
    bool poisson_test_debug_masks =
        pin->GetOrAddBoolean("poisson_test", "debug_masks", false);
    poisson_test_debug_composite_masks_ =
        pin->GetOrAddBoolean("poisson_test", "debug_composite_masks",
                             poisson_test_debug_masks);
    poisson_test_debug_residual_split_ =
        pin->GetOrAddBoolean("poisson_test", "debug_residual_split", false);
    eps_ = pin->GetOrAddReal("gravity", "threshold", -1.0);
    niter_ = pin->GetOrAddInteger("gravity", "niteration", -1);
    npresmooth_ = pin->GetOrAddReal("gravity", "npresmooth", npresmooth_);
    npostsmooth_ = pin->GetOrAddReal("gravity", "npostsmooth", npostsmooth_);
    full_multigrid_ = pin->GetOrAddBoolean("gravity", "full_multigrid", false);
    fmg_ncycle_ = pin->GetOrAddInteger("gravity", "fmg_ncycle", 1);
    fshowdef_ = pin->GetOrAddBoolean("gravity", "show_defect", false);
    fsubtract_average_ = pin->GetOrAddBoolean("gravity", "subtract_average", true);
    if (eps_ < 0.0 && niter_ < 0) {
        std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Either \"threshold\" or \"niteration\" parameter must be set "
        << "in the <gravity> block." << std::endl
        << "When both parameters are specified, \"niteration\" is ignored." << std::endl
        << "Set \"threshold = 0.0\" for automatic convergence control." << std::endl;
        exit(EXIT_FAILURE);
  }
  if (four_pi_G_ < 0.0) {
    std::cout<< "### FATAL ERROR in MGGravityDriver::MGGravityDriver" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
    exit(EXIT_FAILURE);
  }
  // Override MG boundary conditions if specified in input file.
  // Options: "zerofixed" (Dirichlet zero), "zerograd" (Neumann zero), "multipole".
  std::string mg_bc_str = pin->GetOrAddString("gravity", "mg_bc", "none");
  if (mg_bc_str != "none") {
    BoundaryFlag mg_bc;
    if (mg_bc_str == "zerofixed") {
      mg_bc = BoundaryFlag::mg_zerofixed;
    } else if (mg_bc_str == "zerograd") {
      mg_bc = BoundaryFlag::mg_zerograd;
    } else if (mg_bc_str == "multipole") {
      mg_bc = BoundaryFlag::mg_multipole;
    } else {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "Unknown mg_bc type: " << mg_bc_str << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int f = 0; f < 6; ++f) {
      if (mg_mesh_bcs_[f] != BoundaryFlag::periodic) {
        mg_mesh_bcs_[f] = mg_bc;
      }
    }
  }
  if (!pmy_mesh_->strictly_periodic) {
    fsubtract_average_ = false;
  }

  // Check if multipole BCs are active and configure
  for (int f = 0; f < 6; ++f) {
    if (mg_mesh_bcs_[f] == BoundaryFlag::mg_multipole) {
      mporder_ = 0;  // mark as detected
      break;
    }
  }
  if (mporder_ >= 0) {
    mporder_ = pin->GetOrAddInteger("gravity", "mporder", 4);
    autompo_ = pin->GetOrAddBoolean("gravity", "auto_mporigin", true);
    nodipole_ = pin->GetOrAddBoolean("gravity", "nodipole", false);
    if (mporder_ != 2 && mporder_ != 4) {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "mporder must be 2 (quadrupole) or 4 (hexadecapole)." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (autompo_ && nodipole_) {
      std::cout << "### FATAL ERROR in MGGravityDriver" << std::endl
                << "auto_mporigin and nodipole cannot be used together." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!autompo_) {
      mpo_[0] = pin->GetReal("gravity", "mporigin_x1");
      mpo_[1] = pin->GetReal("gravity", "mporigin_x2");
      mpo_[2] = pin->GetReal("gravity", "mporigin_x3");
    }
    AllocateMultipoleCoefficients();
    fsubtract_average_ = false;
  }

  // Source masking parameters
  mask_radius_ = pin->GetOrAddReal("gravity", "mask_radius", -1.0);
  mask_origin_[0] = pin->GetOrAddReal("gravity", "mask_origin_x1", 0.0);
  mask_origin_[1] = pin->GetOrAddReal("gravity", "mask_origin_x2", 0.0);
  mask_origin_[2] = pin->GetOrAddReal("gravity", "mask_origin_x3", 0.0);

  // Allocate the root multigrid
  int nghost = pin->GetOrAddInteger("gravity", "mg_nghost", 1);
  bool root_on_host = pin->GetOrAddBoolean("gravity", "root_on_host", false);
  mgroot_ = new MGGravity(this, nullptr, nghost, root_on_host);
  mglevels_ = new MGGravity(this, pmbp, nghost);
  // allocate boundary buffers
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers((nvar_));
  mglevels_->pbval->RemapIndicesForMG();
}


//----------------------------------------------------------------------------------------
//! \fn MGGravityDriver::~MGGravityDriver()
//! \brief MGGravityDriver destructor

MGGravityDriver::~MGGravityDriver() {
  delete mgroot_;
  delete mglevels_;
}

void MGGravityDriver::SetFourPiG(Real four_pi_G) {
  four_pi_G_ = four_pi_G;
}

bool MGGravityDriver::PoissonCellCoveredByFiner(int level, int ix, int iy, int iz) const {
  const int nx = pmy_mesh_->mb_indcs.nx1;
  for (int n = 0; n < nbtotal_; ++n) {
    const LogicalLocation &loc = pmy_mesh_->lloc_eachmb[n];
    if (loc.level <= level) continue;
    const int shift = loc.level - level;
    const int factor = 1 << shift;
    const int fx0 = ix*factor, fx1 = (ix + 1)*factor - 1;
    const int fy0 = iy*factor, fy1 = (iy + 1)*factor - 1;
    const int fz0 = iz*factor, fz1 = (iz + 1)*factor - 1;
    const int bx0 = static_cast<int>(loc.lx1)*nx;
    const int by0 = static_cast<int>(loc.lx2)*nx;
    const int bz0 = static_cast<int>(loc.lx3)*nx;
    const int bx1 = bx0 + nx - 1;
    const int by1 = by0 + nx - 1;
    const int bz1 = bz0 + nx - 1;
    if (fx0 <= bx1 && fx1 >= bx0 &&
        fy0 <= by1 && fy1 >= by0 &&
        fz0 <= bz1 && fz1 >= bz0) {
      return true;
    }
  }
  return false;
}

bool MGGravityDriver::PoissonCellCoveredAtOrAbove(int level, int ix, int iy, int iz) const {
  const int nx = pmy_mesh_->mb_indcs.nx1;
  for (int n = 0; n < nbtotal_; ++n) {
    const LogicalLocation &loc = pmy_mesh_->lloc_eachmb[n];
    if (loc.level < level) continue;
    const int shift = loc.level - level;
    const int factor = 1 << shift;
    const int fx0 = ix*factor, fx1 = (ix + 1)*factor - 1;
    const int fy0 = iy*factor, fy1 = (iy + 1)*factor - 1;
    const int fz0 = iz*factor, fz1 = (iz + 1)*factor - 1;
    const int bx0 = static_cast<int>(loc.lx1)*nx;
    const int by0 = static_cast<int>(loc.lx2)*nx;
    const int bz0 = static_cast<int>(loc.lx3)*nx;
    const int bx1 = bx0 + nx - 1;
    const int by1 = by0 + nx - 1;
    const int bz1 = bz0 + nx - 1;
    if (fx0 <= bx1 && fx1 >= bx0 &&
        fy0 <= by1 && fy1 >= by0 &&
        fz0 <= bz1 && fz1 >= bz0) {
      return true;
    }
  }
  return false;
}

bool MGGravityDriver::PoissonCellNeedsReset(int level, int ix, int iy, int iz) const {
  if (level <= pmy_mesh_->root_level) return false;
  const int nx = pmy_mesh_->mb_indcs.nx1;
  const int ncellx = nrbx1_*nx*(1 << (level - pmy_mesh_->root_level));
  const int ncelly = nrbx2_*nx*(1 << (level - pmy_mesh_->root_level));
  const int ncellz = nrbx3_*nx*(1 << (level - pmy_mesh_->root_level));
  const std::array<std::array<int, 3>, 6> dirs{{
      {{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}},
      {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}}}};
  for (const auto &dir : dirs) {
    int ni = ix + dir[0];
    int nj = iy + dir[1];
    int nk = iz + dir[2];
    if (ni < 0 || ni >= ncellx || nj < 0 || nj >= ncelly ||
        nk < 0 || nk >= ncellz) {
      if (pmy_mesh_->strictly_periodic) {
        ni = (ni + ncellx) % ncellx;
        nj = (nj + ncelly) % ncelly;
        nk = (nk + ncellz) % ncellz;
      } else {
        continue;
      }
    }
    if (!PoissonCellCoveredAtOrAbove(level, ni, nj, nk)) return true;
  }
  return false;
}

void MGGravityDriver::BuildPoissonCompositeMasks() {
  mglevels_->ClearCompositeMasks();
  mgroot_->ClearCompositeMasks();
  for (int lev = 0; lev < nreflevel_; ++lev) {
    for (int o = 0; o < noctets_[lev]; ++o) {
      octets_[lev][o].ZeroClearMask();
    }
  }

  const int nx = pmy_mesh_->mb_indcs.nx1;
  const int root_level = pmy_mesh_->root_level;
  const int ngh = mglevels_->GetGhostCells();
  auto gids = pmy_pack_->pmb->mb_gid.h_view;

  for (int level = 0; level < mglevels_->GetNumberOfLevels(); ++level) {
    auto mask = mglevels_->GetCompositeMaskLevel_h(level);
    const int ncells = mglevels_->GetLevelActiveCells(level);
    const int ll = mglevels_->GetNumberOfLevels() - 1 - level;
    for (int m = 0; m < mglevels_->GetNumMeshBlocks(); ++m) {
      const LogicalLocation &loc = pmy_mesh_->lloc_eachmb[gids(m)];
      const int cell_level = std::max(root_level, loc.level - ll);
      const int coord_shift = loc.level - cell_level;
      for (int k = 0; k < mask.extent_int(2); ++k) {
        for (int j = 0; j < mask.extent_int(3); ++j) {
          for (int i = 0; i < mask.extent_int(4); ++i) {
            const bool active = i >= ngh && i < ngh + ncells &&
                                j >= ngh && j < ngh + ncells &&
                                k >= ngh && k < ngh + ncells;
            if (!active) {
              mask(m, COMP_STENCIL, k, j, i) = 1;
              continue;
            }
            const int fi = static_cast<int>(loc.lx1)*nx + ((i - ngh) << ll);
            const int fj = static_cast<int>(loc.lx2)*nx + ((j - ngh) << ll);
            const int fk = static_cast<int>(loc.lx3)*nx + ((k - ngh) << ll);
            const int ci = (coord_shift > 0) ? (fi >> coord_shift) : fi;
            const int cj = (coord_shift > 0) ? (fj >> coord_shift) : fj;
            const int ck = (coord_shift > 0) ? (fk >> coord_shift) : fk;
            if (PoissonCellCoveredByFiner(cell_level, ci, cj, ck)) {
              mask(m, COMP_COVERED, k, j, i) = 1;
            } else if (PoissonCellNeedsReset(cell_level, ci, cj, ck)) {
              mask(m, COMP_RESET, k, j, i) = 1;
              mask(m, COMP_STENCIL, k, j, i) = 1;
            } else {
              mask(m, COMP_SOLVE, k, j, i) = 1;
              mask(m, COMP_RESID, k, j, i) = 1;
            }
          }
        }
      }
    }
    mglevels_->ModifyCompositeMaskLevelOnHost(level);
    mglevels_->SyncCompositeMaskLevelToDevice(level);
  }

  for (int level = 0; level < mgroot_->GetNumberOfLevels(); ++level) {
    auto mask = mgroot_->GetCompositeMaskLevel_h(level);
    const int ncells = mgroot_->GetLevelActiveCells(level);
    const int ll = mgroot_->GetNumberOfLevels() - 1 - level;
    const int cell_level = root_level;
    for (int k = 0; k < mask.extent_int(2); ++k) {
      for (int j = 0; j < mask.extent_int(3); ++j) {
        for (int i = 0; i < mask.extent_int(4); ++i) {
          const bool active = i >= ngh && i < ngh + ncells &&
                              j >= ngh && j < ngh + ncells &&
                              k >= ngh && k < ngh + ncells;
          if (!active) {
            mask(0, COMP_STENCIL, k, j, i) = 1;
            continue;
          }
          const int ci = ((i - ngh) << ll);
          const int cj = ((j - ngh) << ll);
          const int ck = ((k - ngh) << ll);
          if (PoissonCellCoveredByFiner(cell_level, ci, cj, ck)) {
            mask(0, COMP_COVERED, k, j, i) = 1;
          } else {
            mask(0, COMP_SOLVE, k, j, i) = 1;
            mask(0, COMP_RESID, k, j, i) = 1;
          }
        }
      }
    }
    mgroot_->ModifyCompositeMaskLevelOnHost(level);
    mgroot_->SyncCompositeMaskLevelToDevice(level);
  }

  for (int level = 0; level < nreflevel_; ++level) {
    const int cell_level = root_level + level + 1;
    for (int o = 0; o < noctets_[level]; ++o) {
      MGOctet &oct = octets_[level][o];
      for (int k = 0; k < oct.nc; ++k) {
        for (int j = 0; j < oct.nc; ++j) {
          for (int i = 0; i < oct.nc; ++i) {
            const bool active = i >= ngh && i <= ngh + 1 &&
                                j >= ngh && j <= ngh + 1 &&
                                k >= ngh && k <= ngh + 1;
            if (!active) {
              oct.Mask(COMP_STENCIL, k, j, i) = 1;
              continue;
            }
            const int ci = static_cast<int>(oct.loc.lx1)*2 + (i - ngh);
            const int cj = static_cast<int>(oct.loc.lx2)*2 + (j - ngh);
            const int ck = static_cast<int>(oct.loc.lx3)*2 + (k - ngh);
            if (PoissonCellCoveredByFiner(cell_level, ci, cj, ck)) {
              oct.Mask(COMP_COVERED, k, j, i) = 1;
            } else if (PoissonCellNeedsReset(cell_level, ci, cj, ck)) {
              oct.Mask(COMP_RESET, k, j, i) = 1;
              oct.Mask(COMP_STENCIL, k, j, i) = 1;
            } else {
              oct.Mask(COMP_SOLVE, k, j, i) = 1;
              oct.Mask(COMP_RESID, k, j, i) = 1;
            }
          }
        }
      }
    }
  }
}

PoissonCompositeMaskCounts MGGravityDriver::CountPoissonMeshBlockMasks(
    int level, bool active_only) const {
  PoissonCompositeMaskCounts counts;
  auto mask = mglevels_->GetCompositeMaskLevel_h(level);
  int il = 0, iu = mask.extent_int(4) - 1;
  int jl = 0, ju = mask.extent_int(3) - 1;
  int kl = 0, ku = mask.extent_int(2) - 1;
  if (active_only) {
    const int ncells = mglevels_->GetLevelActiveCells(level);
    il = jl = kl = mglevels_->GetGhostCells();
    iu = il + ncells - 1;
    ju = jl + ncells - 1;
    ku = kl + ncells - 1;
  }
  for (int m = 0; m < mglevels_->GetNumMeshBlocks(); ++m) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          counts.solve += mask(m, COMP_SOLVE, k, j, i);
          counts.resid += mask(m, COMP_RESID, k, j, i);
          counts.reset += mask(m, COMP_RESET, k, j, i);
          counts.stencil += mask(m, COMP_STENCIL, k, j, i);
          counts.covered += mask(m, COMP_COVERED, k, j, i);
        }
      }
    }
  }
  return counts;
}

PoissonCompositeMaskCounts MGGravityDriver::CountPoissonRootMasks(
    int level, bool active_only) const {
  PoissonCompositeMaskCounts counts;
  if (global_variable::my_rank != 0) return counts;
  auto mask = mgroot_->GetCompositeMaskLevel_h(level);
  int il = 0, iu = mask.extent_int(4) - 1;
  int jl = 0, ju = mask.extent_int(3) - 1;
  int kl = 0, ku = mask.extent_int(2) - 1;
  if (active_only) {
    const int ncells = mgroot_->GetLevelActiveCells(level);
    il = jl = kl = mgroot_->GetGhostCells();
    iu = il + ncells - 1;
    ju = jl + ncells - 1;
    ku = kl + ncells - 1;
  }
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        counts.solve += mask(0, COMP_SOLVE, k, j, i);
        counts.resid += mask(0, COMP_RESID, k, j, i);
        counts.reset += mask(0, COMP_RESET, k, j, i);
        counts.stencil += mask(0, COMP_STENCIL, k, j, i);
        counts.covered += mask(0, COMP_COVERED, k, j, i);
      }
    }
  }
  return counts;
}

PoissonCompositeMaskCounts MGGravityDriver::CountPoissonOctetMasks(
    int level, bool active_only) const {
  PoissonCompositeMaskCounts counts;
  if (global_variable::my_rank != 0 || level < 0 || level >= nreflevel_) return counts;
  const int ngh = mgroot_->GetGhostCells();
  for (int o = 0; o < noctets_[level]; ++o) {
    const MGOctet &oct = octets_[level][o];
    for (int k = 0; k < oct.nc; ++k) {
      for (int j = 0; j < oct.nc; ++j) {
        for (int i = 0; i < oct.nc; ++i) {
          const bool active = i >= ngh && i <= ngh + 1 &&
                              j >= ngh && j <= ngh + 1 &&
                              k >= ngh && k <= ngh + 1;
          if (active_only && !active) continue;
          counts.solve += oct.Mask(COMP_SOLVE, k, j, i);
          counts.resid += oct.Mask(COMP_RESID, k, j, i);
          counts.reset += oct.Mask(COMP_RESET, k, j, i);
          counts.stencil += oct.Mask(COMP_STENCIL, k, j, i);
          counts.covered += oct.Mask(COMP_COVERED, k, j, i);
        }
      }
    }
  }
  return counts;
}

void MGGravityDriver::PrintPoissonCompositeMaskDiagnostics() const {
  PoissonCompositeMaskCounts total;
  auto add_total = [&total](const PoissonCompositeMaskCounts &c) {
    total.solve += c.solve;
    total.resid += c.resid;
    total.reset += c.reset;
    total.stencil += c.stencil;
    total.covered += c.covered;
  };
  for (int level = 0; level < mglevels_->GetNumberOfLevels(); ++level) {
    const auto active = ReducePoissonMaskCounts(CountPoissonMeshBlockMasks(level, true));
    const auto all = ReducePoissonMaskCounts(CountPoissonMeshBlockMasks(level, false));
    const auto staging = SubtractPoissonMaskCounts(all, active);
    add_total(active);
    std::ostringstream prefix;
    prefix << "Poisson composite masks: entity=meshblock level=" << level
           << " region=active";
    PrintPoissonMaskCounts(prefix.str(), active);
    prefix.str("");
    prefix << "Poisson composite masks: entity=meshblock level=" << level
           << " region=staging";
    PrintPoissonMaskCounts(prefix.str(), staging);
  }
  for (int level = 0; level < mgroot_->GetNumberOfLevels(); ++level) {
    const auto active = ReducePoissonMaskCounts(CountPoissonRootMasks(level, true));
    const auto all = ReducePoissonMaskCounts(CountPoissonRootMasks(level, false));
    const auto staging = SubtractPoissonMaskCounts(all, active);
    add_total(active);
    std::ostringstream prefix;
    prefix << "Poisson composite masks: entity=root level=" << level
           << " region=active";
    PrintPoissonMaskCounts(prefix.str(), active);
    prefix.str("");
    prefix << "Poisson composite masks: entity=root level=" << level
           << " region=staging";
    PrintPoissonMaskCounts(prefix.str(), staging);
  }
  for (int level = 0; level < nreflevel_; ++level) {
    const auto active = ReducePoissonMaskCounts(CountPoissonOctetMasks(level, true));
    const auto all = ReducePoissonMaskCounts(CountPoissonOctetMasks(level, false));
    const auto staging = SubtractPoissonMaskCounts(all, active);
    add_total(active);
    std::ostringstream prefix;
    prefix << "Poisson composite masks: entity=octet level=" << level
           << " region=active";
    PrintPoissonMaskCounts(prefix.str(), active);
    prefix.str("");
    prefix << "Poisson composite masks: entity=octet level=" << level
           << " region=staging";
    PrintPoissonMaskCounts(prefix.str(), staging);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "Poisson composite masks: decomposition_check key=solve count="
              << total.solve << std::endl;
    std::cout << "Poisson composite masks: decomposition_check key=resid count="
              << total.resid << std::endl;
    std::cout << "Poisson composite masks: decomposition_check key=reset count="
              << total.reset << std::endl;
    std::cout << "Poisson composite masks: decomposition_check key=stencil count="
              << total.stencil << std::endl;
    std::cout << "Poisson composite masks: decomposition_check key=covered count="
              << total.covered << std::endl;
  }
}

void MGGravityDriver::PrintPoissonResidualSplit(const char *label) {
  const int level = mglevels_->GetNumberOfLevels() - 1;
  mglevels_->SetCurrentLevel(level);
  mglevels_->CalculateDefectPack();
  Kokkos::fence();
  mglevels_->SyncDefectLevelToHost(level);
  auto def = mglevels_->GetDefectLevel_h(level);
  auto mask = mglevels_->GetCompositeMaskLevel_h(level);
  const int ngh = mglevels_->GetGhostCells();
  const int ncells = mglevels_->GetLevelActiveCells(level);
  PoissonResidualSplit local;
  auto add = [](PoissonResidualCategory &cat, Real value) {
    const Real av = std::abs(value);
    ++cat.count;
    cat.sum2 += value*value;
    cat.maxabs = std::max(cat.maxabs, av);
  };
  for (int m = 0; m < mglevels_->GetNumMeshBlocks(); ++m) {
    for (int k = ngh; k < ngh + ncells; ++k) {
      for (int j = ngh; j < ngh + ncells; ++j) {
        for (int i = ngh; i < ngh + ncells; ++i) {
          const Real r = def(m, 0, k, j, i);
          if (mask(m, COMP_RESID, k, j, i) != 0) add(local.accepted, r);
          if (mask(m, COMP_COVERED, k, j, i) != 0) {
            add(local.covered, r);
          } else if (mask(m, COMP_RESET, k, j, i) != 0) {
            add(local.reset, r);
          } else if (mask(m, COMP_SOLVE, k, j, i) != 0) {
            add(local.solve, r);
          } else if (mask(m, COMP_STENCIL, k, j, i) != 0) {
            add(local.stencil, r);
          }
        }
      }
    }
  }
  const auto global = ReducePoissonResidualSplit(local);
  if (global_variable::my_rank == 0) {
    std::cout << "Poisson residual split: label=" << label;
    PrintPoissonResidualCategory("solve_interior", global.solve);
    PrintPoissonResidualCategory("reset_interface", global.reset);
    PrintPoissonResidualCategory("stencil_only", global.stencil);
    PrintPoissonResidualCategory("covered", global.covered);
    PrintPoissonResidualCategory("resid_accepted", global.accepted);
    std::cout << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn MGGravity::MGGravity(MultigridDriver *pmd, MeshBlock *pmb)
//! \brief MGGravity constructor

MGGravity::MGGravity(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                     bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host) {
}


//----------------------------------------------------------------------------------------
//! \fn MGGravity::~MGGravity()
//! \brief MGGravity deconstructor

MGGravity::~MGGravity() {
  //delete pmgbval;
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::Solve(int stage, Real dt)
//! \brief load the data and solve

void MGGravityDriver::Solve(Driver *pdriver, int stage, Real dt) {
  RegionIndcs &indcs_ = pmy_pack_->pmesh->mb_indcs;

  // Reallocate MG arrays and phi if AMR has changed the mesh
  PrepareForAMR();
  if (poisson_test_enabled_ && global_variable::my_rank == 0) {
    std::cout << "Poisson MG test: mode="
              << (pmy_pack_->pmesh->multilevel ? "SMR" : "unigrid")
              << " composite_fas=" << (poisson_test_composite_fas_ ? 1 : 0)
              << " ranks=" << global_variable::nranks
              << " nreflevel=" << nreflevel_
              << " meshblock_transfer_level=" << MeshBlockTransferLevel()
              << " root_octet_bridge_used=" << ((nreflevel_ > 0) ? 1 : 0)
              << std::endl;
    if (poisson_test_composite_fas_) {
      std::cout << "Poisson MG test: composite scaffold uses generic "
                << "Multigrid traversal; invariant masks and AMR boundary "
                << "contract are pending later stages." << std::endl;
    }
  }
  {
    int nmb = pmy_pack_->nmb_thispack;
    if (static_cast<int>(pmy_pack_->pgrav->phi.extent_int(0)) != nmb) {
      int ncells1 = indcs_.nx1 + 2*indcs_.ng;
      int ncells2 = (indcs_.nx2 > 1) ? (indcs_.nx2 + 2*indcs_.ng) : 1;
      int ncells3 = (indcs_.nx3 > 1) ? (indcs_.nx3 + 2*indcs_.ng) : 1;
      Kokkos::realloc(pmy_pack_->pgrav->phi, nmb, 1, ncells3, ncells2, ncells1);
    }
  }

  // mglevels_ points to the Multigrid object for all MeshBlocks
  // The MG smoother solves -∇²u = src (note the minus sign from the Laplacian
  // convention: Laplacian(u) = 6u - neighbors = -dx²∇²u).  To obtain the
  // standard Poisson equation ∇²φ = 4πGρ we must load the source with a
  // negative sign so that -∇²φ = -4πGρ, i.e. ∇²φ = +4πGρ.
  auto &u0 = (pmy_pack_->pmhd != nullptr) ? pmy_pack_->pmhd->u0
                                            : pmy_pack_->phydro->u0;
  mglevels_->LoadSource(u0, IDN, indcs_.ng, -four_pi_G_);

  // Apply source mask (zero source outside mask_radius_)
  mglevels_->ApplyMask();

  // iterative mode - load initial guess
  if(!full_multigrid_) 
    mglevels_->LoadFinestData(pmy_pack_->pgrav->phi, 0, indcs_.ng);

  // Finalize setup (SubtractAverage, level counts) after data is loaded
  SetupMultigrid(dt, false);
  if (poisson_test_enabled_) {
    BuildPoissonCompositeMasks();
    if (poisson_test_debug_composite_masks_) {
      PrintPoissonCompositeMaskDiagnostics();
    }
    if (poisson_test_debug_residual_split_) {
      PrintPoissonResidualSplit("initial");
    }
  }

  // Compute multipole coefficients for isolated boundaries
  if (mporder_ > 0) {
    if (autompo_) CalculateCenterOfMass();
    CalculateMultipoleCoefficients();
    SyncMultipoleToDevice();
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  if (full_multigrid_)
    SolveFMG(pdriver);
  else
    SolveMG(pdriver);

  Kokkos::fence();
  if (poisson_test_enabled_ && poisson_test_debug_residual_split_) {
    PrintPoissonResidualSplit("final");
  }

  if (fshowdef_) {
    auto t_end = std::chrono::high_resolution_clock::now();
    double mg_elapsed = std::chrono::duration<double>(t_end - t_start).count();
    Real norm = CalculateDefectNorm(MGNormType::l2, 0);
    if (global_variable::my_rank == 0) {
      std::cout << "mg_solve_time = " << std::scientific << std::setprecision(6)
                << mg_elapsed << std::endl;
      std::cout << "MGGravityDriver::Solve: Final defect norm = " << norm << std::endl;
    }
  }

  mglevels_->RetrieveResult(pmy_pack_->pgrav->phi, 0, indcs_.ng);

  return;
}

void MGGravity::SmoothPack(int color) {
  color ^= pmy_driver_->GetCoffset();
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{static_cast<MGGravityDriver*>(pmy_driver_)->omega_/6.0};
  if (on_host_) {
    Smooth(u_[current_level_].h_view, src_[current_level_].h_view,
           coeff_[current_level_].h_view, matrix_[current_level_].h_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  } else {
    Smooth(u_[current_level_].d_view, src_[current_level_].d_view,
           coeff_[current_level_].d_view, matrix_[current_level_].d_view,
           stencil, -ll, is, ie, js, je, ks, ke, color, false);
  }
}

void MGGravity::CalculateDefectPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{0.0};
  if (on_host_) {
    CalculateDefect(def_[current_level_].h_view, u_[current_level_].h_view,
                    src_[current_level_].h_view, coeff_[current_level_].h_view,
                    matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateDefect(def_[current_level_].d_view, u_[current_level_].d_view,
                    src_[current_level_].d_view, coeff_[current_level_].d_view,
                    matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}

void MGGravity::CalculateFASRHSPack() {
  int ll = nlevel_-1-current_level_;
  int is = ngh_, ie = is+(indcs_.nx1>>ll)-1;
  int js = ngh_, je = js+(indcs_.nx2>>ll)-1;
  int ks = ngh_, ke = ks+(indcs_.nx3>>ll)-1;
  GravityStencil stencil{0.0};
  if (on_host_) {
    CalculateFASRHS(src_[current_level_].h_view, u_[current_level_].h_view,
                    coeff_[current_level_].h_view, matrix_[current_level_].h_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  } else {
    CalculateFASRHS(src_[current_level_].d_view, u_[current_level_].d_view,
                    coeff_[current_level_].d_view, matrix_[current_level_].d_view,
                    stencil, -ll, is, ie, js, je, ks, ke, false);
  }
}


//----------------------------------------------------------------------------------------
// Host-side octet physics for MGGravityDriver

static inline Real OctLaplacian(const MGOctet &o, int v, int k, int j, int i) {
  return (6.0*o.U(v,k,j,i) - o.U(v,k+1,j,i) - o.U(v,k,j+1,i)
          - o.U(v,k,j,i+1) - o.U(v,k-1,j,i) - o.U(v,k,j-1,i)
          - o.U(v,k,j,i-1));
}

void MGGravityDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real dx2 = dx * dx;
  Real isix = omega_ / 6.0;
  int c = color ^ coffset_;
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh + ((c^k^j)&1); i <= ngh+1; i += 2) {
        Real lap = OctLaplacian(oct, 0, k, j, i);
        oct.U(0,k,j,i) -= (lap - oct.Src(0,k,j,i)*dx2)*isix;
      }
    }
  }
}

void MGGravityDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Def(0,k,j,i) = oct.Src(0,k,j,i) - OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}

void MGGravityDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx / static_cast<Real>(1 << rlev);
  Real idx2 = 1.0 / (dx * dx);
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        oct.Src(0,k,j,i) += OctLaplacian(oct, 0, k, j, i) * idx2;
      }
    }
  }
}


//----------------------------------------------------------------------------------------
//! \fn void MGGravityDriver::ProlongateOctetBoundariesFluxCons(...)
//! \brief Conservative prolongation of face ghost cells at fine-coarse level boundaries.
//!        Implements the "Conservative Formulation" from Tomida & Stone (2023) Eq. 24-27.
//!        Ghost = (2/3)*coarse_interpolated + (1/3)*fine_active, where transverse
//!        gradients from the coarse buffer provide sub-cell interpolation.
//!        Only face neighbors are handled (edges/corners are unused by the 7-point stencil).

void MGGravityDriver::ProlongateOctetBoundariesFluxCons(MGOctet &oct,
     std::vector<Real> &cbuf, const std::vector<bool> &ncoarse) {
  constexpr Real ot = 1.0/3.0;
  const int ngh = mgroot_->GetGhostCells();
  const int l = ngh, r = ngh + 1;

  // x1 face
  for (int ox1 = -1; ox1 <= 1; ox1 += 2) {
    if (ncoarse[1*9 + 1*3 + (ox1+1)]) {
      int i, fi, fig;
      if (ox1 > 0) { i = ngh + 1; fi = ngh + 1; fig = ngh + 2; }
      else         { i = ngh - 1; fi = ngh;     fig = ngh - 1; }
      Real ccval = BufRef(cbuf, 3, 0, ngh, ngh, i);
      Real gx2c = 0.125*(BufRef(cbuf, 3, 0, ngh, ngh+1, i)
                        - BufRef(cbuf, 3, 0, ngh, ngh-1, i));
      Real gx3c = 0.125*(BufRef(cbuf, 3, 0, ngh+1, ngh, i)
                        - BufRef(cbuf, 3, 0, ngh-1, ngh, i));
      oct.U(0, l, l, fig) = ot*(2.0*(ccval - gx2c - gx3c) + oct.U(0, l, l, fi));
      oct.U(0, l, r, fig) = ot*(2.0*(ccval + gx2c - gx3c) + oct.U(0, l, r, fi));
      oct.U(0, r, l, fig) = ot*(2.0*(ccval - gx2c + gx3c) + oct.U(0, r, l, fi));
      oct.U(0, r, r, fig) = ot*(2.0*(ccval + gx2c + gx3c) + oct.U(0, r, r, fi));
    }
  }

  // x2 face
  for (int ox2 = -1; ox2 <= 1; ox2 += 2) {
    if (ncoarse[1*9 + (ox2+1)*3 + 1]) {
      int j, fj, fjg;
      if (ox2 > 0) { j = ngh + 1; fj = ngh + 1; fjg = ngh + 2; }
      else         { j = ngh - 1; fj = ngh;     fjg = ngh - 1; }
      Real ccval = BufRef(cbuf, 3, 0, ngh, j, ngh);
      Real gx1c = 0.125*(BufRef(cbuf, 3, 0, ngh, j, ngh+1)
                        - BufRef(cbuf, 3, 0, ngh, j, ngh-1));
      Real gx3c = 0.125*(BufRef(cbuf, 3, 0, ngh+1, j, ngh)
                        - BufRef(cbuf, 3, 0, ngh-1, j, ngh));
      oct.U(0, l, fjg, l) = ot*(2.0*(ccval - gx1c - gx3c) + oct.U(0, l, fj, l));
      oct.U(0, l, fjg, r) = ot*(2.0*(ccval + gx1c - gx3c) + oct.U(0, l, fj, r));
      oct.U(0, r, fjg, l) = ot*(2.0*(ccval - gx1c + gx3c) + oct.U(0, r, fj, l));
      oct.U(0, r, fjg, r) = ot*(2.0*(ccval + gx1c + gx3c) + oct.U(0, r, fj, r));
    }
  }

  // x3 face
  for (int ox3 = -1; ox3 <= 1; ox3 += 2) {
    if (ncoarse[(ox3+1)*9 + 1*3 + 1]) {
      int k, fk, fkg;
      if (ox3 > 0) { k = ngh + 1; fk = ngh + 1; fkg = ngh + 2; }
      else         { k = ngh - 1; fk = ngh;     fkg = ngh - 1; }
      Real ccval = BufRef(cbuf, 3, 0, k, ngh, ngh);
      Real gx1c = 0.125*(BufRef(cbuf, 3, 0, k, ngh, ngh+1)
                        - BufRef(cbuf, 3, 0, k, ngh, ngh-1));
      Real gx2c = 0.125*(BufRef(cbuf, 3, 0, k, ngh+1, ngh)
                        - BufRef(cbuf, 3, 0, k, ngh-1, ngh));
      oct.U(0, fkg, l, l) = ot*(2.0*(ccval - gx1c - gx2c) + oct.U(0, fk, l, l));
      oct.U(0, fkg, l, r) = ot*(2.0*(ccval + gx1c - gx2c) + oct.U(0, fk, l, r));
      oct.U(0, fkg, r, l) = ot*(2.0*(ccval - gx1c + gx2c) + oct.U(0, fk, r, l));
      oct.U(0, fkg, r, r) = ot*(2.0*(ccval + gx1c + gx2c) + oct.U(0, fk, r, r));
    }
  }
}
