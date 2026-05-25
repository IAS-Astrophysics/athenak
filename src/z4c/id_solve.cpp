#include "z4c/id_solve.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "utils/finite_diff.hpp"
#include "z4c/z4c.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {
namespace {

constexpr Real kPi = 3.141592653589793238462643383279502884;

KOKKOS_INLINE_FUNCTION
int SymIdx(int a, int b) {
  if (a > b) {
    int t = a;
    a = b;
    b = t;
  }
  if (a == 0 && b == 0) return 0;
  if (a == 0 && b == 1) return 1;
  if (a == 0 && b == 2) return 2;
  if (a == 1 && b == 1) return 3;
  if (a == 1 && b == 2) return 4;
  return 5;
}

KOKKOS_INLINE_FUNCTION
Real LeviCivita(int i, int j, int k) {
  if (i == j || j == k || i == k) return 0.0;
  if ((i == 0 && j == 1 && k == 2) ||
      (i == 1 && j == 2 && k == 0) ||
      (i == 2 && j == 0 && k == 1)) return 1.0;
  return -1.0;
}

KOKKOS_INLINE_FUNCTION
void Cross(const Real a[3], const Real b[3], Real axb[3]) {
  axb[0] = a[1]*b[2] - a[2]*b[1];
  axb[1] = a[2]*b[0] - a[0]*b[2];
  axb[2] = a[0]*b[1] - a[1]*b[0];
}

KOKKOS_INLINE_FUNCTION
void AddBowenYorkAhat(const Real x[3], const Real xp[3], const Real p[3],
                      const Real s[3], Real a[3][3]) {
  Real rvec[3] = {x[0] - xp[0], x[1] - xp[1], x[2] - xp[2]};
  Real r2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
  Real r = std::sqrt(std::max(r2, static_cast<Real>(1.0e-24)));
  Real n[3] = {rvec[0]/r, rvec[1]/r, rvec[2]/r};
  Real pdotn = p[0]*n[0] + p[1]*n[1] + p[2]*n[2];
  Real sxn[3];
  Cross(s, n, sxn);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Real delta = (i == j) ? 1.0 : 0.0;
      a[i][j] += (3.0/(2.0*r2))*
          (p[i]*n[j] + p[j]*n[i] - (delta - n[i]*n[j])*pdotn);
      a[i][j] += (3.0/(r2*r))*(n[i]*sxn[j] + n[j]*sxn[i]);
    }
  }
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real HamiltonianResidual(const IDConformalThinSandwich::RelaxVars &relax,
                         const IDConformalThinSandwich::FreeVars &free,
                         const Real idx[3], int m, int k, int j, int i) {
  Real lap = Dxx<NGHOST>(0, idx, relax.u, m, k, j, i)
           + Dxx<NGHOST>(1, idx, relax.u, m, k, j, i)
           + Dxx<NGHOST>(2, idx, relax.u, m, k, j, i);
  Real psi = std::max(free.psi_singular(m,k,j,i) + relax.u(m,k,j,i),
                      static_cast<Real>(1.0e-10));
  return lap + 0.125*free.ahat2(m,k,j,i)/std::pow(psi, 7.0);
}

template <typename ViewType>
void DeepCopy5D(ViewType dst, ViewType src) {
  Kokkos::deep_copy(DevExeSpace(), dst, src);
}

Real BlockCornerRadius(const RegionSize &rs, const Real center[3]) {
  Real rmax = 0.0;
  for (int i1 = 0; i1 < 2; ++i1) {
    Real x1 = (i1 == 0 ? rs.x1min : rs.x1max) - center[0];
    for (int i2 = 0; i2 < 2; ++i2) {
      Real x2 = (i2 == 0 ? rs.x2min : rs.x2max) - center[1];
      for (int i3 = 0; i3 < 2; ++i3) {
        Real x3 = (i3 == 0 ? rs.x3min : rs.x3max) - center[2];
        rmax = std::max(rmax, std::sqrt(x1*x1 + x2*x2 + x3*x3));
      }
    }
  }
  return rmax;
}

RegionSize MeshBlockRegion(const Mesh *pmesh, int gid) {
  RegionSize rs;
  const RegionSize &ms = pmesh->mesh_size;
  const LogicalLocation &lloc = pmesh->lloc_eachmb[gid];
  int level = lloc.level - pmesh->root_level;
  int nmbx1 = pmesh->nmb_rootx1 << level;
  rs.x1min = (lloc.lx1 == 0) ? ms.x1min : LeftEdgeX(lloc.lx1, nmbx1,
                                                     ms.x1min, ms.x1max);
  rs.x1max = (lloc.lx1 == nmbx1 - 1) ? ms.x1max : LeftEdgeX(lloc.lx1 + 1, nmbx1,
                                                            ms.x1min, ms.x1max);
  if (!pmesh->multi_d) {
    rs.x2min = ms.x2min;
    rs.x2max = ms.x2max;
  } else {
    int nmbx2 = pmesh->nmb_rootx2 << level;
    rs.x2min = (lloc.lx2 == 0) ? ms.x2min : LeftEdgeX(lloc.lx2, nmbx2,
                                                       ms.x2min, ms.x2max);
    rs.x2max = (lloc.lx2 == nmbx2 - 1) ? ms.x2max : LeftEdgeX(lloc.lx2 + 1, nmbx2,
                                                              ms.x2min, ms.x2max);
  }
  if (!pmesh->three_d) {
    rs.x3min = ms.x3min;
    rs.x3max = ms.x3max;
  } else {
    int nmbx3 = pmesh->nmb_rootx3 << level;
    rs.x3min = (lloc.lx3 == 0) ? ms.x3min : LeftEdgeX(lloc.lx3, nmbx3,
                                                       ms.x3min, ms.x3max);
    rs.x3max = (lloc.lx3 == nmbx3 - 1) ? ms.x3max : LeftEdgeX(lloc.lx3 + 1, nmbx3,
                                                              ms.x3min, ms.x3max);
  }
  rs.dx1 = (rs.x1max - rs.x1min)/static_cast<Real>(pmesh->mb_indcs.nx1);
  rs.dx2 = (rs.x2max - rs.x2min)/static_cast<Real>(pmesh->mb_indcs.nx2);
  rs.dx3 = (rs.x3max - rs.x3min)/static_cast<Real>(pmesh->mb_indcs.nx3);
  return rs;
}

} // namespace

IDConformalThinSandwich::IDConformalThinSandwich(MeshBlockPack *pmbp,
                                                 ParameterInput *pin)
    : pmy_pack_(pmbp), pbval_relax_(nullptr), enabled_(true), solved_(false),
      history_file_(nullptr) {
  enabled_ = pin->GetOrAddBoolean("id_solve", "enable", true);
  std::string method = pin->GetOrAddString("id_solve", "method", "hyperbolic_relaxation");
  std::string formulation = pin->GetOrAddString("id_solve", "formulation", "ctt");
  if (method != "hyperbolic_relaxation" || formulation != "ctt") {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "This branch supports only <id_solve>/method=hyperbolic_relaxation "
              << "and formulation=ctt." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  solve_once_ = pin->GetOrAddBoolean("id_solve", "solve_once", true);
  run_on_restart_ = pin->GetOrAddBoolean("id_solve", "run_on_restart", false);
  stop_after_solve_ = pin->GetOrAddBoolean("id_solve", "stop_after_solve", false);
  skip_initial_output_ = pin->GetOrAddBoolean("id_solve", "skip_initial_output",
                                              stop_after_solve_);
  reject_worse_ = pin->GetOrAddBoolean("id_solve", "reject_worse", true);
  max_steps_ = pin->GetOrAddInteger("id_solve", "max_steps", 2000);
  history_every_ = std::max(1, pin->GetOrAddInteger("id_solve", "history_every", 10));
  tolerance_ = pin->GetOrAddReal("id_solve", "tolerance", 1.0e-8);
  relax_cfl_ = pin->GetOrAddReal("id_solve", "relax_cfl", 0.25);
  eta_ = pin->GetOrAddReal("id_solve", "eta", 0.0);
  wavespeed_scale_ = pin->GetOrAddReal("id_solve", "wavespeed_scale", 1.0);
  wavespeed_mode_ = pin->GetOrAddString("id_solve", "wavespeed_mode", "smooth_box");
  if (wavespeed_mode_ != "local_dx" && wavespeed_mode_ != "smooth_box") {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "Supported <id_solve>/wavespeed_mode values are local_dx and "
              << "smooth_box." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  history_name_ = pin->GetString("job", "basename") + ".id_solve.hst";

  std::string block = "problem";
  Real b = pin->GetOrAddReal(block, "par_b", 1.0);
  bare_mass_[0] = pin->GetOrAddReal(block, "par_m_plus", 1.0);
  bare_mass_[1] = pin->GetOrAddReal(block, "par_m_minus", 1.0);
  Real center[3] = {
      pin->GetOrAddReal(block, "center_offset1", 0.0),
      pin->GetOrAddReal(block, "center_offset2", 0.0),
      pin->GetOrAddReal(block, "center_offset3", 0.0)};
  wavespeed_center_[0] = pin->GetOrAddReal("id_solve", "wavespeed_center1", center[0]);
  wavespeed_center_[1] = pin->GetOrAddReal("id_solve", "wavespeed_center2", center[1]);
  wavespeed_center_[2] = pin->GetOrAddReal("id_solve", "wavespeed_center3", center[2]);
  pos_[0][0] = center[0] + b; pos_[0][1] = center[1]; pos_[0][2] = center[2];
  pos_[1][0] = center[0] - b; pos_[1][1] = center[1]; pos_[1][2] = center[2];
  for (int a = 0; a < 3; ++a) {
    mom_[0][a] = pin->GetOrAddReal(block, "par_P_plus" + std::to_string(a + 1), 0.0);
    mom_[1][a] = pin->GetOrAddReal(block, "par_P_minus" + std::to_string(a + 1), 0.0);
    spin_[0][a] = pin->GetOrAddReal(block, "par_S_plus" + std::to_string(a + 1), 0.0);
    spin_[1][a] = pin->GetOrAddReal(block, "par_S_minus" + std::to_string(a + 1), 0.0);
  }

  int nmb = std::max(pmbp->nmb_thispack, pmbp->pmesh->nmb_maxperrank);
  auto &indcs = pmbp->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(u_relax, nmb, ID_RELAX_NVAR, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_relax_tmp, nmb, ID_RELAX_NVAR, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_rhs, nmb, ID_RELAX_NVAR, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_free, nmb, ID_RELAX_NFREE, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(u_relax, 0.0);
  Kokkos::deep_copy(u_rhs, 0.0);
  if (pmbp->pmesh->multilevel) {
    int nccells1 = indcs.cnx1 + 2*indcs.ng;
    int nccells2 = (indcs.cnx2 > 1) ? indcs.cnx2 + 2*indcs.ng : 1;
    int nccells3 = (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
    Kokkos::realloc(coarse_u_relax, nmb, ID_RELAX_NVAR, nccells3, nccells2, nccells1);
  }

  relax_.u.InitWithShallowSlice(u_relax, ID_RELAX_U);
  relax_.v.InitWithShallowSlice(u_relax, ID_RELAX_V);
  rhs_.u.InitWithShallowSlice(u_rhs, ID_RELAX_U);
  rhs_.v.InitWithShallowSlice(u_rhs, ID_RELAX_V);
  free_.psi_singular.InitWithShallowSlice(u_free, ID_RELAX_PSI_SINGULAR);
  free_.ahat2.InitWithShallowSlice(u_free, ID_RELAX_AHAT2);
  free_.residual.InitWithShallowSlice(u_free, ID_RELAX_RESIDUAL);
  free_.wavespeed.InitWithShallowSlice(u_free, ID_RELAX_WAVESPEED);

  pbval_relax_ = new MeshBoundaryValuesCC(pmbp, pin, true);
  pbval_relax_->InitializeBuffers(ID_RELAX_NVAR);
}

IDConformalThinSandwich::~IDConformalThinSandwich() {
  if (history_file_ != nullptr) std::fclose(history_file_);
  delete pbval_relax_;
}

void IDConformalThinSandwich::PrepareForRestart() {
  if (!run_on_restart_) {
    solved_ = true;
    solve_once_ = true;
  }
}

TaskStatus IDConformalThinSandwich::SolveTask(Driver *pdriver, int stage) {
  if (!enabled_) return TaskStatus::complete;
  if (solve_once_ && solved_) return TaskStatus::complete;
  SolveRelaxation(pdriver);
  RefreshZ4cBoundariesAfterSolve(pdriver);
  RecomputeConstraintsAfterSolve();
  solved_ = true;
  return TaskStatus::complete;
}

TaskStatus IDConformalThinSandwich::InitRecv() {
  return pbval_relax_->InitRecv(ID_RELAX_NVAR);
}

TaskStatus IDConformalThinSandwich::Send() {
  return pbval_relax_->PackAndSendCC(u_relax, coarse_u_relax);
}

TaskStatus IDConformalThinSandwich::Recv() {
  return pbval_relax_->RecvAndUnpackCC(u_relax, coarse_u_relax);
}

TaskStatus IDConformalThinSandwich::ClearSend() {
  return pbval_relax_->ClearSend();
}

TaskStatus IDConformalThinSandwich::ClearRecv() {
  return pbval_relax_->ClearRecv();
}

void IDConformalThinSandwich::Restrict() {
  if (pmy_pack_->pmesh->multilevel) {
    pmy_pack_->pmesh->pmr->RestrictCC(u_relax, coarse_u_relax, true);
  }
}

void IDConformalThinSandwich::Prolongate() {
  if (pmy_pack_->pmesh->multilevel) {
    pbval_relax_->ProlongateCC(u_relax, coarse_u_relax, true);
  }
}

void IDConformalThinSandwich::ApplyPhysicalBCs() {
  if (!pmy_pack_->pmesh->strictly_periodic) {
    pbval_relax_->Z4cBCs(pmy_pack_, pbval_relax_->u_in, u_relax, coarse_u_relax);
  }
}

void IDConformalThinSandwich::BuildCTTFreeData() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  int isg = indcs.is - indcs.ng, ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng, jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng, keg = indcs.ke + indcs.ng;
  int nmb = pmy_pack_->nmb_thispack;
  auto free = free_;
  Real m0 = bare_mass_[0], m1 = bare_mass_[1];
  Real x0[3] = {pos_[0][0], pos_[0][1], pos_[0][2]};
  Real x1[3] = {pos_[1][0], pos_[1][1], pos_[1][2]};
  Real p0[3] = {mom_[0][0], mom_[0][1], mom_[0][2]};
  Real p1[3] = {mom_[1][0], mom_[1][1], mom_[1][2]};
  Real s0[3] = {spin_[0][0], spin_[0][1], spin_[0][2]};
  Real s1[3] = {spin_[1][0], spin_[1][1], spin_[1][2]};

  par_for("IDCTT::BuildFreeData", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x[3] = {
        CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                    size.d_view(m).x1max),
        CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                    size.d_view(m).x2max),
        CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                    size.d_view(m).x3max)};
    Real r0 = std::sqrt(std::max(SQR(x[0] - x0[0]) + SQR(x[1] - x0[1]) +
                                 SQR(x[2] - x0[2]), static_cast<Real>(1.0e-24)));
    Real r1 = std::sqrt(std::max(SQR(x[0] - x1[0]) + SQR(x[1] - x1[1]) +
                                 SQR(x[2] - x1[2]), static_cast<Real>(1.0e-24)));
    free.psi_singular(m,k,j,i) = 1.0 + 0.5*m0/r0 + 0.5*m1/r1;
    Real a[3][3];
    for (int q = 0; q < 3; ++q)
      for (int r = 0; r < 3; ++r) a[q][r] = 0.0;
    AddBowenYorkAhat(x, x0, p0, s0, a);
    AddBowenYorkAhat(x, x1, p1, s1, a);
    Real ahat2 = 0.0;
    for (int q = 0; q < 3; ++q)
      for (int r = 0; r < 3; ++r) ahat2 += a[q][r]*a[q][r];
    free.ahat2(m,k,j,i) = ahat2;
    free.residual(m,k,j,i) = 0.0;
  });
}

void IDConformalThinSandwich::BuildWaveSpeedProfile(Real dx_min) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  int isg = indcs.is - indcs.ng, ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng, jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng, keg = indcs.ke + indcs.ng;
  int nmb = pmy_pack_->nmb_thispack;
  auto free = free_;
  Real scale = wavespeed_scale_;

  if (wavespeed_mode_ == "local_dx") {
    par_for("IDCTT::BuildLocalDxWaveSpeed", DevExeSpace(), 0, nmb-1,
            ksg, keg, jsg, jeg, isg, ieg,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dxloc = fmin(size.d_view(m).dx1,
                        fmin(size.d_view(m).dx2, size.d_view(m).dx3));
      free.wavespeed(m,k,j,i) = scale*dxloc/dx_min;
    });
    return;
  }

  Mesh *pmesh = pmy_pack_->pmesh;
  int max_phys_level = std::max(0, pmesh->max_level - pmesh->root_level);
  std::vector<Real> radii(max_phys_level + 1, 0.0);
  for (int gid = 0; gid < pmesh->nmb_total; ++gid) {
    int phys_level = pmesh->lloc_eachmb[gid].level - pmesh->root_level;
    RegionSize rs = MeshBlockRegion(pmesh, gid);
    Real rmax = BlockCornerRadius(rs, wavespeed_center_);
    for (int lev = 0; lev <= phys_level; ++lev) {
      radii[lev] = std::max(radii[lev], rmax);
    }
  }
  for (int lev = 1; lev <= max_phys_level; ++lev) {
    if (radii[lev] <= 0.0) radii[lev] = radii[lev - 1];
  }

  Kokkos::realloc(wavespeed_radii_, max_phys_level + 1);
  for (int lev = 0; lev <= max_phys_level; ++lev) {
    wavespeed_radii_.h_view(lev) = radii[lev];
  }
  wavespeed_radii_.template modify<HostMemSpace>();
  wavespeed_radii_.template sync<DevExeSpace>();

  auto radii_d = wavespeed_radii_.d_view;
  Real center[3] = {wavespeed_center_[0], wavespeed_center_[1], wavespeed_center_[2]};
  par_for("IDCTT::BuildSmoothBoxWaveSpeed", DevExeSpace(), 0, nmb-1,
          ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dxloc = fmin(size.d_view(m).dx1,
                      fmin(size.d_view(m).dx2, size.d_view(m).dx3));
    Real cap = scale*dxloc/dx_min;
    Real x1 = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                          size.d_view(m).x1max) - center[0];
    Real x2 = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                          size.d_view(m).x2max) - center[1];
    Real x3 = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                          size.d_view(m).x3max) - center[2];
    Real r = sqrt(x1*x1 + x2*x2 + x3*x3);
    Real profile = scale;
    if (max_phys_level > 0) {
      if (r <= radii_d(max_phys_level)) {
        profile = scale;
      } else {
        profile = scale*static_cast<Real>(1 << max_phys_level);
        for (int lev = max_phys_level; lev >= 1; --lev) {
          Real r_inner = radii_d(lev);
          Real r_outer = radii_d(lev - 1);
          if (r <= r_outer) {
            Real c_inner = scale*static_cast<Real>(1 << (max_phys_level - lev));
            Real c_outer = scale*static_cast<Real>(1 << (max_phys_level - lev + 1));
            Real width = fmax(r_outer - r_inner, static_cast<Real>(1.0e-20));
            Real s = fmin(static_cast<Real>(1.0),
                          fmax(static_cast<Real>(0.0), (r - r_inner)/width));
            Real smooth = s*s*(static_cast<Real>(3.0) - static_cast<Real>(2.0)*s);
            profile = c_inner + (c_outer - c_inner)*smooth;
            break;
          }
        }
      }
    }
    free.wavespeed(m,k,j,i) = fmin(cap, profile);
  });

  if (global_variable::my_rank == 0) {
    std::cout << "ID CTT relaxation wave speed mode = " << wavespeed_mode_
              << ", levels = " << max_phys_level + 1 << ", radii:";
    for (int lev = 0; lev <= max_phys_level; ++lev) {
      std::cout << " L" << lev << "=" << radii[lev];
    }
    std::cout << std::endl;
  }
}

template <int NGHOST>
void IDConformalThinSandwich::ComputeResidual() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  int is = indcs.is - indcs.ng, ie = indcs.ie + indcs.ng;
  int js = indcs.js - indcs.ng, je = indcs.je + indcs.ng;
  int ks = indcs.ks - indcs.ng, ke = indcs.ke + indcs.ng;
  int nmb = pmy_pack_->nmb_thispack;
  auto relax = relax_;
  auto free = free_;
  par_for("IDCTT::ComputeResidual", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    free.residual(m,k,j,i) = HamiltonianResidual<NGHOST>(relax, free, idx, m, k, j, i);
  });
}

template <int NGHOST>
void IDConformalThinSandwich::RKStep(Real dtau, Real dx_min) {
  (void) dx_min;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  auto u = u_relax;
  auto base = u_relax_tmp;
  auto free = free_;
  Real eta = eta_;

  DeepCopy5D(base, u);
  auto do_stage = [&](Real a_base, Real a_trial) {
    ComputeResidual<NGHOST>();
    par_for("IDCTT::RKStage", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real c = free.wavespeed(m,k,j,i);
      Real u_old = u(m, ID_RELAX_U, k,j,i);
      Real v_old = u(m, ID_RELAX_V, k,j,i);
      Real du = v_old - eta*u_old;
      Real dv = c*c*free.residual(m,k,j,i);
      u(m, ID_RELAX_U, k,j,i) =
          a_base*base(m, ID_RELAX_U, k,j,i) + a_trial*(u_old + dtau*du);
      u(m, ID_RELAX_V, k,j,i) =
          a_base*base(m, ID_RELAX_V, k,j,i) + a_trial*(v_old + dtau*dv);
    });
    InitRecv();
    Restrict();
    Send();
    ClearSend();
    ClearRecv();
    Recv();
    ApplyPhysicalBCs();
    Prolongate();
  };

  do_stage(0.0, 1.0);
  do_stage(0.75, 0.25);
  do_stage(1.0/3.0, 2.0/3.0);
}

void IDConformalThinSandwich::OpenHistory() {
  if (history_file_ != nullptr || global_variable::my_rank != 0) return;
  history_file_ = std::fopen(history_name_.c_str(), "w");
  if (history_file_ != nullptr) {
    std::fprintf(history_file_, "# iter residual_l2 max_update\n");
  }
}

void IDConformalThinSandwich::RecordHistory(int iter, Real residual_l2,
                                            Real max_update) {
  if (global_variable::my_rank != 0) return;
  OpenHistory();
  if (history_file_ == nullptr) return;
  std::fprintf(history_file_, "%d %.16e %.16e\n", iter,
               static_cast<double>(residual_l2), static_cast<double>(max_update));
  std::fflush(history_file_);
}

void IDConformalThinSandwich::SolveRelaxation(Driver *pdriver) {
  BuildCTTFreeData();
  Kokkos::deep_copy(u_relax, 0.0);
  InitRecv();
  Restrict();
  Send();
  ClearSend();
  ClearRecv();
  Recv();
  ApplyPhysicalBCs();
  Prolongate();

  auto &size = pmy_pack_->pmb->mb_size;
  int nmb = pmy_pack_->nmb_thispack;
  Real dx_min = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("IDCTT::DxMin", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb),
  KOKKOS_LAMBDA(int m, Real &min_dx) {
    min_dx = fmin(min_dx, fmin(size.d_view(m).dx1,
                               fmin(size.d_view(m).dx2, size.d_view(m).dx3)));
  }, Kokkos::Min<Real>(dx_min));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &dx_min, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  Real dtau = relax_cfl_*dx_min/std::max(wavespeed_scale_, static_cast<Real>(1.0e-12));
  BuildWaveSpeedProfile(dx_min);

  auto reduce_norms = [&]() {
    Real sum2 = 0.0;
    Real maxabs = 0.0;
    Real maxupd = 0.0;
    auto &indcs = pmy_pack_->pmesh->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    auto free = free_;
    auto u = u_relax;
    Kokkos::parallel_reduce("IDCTT::ResidualNorm",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>, Kokkos::IndexType<int>>(
          {0, ks, js, is}, {nmb, ke + 1, je + 1, ie + 1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    Real &lsum, Real &lmax, Real &lupd) {
        Real r = std::abs(free.residual(m,k,j,i));
        lsum += r*r;
        lmax = fmax(lmax, r);
        lupd = fmax(lupd, std::abs(u(m, ID_RELAX_V, k,j,i)));
      }, Kokkos::Sum<Real>(sum2), Kokkos::Max<Real>(maxabs),
         Kokkos::Max<Real>(maxupd));
    Real vals[3] = {sum2, maxabs, maxupd};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &vals[0], 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &vals[1], 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &vals[2], 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    Real ncell = static_cast<Real>(pmy_pack_->pmesh->nmb_total*
                                   indcs.nx1*indcs.nx2*indcs.nx3);
    vals[0] = std::sqrt(vals[0]/std::max(ncell, static_cast<Real>(1.0)));
    return std::array<Real,3>{vals[0], vals[1], vals[2]};
  };

  Real initial = std::numeric_limits<Real>::max();
  Real best = initial;
  int fd = pmy_pack_->pz4c->opt.fd_stencil;
  for (int iter = 0; iter <= max_steps_; ++iter) {
    if (fd == 2) ComputeResidual<2>();
    else if (fd == 3) ComputeResidual<3>();
    else ComputeResidual<4>();
    auto norms = reduce_norms();
    if (iter == 0) {
      initial = norms[0];
      best = norms[0];
    } else {
      best = std::min(best, norms[0]);
    }
    if (iter % history_every_ == 0 || iter == 0 || norms[0] <= tolerance_) {
      RecordHistory(iter, norms[0], norms[2]);
      if (global_variable::my_rank == 0) {
        std::cout << "ID CTT relaxation iter " << iter
                  << ": residual_l2 = " << norms[0]
                  << ", residual_max = " << norms[1]
                  << ", max_v = " << norms[2] << std::endl;
      }
    }
    if (norms[0] <= tolerance_) break;
    if (iter == max_steps_) break;
    if (fd == 2) RKStep<2>(dtau, dx_min);
    else if (fd == 3) RKStep<3>(dtau, dx_min);
    else RKStep<4>(dtau, dx_min);
  }

  if (reject_worse_) {
    if (fd == 2) ComputeResidual<2>();
    else if (fd == 3) ComputeResidual<3>();
    else ComputeResidual<4>();
    auto final_norms = reduce_norms();
    if (!std::isfinite(final_norms[0]) || final_norms[0] > initial) {
      if (global_variable::my_rank == 0) {
        std::cout << "### WARNING in IDConformalThinSandwich::SolveRelaxation"
                  << std::endl
                  << "Rejecting CTT relaxation result because the residual did not "
                  << "decrease." << std::endl;
      }
      Kokkos::deep_copy(u_relax, 0.0);
    }
  }
  ApplySolution();
}

void IDConformalThinSandwich::ApplySolution() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  auto rel = relax_;
  auto free = free_;
  auto &admvars = pmy_pack_->padm->adm;
  Real x0[3] = {pos_[0][0], pos_[0][1], pos_[0][2]};
  Real x1[3] = {pos_[1][0], pos_[1][1], pos_[1][2]};
  Real p0[3] = {mom_[0][0], mom_[0][1], mom_[0][2]};
  Real p1[3] = {mom_[1][0], mom_[1][1], mom_[1][2]};
  Real s0[3] = {spin_[0][0], spin_[0][1], spin_[0][2]};
  Real s1[3] = {spin_[1][0], spin_[1][1], spin_[1][2]};
  auto &size = pmy_pack_->pmb->mb_size;

  par_for("IDCTT::ApplySolution", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x[3] = {
        CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                    size.d_view(m).x1max),
        CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                    size.d_view(m).x2max),
        CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                    size.d_view(m).x3max)};
    Real psi = std::max(free.psi_singular(m,k,j,i) + rel.u(m,k,j,i),
                        static_cast<Real>(1.0e-10));
    Real psi2 = psi*psi;
    Real psi4 = psi2*psi2;
    admvars.psi4(m,k,j,i) = psi4;
    admvars.alpha(m,k,j,i) = 1.0/psi2;
    for (int a = 0; a < 3; ++a) admvars.beta_u(m,a,k,j,i) = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        admvars.g_dd(m,a,b,k,j,i) = (a == b) ? psi4 : 0.0;
      }
    }

    Real ahat[3][3];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) ahat[a][b] = 0.0;
    AddBowenYorkAhat(x, x0, p0, s0, ahat);
    AddBowenYorkAhat(x, x1, p1, s1, ahat);
    Real invpsi2 = 1.0/psi2;
    for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b)
        admvars.vK_dd(m,a,b,k,j,i) = invpsi2*ahat[a][b];
  });

  int fd = pmy_pack_->pz4c->opt.fd_stencil;
  if (fd == 2) pmy_pack_->pz4c->ADMToZ4c<2>(pmy_pack_, nullptr);
  else if (fd == 3) pmy_pack_->pz4c->ADMToZ4c<3>(pmy_pack_, nullptr);
  else pmy_pack_->pz4c->ADMToZ4c<4>(pmy_pack_, nullptr);
  pmy_pack_->pz4c->Z4cToADM(pmy_pack_);
}

void IDConformalThinSandwich::RefreshZ4cBoundariesAfterSolve(Driver *pdriver) {
  Z4c *pz4c = pmy_pack_->pz4c;
  if (pz4c == nullptr) return;
  (void) pz4c->RestrictU(pdriver, 0);
  (void) pz4c->SendU(pdriver, 0);
  (void) pz4c->ClearSend(pdriver, -1);
  (void) pz4c->ClearRecv(pdriver, -1);
  (void) pz4c->RecvU(pdriver, 0);
  (void) pz4c->Z4cBoundaryRHS(pdriver, 0);
  (void) pz4c->ApplyPhysicalBCs(pdriver, 0);
  (void) pz4c->Prolongate(pdriver, 0);
  if (!stop_after_solve_) {
    (void) pz4c->InitRecv(pdriver, -1);
  }
}

void IDConformalThinSandwich::RecomputeConstraintsAfterSolve() {
  Z4c *pz4c = pmy_pack_->pz4c;
  if (pz4c == nullptr) return;
  pz4c->Z4cToADM(pmy_pack_);
  int fd = pz4c->opt.fd_stencil;
  if (fd == 2) pz4c->ADMConstraints<2>(pmy_pack_);
  else if (fd == 3) pz4c->ADMConstraints<3>(pmy_pack_);
  else pz4c->ADMConstraints<4>(pmy_pack_);
}

template void IDConformalThinSandwich::ComputeResidual<2>();
template void IDConformalThinSandwich::ComputeResidual<3>();
template void IDConformalThinSandwich::ComputeResidual<4>();
template void IDConformalThinSandwich::RKStep<2>(Real, Real);
template void IDConformalThinSandwich::RKStep<3>(Real, Real);
template void IDConformalThinSandwich::RKStep<4>(Real, Real);

} // namespace z4c
