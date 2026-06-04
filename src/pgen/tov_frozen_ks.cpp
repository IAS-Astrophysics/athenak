//========================================================================================
// AthenaXXX astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tov_frozen_ks.cpp
//! \brief TOV fluid on a frozen analytic Kerr-Schild ADM background.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "pgen/pgen.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "mhd/mhd.hpp"
#include "utils/tov/tov.hpp"
#include "utils/tov/tov_polytrope.hpp"

namespace {

Real star_center_x1 = 20.0;
Real star_center_x2 = 0.0;
Real star_center_x3 = 0.0;
Real star_boost_x = 0.0;
Real star_boost_y = 0.0;
Real star_boost_z = 0.0;
bool star_isotropic = true;
bool amr_rho_slope_refine = true;
Real amr_rho_slope_threshold = 0.25;
Real amr_rho_min = 1.0e-14;
bool amr_star_refine = false;
Real amr_star_refine_radius = 0.0;
int amr_star_refine_level = -1;

template <class TOVEOS>
KOKKOS_INLINE_FUNCTION
void SampleIsotropicTOV(const TOVEOS &eos, const tov::TOVStar &tov_star, Real r,
                        Real &rho, Real &p) {
  Real mass, alp;
  tov_star.GetPrimitivesAtIsoPoint(eos, r, rho, p, mass, alp);
}

bool TouchesGlobalBoundary(const RegionSize &mb_size, const RegionSize &mesh_size,
                           bool multi_d, bool three_d) {
  const Real x1scale = fmax(fabs(mesh_size.x1min), fabs(mesh_size.x1max));
  const Real x2scale = fmax(fabs(mesh_size.x2min), fabs(mesh_size.x2max));
  const Real x3scale = fmax(fabs(mesh_size.x3min), fabs(mesh_size.x3max));
  const Real tol1 = 10.0*std::numeric_limits<Real>::epsilon()*fmax(x1scale, 1.0);
  const Real tol2 = 10.0*std::numeric_limits<Real>::epsilon()*fmax(x2scale, 1.0);
  const Real tol3 = 10.0*std::numeric_limits<Real>::epsilon()*fmax(x3scale, 1.0);
  bool touches = (mb_size.x1min <= mesh_size.x1min + tol1) ||
                 (mb_size.x1max >= mesh_size.x1max - tol1);
  if (multi_d) {
    touches = touches || (mb_size.x2min <= mesh_size.x2min + tol2) ||
              (mb_size.x2max >= mesh_size.x2max - tol2);
  }
  if (three_d) {
    touches = touches || (mb_size.x3min <= mesh_size.x3min + tol3) ||
              (mb_size.x3max >= mesh_size.x3max - tol3);
  }
  return touches;
}

void TOVFrozenKerrSchildHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &w0 = pmbp->pmhd->w0;
  auto &adm = pmbp->padm->adm;
  const int nmb = pmbp->nmb_thispack;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ni = ie - is + 1;
  const int nj = je - js + 1;
  const int nk = ke - ks + 1;
  const int nji = nj*ni;
  const int nkji = nk*nji;
  const int nmkji = nmb*nkji;

  Real rho_max = -std::numeric_limits<Real>::max();
  Real alpha_min = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce(
      "TOVFrozenKerrSchildHistory", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &rho_local, Real &alpha_local) {
        int m = idx/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/ni;
        int i = idx - m*nkji - k*nji - j*ni;
        k += ks;
        j += js;
        i += is;
        rho_local = fmax(rho_local, w0(m,IDN,k,j,i));
        alpha_local = fmin(alpha_local, adm.alpha(m,k,j,i));
      }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

void RefinementCondition(MeshBlockPack *pmbp) {
  if (pmbp->pmhd == nullptr || pmbp->pmesh->pmr == nullptr) return;

  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  const int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nji = nx2*nx1;
  const int nkji = nx3*nji;
  const int nmb = pmbp->nmb_thispack;
  const bool multi_d = pmbp->pmesh->multi_d;
  const bool three_d = pmbp->pmesh->three_d;
  auto &w0 = pmbp->pmhd->w0;
  auto &size = pmbp->pmb->mb_size;
  const bool rho_slope_refine_l = amr_rho_slope_refine;
  const Real rho_slope_threshold_l = amr_rho_slope_threshold;
  const Real rho_min_l = amr_rho_min;

  par_for_outer("tov_frozen_ks_rho_gradient_refinement", DevExeSpace(), 0, 0,
                0, nmb - 1, KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    int &flag = refine_flag.d_view(m + mbs);
    if (!rho_slope_refine_l) {
      if (flag == 0) flag = -1;
      return;
    }

    Real team_dqmax = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
    [=](const int idx, Real &dqmax) {
      int k = idx/nji;
      int j = (idx - k*nji)/nx1;
      int i = idx - k*nji - j*nx1;
      k += ks;
      j += js;
      i += is;
      const Real rho = w0(m,IDN,k,j,i);
      if (!isfinite(rho) || rho <= rho_min_l) return;
      Real d2 = SQR(w0(m,IDN,k,j,i+1) - w0(m,IDN,k,j,i-1));
      if (multi_d) d2 += SQR(w0(m,IDN,k,j+1,i) - w0(m,IDN,k,j-1,i));
      if (three_d) d2 += SQR(w0(m,IDN,k+1,j,i) - w0(m,IDN,k-1,j,i));
      dqmax = fmax(0.5*sqrt(d2)/fmax(rho, rho_min_l), dqmax);
    }, Kokkos::Max<Real>(team_dqmax));

    if (team_dqmax > rho_slope_threshold_l) {
      flag = 1;
    } else if (flag == 0) {
      flag = -1;
    }
  });
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();

  if (amr_star_refine && amr_star_refine_radius > 0.0 && amr_star_refine_level >= 0) {
    const Real refine_radius2 = SQR(amr_star_refine_radius);
    for (int m = 0; m < nmb; ++m) {
      const int level = pmbp->pmesh->lloc_eachmb[m + mbs].level - pmbp->pmesh->root_level;
      const auto &mb = size.h_view(m);
      const Real closest_x = std::max(mb.x1min, std::min(star_center_x1, mb.x1max));
      const Real closest_y = std::max(mb.x2min, std::min(star_center_x2, mb.x2max));
      const Real closest_z = std::max(mb.x3min, std::min(star_center_x3, mb.x3max));
      const Real dmin2 = SQR(closest_x - star_center_x1) +
                         SQR(closest_y - star_center_x2) +
                         SQR(closest_z - star_center_x3);
      if (dmin2 < refine_radius2 && level < amr_star_refine_level) {
        refine_flag.h_view(m + mbs) = 1;
      }
    }
    refine_flag.template modify<HostMemSpace>();
    refine_flag.template sync<DevExeSpace>();
  }

  const auto &mesh_size = pmbp->pmesh->mesh_size;
  for (int m = 0; m < nmb; ++m) {
    if (refine_flag.h_view(m + mbs) > 0 &&
        TouchesGlobalBoundary(size.h_view(m), mesh_size, multi_d, three_d)) {
      refine_flag.h_view(m + mbs) = 0;
    }
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

template <class TOVEOS>
void SetupTOVFluidOnFrozenKS(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  TOVEOS eos{pin};
  auto tov_star = tov::TOVStar::ConstructTOV(pin, eos);

  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = pmbp->pmhd->w0;
  const int nmb = pmbp->nmb_thispack;
  const int isg = indcs.is - indcs.ng;
  const int ieg = indcs.ie + indcs.ng;
  const int jsg = indcs.js - indcs.ng;
  const int jeg = indcs.je + indcs.ng;
  const int ksg = indcs.ks - indcs.ng;
  const int keg = indcs.ke + indcs.ng;
  const Real dfloor = pin->GetOrAddReal("mhd", "dfloor", tov_star.dfloor);
  const Real pfloor = pin->GetOrAddReal("mhd", "pfloor", tov_star.pfloor);
  const Real boost_x = star_boost_x;
  const Real boost_y = star_boost_y;
  const Real boost_z = star_boost_z;
  const Real boost2 = SQR(boost_x) + SQR(boost_y) + SQR(boost_z);
  const Real lorentz = 1.0/std::sqrt(fmax(1.0e-16, 1.0 - boost2));
  const bool isotropic = star_isotropic;
  const Real scx1 = star_center_x1;
  const Real scx2 = star_center_x2;
  const Real scx3 = star_center_x3;
  auto &eos_ = eos;
  auto &tov_star_ = tov_star;
  const int nvars = pmbp->pmhd->nmhd;
  const int nscalars = pmbp->pmhd->nscalars;
  const Real ye_atmo = pin->GetOrAddReal("mhd", "s0_atmosphere", 0.5);

  par_for("tov_frozen_ks_star", DevExeSpace(), 0, nmb - 1, ksg, keg, jsg, jeg,
          isg, ieg, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    const Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - scx1;
    const Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - scx2;
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - scx3;
    const Real r = sqrt(SQR(x) + SQR(y) + SQR(z));

    Real rho, p;
    if (isotropic) {
      SampleIsotropicTOV(eos_, tov_star_, r, rho, p);
    } else {
      Real mass, alp;
      tov_star_.GetPrimitivesAtPoint(eos_, r, rho, p, mass, alp);
    }

    w0(m,IDN,k,j,i) = fmax(rho, dfloor);
    w0(m,IPR,k,j,i) = fmax(p, pfloor);
    w0(m,IVX,k,j,i) = lorentz*boost_x;
    w0(m,IVY,k,j,i) = lorentz*boost_y;
    w0(m,IVZ,k,j,i) = lorentz*boost_z;
    for (int n = 0; n < nscalars; ++n) {
      w0(m,nvars+n,k,j,i) = ye_atmo;
    }
  });

  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->b0.x3f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->bcc0, 0.0);

  pmbp->padm->SetADMVariables(pmbp);
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
  pmbp->pdyngr->PrimToConInit(0, n1 - 1, 0, n2 - 1, 0, n3 - 1);
}

} // namespace

void ProblemGenerator::TOVFrozenKerrSchild(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic || pmbp->pmhd == nullptr ||
      pmbp->pdyngr == nullptr || pmbp->padm == nullptr || pmbp->pz4c != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "tov_frozen_ks requires <mhd> and <adm>, but no <z4c>."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(pmbp->padm->is_dynamic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "tov_frozen_ks requires <adm>/dynamic = true so "
              << "analytic ADM fields are refreshed during evolution." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pdyngr->eos_policy != DynGRMHD_EOS::eos_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "tov_frozen_ks currently supports <mhd>/dyn_eos = ideal."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  user_hist_func = &TOVFrozenKerrSchildHistory;
  user_ref_func = &RefinementCondition;
  pmbp->padm->SetADMVariables = &adm::ADM::SetADMVariablesToKerrSchild;

  star_center_x1 = pin->GetOrAddReal("problem", "star_center_x1", 20.0);
  star_center_x2 = pin->GetOrAddReal("problem", "star_center_x2", 0.0);
  star_center_x3 = pin->GetOrAddReal("problem", "star_center_x3", 0.0);
  star_boost_x = pin->GetOrAddReal("problem", "star_boost_x", 0.0);
  star_boost_y = pin->GetOrAddReal("problem", "star_boost_y", 0.0);
  star_boost_z = pin->GetOrAddReal("problem", "star_boost_z", 0.0);
  star_isotropic = pin->GetOrAddBoolean("problem", "isotropic", true);
  const Real boost2 = SQR(star_boost_x) + SQR(star_boost_y) + SQR(star_boost_z);
  if (boost2 >= 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "tov_frozen_ks requires star boost speed < 1."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  amr_rho_slope_refine = pin->GetOrAddBoolean("problem", "amr_rho_slope_refine", true);
  amr_rho_slope_threshold =
      pin->GetOrAddReal("problem", "amr_rho_slope_threshold", 0.25);
  amr_rho_min = pin->GetOrAddReal("problem", "amr_rho_min", 1.0e-14);
  amr_star_refine = pin->GetOrAddBoolean("problem", "amr_star_refine", false);
  amr_star_refine_radius = pin->GetOrAddReal("problem", "amr_star_refine_radius", 0.0);
  amr_star_refine_level = pin->GetOrAddInteger("problem", "amr_star_refine_level", -1);

  if (restart) {
    pmbp->padm->SetADMVariables(pmbp);
    return;
  }

  SetupTOVFluidOnFrozenKS<tov::PolytropeEOS>(pin, pmy_mesh_);
}

#if USER_PROBLEM_ENABLED
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  TOVFrozenKerrSchild(pin, restart);
}
#endif
