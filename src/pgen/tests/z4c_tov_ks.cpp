//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_tov_ks.cpp
//! \brief TOV-star residual data on top of an analytic Kerr-Schild background

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "utils/tov/tov.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_tabulated.hpp"

namespace {

Real bh_spin = 0.0;
Real bh_center_x1 = 0.0;
Real bh_center_x2 = 0.0;
Real bh_center_x3 = 0.0;
Real bh_horizon_radius = 1.0;
Real star_center_x1 = 8.0;
Real star_center_x2 = 0.0;
Real star_center_x3 = 0.0;
bool star_isotropic = true;
Real excision_damp_rate = 50.0;
bool excision_project_state = true;
Real excision_freeze_radius = 0.0;
Real excision_ramp_radius = 0.0;
Real excision_atmo_density = 0.0;
Real excision_atmo_energy = 0.0;

KOKKOS_INLINE_FUNCTION
Real KerrSchildRadius(Real x, Real y, Real z, Real a) {
  Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  Real discr = SQR(SQR(rad) - SQR(a)) + 4.0*SQR(a)*SQR(z);
  Real r = sqrt((SQR(rad) - SQR(a) + sqrt(discr))/2.0);
  Real eps = 1.0e-6;
  if (r < eps) {
    r = 0.5*(eps + r*r/eps);
  }
  return r;
}

KOKKOS_INLINE_FUNCTION
Real SmootherStep(Real q) {
  q = fmax(0.0, fmin(1.0, q));
  return q*q*q*(10.0 + q*(-15.0 + 6.0*q));
}

KOKKOS_INLINE_FUNCTION
Real InnerExcisionRamp(Real x, Real y, Real z) {
  Real r_ks = KerrSchildRadius(x, y, z, bh_spin);
  if (r_ks <= excision_freeze_radius) {
    return 0.0;
  }
  if (r_ks >= excision_ramp_radius) {
    return 1.0;
  }
  Real width = excision_ramp_radius - excision_freeze_radius;
  if (width <= 0.0) {
    return 1.0;
  }
  return SmootherStep((r_ks - excision_freeze_radius)/width);
}

KOKKOS_INLINE_FUNCTION
Real BlendFiniteToTarget(Real value, Real target, Real ramp) {
  if (!(isfinite(value))) {
    return target;
  }
  return ramp*value + (1.0 - ramp)*target;
}

void ApplyInnerExcision(Mesh *pm, Real bdt, bool project_mhd) {
  if (excision_damp_rate <= 0.0 && !excision_project_state) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  int is = indcs.is;
  int ie = indcs.ie;
  int js = indcs.js;
  int je = indcs.je;
  int ks = indcs.ks;
  int ke = indcs.ke;

  if (project_mhd && pmbp->pmhd != nullptr) {
    auto &mhd_u0 = pmbp->pmhd->u0;
    auto &mhd_u1 = pmbp->pmhd->u1;
    int nmhd = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    int nbase = pmbp->pmhd->nmhd;
    par_for("z4c_tov_ks_inner_excision_mhd", DevExeSpace(), 0, nmb - 1, 0, nmhd - 1,
            ks, ke, js, je, is, ie,
            KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - bh_center_x1;
      Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - bh_center_x2;
      Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - bh_center_x3;
      Real ramp = InnerExcisionRamp(x, y, z);
      if (ramp >= 1.0 && isfinite(mhd_u0(m,n,k,j,i)) && isfinite(mhd_u1(m,n,k,j,i))) {
        return;
      }

      Real target = 0.0;
      if (n == IDN) {
        target = excision_atmo_density;
      } else if (n == IEN && nbase > IEN) {
        target = excision_atmo_energy;
      }

      // Only cell-centered hydrodynamic conserved variables are projected here.
      // Face- and cell-centered magnetic fields are intentionally untouched.
      if (excision_project_state) {
        mhd_u0(m,n,k,j,i) = BlendFiniteToTarget(mhd_u0(m,n,k,j,i), target, ramp);
        mhd_u1(m,n,k,j,i) = BlendFiniteToTarget(mhd_u1(m,n,k,j,i), target, ramp);
      } else if (!(isfinite(mhd_u0(m,n,k,j,i)))) {
        mhd_u0(m,n,k,j,i) = target;
      }
    });
  }

  auto &z4c_u0 = pmbp->pz4c->u0;
  auto &z4c_u1 = pmbp->pz4c->u1;
  auto &z4c_rhs = pmbp->pz4c->u_rhs;
  int nz4c = pmbp->pz4c->nz4c;
  par_for("z4c_tov_ks_inner_excision_z4c", DevExeSpace(), 0, nmb - 1, 0, nz4c - 1,
          ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - bh_center_x1;
    Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - bh_center_x2;
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - bh_center_x3;
    Real ramp = InnerExcisionRamp(x, y, z);
    if (ramp >= 1.0 && isfinite(z4c_rhs(m,n,k,j,i)) && isfinite(z4c_u0(m,n,k,j,i)) &&
        isfinite(z4c_u1(m,n,k,j,i))) {
      return;
    }
    if (excision_damp_rate > 0.0) {
      Real damp = fmax(0.0, 1.0 - bdt*excision_damp_rate*(1.0 - ramp));
      z4c_rhs(m,n,k,j,i) = isfinite(z4c_rhs(m,n,k,j,i)) ?
                            damp*ramp*z4c_rhs(m,n,k,j,i) : 0.0;
    } else {
      z4c_rhs(m,n,k,j,i) = isfinite(z4c_rhs(m,n,k,j,i)) ?
                            ramp*z4c_rhs(m,n,k,j,i) : 0.0;
    }
    if (excision_project_state) {
      z4c_u0(m,n,k,j,i) = isfinite(z4c_u0(m,n,k,j,i)) ?
                           ramp*z4c_u0(m,n,k,j,i) : 0.0;
      z4c_u1(m,n,k,j,i) = isfinite(z4c_u1(m,n,k,j,i)) ?
                           ramp*z4c_u1(m,n,k,j,i) : 0.0;
    }
  });
}

void ApplyInnerExcision(Mesh *pm, Real bdt) {
  ApplyInnerExcision(pm, bdt, true);
}

template <typename ADMState>
void FillFlatADM(MeshBlockPack *pmbp, ADMState &adm_state) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;

  par_for("z4c_tov_ks_flat_adm", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    adm_state.alpha(m,k,j,i) = 1.0;
    adm_state.beta_u(m,0,k,j,i) = 0.0;
    adm_state.beta_u(m,1,k,j,i) = 0.0;
    adm_state.beta_u(m,2,k,j,i) = 0.0;
    adm_state.psi4(m,k,j,i) = 1.0;

    adm_state.g_dd(m,0,0,k,j,i) = 1.0;
    adm_state.g_dd(m,0,1,k,j,i) = 0.0;
    adm_state.g_dd(m,0,2,k,j,i) = 0.0;
    adm_state.g_dd(m,1,1,k,j,i) = 1.0;
    adm_state.g_dd(m,1,2,k,j,i) = 0.0;
    adm_state.g_dd(m,2,2,k,j,i) = 1.0;

    adm_state.vK_dd(m,0,0,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,2,2,k,j,i) = 0.0;
  });
}

template <typename ADMState>
void FillKerrSchildADM(MeshBlockPack *pmbp, ADMState &adm_state) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;

  par_for("z4c_tov_ks_background", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - bh_center_x1;
    Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - bh_center_x2;
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - bh_center_x3;
    Real rad = sqrt(SQR(x) + SQR(y) + SQR(z));
    Real r_ks = KerrSchildRadius(x, y, z, bh_spin);
    if (excision_project_state && r_ks < excision_freeze_radius) {
      Real scale = excision_freeze_radius/fmax(rad, 1.0e-12);
      x = (rad > 1.0e-12) ? x*scale : excision_freeze_radius;
      y = (rad > 1.0e-12) ? y*scale : 0.0;
      z = (rad > 1.0e-12) ? z*scale : 0.0;
    }

    ComputeADMDecomposition(
        x, y, z, false, bh_spin,
        &adm_state.alpha(m,k,j,i),
        &adm_state.beta_u(m,0,k,j,i), &adm_state.beta_u(m,1,k,j,i),
        &adm_state.beta_u(m,2,k,j,i), &adm_state.psi4(m,k,j,i),
        &adm_state.g_dd(m,0,0,k,j,i), &adm_state.g_dd(m,0,1,k,j,i),
        &adm_state.g_dd(m,0,2,k,j,i), &adm_state.g_dd(m,1,1,k,j,i),
        &adm_state.g_dd(m,1,2,k,j,i), &adm_state.g_dd(m,2,2,k,j,i),
        &adm_state.vK_dd(m,0,0,k,j,i), &adm_state.vK_dd(m,0,1,k,j,i),
        &adm_state.vK_dd(m,0,2,k,j,i), &adm_state.vK_dd(m,1,1,k,j,i),
        &adm_state.vK_dd(m,1,2,k,j,i), &adm_state.vK_dd(m,2,2,k,j,i));
  });
}

void SetADMBackgroundKerrSchild(MeshBlockPack *pmbp, Real /*time*/) {
  FillKerrSchildADM(pmbp, pmbp->pz4c->adm_bg);
}

template <class TOVEOS>
void FillTOVPrimitivesAndADM(ParameterInput *pin, Mesh *pmy_mesh, TOVEOS &eos,
                             const tov::TOVStar &tov_star) {
  MeshBlockPack *pmbp = pmy_mesh->pmb_pack;
  auto &w0 = pmbp->pmhd->w0;
  auto &adm_state = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmy_mesh->mb_indcs;

  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;

  Real dfloor = pin->GetOrAddReal("mhd", "dfloor", tov_star.dfloor);
  Real pfloor = pin->GetOrAddReal("mhd", "pfloor", tov_star.pfloor);
  constexpr bool use_ye = tov::UsesYe<TOVEOS>;
  Real ye_atmo = pin->GetOrAddReal("mhd", "s0_atmosphere", 0.5);
  int nvars = pmbp->pmhd->nmhd;
  int nscalars = pmbp->pmhd->nscalars;

  par_for("z4c_tov_ks_star", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - indcs.is, indcs.nx1, x1min, x1max) - star_center_x1;
    Real y = CellCenterX(j - indcs.js, indcs.nx2, x2min, x2max) - star_center_x2;
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, x3min, x3max) - star_center_x3;

    Real r = sqrt(SQR(x) + SQR(y) + SQR(z));
    Real rho, p, mass, alp;
    if (star_isotropic) {
      tov_star.GetPrimitivesAtIsoPoint(eos, r, rho, p, mass, alp);
    } else {
      tov_star.GetPrimitivesAtPoint(eos, r, rho, p, mass, alp);
    }

    w0(m,IDN,k,j,i) = fmax(rho, dfloor);
    w0(m,IPR,k,j,i) = fmax(p, pfloor);
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    if constexpr (use_ye) {
      if (nscalars >= 1) {
        Real ye = ye_atmo;
        if (rho > 0.0) {
          ye = eos.template GetYeFromRho<tov::LocationTag::Device>(rho);
        }
        w0(m,nvars,k,j,i) = ye;
      }
    }

    adm_state.alpha(m,k,j,i) = alp;
    adm_state.beta_u(m,0,k,j,i) = 0.0;
    adm_state.beta_u(m,1,k,j,i) = 0.0;
    adm_state.beta_u(m,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,0,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,0,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,1,k,j,i) = 0.0;
    adm_state.vK_dd(m,1,2,k,j,i) = 0.0;
    adm_state.vK_dd(m,2,2,k,j,i) = 0.0;

    if (star_isotropic) {
      Real r_schw = tov_star.FindSchwarzschildR(r, mass);
      Real fmet = 1.0;
      if (r > 0.0) {
        fmet = r_schw/r;
      }
      Real psi4 = fmet*fmet;
      adm_state.g_dd(m,0,0,k,j,i) = psi4;
      adm_state.g_dd(m,0,1,k,j,i) = 0.0;
      adm_state.g_dd(m,0,2,k,j,i) = 0.0;
      adm_state.g_dd(m,1,1,k,j,i) = psi4;
      adm_state.g_dd(m,1,2,k,j,i) = 0.0;
      adm_state.g_dd(m,2,2,k,j,i) = psi4;
      adm_state.psi4(m,k,j,i) = psi4;
    } else {
      Real fmet = 0.0;
      if (r > 0.0) {
        fmet = (1.0/(1.0 - 2.0*mass/r) - 1.0)/(r*r);
      }
      adm_state.g_dd(m,0,0,k,j,i) = x*x*fmet + 1.0;
      adm_state.g_dd(m,0,1,k,j,i) = x*y*fmet;
      adm_state.g_dd(m,0,2,k,j,i) = x*z*fmet;
      adm_state.g_dd(m,1,1,k,j,i) = y*y*fmet + 1.0;
      adm_state.g_dd(m,1,2,k,j,i) = y*z*fmet;
      adm_state.g_dd(m,2,2,k,j,i) = z*z*fmet + 1.0;
      Real det = adm::SpatialDet(adm_state.g_dd(m,0,0,k,j,i),
                                 adm_state.g_dd(m,0,1,k,j,i),
                                 adm_state.g_dd(m,0,2,k,j,i),
                                 adm_state.g_dd(m,1,1,k,j,i),
                                 adm_state.g_dd(m,1,2,k,j,i),
                                 adm_state.g_dd(m,2,2,k,j,i));
      adm_state.psi4(m,k,j,i) = pow(det, 1.0/3.0);
    }
  });
}

template <int NGHOST>
void ConvertADMToResidualOnBackground(MeshBlockPack *pmbp, ParameterInput *pin) {
  auto *pz4c = pmbp->pz4c;
  pz4c->ADMToZ4c<NGHOST>(pmbp, pin);
  Kokkos::deep_copy(DevExeSpace(), pz4c->u_full, pz4c->u0);

  FillFlatADM(pmbp, pmbp->padm->adm);
  pz4c->ADMToZ4c<NGHOST>(pmbp, pin);
  Kokkos::deep_copy(DevExeSpace(), pz4c->u_bg, pz4c->u0);

  auto &indcs = pmbp->pmesh->mb_indcs;
  int isg = indcs.is - indcs.ng;
  int ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng;
  int jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng;
  int keg = indcs.ke + indcs.ng;
  int nmb = pmbp->nmb_thispack;
  int nz4c = pz4c->nz4c;

  int ialpha = pz4c->I_Z4C_ALPHA;
  int ibetax = pz4c->I_Z4C_BETAX;
  int ibetaz = pz4c->I_Z4C_BETAZ;
  int ibx = pz4c->I_Z4C_BX;
  int ibz = pz4c->I_Z4C_BZ;
  auto &u0 = pz4c->u0;
  auto &u_full = pz4c->u_full;
  auto &u_bg = pz4c->u_bg;

  par_for("z4c_tov_ks_residual", DevExeSpace(), 0, nmb - 1, 0, nz4c - 1,
          ksg, keg, jsg, jeg, isg, ieg,
          KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    u0(m,n,k,j,i) = u_full(m,n,k,j,i) - u_bg(m,n,k,j,i);
    if (n == ialpha || (n >= ibetax && n <= ibetaz) || (n >= ibx && n <= ibz)) {
      u0(m,n,k,j,i) = 0.0;
    }
  });
}

void ZeroMagneticFields(MeshBlockPack *pmbp) {
  if (pmbp->pmhd == nullptr) {
    return;
  }
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->b0.x3f, 0.0);
  Kokkos::deep_copy(DevExeSpace(), pmbp->pmhd->bcc0, 0.0);
}

template <class TOVEOS>
void SetupTOVKerrSchild(ParameterInput *pin, Mesh *pmy_mesh) {
  MeshBlockPack *pmbp = pmy_mesh->pmb_pack;
  TOVEOS eos{pin};
  auto tov_star = tov::TOVStar::ConstructTOV(pin, eos);

  FillTOVPrimitivesAndADM(pin, pmy_mesh, eos, tov_star);

  auto &indcs = pmy_mesh->mb_indcs;
  switch (indcs.ng) {
    case 2:
      ConvertADMToResidualOnBackground<2>(pmbp, pin);
      break;
    case 3:
      ConvertADMToResidualOnBackground<3>(pmbp, pin);
      break;
    case 4:
      ConvertADMToResidualOnBackground<4>(pmbp, pin);
      break;
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unsupported nghost for z4c_tov_ks" << std::endl;
      std::exit(EXIT_FAILURE);
  }

  auto *pz4c = pmbp->pz4c;
  pz4c->SetADMBackground = &SetADMBackgroundKerrSchild;
  pz4c->UpdateBackgroundState(pmy_mesh->time);
  pz4c->ReconstructFullState();
  pz4c->EnforceAlgConstrOn(pz4c->full);
  pz4c->RecastResidualState();
  pz4c->PrescribeGaugeResidual();
  ApplyInnerExcision(pmy_mesh, 0.0, false);
  pz4c->Z4cToADM(pmbp);

  ZeroMagneticFields(pmbp);

  int ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  pmbp->pdyngr->PrimToConInit(0, n1 - 1, 0, n2 - 1, 0, n3 - 1);
  ApplyInnerExcision(pmy_mesh, 0.0);

  switch (indcs.ng) {
    case 2:
      pmbp->pz4c->ADMConstraints<2>(pmbp);
      break;
    case 3:
      pmbp->pz4c->ADMConstraints<3>(pmbp);
      break;
    case 4:
      pmbp->pz4c->ADMConstraints<4>(pmbp);
      break;
  }
}

}  // namespace

void ProblemGenerator::Z4cTovKerrSchild(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic || pmbp->pdyngr == nullptr ||
      pmbp->padm == nullptr || pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks requires dynamical GR, ADM, DynGRMHD, and Z4c"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmbp->pz4c->use_analytic_background) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks requires <z4c>/use_analytic_background = true"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  Real bh_mass = pin->GetOrAddReal("problem", "bh_mass", 1.0);
  if (fabs(bh_mass - 1.0) > 1.0e-12) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks currently assumes bh_mass = 1 because the "
              << "Cartesian Kerr-Schild helper is unit-mass normalized." << std::endl;
    exit(EXIT_FAILURE);
  }

  bh_spin = pin->GetOrAddReal("problem", "bh_spin", 0.0);
  Real coord_spin = pin->GetReal("coord", "a");
  if (fabs(coord_spin - bh_spin) > 1.0e-12) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks requires <coord>/a to match "
              << "<problem>/bh_spin so the coordinate source terms and analytic "
              << "background use the same Kerr-Schild spin." << std::endl;
    exit(EXIT_FAILURE);
  }
  bh_center_x1 = pin->GetOrAddReal("problem", "bh_center_x1", 0.0);
  bh_center_x2 = pin->GetOrAddReal("problem", "bh_center_x2", 0.0);
  bh_center_x3 = pin->GetOrAddReal("problem", "bh_center_x3", 0.0);
  if (fabs(bh_center_x1) > 1.0e-12 || fabs(bh_center_x2) > 1.0e-12 ||
      fabs(bh_center_x3) > 1.0e-12) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks currently requires the BH to remain at the "
              << "grid origin so the coordinate metric and analytic background agree."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  star_center_x1 = pin->GetOrAddReal("problem", "star_center_x1", 8.0);
  star_center_x2 = pin->GetOrAddReal("problem", "star_center_x2", 0.0);
  star_center_x3 = pin->GetOrAddReal("problem", "star_center_x3", 0.0);
  star_isotropic = pin->GetOrAddBoolean("problem", "isotropic", true);
  bh_horizon_radius = 1.0 + sqrt(fmax(0.0, 1.0 - SQR(bh_spin)));
  excision_damp_rate = pin->GetOrAddReal("problem", "excision_damp_rate", 50.0);
  excision_project_state = pin->GetOrAddBoolean("problem", "excision_project_state", true);
  excision_freeze_radius =
      pin->GetOrAddReal("problem", "excision_freeze_radius", 0.85*bh_horizon_radius);
  excision_ramp_radius =
      pin->GetOrAddReal("problem", "excision_ramp_radius", 0.95*bh_horizon_radius);
  if (excision_freeze_radius < 0.0 || excision_ramp_radius < excision_freeze_radius ||
      excision_ramp_radius > bh_horizon_radius) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "z4c_tov_ks requires 0 <= excision_freeze_radius <= "
              << "excision_ramp_radius <= Kerr-Schild horizon radius." << std::endl;
    exit(EXIT_FAILURE);
  }
  Real dfloor = pin->GetOrAddReal("mhd", "dfloor", 1.0e-16);
  Real pfloor = pin->GetOrAddReal("mhd", "pfloor", 1.0e-22);
  Real gamma = pin->GetOrAddReal("mhd", "gamma", 5.0/3.0);
  excision_atmo_density = pin->GetOrAddReal("problem", "excision_atmo_density", dfloor);
  excision_atmo_energy =
      pin->GetOrAddReal("problem", "excision_atmo_energy", pfloor/fmax(gamma - 1.0, 1.0e-12));
  user_srcs = true;
  user_srcs_func = &ApplyInnerExcision;
  pmbp->pz4c->SetADMBackground = &SetADMBackgroundKerrSchild;

  if (restart) {
    pmbp->pz4c->UpdateBackgroundState(pmy_mesh_->time);
    pmbp->pz4c->ReconstructFullState();
    pmbp->pz4c->EnforceAlgConstrOn(pmbp->pz4c->full);
    pmbp->pz4c->RecastResidualState();
    pmbp->pz4c->PrescribeGaugeResidual();
    ApplyInnerExcision(pmy_mesh_, 0.0, false);
    pmbp->pz4c->Z4cToADM(pmbp);
    ApplyInnerExcision(pmy_mesh_, 0.0);
    return;
  }

  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupTOVKerrSchild<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupTOVKerrSchild<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
    SetupTOVKerrSchild<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupTOVKerrSchild<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unsupported EOS policy for z4c_tov_ks" << std::endl;
    exit(EXIT_FAILURE);
  }
}
