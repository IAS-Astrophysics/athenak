//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dynbbh.cpp
//! \brief Problem generator for superimposed Kerr-Schild black holes

#include <math.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <iostream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"

#define h 5e-5
#define D2(comp, h) ((met_p1.g).comp - (met_m1.g).comp) / (2*h)

namespace {

enum {
  TT, XX, YY, ZZ, NDIM
};

enum {
  X1, Y1, Z1, X2, Y2, Z2,
  VX1, VY1, VZ1, VX2, VY2, VZ2,
  AX1, AY1, AZ1, AX2, AY2, AZ2,
  M1T, M2T, NTRAJ
};

struct dd_sym {
  Real tt;
  Real tx;
  Real ty;
  Real tz;
  Real xx;
  Real xy;
  Real xz;
  Real yy;
  Real yz;
  Real zz;
};

struct four_metric {
  struct dd_sym g;
  struct dd_sym g_t;
  struct dd_sym g_x;
  struct dd_sym g_y;
  struct dd_sym g_z;
};

struct three_metric {
  Real gxx;
  Real gxy;
  Real gxz;
  Real gyy;
  Real gyz;
  Real gzz;
  Real alpha;
  Real betax;
  Real betay;
  Real betaz;
  Real kxx;
  Real kxy;
  Real kxz;
  Real kyy;
  Real kyz;
  Real kzz;
};

struct bbh_pgen {
  Real sep;
  Real om;
  Real q;
  Real a1, a2;
  Real th_a1, th_a2;
  Real ph_a1, ph_a2;
  Real dfloor;
  Real pfloor;
  Real gamma_adi;
  Real a1_buffer, a2_buffer;
  Real adjust_mass1, adjust_mass2;
  Real cutoff_floor;
  Real alpha_thr;
  Real radius_thr;
};

struct bbh_pgen bbh;

/* Declare functions */
void find_traj_t(Real tt, Real traj_array[NTRAJ]);

KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
    const Real z, struct four_metric &outmet,
    const Real nz_m1[NTRAJ], const Real nz_0[NTRAJ], const Real nz_p1[NTRAJ],
    const bbh_pgen& bbh_);
KOKKOS_INLINE_FUNCTION
int four_metric_to_three_metric(const struct four_metric &met, struct three_metric &gam);
KOKKOS_INLINE_FUNCTION
void get_metric(const Real t, const Real x, const Real y, const Real z,
                struct four_metric &met, const Real bbh_traj_loc[NTRAJ],
                const bbh_pgen& bbh_);
KOKKOS_INLINE_FUNCTION
void SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
                   Real gcov[][NDIM], const Real traj_array[NTRAJ], const bbh_pgen& bbh_);
void SetADMVariablesToBBH(MeshBlockPack *pmbp);
void RefineAlphaMin(MeshBlockPack* pmbp);
void RefineTracker(MeshBlockPack* pmbp);

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::ShockTube_()
//! \brief Problem Generator for the shock tube (Riemann problem) tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic &&
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "BBH problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string amr_cond = pin->GetOrAddString("problem", "amr_condition", "track");
  if (amr_cond == "alpha_min") {
    user_ref_func = RefineAlphaMin;
  } else {
    user_ref_func = RefineTracker;
  }
  pmbp->padm->SetADMVariables = &SetADMVariablesToBBH;

  if (restart) return;

  bbh.sep = pin->GetOrAddReal("problem", "sep", 20.0);
  bbh.om = std::pow(bbh.sep, -1.5);
  bbh.q = pin->GetOrAddReal("problem", "q", 1.0);
  bbh.a1 = pin->GetOrAddReal("problem", "a1", 0.0);
  bbh.a2 = pin->GetOrAddReal("problem", "a2", 0.0);
  bbh.th_a1 = pin->GetOrAddReal("problem", "th_a1", 0.0);
  bbh.th_a2 = pin->GetOrAddReal("problem", "th_a2", 0.0);
  bbh.ph_a1 = pin->GetOrAddReal("problem", "ph_a1", 0.0);
  bbh.ph_a2 = pin->GetOrAddReal("problem", "ph_a2", 0.0);
  bbh.dfloor = pin->GetOrAddReal("problem", "dfloor", (FLT_MIN));
  bbh.pfloor = pin->GetOrAddReal("problem", "pfloor", (FLT_MIN));
  bbh.adjust_mass1 = pin->GetOrAddReal("problem", "adjust_mass1", 1.0);
  bbh.adjust_mass2 = pin->GetOrAddReal("problem", "adjust_mass2", 1.0);
  bbh.a1_buffer = pin->GetOrAddReal("problem", "a1_buffer", 0.0);
  bbh.a2_buffer = pin->GetOrAddReal("problem", "a2_buffer", 0.0);
  bbh.cutoff_floor = pin->GetOrAddReal("problem", "cutoff_floor", 1e-10);
  bbh.alpha_thr = pin->GetOrAddReal("problem", "alpha_thr", 0.6);
  bbh.radius_thr = pin->GetOrAddReal("problem", "radius_thr", 6.0);

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int nmb = pmbp->nmb_thispack;
  auto &bbh_ = bbh;

  if (pmbp->phydro != nullptr) {
    auto &eos = pmbp->phydro->peos->eos_data;
    auto &w0 = pmbp->phydro->w0;
    auto &nscal = pmbp->phydro->nscalars;
    par_for("pgen_hydro", DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      w0(m,IDN,k,j,i) = bbh_.dfloor;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IPR,k,j,i) = bbh_.pfloor; //bbh.fluid.pfloor;
      for (int r=0; r<nscal; ++r) {
        w0(m,IYF+r,k,j,i) = 0.0;
      }
    });

    // Convert primitives to conserved
    auto &u0 = pmbp->phydro->u0;
    if (pmbp->padm == nullptr) {
      pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);
    }
  } // End initialization of Hydro variables

  // Initialize MHD variables -------------------------------
  if (pmbp->pmhd != nullptr) {
    auto &eos = pmbp->pmhd->peos->eos_data;
    auto &w0 = pmbp->pmhd->w0;
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    auto &nscal = pmbp->pmhd->nscalars;
    par_for("pgen_shock1", DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      w0(m,IDN,k,j,i) = bbh_.dfloor;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IPR,k,j,i) = bbh_.pfloor; //bbh.fluid.pfloor;
      for (int r=0; r<nscal; ++r) {
        w0(m,IYF+r,k,j,i) = 0.0;
      }
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;
      bcc0(m,IBX,k,j,i) = 0.0;
      bcc0(m,IBY,k,j,i) = 0.0;
      bcc0(m,IBZ,k,j,i) = 0.0;
    });
    // Convert primitives to conserved
    auto &u0 = pmbp->pmhd->u0;
    if (!pmbp->pcoord->is_dynamical_relativistic) {
      pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);
    }
  } // End initialization of MHD variables

  // Initialize ADM variables -------------------------------
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
    // If we're using the ADM variables, then we've got dynamic GR enabled.
    // Because we need the metric, we can't initialize the conserved variables
    // until we've filled out the ADM variables.
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }
  return;
}

namespace {

void SetADMVariablesToBBH(MeshBlockPack *pmbp) {
  const Real tt = pmbp->pmesh->time;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  Real bbh_traj_p1[NTRAJ];
  Real bbh_traj_0[NTRAJ];
  Real bbh_traj_m1[NTRAJ];
  auto& bbh_ = bbh;

  /* Load trajectories */

  /* Whether we load traj from a table or we compute analytical trajectories */
  find_traj_t(tt+h, bbh_traj_p1);
  find_traj_t(tt, bbh_traj_0);
  find_traj_t(tt-h, bbh_traj_m1);


  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    struct four_metric met4;
    struct three_metric met3;
    numerical_4metric(tt, x1v, x2v, x3v, met4, bbh_traj_m1, bbh_traj_0, bbh_traj_p1,
                      bbh_);

    /* Transform 4D metric to 3+1 variables*/
    four_metric_to_three_metric(met4, met3);

    /* Load (Cartesian) components of the metric and curvature */
    // g_ab
    adm.g_dd(m,0,0,k,j,i) = met3.gxx;
    adm.g_dd(m,0,1,k,j,i) = met3.gxy;
    adm.g_dd(m,0,2,k,j,i) = met3.gxz;
    adm.g_dd(m,1,1,k,j,i) = met3.gyy;
    adm.g_dd(m,1,2,k,j,i) = met3.gyz;
    adm.g_dd(m,2,2,k,j,i) = met3.gzz;

    adm.vK_dd(m,0,0,k,j,i) = met3.kxx;
    adm.vK_dd(m,0,1,k,j,i) = met3.kxy;
    adm.vK_dd(m,0,2,k,j,i) = met3.kxz;
    adm.vK_dd(m,1,1,k,j,i) = met3.kyy;
    adm.vK_dd(m,1,2,k,j,i) = met3.kyz;
    adm.vK_dd(m,2,2,k,j,i) = met3.kzz;

    adm.alpha(m,k,j,i) = met3.alpha;
    adm.beta_u(m,0,k,j,i) = met3.betax;
    adm.beta_u(m,1,k,j,i) = met3.betay;
    adm.beta_u(m,2,k,j,i) = met3.betaz;
  });
  return;
}

KOKKOS_INLINE_FUNCTION
void numerical_4metric(const Real t, const Real x, const Real y,
    const Real z, struct four_metric &outmet,
    const Real nz_m1[NTRAJ], const Real nz_0[NTRAJ], const Real nz_p1[NTRAJ],
    const bbh_pgen& bbh_) {
  struct four_metric met_m1;
  struct four_metric met_p1;

  // Time
  get_metric(t-1*h, x, y, z, met_m1, nz_m1, bbh_);
  get_metric(t+1*h, x, y, z, met_p1, nz_p1, bbh_);
  get_metric(t, x, y, z, outmet, nz_0, bbh_);

  outmet.g_t.tt = D2(tt, h);
  outmet.g_t.tx = D2(tx, h);
  outmet.g_t.ty = D2(ty, h);
  outmet.g_t.tz = D2(tz, h);
  outmet.g_t.xx = D2(xx, h);
  outmet.g_t.xy = D2(xy, h);
  outmet.g_t.xz = D2(xz, h);
  outmet.g_t.yy = D2(yy, h);
  outmet.g_t.yz = D2(yz, h);
  outmet.g_t.zz = D2(zz, h);

  // X
  get_metric(t, x-1*h, y, z, met_m1, nz_0, bbh_);
  get_metric(t, x+1*h, y, z, met_p1, nz_0, bbh_);

  outmet.g_x.tt = D2(tt, h);
  outmet.g_x.tx = D2(tx, h);
  outmet.g_x.ty = D2(ty, h);
  outmet.g_x.tz = D2(tz, h);
  outmet.g_x.xx = D2(xx, h);
  outmet.g_x.xy = D2(xy, h);
  outmet.g_x.xz = D2(xz, h);
  outmet.g_x.yy = D2(yy, h);
  outmet.g_x.yz = D2(yz, h);
  outmet.g_x.zz = D2(zz, h);

  // Y
  get_metric(t, x, y-1*h, z, met_m1, nz_0, bbh_);
  get_metric(t, x, y+1*h, z, met_p1, nz_0, bbh_);

  outmet.g_y.tt = D2(tt, h);
  outmet.g_y.tx = D2(tx, h);
  outmet.g_y.ty = D2(ty, h);
  outmet.g_y.tz = D2(tz, h);
  outmet.g_y.xx = D2(xx, h);
  outmet.g_y.xy = D2(xy, h);
  outmet.g_y.xz = D2(xz, h);
  outmet.g_y.yy = D2(yy, h);
  outmet.g_y.yz = D2(yz, h);
  outmet.g_y.zz = D2(zz, h);

  // Z
  get_metric(t, x, y, z-1*h, met_m1, nz_0, bbh_);
  get_metric(t, x, y, z+1*h, met_p1, nz_0, bbh_);

  outmet.g_z.tt = D2(tt, h);
  outmet.g_z.tx = D2(tx, h);
  outmet.g_z.ty = D2(ty, h);
  outmet.g_z.tz = D2(tz, h);
  outmet.g_z.xx = D2(xx, h);
  outmet.g_z.xy = D2(xy, h);
  outmet.g_z.xz = D2(xz, h);
  outmet.g_z.yy = D2(yy, h);
  outmet.g_z.yz = D2(yz, h);
  outmet.g_z.zz = D2(zz, h);

  return;
}

KOKKOS_INLINE_FUNCTION
int four_metric_to_three_metric(const struct four_metric &met,
                                struct three_metric &gam) {
  /* Check determinant first */
  gam.gxx = met.g.xx;
  gam.gxy = met.g.xy;
  gam.gxz = met.g.xz;
  gam.gyy = met.g.yy;
  gam.gyz = met.g.yz;
  gam.gzz = met.g.zz;

  Real det = adm::SpatialDet(gam.gxx, gam.gxy, gam.gxz,
                                   gam.gyy, gam.gyz, gam.gzz);

  /* If determinant is not >0  something is wrong with the metric */
  /* This could occur during the transition to merger at certain points
     so here we restart to Minkowski */
  if (!(det > 0)) {
    //std::fprintf(stderr, "det < 0: %e\n", det);
    //std::fprintf(stderr, "%e %e %e\n", gam.gxx, gam.gxy, gam.gxz);
    //std::fprintf(stderr, "%e %e %e\n", gam.gyy, gam.gyz, gam.gzz);
    //std::fflush(stderr);
    Kokkos::printf("det < 0: %e\n" // NOLINT
                   "%e %e %e\n"
                   "%e %e %e\n",
                   det, gam.gxx, gam.gxy, gam.gxz, gam.gyy, gam.gyz, gam.gzz);
    det = 1.0;
    gam.gxx = 1.0;
    gam.gxy = 0.0;
    gam.gxz = 0.0;
    gam.gyy = 1.0;
    gam.gyz = 0.0;
    gam.gzz = 1.0;
    Real betadownx = 0.0;
    Real betadowny = 0.0;
    Real betadownz = 0.0;

    Real dbetadownxx = 0.0;
    Real dbetadownyx = 0.0;
    Real dbetadownzx = 0.0;

    Real dbetadownxy = 0.0;
    Real dbetadownyy = 0.0;
    Real dbetadownzy = 0.0;

    Real dbetadownxz = 0.0;
    Real dbetadownyz = 0.0;
    Real dbetadownzz = 0.0;

    Real dtgxx = 0.0;
    Real dtgxy = 0.0;
    Real dtgxz = 0.0;
    Real dtgyy = 0.0;
    Real dtgyz = 0.0;
    Real dtgzz = 0.0;

    Real dgxxx = 0.0;
    Real dgxyx = 0.0;
    Real dgxzx = 0.0;
    Real dgyyx = 0.0;
    Real dgyzx = 0.0;
    Real dgzzx = 0.0;

    Real dgxxy = 0.0;
    Real dgxyy = 0.0;
    Real dgxzy = 0.0;
    Real dgyyy = 0.0;
    Real dgyzy = 0.0;
    Real dgzzy = 0.0;

    Real dgxxz = 0.0;
    Real dgxyz = 0.0;
    Real dgxzz = 0.0;
    Real dgyyz = 0.0;
    Real dgyzz = 0.0;
    Real dgzzz = 0.0;

    Real idetgxx = -gam.gyz * gam.gyz + gam.gyy * gam.gzz;
    Real idetgxy = gam.gxz * gam.gyz - gam.gxy * gam.gzz;
    Real idetgxz = -(gam.gxz * gam.gyy) + gam.gxy * gam.gyz;
    Real idetgyy = -gam.gxz * gam.gxz + gam.gxx * gam.gzz;
    Real idetgyz = gam.gxy * gam.gxz - gam.gxx * gam.gyz;
    Real idetgzz = -gam.gxy * gam.gxy + gam.gxx * gam.gyy;
    Real invgxx = idetgxx / det;
    Real invgxy = idetgxy / det;
    Real invgxz = idetgxz / det;
    Real invgyy = idetgyy / det;
    Real invgyz = idetgyz / det;
    Real invgzz = idetgzz / det;

    gam.betax = 0.0;
    gam.betay = 0.0;
    gam.betaz = 0.0;

    gam.alpha = 1.0;
    gam.kxx = 0.0;
    gam.kxy = 0.0;
    gam.kxz = 0.0;
    gam.kyy = 0.0;
    gam.kyz = 0.0;
    gam.kzz = 0.0;

  } else {
    /* Compute components if detg is not <0 */
    Real betadownx = met.g.tx;
    Real betadowny = met.g.ty;
    Real betadownz = met.g.tz;

    Real dbetadownxx = met.g_x.tx;
    Real dbetadownyx = met.g_x.ty;
    Real dbetadownzx = met.g_x.tz;

    Real dbetadownxy = met.g_y.tx;
    Real dbetadownyy = met.g_y.ty;
    Real dbetadownzy = met.g_y.tz;

    Real dbetadownxz = met.g_z.tx;
    Real dbetadownyz = met.g_z.ty;
    Real dbetadownzz = met.g_z.tz;

    Real dtgxx = met.g_t.xx;
    Real dtgxy = met.g_t.xy;
    Real dtgxz = met.g_t.xz;
    Real dtgyy = met.g_t.yy;
    Real dtgyz = met.g_t.yz;
    Real dtgzz = met.g_t.zz;

    Real dgxxx = met.g_x.xx;
    Real dgxyx = met.g_x.xy;
    Real dgxzx = met.g_x.xz;
    Real dgyyx = met.g_x.yy;
    Real dgyzx = met.g_x.yz;
    Real dgzzx = met.g_x.zz;

    Real dgxxy = met.g_y.xx;
    Real dgxyy = met.g_y.xy;
    Real dgxzy = met.g_y.xz;
    Real dgyyy = met.g_y.yy;
    Real dgyzy = met.g_y.yz;
    Real dgzzy = met.g_y.zz;

    Real dgxxz = met.g_z.xx;
    Real dgxyz = met.g_z.xy;
    Real dgxzz = met.g_z.xz;
    Real dgyyz = met.g_z.yy;
    Real dgyzz = met.g_z.yz;
    Real dgzzz = met.g_z.zz;

    Real idetgxx = -gam.gyz * gam.gyz + gam.gyy * gam.gzz;
    Real idetgxy = gam.gxz * gam.gyz - gam.gxy * gam.gzz;
    Real idetgxz = -(gam.gxz * gam.gyy) + gam.gxy * gam.gyz;
    Real idetgyy = -gam.gxz * gam.gxz + gam.gxx * gam.gzz;
    Real idetgyz = gam.gxy * gam.gxz - gam.gxx * gam.gyz;
    Real idetgzz = -gam.gxy * gam.gxy + gam.gxx * gam.gyy;

    Real invgxx = idetgxx / det;
    Real invgxy = idetgxy / det;
    Real invgxz = idetgxz / det;
    Real invgyy = idetgyy / det;
    Real invgyz = idetgyz / det;
    Real invgzz = idetgzz / det;

    gam.betax =
      betadownx * invgxx + betadowny * invgxy + betadownz * invgxz;

    gam.betay =
      betadownx * invgxy + betadowny * invgyy + betadownz * invgyz;

    gam.betaz =
      betadownx * invgxz + betadowny * invgyz + betadownz * invgzz;

    Real b2 =
      betadownx * gam.betax + betadowny * gam.betay +
      betadownz * gam.betaz;


    gam.alpha = sqrt(fabs(b2 - met.g.tt));

    gam.kxx = -(-2 * dbetadownxx - gam.betax * dgxxx - gam.betay * dgxxy -
      gam.betaz * dgxxz + 2 * (gam.betax * dgxxx + gam.betay * dgxyx +
        gam.betaz * dgxzx) + dtgxx) / (2. * gam.alpha);

    gam.kxy = -(-dbetadownxy - dbetadownyx + gam.betax * dgxxy -
      gam.betaz * dgxyz + gam.betaz * dgxzy + gam.betay * dgyyx +
      gam.betaz * dgyzx + dtgxy) / (2. * gam.alpha);

    gam.kxz = -(-dbetadownxz - dbetadownzx + gam.betax * dgxxz +
      gam.betay * dgxyz - gam.betay * dgxzy + gam.betay * dgyzx +
      gam.betaz * dgzzx + dtgxz) / (2. * gam.alpha);

    gam.kyy = -(-2 * dbetadownyy - gam.betax * dgyyx - gam.betay * dgyyy -
      gam.betaz * dgyyz + 2 * (gam.betax * dgxyy + gam.betay * dgyyy +
        gam.betaz * dgyzy) + dtgyy) / (2. * gam.alpha);

    gam.kyz = -(-dbetadownyz - dbetadownzy + gam.betax * dgxyz +
      gam.betax * dgxzy + gam.betay * dgyyz - gam.betax * dgyzx +
      gam.betaz * dgzzy + dtgyz) / (2. * gam.alpha);

    gam.kzz = -(-2 * dbetadownzz - gam.betax * dgzzx - gam.betay * dgzzy -
      gam.betaz * dgzzz + 2 * (gam.betax * dgxzz + gam.betay * dgyzz +
        gam.betaz * dgzzz) + dtgzz) / (2. * gam.alpha);
  }
  return 0;
}

// Function to calculate the position and velocity of m1 and m2 at time t
void find_traj_t(Real t, Real bbh_t[NTRAJ]) {
  Real const r_BH1_0 = bbh.q/(1.0+bbh.q)*bbh.sep;
  Real const r_BH2_0 = -bbh.sep/(1.0+bbh.q);
  bbh_t[X1] = r_BH1_0*std::cos(bbh.om*t);
  bbh_t[Y1] = r_BH1_0*std::sin(bbh.om*t);
  bbh_t[Z1] = 0.0;
  bbh_t[X2] = r_BH1_0*std::cos(bbh.om*t);
  bbh_t[Y2] = r_BH2_0*std::sin(bbh.om*t);
  bbh_t[Z2] = 0.0;
  bbh_t[VX1] = -r_BH1_0*bbh.om*std::sin(bbh.om*t);
  bbh_t[VY1] = r_BH1_0*bbh.om*std::cos(bbh.om*t);
  bbh_t[VZ1] = 0.0;
  bbh_t[VX2] = -r_BH2_0*bbh.om*std::sin(bbh.om*t);
  bbh_t[VY2] = r_BH2_0*bbh.om*std::cos(bbh.om*t);
  bbh_t[VZ2] = 0.0;
  bbh_t[AX1] = bbh.a1*std::sin(bbh.th_a1)*std::cos(bbh.ph_a1);
  bbh_t[AY1] = bbh.a1*std::sin(bbh.th_a1)*std::sin(bbh.ph_a1);
  bbh_t[AZ1] = bbh.a1*std::cos(bbh.th_a1);
  bbh_t[AX2] = bbh.a1*std::sin(bbh.th_a2)*std::cos(bbh.ph_a2);
  bbh_t[AY2] = bbh.a1*std::sin(bbh.th_a2)*std::sin(bbh.ph_a2);
  bbh_t[AZ2] = bbh.a1*std::cos(bbh.th_a2);
  bbh_t[M1T] = 1.0/(bbh.q+1.0);
  bbh_t[M2T] = 1.0 - bbh_t[M1T];
}

KOKKOS_INLINE_FUNCTION
void SuperposedBBH(const Real time, const Real x, const Real y, const Real z,
                  Real gcov[][NDIM], const Real traj_array[NTRAJ], const bbh_pgen& bbh_) {
  /* Superposition components*/
  Real KS1[NDIM][NDIM];
  Real KS2[NDIM][NDIM];
  Real J1[NDIM][NDIM];
  Real J2[NDIM][NDIM];

  /* Load trajectories */
  Real xi1x = traj_array[X1];
  Real xi1y = traj_array[Y1];
  Real xi1z = traj_array[Z1];
  Real xi2x = traj_array[X2];
  Real xi2y = traj_array[Y2];
  Real xi2z = traj_array[Z2];
  Real v1x  = traj_array[VX1] + 1e-40;
  Real v1y  = traj_array[VY1] + 1e-40;
  Real v1z  = traj_array[VZ1] + 1e-40;
  Real v2x =  traj_array[VX2] + 1e-40;
  Real v2y =  traj_array[VY2] + 1e-40;
  Real v2z =  traj_array[VZ2] + 1e-40;

  Real v2  =  sqrt( v2x * v2x + v2y * v2y + v2z * v2z );
  Real v1  =  sqrt( v1x * v1x + v1y * v1y + v1z * v1z );

  Real a1x  = traj_array[AX1];
  Real a1y  = traj_array[AY1];
  Real a1z  = traj_array[AZ1];

  Real a2x =  traj_array[AX2];
  Real a2y =  traj_array[AY2];
  Real a2z =  traj_array[AZ2];

  Real m1_t = traj_array[M1T];
  Real m2_t = traj_array[M2T];

  Real a1_t = sqrt( a1x*a1x + a1y*a1y + a1z*a1z + 1e-40);
  Real a2_t = sqrt( a2x*a2x + a2y*a2y + a2z*a2z + 1e-40);

  /* Load coordinates */

  Real oo1 = v1 * v1;
  Real oo2 = oo1 * -1;
  Real oo3 = 1 + oo2;
  Real oo4 = sqrt(oo3);
  Real oo5 = 1 / oo4;
  Real oo6 = x * -1;
  Real oo7 = oo6 + xi1x;
  Real oo8 = v1x * oo7;
  Real oo9 = y * -1;
  Real oo10 = z * -1;
  Real oo11 = v2 * v2;
  Real oo12 = oo11 * -1;
  Real oo13 = 1 + oo12;
  Real oo14 = sqrt(oo13);
  Real oo15 = 1 / oo14;
  Real oo16 = oo6 + xi2x;
  Real oo17 = v2x * oo16;
  Real oo18 = xi1x * -1;
  Real oo19 = 1 / oo1;
  Real oo20 = -1 + oo4;
  Real oo21 = xi1y * -1;
  Real oo22 = xi1z * -1;
  Real oo23 = xi2x * -1;
  Real oo24 = 1 / oo11;
  Real oo25 = -1 + oo14;
  Real oo26 = xi2y * -1;
  Real oo27 = xi2z * -1;
  Real oo28 = xi1y * v1y;
  Real oo29 = xi1z * v1z;
  Real oo30 = v1y * (y * -1);
  Real oo31 = v1z * (z * -1);
  Real oo32 = oo28 + (oo29 + (oo30 + (oo31 + oo8)));
  Real oo33 = xi2y * v2y;
  Real oo34 = xi2z * v2z;
  Real oo35 = v2y * (y * -1);
  Real oo36 = v2z * (z * -1);
  Real oo37 = oo17 + (oo33 + (oo34 + (oo35 + oo36)));
  //Real x0BH1 = (oo8 + ((oo9 + xi1y) * v1y + (oo10 + xi1z) * v1z)) * oo5;
  //Real x0BH2 = (oo17 + ((oo9 + xi2y) * v2y + (oo10 + xi2z) * v2z)) * oo15;
  Real x1BH1 = (oo18 + x) - oo20 * (oo5 * (v1x * (((oo18 + x) * v1x + ((oo21 + y) * v1y +
                                                   (oo22 + z) * v1z)) * oo19)));
  Real x1BH2 = (oo23 + x) - oo24 * (oo25 * (v2x * (((oo23 + x) * v2x + ((oo26 + y) * v2y +
                                                    (oo27 + z) * v2z)) * oo15)));
  Real x2BH1 = oo21 + (oo20 * (oo32 * (oo5 * (v1y * oo19))) + y);
  Real x2BH2 = oo26 + (oo24 * (oo25 * (oo37 * (v2y * oo15))) + y);
  Real x3BH1 = oo22 + (oo20 * (oo32 * (oo5 * (v1z * oo19))) + z);
  Real x3BH2 = oo27 + (oo24 * (oo25 * (oo37 * (v2z * oo15))) + z);


  /* Adjust mass */
  /* This is useful for reducing the effective mass of each BH */
  /* Adjust by hand to get the correct irreducible mass of the BH */
  Real a1 = a1_t * bbh_.adjust_mass1;
  Real m1 = m1_t * bbh_.adjust_mass1;
  Real a2 = a2_t * bbh_.adjust_mass2;
  Real m2 = m2_t * bbh_.adjust_mass2;

  //============================================//
  // Regularize horizon and apply excision mask //
  //============================================//

  /* Define radius with respect to BH frame */
  Real rBH1 = sqrt( x1BH1*x1BH1 + x2BH1*x2BH1 + x3BH1*x3BH1);
  Real rBH2 = sqrt( x1BH2*x1BH2 + x2BH2*x2BH2 + x3BH2*x3BH2);

  /* Define radius cutoff */
  Real rBH1_Cutoff = fabs(a1) * ( 1.0 + bbh_.a1_buffer) + bbh_.cutoff_floor;
  Real rBH2_Cutoff = fabs(a2) * ( 1.0 + bbh_.a2_buffer) + bbh_.cutoff_floor;

  /* Apply excision */
  if ((rBH1) < rBH1_Cutoff) {
    if(x3BH1>0) {
      x3BH1 = rBH1_Cutoff;
    } else {
      x3BH1 = -1.0*rBH1_Cutoff;
    }
  }
  if ((rBH2) < rBH2_Cutoff) {
    if(x3BH2>0) {
      x3BH2 = rBH2_Cutoff;
    } else {
      x3BH2 = -1.0*rBH2_Cutoff;
    }
  }

  //=================//
  //     Metric      //
  //=================//
  Real o1 = 1.4142135623730951;
  Real o2 = 1 / o1;
  Real o3 = a1x * a1x;
  Real o4 = o3 * -1;
  Real o5 = a1z * a1z;
  Real o6 = o5 * -1;
  Real o7 = a2x * a2x;
  Real o8 = o7 * -1;
  Real o9 = x1BH1 * x1BH1;
  Real o10 = x2BH1 * x2BH1;
  Real o11 = x3BH1 * x3BH1;
  Real o12 = x1BH1 * a1x;
  Real o13 = x2BH1 * a2x;
  Real o14 = x3BH1 * a1z;
  Real o15 = o12 + (o13 + o14);
  Real o16 = o15 * o15;
  Real o17 = o16 * 4;
  Real o18 = o10 + (o11 + (o4 + (o6 + (o8 + o9))));
  Real o19 = o18 * o18;
  Real o20 = o17 + o19;
  Real o21 = sqrt(o20);
  Real o22 = o10 + (o11 + (o21 + (o4 + (o6 + (o8 + o9)))));
  Real o23 = pow(o22, 1.5);
  Real o24 = o22 * o22;
  Real o25 = o24 * 0.25;
  Real o26 = o16 + o25;
  Real o27 = 1 / o26;
  Real o28 = x2BH1 * a1z;
  Real o29 = a2x * (x3BH1 * -1);
  Real o30 = sqrt(o22);
  Real o31 = 1 / o30;
  Real o32 = o1 * (o15 * (o31 * a1x));
  Real o33 = o30 * (x1BH1 * o2);
  Real o34 = o28 + (o29 + (o32 + o33));
  Real o35 = o22 * 0.5;
  Real o36 = o3 + (o35 + (o5 + o7));
  Real o37 = 1 / o36;
  Real o38 = o2 * (o23 * (o27 * (o34 * (o37 * m1))));
  Real o39 = a1z * (x1BH1 * -1);
  Real o40 = x3BH1 * a1x;
  Real o41 = o1 * (o15 * (o31 * a2x));
  Real o42 = o30 * (x2BH1 * o2);
  Real o43 = o39 + (o40 + (o41 + o42));
  Real o44 = o2 * (o23 * (o27 * (o37 * (o43 * m1))));
  Real o45 = x1BH1 * a2x;
  Real o46 = a1x * (x2BH1 * -1);
  Real o47 = o1 * (o15 * (o31 * a1z));
  Real o48 = o30 * (x3BH1 * o2);
  Real o49 = o45 + (o46 + (o47 + o48));
  Real o50 = o2 * (o23 * (o27 * (o37 * (o49 * m1))));
  Real o51 = o36 * o36;
  Real o52 = 1 / o51;
  Real o53 = o2 * (o23 * (o27 * (o34 * (o43 * (o52 * m1)))));
  Real o54 = o2 * (o23 * (o27 * (o34 * (o49 * (o52 * m1)))));
  Real o55 = o2 * (o23 * (o27 * (o43 * (o49 * (o52 * m1)))));
  Real o56 = a2y * a2y;
  Real o57 = o56 * -1;
  Real o58 = a2z * a2z;
  Real o59 = o58 * -1;
  Real o60 = x1BH2 * x1BH2;
  Real o61 = x2BH2 * x2BH2;
  Real o62 = x3BH2 * x3BH2;
  Real o63 = x1BH2 * a2x;
  Real o64 = x2BH2 * a2y;
  Real o65 = x3BH2 * a2z;
  Real o66 = o63 + (o64 + o65);
  Real o67 = o66 * o66;
  Real o68 = o67 * 4;
  Real o69 = o57 + (o59 + (o60 + (o61 + (o62 + o8))));
  Real o70 = o69 * o69;
  Real o71 = o68 + o70;
  Real o72 = sqrt(o71);
  Real o73 = o57 + (o59 + (o60 + (o61 + (o62 + (o72 + o8)))));
  Real o74 = pow(o73, 1.5);
  Real o75 = o73 * o73;
  Real o76 = o75 * 0.25;
  Real o77 = o67 + o76;
  Real o78 = 1 / o77;
  Real o79 = x2BH2 * a2z;
  Real o80 = a2y * (x3BH2 * -1);
  Real o81 = sqrt(o73);
  Real o82 = 1 / o81;
  Real o83 = o1 * (o66 * (o82 * a2x));
  Real o84 = o81 * (x1BH2 * o2);
  Real o85 = o79 + (o80 + (o83 + o84));
  Real o86 = o73 * 0.5;
  Real o87 = o56 + (o58 + (o7 + o86));
  Real o88 = 1 / o87;
  Real o89 = o2 * (o74 * (o78 * (o85 * (o88 * m2))));
  Real o90 = a2z * (x1BH2 * -1);
  Real o91 = x3BH2 * a2x;
  Real o92 = o1 * (o66 * (o82 * a2y));
  Real o93 = o81 * (x2BH2 * o2);
  Real o94 = o90 + (o91 + (o92 + o93));
  Real o95 = o2 * (o74 * (o78 * (o88 * (o94 * m2))));
  Real o96 = x1BH2 * a2y;
  Real o97 = a2x * (x2BH2 * -1);
  Real o98 = o1 * (o66 * (o82 * a2z));
  Real o99 = o81 * (x3BH2 * o2);
  Real o100 = o96 + (o97 + (o98 + o99));
  Real o101 = o100 * (o2 * (o74 * (o78 * (o88 * m2))));
  Real o102 = o87 * o87;
  Real o103 = 1 / o102;
  Real o104 = o103 * (o2 * (o74 * (o78 * (o85 * (o94 * m2)))));
  Real o105 = o100 * (o103 * (o2 * (o74 * (o78 * (o85 * m2)))));
  Real o106 = o100 * (o103 * (o2 * (o74 * (o78 * (o94 * m2)))));
  Real o107 = v1 * v1;
  Real o108 = o107 * -1;
  Real o109 = 1 + o108;
  Real o110 = sqrt(o109);
  Real o111 = 1 / o110;
  Real o112 = o111 * (v1x * -1);
  Real o113 = o111 * (v1y * -1);
  Real o114 = o111 * (v1z * -1);
  Real o115 = 1 / o107;
  Real o116 = -1 + o111;
  Real o117 = o116 * (v1x * (v1y * o115));
  Real o118 = o116 * (v1x * (v1z * o115));
  Real o119 = o116 * (v1y * (v1z * o115));
  Real o120 = v2 * v2;
  Real o121 = o120 * -1;
  Real o122 = 1 + o121;
  Real o123 = sqrt(o122);
  Real o124 = 1 / o123;
  Real o125 = o124 * (v2x * -1);
  Real o126 = o124 * (v2y * -1);
  Real o127 = o124 * (v2z * -1);
  Real o128 = 1 / o120;
  Real o129 = -1 + o124;
  Real o130 = o129 * (v2x * (v2y * o128));
  Real o131 = o129 * (v2x * (v2z * o128));
  Real o132 = o129 * (v2y * (v2z * o128));
  KS1[0][0] = o2 * (o23 * (o27 * m1));
  KS1[0][1] = o38;
  KS1[0][2] = o44;
  KS1[0][3] = o50;
  KS1[1][0] = o38;
  KS1[1][1] = o2 * (o23 * (o27 * ((o34 * o34) * (o52 * m1))));
  KS1[1][2] = o53;
  KS1[1][3] = o54;
  KS1[2][0] = o44;
  KS1[2][1] = o53;
  KS1[2][2] = o2 * (o23 * (o27 * ((o43 * o43) * (o52 * m1))));
  KS1[2][3] = o55;
  KS1[3][0] = o50;
  KS1[3][1] = o54;
  KS1[3][2] = o55;
  KS1[3][3] = o2 * (o23 * (o27 * ((o49 * o49) * (o52 * m1))));
  KS2[0][0] = o2 * (o74 * (o78 * m2));
  KS2[0][1] = o89;
  KS2[0][2] = o95;
  KS2[0][3] = o101;
  KS2[1][0] = o89;
  KS2[1][1] = o103 * (o2 * (o74 * (o78 * ((o85 * o85) * m2))));
  KS2[1][2] = o104;
  KS2[1][3] = o105;
  KS2[2][0] = o95;
  KS2[2][1] = o104;
  KS2[2][2] = o103 * (o2 * (o74 * (o78 * ((o94 * o94) * m2))));
  KS2[2][3] = o106;
  KS2[3][0] = o101;
  KS2[3][1] = o105;
  KS2[3][2] = o106;
  KS2[3][3] = (o100 * o100) * (o103 * (o2 * (o74 * (o78 * m2))));
  J1[0][0] = o111;
  J1[0][1] = o112;
  J1[0][2] = o113;
  J1[0][3] = o114;
  J1[1][0] = o112;
  J1[1][1] = 1 + o116 * ((v1x * v1x) * o115);
  J1[1][2] = o117;
  J1[1][3] = o118;
  J1[2][0] = o113;
  J1[2][1] = o117;
  J1[2][2] = 1 + o116 * ((v1y * v1y) * o115);
  J1[2][3] = o119;
  J1[3][0] = o114;
  J1[3][1] = o118;
  J1[3][2] = o119;
  J1[3][3] = 1 + o116 * ((v1z * v1z) * o115);
  J2[0][0] = o124;
  J2[0][1] = o125;
  J2[0][2] = o126;
  J2[0][3] = o127;
  J2[1][0] = o125;
  J2[1][1] = 1 + o129 * ((v2x * v2x) * o128);
  J2[1][2] = o130;
  J2[1][3] = o131;
  J2[2][0] = o126;
  J2[2][1] = o130;
  J2[2][2] = 1 + o129 * ((v2y * v2y) * o128);
  J2[2][3] = o132;
  J2[3][0] = o127;
  J2[3][1] = o131;
  J2[3][2] = o132;
  J2[3][3] = 1 + o129 * ((v2z * v2z) * o128);
  /* Initialize the flat part */
  Real eta[4][4] = {
    {-1,0,0,0},
    {0,1,0,0},
    {0,0,1,0},
    {0,0,0,1}
  };
  for (int i=0; i < 4; i++ ) {
    for (int j=0; j < 4; j++ ) {
      gcov[i][j] = eta[i][j];
    }
  }

  /* Load symmetric gcov (from chatGPT3)*/
  for (int i = 0; i < 4; ++i) {
    for (int j = i; j < 4; ++j) {
      Real sum = 0.0;
      for (int m = 0; m < 4; ++m) {
        Real term1 = J2[m][i];
        Real term2 = J1[m][i];
        for (int n = 0; n < 4; ++n) {
          Real term3 = J2[n][j];
          Real term4 = J1[n][j];

          sum += (term1 * term3 * KS2[m][n] + term2 * term4 * KS1[m][n]);
        }
      }

      gcov[i][j] += sum;
      gcov[j][i] = gcov[i][j];
    }
  }

  return;
}

KOKKOS_INLINE_FUNCTION
void get_metric(const Real t,
                const Real x,
                const Real y,
                const Real z,
                struct four_metric &met,
                const Real bbh_traj_loc[NTRAJ],
                const bbh_pgen& bbh_) {
  Real gcov[NDIM][NDIM];

  SuperposedBBH(t, x, y, z, gcov, bbh_traj_loc, bbh_);

  met.g.tt = gcov[TT][TT];
  met.g.tx = gcov[TT][XX];
  met.g.ty = gcov[TT][YY];
  met.g.tz = gcov[TT][ZZ];
  met.g.xx = gcov[XX][XX];
  met.g.xy = gcov[XX][YY];
  met.g.xz = gcov[XX][ZZ];
  met.g.yy = gcov[YY][YY];
  met.g.yz = gcov[YY][ZZ];
  met.g.zz = gcov[ZZ][ZZ];

  return;
}

// refine region within a certain distance from each compact object
void RefineAlphaMin(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &indcs       = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->padm->u_adm;
  int I_ADM_ALPHA  = pmbp->padm->I_ADM_ALPHA;
  // note: we need this to prevent capture by this in the lambda expr.
  auto &bbh_ = bbh;

  par_for_outer(
  "AMR::ChiMin", DevExeSpace(), 0, 0, 0, (nmb - 1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_dmin;
    Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &dmin) {
        int k = (idx) / nji;
        int j = (idx - k * nji) / nx1;
        int i = (idx - k * nji - j * nx1) + is;
        j += js;
        k += ks;
        dmin = fmin(u0(m, I_ADM_ALPHA, k, j, i), dmin);
      },
      Kokkos::Min<Real>(team_dmin));

    if (team_dmin < bbh_.alpha_thr) {
      refine_flag.d_view(m + mbs) = 1;
    }
    if (team_dmin > 1.25 * bbh_.alpha_thr) {
      refine_flag.d_view(m + mbs) = -1;
    }
  });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

void RefineTracker(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  Real bbh_traj[NTRAJ];

  Real tt = pmesh->time;
  find_traj_t(tt, bbh_traj);
  Real x1_BH1 = bbh_traj[X1];
  Real x2_BH1 = bbh_traj[Y1];
  Real x3_BH1 = bbh_traj[Z1];
  Real x1_BH2 = bbh_traj[X2];
  Real x2_BH2 = bbh_traj[Y2];
  Real x3_BH2 = bbh_traj[Z2];
  for (int m = 0; m < nmb; ++m) {
    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    Real d2_bh1[8] = {
      SQR(x1min - x1_BH1) + SQR(x2min - x2_BH1) + SQR(x3min - x3_BH1),
      SQR(x1max - x1_BH1) + SQR(x2min - x2_BH1) + SQR(x3min - x3_BH1),
      SQR(x1min - x1_BH1) + SQR(x2max - x2_BH1) + SQR(x3min - x3_BH1),
      SQR(x1max - x1_BH1) + SQR(x2max - x2_BH1) + SQR(x3min - x3_BH1),
      SQR(x1min - x1_BH1) + SQR(x2min - x2_BH1) + SQR(x3max - x3_BH1),
      SQR(x1max - x1_BH1) + SQR(x2min - x2_BH1) + SQR(x3max - x3_BH1),
      SQR(x1min - x1_BH1) + SQR(x2max - x2_BH1) + SQR(x3max - x3_BH1),
      SQR(x1max - x1_BH1) + SQR(x2max - x2_BH1) + SQR(x3max - x3_BH1),
    };

    Real d2_bh2[8] = {
      SQR(x1min - x1_BH2) + SQR(x2min - x2_BH2) + SQR(x3min - x3_BH2),
      SQR(x1max - x1_BH2) + SQR(x2min - x2_BH2) + SQR(x3min - x3_BH2),
      SQR(x1min - x1_BH2) + SQR(x2max - x2_BH2) + SQR(x3min - x3_BH2),
      SQR(x1max - x1_BH2) + SQR(x2max - x2_BH2) + SQR(x3min - x3_BH2),
      SQR(x1min - x1_BH2) + SQR(x2min - x2_BH2) + SQR(x3max - x3_BH2),
      SQR(x1max - x1_BH2) + SQR(x2min - x2_BH2) + SQR(x3max - x3_BH2),
      SQR(x1min - x1_BH2) + SQR(x2max - x2_BH2) + SQR(x3max - x3_BH2),
      SQR(x1max - x1_BH2) + SQR(x2max - x2_BH2) + SQR(x3max - x3_BH2),
    };
    Real dmin2_bh1 = *std::min_element(&d2_bh1[0], &d2_bh1[8]);
    Real dmin2_bh2 = *std::min_element(&d2_bh2[0], &d2_bh2[8]);
    bool iscontained_bh1 =
      (x1_BH1 >= x1min && x1_BH1 <= x1max) &&
      (x2_BH1 >= x2min && x2_BH1 <= x2max) &&
      (x3_BH1 >= x3min && x3_BH1 <= x3max);
    bool iscontained_bh2 =
      (x1_BH2 >= x1min && x1_BH2 <= x1max) &&
      (x2_BH2 >= x2min && x2_BH2 <= x2max) &&
      (x3_BH2 >= x3min && x3_BH2 <= x3max);

    if (dmin2_bh1 < SQR(bbh.radius_thr) || dmin2_bh2 < SQR(bbh.radius_thr) ||
        iscontained_bh1 || iscontained_bh2) {
      refine_flag.d_view(m + mbs) = 1;
    } else {
      refine_flag.d_view(m + mbs) = -1;
    }
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

} // namespace
