//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rns_pbh_ns.cpp
//  \brief Problem generator for a head-on collision between a boosted PBH and RNS NS.

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

#include "RNS.h"

void PBHRefinementCondition(MeshBlockPack* pmbp);

void LoadRNSData(MeshBlockPack *pmbp, ParameterInput *pin);

void AddBoostedPuncture(MeshBlockPack *pmbp, ParameterInput *pin);

KOKKOS_INLINE_FUNCTION
void ConstructXBoost(Real lam[4][4], const Real g3d[NSPMETRIC], Real velx);
KOKKOS_INLINE_FUNCTION
void SpacetimeMetric(Real g[4][4], const Real g3d[NSPMETRIC], const Real beta_u[3],
                     Real alpha);
KOKKOS_INLINE_FUNCTION
void SpatialMetric(Real g3d[NSPMETRIC], Real beta_u[3], Real& alpha, const Real g[4][4]);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Problem Generator for boosted puncture superimposed with RNS data
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func = PBHRefinementCondition;
  if (restart) {
    return;
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr || pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "PBH-NS problem requires dyn_grmhd and z4c, but an <mhd> or <z4c> block"
              << "is missing in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // First load the RNS data
  LoadRNSData(pmbp, pin);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;

  // TODO(JMF): Add magnetic field
  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b0.x1f(m, k, j, i) = 0.0;
    b0.x2f(m, k, j, i) = 0.0;
    b0.x3f(m, k, j, i) = 0.0;

    if (i == ie) {
      b0.x1f(m, k, j, i+1) = 0.0;
    }
    if (j == je) {
      b0.x2f(m, k, j+1, i) = 0.0;
    }
    if (k == ke) {
      b0.x3f(m, k+1, j, i) = 0.0;
    }
  });

  // Compute cell-centered fields
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_bcc", DevExeSpace(), 0, nmb-1, ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0(m, IBX, k, j, i) = 0.5*(b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
    bcc0(m, IBY, k, j, i) = 0.5*(b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
    bcc0(m, IBZ, k, j, i) = 0.5*(b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
  });

  // Now add the puncture data
  AddBoostedPuncture(pmbp, pin);

  // Update the excision mask
  pmbp->pcoord->UpdateExcisionMasks();

  // Calculate conserved variables and Z4c variables
  pmbp->pdyngr->PrimToConInit(0, (ncells1-1), 0, (ncells2-1), 0, (ncells3-1));
  switch(indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }

  return;
}

void PBHRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

//----------------------------------------------------------------------------------------
//! \fn LoadRNSData()
//! \brief Load RNS data from parameter input
void LoadRNSData(MeshBlockPack *pmbp, ParameterInput *pin) {
  // Create RNS data
  std::string input_file = pin->GetOrAddString("problem", "filename", "tovgamma2.par");
  RNS_params_set_inputfile((char *) input_file.c_str());
  // Replace the atmosphere value of density with whatever AthenaK is using.
  RNS_params_set_Real("atm_level_rho", pin->GetOrAddReal("mhd", "dfloor", 1.28e-13));
  ini_data *data = RNS_make_initial_data();

  // Some utility information we need
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int imin[3] = {0, 0, 0};
  int n[3] = {ncells1, ncells2, ncells3};
  int nmb = pmbp->nmb_thispack;
  int width = ncells1*ncells2*ncells3;

  auto &u_adm = pmbp->padm->u_adm;
  auto &adm   = pmbp->padm->adm;
  auto &w0    = pmbp->pmhd->w0;
  auto &u_z4c = pmbp->pz4c->u0;

  // Mirrors for AthenaK device data
  HostArray5D<Real>::HostMirror host_u_adm = create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0 = create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c = create_mirror_view(u_z4c);
  adm::ADM::ADMhost_vars host_adm;
  host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
  host_adm.beta_u.InitWithShallowSlice(host_u_z4c,
      z4c::Z4c::I_Z4C_BETAX, z4c::Z4c::I_Z4C_BETAZ);
  host_adm.g_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_KXX, adm::ADM::I_ADM_KZZ);

  // Create scratch memory for RNS
  Real *gxx = new Real[width], *gyy = new Real[width], *gzz = new Real[width];
  Real *gxy = new Real[width], *gxz = new Real[width], *gyz = new Real[width];

  Real *Kxx = new Real[width], *Kyy = new Real[width], *Kzz = new Real[width];
  Real *Kxy = new Real[width], *Kxz = new Real[width], *Kyz = new Real[width];

  Real *alp = new Real[width];
  Real *betax = new Real[width], *betay = new Real[width], *betaz = new Real[width];

  Real *rho = new Real[width];
  Real *vx = new Real[width], *vy = new Real[width], *vz = new Real[width];
  Real *pgas = new Real[width];

  Real *x = new Real[ncells1];
  Real *y = new Real[ncells2];
  Real *z = new Real[ncells3];

  // Loop over every mesh block
  for (int m = 0; m < nmb; m++) {
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    int nx1 = indcs.nx1;

    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    int nx2 = indcs.nx2;

    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;
    int nx3 = indcs.nx3;

    // Populate coordinates
    for (int i = 0; i < ncells1; i++) {
      x[i] = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    }
    for (int j = 0; j < ncells2; j++) {
      y[j] = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    }
    for (int k = 0; k < ncells3; k++) {
      z[k] = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    }

    // Interpolate geometry
    RNS_Cartesian_interpolation(
      data,  // struct containing the previously calculated solution
      imin,  // min, max indcs of Cartesian grid in three directions
      n,     // ???
      n,     // total number of indices in each direction
      x,     // x coordinates
      y,     // y coordinates
      z,     // z coordinates
      alp,   // lapse
      betax, // shift
      betay, 
      betaz,
      gxx,   // metric components
      gxy,
      gxz,
      gyy,
      gyz,
      gzz,
      Kxx,   // extrinsic curvature components
      Kxy,
      Kxz,
      Kyy,
      Kyz,
      Kzz,
      rho,   // density
      NULL,  // epsilon
      NULL,  // vx
      NULL,  // vy
      NULL,  // vz
      vx,    // Wvx
      vy,    // Wvy
      vz,    // Wvz
      pgas   // pressure
    );

    for (int k = 0; k < ncells3; k++) {
      for (int j = 0; j < ncells2; j++) {
        for (int i = 0; i < ncells1; i++) {
          int idx = i + n[0]*(j + n[1]*k);

          // Extract metric quantities
          host_adm.alpha(m, k, j, i) = alp[idx];
          host_adm.beta_u(m, 0, k, j, i) = betax[idx];
          host_adm.beta_u(m, 1, k, j, i) = betay[idx];
          host_adm.beta_u(m, 2, k, j, i) = betaz[idx];

          host_adm.g_dd(m, 0, 0, k, j, i) = gxx[idx];
          host_adm.g_dd(m, 0, 1, k, j, i) = gxy[idx];
          host_adm.g_dd(m, 0, 2, k, j, i) = gxz[idx];
          host_adm.g_dd(m, 1, 1, k, j, i) = gyy[idx];
          host_adm.g_dd(m, 1, 2, k, j, i) = gyz[idx];
          host_adm.g_dd(m, 2, 2, k, j, i) = gzz[idx];

          host_adm.vK_dd(m, 0, 0, k, j, i) = Kxx[idx];
          host_adm.vK_dd(m, 0, 1, k, j, i) = Kxy[idx];
          host_adm.vK_dd(m, 0, 2, k, j, i) = Kxz[idx];
          host_adm.vK_dd(m, 1, 1, k, j, i) = Kyy[idx];
          host_adm.vK_dd(m, 1, 2, k, j, i) = Kyz[idx];
          host_adm.vK_dd(m, 2, 2, k, j, i) = Kzz[idx];

          // Extract hydro quantities
          host_w0(m, IDN, k, j, i) = rho[idx];
          host_w0(m, IVX, k, j, i) = vx[idx];
          host_w0(m, IVY, k, j, i) = vy[idx];
          host_w0(m, IVZ, k, j, i) = vz[idx];
          host_w0(m, IPR, k, j, i) = pgas[idx];
        }
      }
    }
  }

  // Free RNS data
  delete[] gxx;
  delete[] gxy;
  delete[] gxz;
  delete[] gyy;
  delete[] gyz;
  delete[] gzz;

  delete[] Kxx;
  delete[] Kxy;
  delete[] Kxz;
  delete[] Kyy;
  delete[] Kyz;
  delete[] Kzz;

  delete[] alp;
  delete[] betax;
  delete[] betay;
  delete[] betaz;

  delete[] x;
  delete[] y;
  delete[] z;

  delete[] rho;
  delete[] vx;
  delete[] vy;
  delete[] vz;
  delete[] pgas;

  RNS_finalise(data);

  // Copy data to the GPU
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  Kokkos::deep_copy(u_z4c, host_u_z4c);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddBoostedPuncture()
//! \brief Create a single puncture, boost it, and superimpose it on the solution
void AddBoostedPuncture(MeshBlockPack* pmbp, ParameterInput* pin) {
  Real mass = pin->GetOrAddReal("problem", "bh_mass", 1.0);
  Real sep  = pin->GetOrAddReal("problem", "separation", 20.0);
  Real lapse_floor = pin->GetOrAddReal("problem", "lapse_floor", 1e-3);
  Real bh_boost = pin->GetOrAddReal("problem", "bh_boost", 0.0);

  // Some utility information we need
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int imin[3] = {0, 0, 0};
  int n[3] = {ncells1, ncells2, ncells3};
  int nmb = pmbp->nmb_thispack;
  int width = ncells1*ncells2*ncells3;

  // Compute the metric
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("Boosted puncture",
  DevExeSpace(),0,nmb-1,0,ncells3-1,0,ncells2-1,0,ncells1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    x1v += sep;

    Real r = Kokkos::sqrt(x1v*x1v + x2v*x2v + x3v*x3v);

    adm.psi4(m,k,j,i) = Kokkos::pow(1.0 + 0.5*mass/r,4.0);

    // Construct the puncture solution with precollapsed lapse and shift.
    Real g3d[NSPMETRIC] = {0.0};
    g3d[S11] = adm.psi4(m,k,j,i);
    g3d[S22] = adm.psi4(m,k,j,i);
    g3d[S33] = adm.psi4(m,k,j,i);

    Real alp = 1.0/Kokkos::sqrt(adm.psi4(m,k,j,i));
    Real beta_u[3] = {0.0};

    Real g_dd[4][4];
    SpacetimeMetric(g_dd, g3d, beta_u, alp);

    // Construct the boosted puncture solution. Note that we split the boost into two
    // separate multiplications. Though less compact, it actually costs 1/3 as much.
    // One multiplication: 4^4 loops * 3 FLOPS = 768 FLOPS
    // Two multiplications: 2 * (4^3 loops * 2 FLOPS) = 256 FLOPS
    Real lam_ud[4][4] = {0.0};
    Real g_dd_boost[4][4] = {0.0};
    ConstructXBoost(lam_ud, g3d, bh_boost);
    
    for (int c = 0; c < 4; c++) {
      for (int b = 0; b < 4; b++) {
        for (int d = 0; d < 4; d++) {
          g_dd_boost[c][b] += g_dd[c][d]*lam_ud[d][b];
        }
      }
    }

    for (int a = 0; a < 4; a++) {
      for (int b = 0; b < 4; b++) {
        g_dd[a][b] = 0.0;
        for (int c = 0; c < 4; c++) {
          g_dd[a][b] += lam_ud[c][a]*g_dd_boost[c][b];
        }
      }
    }

    SpatialMetric(g3d, beta_u, alp, g_dd);

    // Superpose the solution and subtract off Minkowski space.
    adm.g_dd(m,0,0,k,j,i) += g3d[S11] - 1.0;
    adm.g_dd(m,0,1,k,j,i) += g3d[S12];
    adm.g_dd(m,0,2,k,j,i) += g3d[S13];
    adm.g_dd(m,1,1,k,j,i) += g3d[S22] - 1.0;
    adm.g_dd(m,1,2,k,j,i) += g3d[S23];
    adm.g_dd(m,2,2,k,j,i) += g3d[S33] - 1.0;

    // Check that the resulting metric is still positive-definite by finding the minors
    Real min1 = adm.g_dd(m,0,0,k,j,i);
    Real min2 = adm.g_dd(m,0,0,k,j,i)*adm.g_dd(m,1,1,k,j,i) - SQR(adm.g_dd(m,0,1,k,j,i));
    Real min3 = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    
    if (min1 <= 0 || min2 <= 0 || min3 <= 0) {
      Kokkos::printf("Metric is no longer positive definite!\n");
    }

    // Assumed precollapsed lapse
    adm.alpha(m,k,j,i) = Kokkos::sqrt(Kokkos::max(lapse_floor,
                            adm.alpha(m,k,j,i)*adm.alpha(m,k,j,i) +
                            alp*alp - 1.0));
  });
}

KOKKOS_INLINE_FUNCTION
void ConstructXBoost(Real lam_ud[4][4], const Real g3d[NSPMETRIC], Real velx) {
  Real W = 1.0/Kokkos::sqrt(1.0 - velx*g3d[S11]*velx);

  lam_ud[0][0] = W;
  lam_ud[0][1] = -W*velx;
  lam_ud[1][0] = -W*velx;
  lam_ud[1][1] = W;
  lam_ud[2][2] = 1.0;
  lam_ud[3][3] = 1.0;
}

KOKKOS_INLINE_FUNCTION
void SpacetimeMetric(Real g_dd[4][4], const Real g3d[NSPMETRIC], const Real beta_u[3],
                     Real alpha) {
  g_dd[1][1] = g3d[S11];
  g_dd[1][2] = g_dd[2][1] = g3d[S12];
  g_dd[1][3] = g_dd[3][1] = g3d[S13];
  g_dd[2][2] = g3d[S22];
  g_dd[2][3] = g_dd[3][2] = g3d[S23];
  g_dd[3][3] = g3d[S33];

  Real beta_d[3] = {0.0};
  Real betasq = 0.0;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      beta_d[a] += beta_u[b]*g_dd[a+1][b+1];
    }
    g_dd[0][a+1] = g_dd[a+1][0] = beta_d[a];
    betasq += beta_d[a]*beta_u[a];
  }

  g_dd[0][0] = -alpha*alpha + betasq;
}

KOKKOS_INLINE_FUNCTION
void SpatialMetric(Real g3d[NSPMETRIC], Real beta_u[3], Real& alpha,
                   const Real g_dd[4][4]) {
  g3d[S11] = g_dd[1][1];
  g3d[S12] = g_dd[1][2];
  g3d[S13] = g_dd[1][3];
  g3d[S22] = g_dd[2][2];
  g3d[S23] = g_dd[2][3];
  g3d[S33] = g_dd[3][3];

  Real g3u[NSPMETRIC] = {0.0};
  Real detg = adm::SpatialDet(g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33]);
  adm::SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                  &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);
  beta_u[0] = g_dd[0][1]*g3u[S11] + g_dd[0][2]*g3u[S12] + g_dd[0][3]*g3u[S13];
  beta_u[1] = g_dd[0][1]*g3u[S12] + g_dd[0][2]*g3u[S22] + g_dd[0][3]*g3u[S23];
  beta_u[2] = g_dd[0][1]*g3u[S13] + g_dd[0][2]*g3u[S23] + g_dd[0][3]*g3u[S33];

  alpha = Kokkos::sqrt(-(g_dd[0][0] -
            beta_u[0]*g_dd[0][1] - beta_u[1]*g_dd[0][2] - beta_u[2]*g_dd[0][3]));
}
