//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file angular_grid.cpp
//! \brief Construction of the Bruls et al. (1999) type-A angular quadrature grid.
//!
//! This is a direct, validated port (host-side, run-once at startup) of the Carlson
//! symmetric S_N method implemented in Athena-C's radiation/angles.c::carlson()
//! (itself following Bruls et al. 1999, A&A 348, 233, as cited by Davis, Stone & Jiang
//! 2012 Sec. 3.2). The polar weights are distributed in azimuth by solving a linear
//! system over "permutation families" of direction-cosine triplets (the exact type-A
//! grid). That solve yields negative weights for n_mu > 6 (Bruls et al. 1999 define the
//! exact grid only up to 6; Athena-C falls back to equal weights there), so n_mu > 6 is
//! rejected.

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "utils/legendre_roots.hpp"
#include "nr_radiation/angular_grid.hpp"

namespace nr_radiation {

namespace {

//----------------------------------------------------------------------------------------
//! \brief Solve the n x n system a x = b by Gaussian elimination with partial pivoting.
//! Host-only, run once at startup on the (nmu-1) x (nmu-1) permutation-family system.

void SolveLinear(std::vector<std::vector<Real>> a, std::vector<Real> b, int n,
                 std::vector<Real> *x) {
  for (int j = 0; j < n; j++) {
    int p = j;
    for (int i = j+1; i < n; i++) {
      if (std::abs(a[i][j]) > std::abs(a[p][j])) p = i;
    }
    if (a[p][j] == 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Singular matrix in SC quadrature setup" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::swap(a[p], a[j]);
    std::swap(b[p], b[j]);
    for (int i = j+1; i < n; i++) {
      Real f = a[i][j] / a[j][j];
      for (int k = j; k < n; k++) a[i][k] -= f * a[j][k];
      b[i] -= f * b[j];
    }
  }
  for (int i = n-1; i >= 0; i--) {
    Real s = b[i];
    for (int k = i+1; k < n; k++) s -= a[i][k] * (*x)[k];
    (*x)[i] = s / a[i][i];
  }
}

//! \brief returns index of matching permutation family in pl[0..np-1], or -1
int MatchPermutation(int i, int j, int k, const std::vector<std::array<int,3>> &pl,
                     int np) {
  for (int l = 0; l < np; l++) {
    for (int m = 0; m < 3; m++) {
      if (i == pl[l][m]) {
        for (int n = 0; n < 3; n++) {
          if (n != m && j == pl[l][n]) {
            for (int o = 0; o < 3; o++) {
              if (o != m && o != n && k == pl[l][o]) return l;
            }
          }
        }
      }
    }
  }
  return -1;
}

}  // anonymous namespace

//----------------------------------------------------------------------------------------
// SCAngularGrid constructor

SCAngularGrid::SCAngularGrid(int ndim_in, int nmu_in) :
    mu("sc_mu",1,1,1),
    wmu("sc_wmu",1),
    ndim(ndim_in), nmu(nmu_in) {
  if (nmu < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/nmu = " << nmu << " must be >= 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nmu > 6) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "<nr_radiation>/nmu = " << nmu
      << " exceeds the Bruls type-A exact-grid limit of 6: the permutation-family solve "
      << "gives negative quadrature weights beyond it. Use nmu <= 6." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ndim == 1) {
    noct = 2;
    nang = nmu;
  } else if (ndim == 2) {
    noct = 4;
    nang = nmu * (nmu + 1) / 2;
  } else {
    noct = 8;
    nang = nmu * (nmu + 1) / 2;
  }
  Kokkos::realloc(mu, noct, nang, 3);
  Kokkos::realloc(wmu, nang);

  if (ndim == 1) {
    BuildCarlson1D();
  } else {
    BuildCarlsonND();
  }

  mu.template modify<HostMemSpace>();
  mu.template sync<DevExeSpace>();
  wmu.template modify<HostMemSpace>();
  wmu.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn SCAngularGrid::BuildCarlson1D
//! \brief 1D angular grid: plain Gauss-Legendre quadrature on mu in [-1,1], split into
//! two octants (mu>0 outgoing, mu<0 incoming) -- Athena-C angles.c::carlson(), nDim==1
//! branch.

void SCAngularGrid::BuildCarlson1D() {
  // 2*nmu Gauss-Legendre nodes on [-1,1]: the nmu positive nodes form octant 0 (mu > 0),
  // their mirror images octant 1. Weights halved so they sum to one.
  auto rw = RootsAndWeights(2*nmu);   // [0] = nodes ascending, [1] = weights
  for (int i = nmu; i < 2*nmu; i++) {
    wmu.h_view(i-nmu) = 0.5 * rw[1][i];
    mu.h_view(0, i-nmu, 0) = rw[0][i];
    mu.h_view(0, i-nmu, 1) = 0.0;
    mu.h_view(0, i-nmu, 2) = 0.0;
    mu.h_view(1, i-nmu, 0) = -rw[0][i];
    mu.h_view(1, i-nmu, 1) = 0.0;
    mu.h_view(1, i-nmu, 2) = 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn SCAngularGrid::BuildCarlsonND
//! \brief 2D/3D angular grid following Bruls et al. (1999) type-A construction --
//! Athena-C angles.c::carlson(), nDim>1 branch, ported verbatim.

void SCAngularGrid::BuildCarlsonND() {
  // --- polar weights/angles (mu2tmp, wtmp); nmu<=6 enforced in ctor ---
  std::vector<Real> mu2tmp(nmu);
  Real deltamu = 2.0 / (2*nmu - 1);
  mu2tmp[0] = 1.0 / (3.0 * (2*nmu - 1));
  for (int i = 1; i < nmu; i++) mu2tmp[i] = mu2tmp[i-1] + deltamu;

  std::vector<Real> Wtmp(std::max(nmu-1,1)), wtmp(nmu);
  Real W2 = 4.0 * mu2tmp[0];
  Real Wsum = Wtmp[0] = std::sqrt(W2);
  for (int i = 1; i < nmu-2; i++) {
    W2 += deltamu;
    Wsum += Wtmp[i] = std::sqrt(W2);
  }
  if (nmu > 2) Wtmp[nmu-2] = 2.0*(nmu-1)/3.0 - Wsum;

  Real wsum = wtmp[0] = Wtmp[0];
  for (int i = 1; i < nmu-1; i++) wsum += wtmp[i] = Wtmp[i] - Wtmp[i-1];
  if (nmu > 1) {
    wtmp[nmu-1] = 1.0 - wsum;
  } else {
    wtmp[0] = 1.0;
  }

  // --- direction cosines: all (i,j,k)>=0 with i+j+k=nmu-1, plus the
  // permutation-family incidence matrix pmat(i,fam) needed to solve for azimuthal
  // weight distribution (Bruls et al. 1999; angles.c::carlson() single combined pass) ---
  std::vector<std::array<Real,3>> mutmp(nang);
  std::vector<int> plab(nang);
  int nfam = std::max(nmu - 1, 1);
  std::vector<std::array<int,3>> pl(nmu, {0,0,0});
  std::vector<std::vector<Real>> pmat(nfam, std::vector<Real>(nfam, 0.0));
  int np = 0, iang = 0;
  for (int i = 0; i < nmu; i++) {
    for (int j = 0; j < nmu; j++) {
      for (int k = 0; k < nmu; k++) {
        if (i + j + k == nmu - 1) {
          mutmp[iang][0] = std::sqrt(mu2tmp[j]);
          mutmp[iang][1] = std::sqrt(mu2tmp[k]);
          mutmp[iang][2] = std::sqrt(mu2tmp[i]);
          int ip = MatchPermutation(i, j, k, pl, np);
          if (ip == -1) {
            pl[np] = {i, j, k};
            if (i < nfam) pmat[i][np] += 1.0;
            plab[iang] = np;
            np++;
          } else {
            if (i < nfam) pmat[i][ip] += 1.0;
            plab[iang] = ip;
          }
          iang++;
        }
      }
    }
  }

  // --- weights: exact type-A permutation-family solve (nmu<=6, enforced in ctor) ---
  std::vector<Real> wang(nang);
  if (nmu > 1) {
    std::vector<Real> wpf(nfam, 0.0);
    SolveLinear(pmat, wtmp, nfam, &wpf);     // pmat * wpf = wtmp: weight per family
    for (int i = 0; i < nang; i++) wang[i] = wpf[plab[i]];
  } else {
    wang[0] = 1.0;
  }

  // --- assign signed direction cosines to octants, and finalize weights ---
  // The octant index carries the sign bits, x at bit 0, y at bit 1, z at bit 2, and every
  // octant holds the same |mu| triple at the same in-octant index. SC::ApplyReflectBCs
  // mirrors a ray by flipping one of those bits (MirrorRay in sc/iteration.cpp), so a
  // change to this encoding must be made there too.
  Real octfac = (ndim == 2) ? 0.25 : 0.125;
  for (int i = 0; i < nang; i++) {
    if (ndim == 2) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          int l = 2*j + k;
          mu.h_view(l, i, 0) = (k == 0 ?  mutmp[i][0] : -mutmp[i][0]);
          mu.h_view(l, i, 1) = (j == 0 ?  mutmp[i][1] : -mutmp[i][1]);
          mu.h_view(l, i, 2) = mutmp[i][2];
        }
      }
    } else {  // ndim == 3
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          for (int l = 0; l < 2; l++) {
            int m = 4*j + 2*k + l;
            mu.h_view(m, i, 0) = (l == 0 ?  mutmp[i][0] : -mutmp[i][0]);
            mu.h_view(m, i, 1) = (k == 0 ?  mutmp[i][1] : -mutmp[i][1]);
            mu.h_view(m, i, 2) = (j == 0 ?  mutmp[i][2] : -mutmp[i][2]);
          }
        }
      }
    }
    wmu.h_view(i) = wang[i] * octfac;
  }
}

}  // namespace nr_radiation
