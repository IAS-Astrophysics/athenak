//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_beam.cpp
//  \brief Beam test for radiation.  Also checks orthonormality of tetrad

// C++ headers
#include <algorithm>  // min, max
#include <fstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <utility>    // pair
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/adm.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "dyn_radiation/dyn_radiation.hpp"
#include "pgen/pgen.hpp"

// Prototypes for user-defined BCs
void ZeroIntensity(Mesh *pm);
void CrossingBeamBoundary(Mesh *pm);
void KerrOrbitBeamSource(Mesh *pm, const Real bdt);
void DynRadPositivityFloorCheck(ParameterInput *pin, Mesh *pm);
void DynRadFLRWRedshiftCheck(ParameterInput *pin, Mesh *pm);
void DynRadLapseGradientCheck(ParameterInput *pin, Mesh *pm);
void DynRadMomentumSourceCheck(ParameterInput *pin, Mesh *pm);

namespace {

struct CrossingBeamData {
  bool enabled = false;
  Real amp = 1.0;
  Real sigma = 0.055;
  Real flux_fraction = 0.995;
  Real x0 = 0.12;
  Real y_lower = 0.15;
  Real y_upper = 0.85;
  Real lower_profile_qx = 1.0;
  Real lower_profile_qy = 0.0;
  Real upper_profile_qx = 1.0;
  Real upper_profile_qy = 0.0;
  DvceArray2D<Real> *angular_weights = nullptr;
};

CrossingBeamData crossing_beams;

struct KerrOrbitBeamData {
  bool enabled = false;
  Real amp = 1.0;
  Real sigma = 0.18;
  Real source_x = 0.0;
  Real source_y = 0.0;
  Real source_z = 0.0;
  DvceArray2D<Real> *angular_weights = nullptr;
};

KerrOrbitBeamData kerr_orbit_beam;

struct ADMFormalTestData {
  Real flrw_h = 0.2;
  Real flrw_t0 = 0.0;
  Real lapse_amp = 0.1;
  Real lapse_k = 2.0*M_PI;
  Real momentum_alpha_amp = 0.05;
  Real momentum_beta_amp = 0.04;
  Real momentum_metric_amp = 0.03;
  Real momentum_k = 2.0*M_PI;
};

ADMFormalTestData adm_formal_test;

KOKKOS_INLINE_FUNCTION
int LocalSym3Index(const int a, const int b) {
  const int i = (a < b) ? a : b;
  const int j = (a < b) ? b : a;
  if (i == 0 && j == 0) return 0;
  if (i == 0 && j == 1) return 1;
  if (i == 0 && j == 2) return 2;
  if (i == 1 && j == 1) return 3;
  if (i == 1 && j == 2) return 4;
  return 5;
}

void SetADMVariablesToFLRWRedshift(MeshBlockPack *pmbp);
void SetADMVariablesToLapseGradient(MeshBlockPack *pmbp);
void SetADMVariablesToMomentumMetric(MeshBlockPack *pmbp);

bool SolveLinear4(Real a[4][5], Real x[4]) {
  for (int col=0; col<4; ++col) {
    int pivot = col;
    Real max_abs = fabs(a[col][col]);
    for (int row=col+1; row<4; ++row) {
      const Real value = fabs(a[row][col]);
      if (value > max_abs) {
        max_abs = value;
        pivot = row;
      }
    }
    if (max_abs < 1.0e-14) { return false; }
    if (pivot != col) {
      for (int c=col; c<5; ++c) {
        std::swap(a[col][c], a[pivot][c]);
      }
    }
    const Real inv_pivot = 1.0/a[col][col];
    for (int c=col; c<5; ++c) {
      a[col][c] *= inv_pivot;
    }
    for (int row=0; row<4; ++row) {
      if (row == col) { continue; }
      const Real factor = a[row][col];
      for (int c=col; c<5; ++c) {
        a[row][c] -= factor*a[col][c];
      }
    }
  }
  for (int row=0; row<4; ++row) {
    x[row] = a[row][4];
  }
  return true;
}

bool SolveLinear3(Real a[3][4], Real x[3]) {
  for (int col=0; col<3; ++col) {
    int pivot = col;
    Real max_abs = fabs(a[col][col]);
    for (int row=col+1; row<3; ++row) {
      const Real value = fabs(a[row][col]);
      if (value > max_abs) {
        max_abs = value;
        pivot = row;
      }
    }
    if (max_abs < 1.0e-14) { return false; }
    if (pivot != col) {
      for (int c=col; c<4; ++c) {
        std::swap(a[col][c], a[pivot][c]);
      }
    }
    const Real inv_pivot = 1.0/a[col][col];
    for (int c=col; c<4; ++c) {
      a[col][c] *= inv_pivot;
    }
    for (int row=0; row<3; ++row) {
      if (row == col) { continue; }
      const Real factor = a[row][col];
      for (int c=col; c<4; ++c) {
        a[row][c] -= factor*a[col][c];
      }
    }
  }
  for (int row=0; row<3; ++row) {
    x[row] = a[row][3];
  }
  return true;
}

template<typename NhView, typename WeightView>
void SetProjectedAngularWeights(NhView nh_c, WeightView weights,
                                const int beam, const int nangles,
                                const Real qx_in, const Real qy_in,
                                const Real qz_in, const char *label) {
  for (int n=0; n<nangles; ++n) {
    weights(beam,n) = 0.0;
  }

  const Real qnorm = sqrt(SQR(qx_in) + SQR(qy_in) + SQR(qz_in));
  if (qnorm <= 0.0) {
    throw std::runtime_error(std::string(label) + " has a zero beam direction");
  }
  const Real qx = qx_in/qnorm;
  const Real qy = qy_in/qnorm;
  const Real qz = qz_in/qnorm;

  std::vector<std::pair<Real, int>> ranked;
  ranked.reserve(nangles);
  for (int n=0; n<nangles; ++n) {
    const Real dot = nh_c.h_view(n,1)*qx + nh_c.h_view(n,2)*qy
                   + nh_c.h_view(n,3)*qz;
    ranked.emplace_back(dot, n);
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  // Project the requested direction onto the convex hull of the angular centers.
  // This gives positive weights with exact zeroth moment and first moment parallel
  // to the requested ray.  The finite angular grid supplies the flux factor.
  const int ncand = std::min(nangles, 80);
  int best[3] = {ranked[0].second, ranked[0].second, ranked[0].second};
  Real best_lam[3] = {1.0, 0.0, 0.0};
  Real best_r = -1.0;
  for (int ia=0; ia<ncand; ++ia) {
    const int aidx = ranked[ia].second;
    for (int ib=ia+1; ib<ncand; ++ib) {
      const int bidx = ranked[ib].second;
      for (int ic=ib+1; ic<ncand; ++ic) {
        const int cidx = ranked[ic].second;
        Real mat[4][5] = {
          {nh_c.h_view(aidx,1), nh_c.h_view(bidx,1), nh_c.h_view(cidx,1), -qx, 0.0},
          {nh_c.h_view(aidx,2), nh_c.h_view(bidx,2), nh_c.h_view(cidx,2), -qy, 0.0},
          {nh_c.h_view(aidx,3), nh_c.h_view(bidx,3), nh_c.h_view(cidx,3), -qz, 0.0},
          {1.0,                 1.0,                 1.0,                 0.0, 1.0},
        };
        Real sol[4];
        if (!(SolveLinear4(mat, sol))) { continue; }
        const Real min_lam = std::min(sol[0], std::min(sol[1], sol[2]));
        const Real r = sol[3];
        if (min_lam >= -1.0e-10 && r > best_r && r <= 1.0 + 1.0e-10) {
          best[0] = aidx;
          best[1] = bidx;
          best[2] = cidx;
          best_lam[0] = std::max(sol[0], 0.0);
          best_lam[1] = std::max(sol[1], 0.0);
          best_lam[2] = std::max(sol[2], 0.0);
          best_r = r;
        }
      }
    }
  }

  const Real sum_lam = best_lam[0] + best_lam[1] + best_lam[2];
  if (best_r <= 0.0 || sum_lam <= 0.0) {
    throw std::runtime_error(std::string(label) + " could not project beam direction "
                             "onto the angular grid");
  }
  for (int n=0; n<3; ++n) {
    weights(beam,best[n]) += best_lam[n]/sum_lam;
  }

  Real sum_w = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  for (int n=0; n<nangles; ++n) {
    const Real w = weights(beam,n);
    sum_w += w;
    mx += w*nh_c.h_view(n,1);
    my += w*nh_c.h_view(n,2);
    mz += w*nh_c.h_view(n,3);
  }
  const Real cx = my*qz - mz*qy;
  const Real cy = mz*qx - mx*qz;
  const Real cz = mx*qy - my*qx;
  if (fabs(sum_w - 1.0) > 1.0e-10 ||
      sqrt(SQR(cx) + SQR(cy) + SQR(cz)) > 1.0e-10) {
    throw std::runtime_error(std::string(label) + " angular projection failed moment check");
  }
}

template<typename NhView, typename SolidAngleView, typename WeightView>
void SetAllAngleMomentWeights(NhView nh_c, SolidAngleView solid_angles, WeightView weights,
                              const int beam, const int nangles, const Real qx_in,
                              const Real qy_in, const Real qz_in,
                              const Real flux_fraction, const char *label) {
  for (int n=0; n<nangles; ++n) {
    weights(beam,n) = 0.0;
  }

  const Real qnorm = sqrt(SQR(qx_in) + SQR(qy_in) + SQR(qz_in));
  if (qnorm <= 0.0) {
    throw std::runtime_error(std::string(label) + " has a zero beam direction");
  }
  const Real qx = qx_in/qnorm;
  const Real qy = qy_in/qnorm;
  const Real qz = qz_in/qnorm;

  std::vector<std::pair<Real, int>> ranked;
  ranked.reserve(nangles);
  for (int n=0; n<nangles; ++n) {
    const Real dot = nh_c.h_view(n,1)*qx + nh_c.h_view(n,2)*qy
                   + nh_c.h_view(n,3)*qz;
    ranked.emplace_back(dot, n);
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  Real best_r = -1.0;
  const int ncand = std::min(nangles, 80);
  for (int ia=0; ia<ncand; ++ia) {
    const int aidx = ranked[ia].second;
    for (int ib=ia+1; ib<ncand; ++ib) {
      const int bidx = ranked[ib].second;
      for (int ic=ib+1; ic<ncand; ++ic) {
        const int cidx = ranked[ic].second;
        Real mat[4][5] = {
          {nh_c.h_view(aidx,1), nh_c.h_view(bidx,1), nh_c.h_view(cidx,1), -qx, 0.0},
          {nh_c.h_view(aidx,2), nh_c.h_view(bidx,2), nh_c.h_view(cidx,2), -qy, 0.0},
          {nh_c.h_view(aidx,3), nh_c.h_view(bidx,3), nh_c.h_view(cidx,3), -qz, 0.0},
          {1.0,                 1.0,                 1.0,                 0.0, 1.0},
        };
        Real sol[4];
        if (!(SolveLinear4(mat, sol))) { continue; }
        const Real min_lam = std::min(sol[0], std::min(sol[1], sol[2]));
        const Real r = sol[3];
        if (min_lam >= -1.0e-10 && r > best_r && r <= 1.0 + 1.0e-10) {
          best_r = r;
        }
      }
    }
  }
  if (best_r <= 0.0) {
    throw std::runtime_error(std::string(label) + " could not find a realizable "
                             "projected angular flux");
  }

  const Real frac = fmin(0.999999, fmax(0.0, flux_fraction));
  const Real target_flux = frac*best_r;
  const Real target[3] = {target_flux*qx, target_flux*qy, target_flux*qz};
  Real lambda[3] = {0.0, 0.0, 0.0};
  bool converged = false;
  for (int iter=0; iter<80; ++iter) {
    Real max_arg = -std::numeric_limits<Real>::max();
    for (int n=0; n<nangles; ++n) {
      const Real arg = lambda[0]*nh_c.h_view(n,1) +
                       lambda[1]*nh_c.h_view(n,2) +
                       lambda[2]*nh_c.h_view(n,3);
      max_arg = fmax(max_arg, arg);
    }

    Real z = 0.0;
    Real moment[3] = {0.0, 0.0, 0.0};
    Real second[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (int n=0; n<nangles; ++n) {
      const Real prior = solid_angles.h_view(n)/(4.0*M_PI);
      const Real arg = lambda[0]*nh_c.h_view(n,1) +
                       lambda[1]*nh_c.h_view(n,2) +
                       lambda[2]*nh_c.h_view(n,3);
      const Real unorm = prior*exp(arg - max_arg);
      z += unorm;
      const Real dir[3] = {nh_c.h_view(n,1), nh_c.h_view(n,2), nh_c.h_view(n,3)};
      for (int a=0; a<3; ++a) {
        moment[a] += unorm*dir[a];
        for (int b=0; b<3; ++b) {
          second[a][b] += unorm*dir[a]*dir[b];
        }
      }
    }
    for (int a=0; a<3; ++a) {
      moment[a] /= z;
      for (int b=0; b<3; ++b) {
        second[a][b] /= z;
      }
    }

    const Real residual[3] = {target[0] - moment[0],
                              target[1] - moment[1],
                              target[2] - moment[2]};
    const Real res_norm = sqrt(SQR(residual[0]) + SQR(residual[1]) +
                               SQR(residual[2]));
    if (res_norm < 1.0e-12) {
      converged = true;
      break;
    }

    Real mat[3][4];
    for (int a=0; a<3; ++a) {
      for (int b=0; b<3; ++b) {
        mat[a][b] = second[a][b] - moment[a]*moment[b];
      }
      mat[a][a] += 1.0e-14;
      mat[a][3] = residual[a];
    }
    Real delta[3];
    if (!(SolveLinear3(mat, delta))) { break; }
    Real max_delta = fmax(fabs(delta[0]), fmax(fabs(delta[1]), fabs(delta[2])));
    const Real step = (max_delta > 8.0) ? (8.0/max_delta) : 1.0;
    for (int a=0; a<3; ++a) {
      lambda[a] += step*delta[a];
    }
  }
  if (!(converged)) {
    throw std::runtime_error(std::string(label) + " all-angle moment projection "
                             "did not converge");
  }

  Real max_arg = -std::numeric_limits<Real>::max();
  for (int n=0; n<nangles; ++n) {
    const Real arg = lambda[0]*nh_c.h_view(n,1) +
                     lambda[1]*nh_c.h_view(n,2) +
                     lambda[2]*nh_c.h_view(n,3);
    max_arg = fmax(max_arg, arg);
  }
  Real z = 0.0;
  for (int n=0; n<nangles; ++n) {
    const Real prior = solid_angles.h_view(n)/(4.0*M_PI);
    const Real arg = lambda[0]*nh_c.h_view(n,1) +
                     lambda[1]*nh_c.h_view(n,2) +
                     lambda[2]*nh_c.h_view(n,3);
    weights(beam,n) = prior*exp(arg - max_arg);
    z += weights(beam,n);
  }
  Real sum_w = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
  for (int n=0; n<nangles; ++n) {
    weights(beam,n) /= z;
    const Real w = weights(beam,n);
    sum_w += w;
    mx += w*nh_c.h_view(n,1);
    my += w*nh_c.h_view(n,2);
    mz += w*nh_c.h_view(n,3);
  }
  const Real moment_err = sqrt(SQR(mx - target[0]) + SQR(my - target[1]) +
                               SQR(mz - target[2]));
  if (fabs(sum_w - 1.0) > 1.0e-11 || moment_err > 1.0e-10) {
    throw std::runtime_error(std::string(label) + " all-angle projection failed "
                             "moment check");
  }
}

Real CounterrotatingPhotonOrbitRadius(const Real spin) {
  const Real arg = fmax(-1.0, fmin(1.0, -spin));
  return 2.0*(1.0 + cos((2.0/3.0)*acos(arg)));
}

void BuildADMSpatialTriadForBeam(const Real gxx, const Real gxy, const Real gxz,
                                  const Real gyy, const Real gyz, const Real gzz,
                                  Real e[3][3]) {
  constexpr Real metric_floor = 1.0e-30;
  const Real l00 = sqrt(fmax(gxx, metric_floor));
  const Real l10 = gxy/l00;
  const Real l20 = gxz/l00;
  const Real l11 = sqrt(fmax(gyy - SQR(l10), metric_floor));
  const Real l21 = (gyz - l20*l10)/l11;
  const Real l22 = sqrt(fmax(gzz - SQR(l20) - SQR(l21), metric_floor));

  e[0][0] = 1.0/l00;
  e[1][0] = 0.0;
  e[2][0] = 0.0;

  e[0][1] = -l10/(l00*l11);
  e[1][1] = 1.0/l11;
  e[2][1] = 0.0;

  e[0][2] = l10*l21/(l00*l11*l22) - l20/(l00*l22);
  e[1][2] = -l21/(l11*l22);
  e[2][2] = 1.0/l22;
}

void CoordinateDirectionToADMTetrad(const Real x, const Real y, const Real z,
                                    const bool flat, const Real spin,
                                    const Real d1, const Real d2, const Real d3,
                                    Real ell[3]) {
  Real alpha, beta[3], psi4, g3d[6], k_dd[6];
  ComputeADMDecomposition(x, y, z, flat, spin, &alpha,
                          &beta[0], &beta[1], &beta[2],
                          &psi4,
                          &g3d[S11], &g3d[S12], &g3d[S13],
                          &g3d[S22], &g3d[S23], &g3d[S33],
                          &k_dd[S11], &k_dd[S12], &k_dd[S13],
                          &k_dd[S22], &k_dd[S23], &k_dd[S33]);

  Real g4[16];
  adm::SpacetimeMetric(alpha, beta[0], beta[1], beta[2],
                       g3d[S11], g3d[S12], g3d[S13],
                       g3d[S22], g3d[S23], g3d[S33], g4);

  const Real temp_a = g4[0];
  const Real temp_b = 2.0*(g4[1]*d1 + g4[2]*d2 + g4[3]*d3);
  const Real temp_c = g4[5]*d1*d1 + 2.0*g4[6]*d1*d2 + 2.0*g4[7]*d1*d3
                    + g4[10]*d2*d2 + 2.0*g4[11]*d2*d3 + g4[15]*d3*d3;
  const Real disc = SQR(temp_b) - 4.0*temp_a*temp_c;
  if (disc <= 0.0) {
    throw std::runtime_error("rad_kerr_orbit_beam ADM direction is not null-realizable");
  }
  const Real d0 = (-temp_b - sqrt(disc))/(2.0*temp_a);

  Real k_cov[4];
  const Real d[4] = {d0, d1, d2, d3};
  for (int mu=0; mu<4; ++mu) {
    k_cov[mu] = 0.0;
    for (int nu=0; nu<4; ++nu) {
      k_cov[mu] += g4[4*mu + nu]*d[nu];
    }
  }

  Real triad[3][3];
  BuildADMSpatialTriadForBeam(g3d[S11], g3d[S12], g3d[S13],
                              g3d[S22], g3d[S23], g3d[S33], triad);
  const Real e0_mu[4] = {1.0/alpha, -beta[0]/alpha, -beta[1]/alpha, -beta[2]/alpha};
  Real energy = 0.0;
  for (int mu=0; mu<4; ++mu) {
    energy -= e0_mu[mu]*k_cov[mu];
  }
  if (energy <= 0.0) {
    throw std::runtime_error("rad_kerr_orbit_beam ADM direction is not future-pointing");
  }
  for (int a=0; a<3; ++a) {
    ell[a] = 0.0;
    for (int i=0; i<3; ++i) {
      ell[a] += triad[i][a]*k_cov[i+1];
    }
    ell[a] /= energy;
  }
}

void CoordinateDirectionToTetrad(const Real x, const Real y, const Real z,
                                 const bool flat, const Real spin,
                                 const Real d1, const Real d2, const Real d3,
                                 Real ell[3]) {
  Real glower[4][4], gupper[4][4];
  Real dgx[4][4], dgy[4][4], dgz[4][4];
  Real e[4][4], e_cov[4][4], omega[4][4][4];
  ComputeMetricAndInverse(x, y, z, flat, spin, glower, gupper);
  ComputeMetricDerivatives(x, y, z, flat, spin, dgx, dgy, dgz);
  ComputeTetrad(x, y, z, flat, spin, glower, gupper, dgx, dgy, dgz,
                e, e_cov, omega);

  const Real temp_a = glower[0][0];
  const Real temp_b = 2.0*(glower[0][1]*d1 + glower[0][2]*d2 +
                           glower[0][3]*d3);
  const Real temp_c = glower[1][1]*d1*d1 + 2.0*glower[1][2]*d1*d2
                    + 2.0*glower[1][3]*d1*d3 + glower[2][2]*d2*d2
                    + 2.0*glower[2][3]*d2*d3 + glower[3][3]*d3*d3;
  const Real disc = SQR(temp_b) - 4.0*temp_a*temp_c;
  if (disc <= 0.0) {
    throw std::runtime_error("rad_kerr_orbit_beam direction is not null-realizable");
  }
  const Real d0 = (-temp_b - sqrt(disc))/(2.0*temp_a);

  Real dc[4];
  for (int mu=0; mu<4; ++mu) {
    dc[mu] = glower[mu][0]*d0 + glower[mu][1]*d1
           + glower[mu][2]*d2 + glower[mu][3]*d3;
  }
  Real dtc0 = 0.0;
  for (int mu=0; mu<4; ++mu) {
    dtc0 += e[0][mu]*dc[mu];
  }
  if (dtc0 >= 0.0) {
    throw std::runtime_error("rad_kerr_orbit_beam produced a non-future-pointing ray");
  }
  for (int a=0; a<3; ++a) {
    ell[a] = 0.0;
    for (int mu=0; mu<4; ++mu) {
      ell[a] += e[a+1][mu]*dc[mu];
    }
    ell[a] /= -dtc0;
  }
}

KOKKOS_INLINE_FUNCTION
Real CrossingBeamProfile(const Real x, const Real y,
                         const Real x0, const Real y0,
                         const Real qx, const Real qy,
                         const Real sigma, const Real amp) {
  const Real dx = x - x0;
  const Real dy = y - y0;
  const Real along = dx*qx + dy*qy;
  if (along < 0.0) { return 0.0; }
  const Real perp = -dx*qy + dy*qx;
  return amp*exp(-0.5*SQR(perp/sigma));
}

void FillCrossingBeams(Mesh *pm, const bool boundaries_only) {
  if (!(crossing_beams.enabled)) { return; }
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie = indcs.ie;
  int &js = indcs.js;  int &je = indcs.je;
  int &ks = indcs.ks;  int &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  int nang1 = -1;
  bool use_adm_geometry = false;
  DvceArray5D<Real> i0;
  DualArray2D<Real> nh_c;
  DualArray1D<Real> solid_angles;
  DvceArray6D<Real> tet_c;
  DvceArray6D<Real> tetcov_c;
  DvceArray4D<Real> sqrt_detg_c;
  if (pmbp->prad != nullptr) {
    nang1 = pmbp->prad->prgeo->nangles - 1;
    i0 = pmbp->prad->i0;
    nh_c = pmbp->prad->nh_c;
    solid_angles = pmbp->prad->prgeo->solid_angles;
    tet_c = pmbp->prad->tet_c;
    tetcov_c = pmbp->prad->tetcov_c;
  } else if (pmbp->pdynrad != nullptr) {
    nang1 = pmbp->pdynrad->prgeo->nangles - 1;
    use_adm_geometry = pmbp->pdynrad->use_adm_geometry;
    i0 = pmbp->pdynrad->i0;
    nh_c = pmbp->pdynrad->nh_c;
    solid_angles = pmbp->pdynrad->prgeo->solid_angles;
    tet_c = pmbp->pdynrad->tet_c;
    tetcov_c = pmbp->pdynrad->tetcov_c;
    sqrt_detg_c = pmbp->pdynrad->sqrt_detg_c;
  } else {
    throw std::runtime_error("rad_crossing_beams requires <radiation> or <dyn_radiation>");
  }

  auto &size = pmbp->pmb->mb_size;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  const Real amp = crossing_beams.amp;
  const Real sigma = crossing_beams.sigma;
  const Real x0 = crossing_beams.x0;
  const Real y_lower = crossing_beams.y_lower;
  const Real y_upper = crossing_beams.y_upper;
  if (crossing_beams.angular_weights == nullptr) {
    throw std::runtime_error("crossing-beam angular weights were not initialized");
  }
  auto angular_weights = *(crossing_beams.angular_weights);
  const Real lower_profile_qx = crossing_beams.lower_profile_qx;
  const Real lower_profile_qy = crossing_beams.lower_profile_qy;
  const Real upper_profile_qx = crossing_beams.upper_profile_qx;
  const Real upper_profile_qy = crossing_beams.upper_profile_qy;

  par_for("crossing_beams_fill",DevExeSpace(),0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    bool fill_cell = !(boundaries_only);
    if (boundaries_only) {
      fill_cell = ((i < is && mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) ||
                   (i > ie && mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) ||
                   (j < js && mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) ||
                   (j > je && mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) ||
                   (k < ks && mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) ||
                   (k > ke && mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user));
    }
    if (!(fill_cell)) { return; }

    const Real x = CellCenterX(i-is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);

    Real intensity = 0.0;
    intensity += CrossingBeamProfile(x, y, x0, y_lower,
                                     lower_profile_qx, lower_profile_qy,
                                     sigma, amp)*angular_weights(0,n)/
                 solid_angles.d_view(n);
    intensity += CrossingBeamProfile(x, y, x0, y_upper,
                                     upper_profile_qx, upper_profile_qy,
                                     sigma, amp)*angular_weights(1,n)/
                 solid_angles.d_view(n);

    Real norm = 1.0;
    if (use_adm_geometry) {
      norm = sqrt_detg_c(m,k,j,i);
    } else {
      Real n_0 = 0.0;
      for (int d=0; d<4; ++d) {
        n_0 += tetcov_c(m,d,0,k,j,i)*nh_c.d_view(n,d);
      }
      norm = tet_c(m,0,0,k,j,i)*n_0;
    }
    i0(m,n,k,j,i) = norm*intensity;
  });
}

void SetADMVariablesToFLRWRedshift(MeshBlockPack *pmbp) {
  const Real t = pmbp->pmesh->time;
  const Real h = adm_formal_test.flrw_h;
  const Real t0 = adm_formal_test.flrw_t0;
  const Real a = 1.0 + h*(t - t0);
  const Real a2 = SQR(a);
  auto &adm = pmbp->padm->adm;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  par_for("adm_flrw_redshift", DevExeSpace(), 0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    adm.g_dd(m,0,0,k,j,i) = a2;
    adm.g_dd(m,0,1,k,j,i) = 0.0;
    adm.g_dd(m,0,2,k,j,i) = 0.0;
    adm.g_dd(m,1,1,k,j,i) = a2;
    adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.g_dd(m,2,2,k,j,i) = a2;

    adm.vK_dd(m,0,0,k,j,i) = -a*h;
    adm.vK_dd(m,0,1,k,j,i) = 0.0;
    adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = -a*h;
    adm.vK_dd(m,1,2,k,j,i) = 0.0;
    adm.vK_dd(m,2,2,k,j,i) = -a*h;

    adm.psi4(m,k,j,i) = a2;
    adm.alpha(m,k,j,i) = 1.0;
    adm.beta_u(m,0,k,j,i) = 0.0;
    adm.beta_u(m,1,k,j,i) = 0.0;
    adm.beta_u(m,2,k,j,i) = 0.0;
  });
}

void SetADMVariablesToLapseGradient(MeshBlockPack *pmbp) {
  const Real amp = adm_formal_test.lapse_amp;
  const Real kwave = adm_formal_test.lapse_k;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int is = indcs.is;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  par_for("adm_lapse_gradient", DevExeSpace(), 0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    adm.g_dd(m,0,0,k,j,i) = 1.0;
    adm.g_dd(m,0,1,k,j,i) = 0.0;
    adm.g_dd(m,0,2,k,j,i) = 0.0;
    adm.g_dd(m,1,1,k,j,i) = 1.0;
    adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.g_dd(m,2,2,k,j,i) = 1.0;

    adm.vK_dd(m,0,0,k,j,i) = 0.0;
    adm.vK_dd(m,0,1,k,j,i) = 0.0;
    adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = 0.0;
    adm.vK_dd(m,1,2,k,j,i) = 0.0;
    adm.vK_dd(m,2,2,k,j,i) = 0.0;

    adm.psi4(m,k,j,i) = 1.0;
    adm.alpha(m,k,j,i) = 1.0 + amp*sin(kwave*x);
    adm.beta_u(m,0,k,j,i) = 0.0;
    adm.beta_u(m,1,k,j,i) = 0.0;
    adm.beta_u(m,2,k,j,i) = 0.0;
  });
}

void SetADMVariablesToMomentumMetric(MeshBlockPack *pmbp) {
  const Real alpha_amp = adm_formal_test.momentum_alpha_amp;
  const Real beta_amp = adm_formal_test.momentum_beta_amp;
  const Real metric_amp = adm_formal_test.momentum_metric_amp;
  const Real kwave = adm_formal_test.momentum_k;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int is = indcs.is;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  par_for("adm_momentum_metric", DevExeSpace(), 0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real sx = sin(kwave*x);
    const Real cx = cos(kwave*x);
    adm.g_dd(m,0,0,k,j,i) = 1.0 + metric_amp*sx;
    adm.g_dd(m,0,1,k,j,i) = 0.0;
    adm.g_dd(m,0,2,k,j,i) = 0.0;
    adm.g_dd(m,1,1,k,j,i) = 1.0 + 0.5*metric_amp*cx;
    adm.g_dd(m,1,2,k,j,i) = 0.0;
    adm.g_dd(m,2,2,k,j,i) = 1.0 + 0.25*metric_amp*sin(kwave*x + 0.3);

    adm.vK_dd(m,0,0,k,j,i) = 0.0;
    adm.vK_dd(m,0,1,k,j,i) = 0.0;
    adm.vK_dd(m,0,2,k,j,i) = 0.0;
    adm.vK_dd(m,1,1,k,j,i) = 0.0;
    adm.vK_dd(m,1,2,k,j,i) = 0.0;
    adm.vK_dd(m,2,2,k,j,i) = 0.0;

    adm.psi4(m,k,j,i) = 1.0;
    adm.alpha(m,k,j,i) = 1.0 + alpha_amp*cx;
    adm.beta_u(m,0,k,j,i) = 0.0;
    adm.beta_u(m,1,k,j,i) = beta_amp*sx;
    adm.beta_u(m,2,k,j,i) = 0.0;
  });
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationBeam(ParameterInput *pin)
//! \brief Checks tetrad is orthonormal.  Beam is introduced as rad_srcterm, so nothing
//! need be done here

void ProblemGenerator::RadiationBeam(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const bool positivity_floor_test =
      pin->GetOrAddBoolean("problem", "positivity_floor_test", false);

  // User boundary function
  user_bcs_func = ZeroIntensity;
  if (positivity_floor_test) {
    pgen_final_func = DynRadPositivityFloorCheck;
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = -1;
  DvceArray6D<Real> tet_c_;
  DvceArray6D<Real> tetcov_c_;
  DvceArray4D<Real> sqrt_detg_c_;
  DualArray2D<Real> nh_c_;
  DvceArray4D<Real> adm_alpha_c_;
  DvceArray5D<Real> adm_beta_u_c_;
  DvceArray6D<Real> adm_g_dd_c_;
  bool use_adm_geometry_ = false;
  if (pmbp->prad != nullptr) {
    nang1 = pmbp->prad->prgeo->nangles - 1;
    tet_c_ = pmbp->prad->tet_c;
  } else if (pmbp->pdynrad != nullptr) {
    nang1 = pmbp->pdynrad->prgeo->nangles - 1;
    tet_c_ = pmbp->pdynrad->tet_c;
    tetcov_c_ = pmbp->pdynrad->tetcov_c;
    sqrt_detg_c_ = pmbp->pdynrad->sqrt_detg_c;
    nh_c_ = pmbp->pdynrad->nh_c;
    use_adm_geometry_ = pmbp->pdynrad->use_adm_geometry;
    adm_alpha_c_ = pmbp->pdynrad->adm_alpha_c;
    adm_beta_u_c_ = pmbp->pdynrad->adm_beta_u_c;
    adm_g_dd_c_ = pmbp->pdynrad->adm_g_dd_c;
  } else {
    throw std::runtime_error("rad_beam requires either <radiation> or <dyn_radiation>");
  }
  auto &size = pmbp->pmb->mb_size;
  auto &flat = pmbp->pcoord->coord_data.is_minkowski;
  auto &spin = pmbp->pcoord->coord_data.bh_spin;
  auto &use_excise = pmbp->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmbp->pcoord->excision_floor;

  par_for("check_tetrad",DevExeSpace(),0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    bool excised = false;
    if (use_excise) {
      if (excision_floor_(m,k,j,i)) {
        excised = true;
      }
    }

    if (!(excised)) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real glower[4][4], gupper[4][4];
      if (use_adm_geometry_) {
        Real g4[16];
        adm::SpacetimeMetric(adm_alpha_c_(m,k,j,i),
                             adm_beta_u_c_(m,0,k,j,i),
                             adm_beta_u_c_(m,1,k,j,i),
                             adm_beta_u_c_(m,2,k,j,i),
                             adm_g_dd_c_(m,0,0,k,j,i),
                             adm_g_dd_c_(m,0,1,k,j,i),
                             adm_g_dd_c_(m,0,2,k,j,i),
                             adm_g_dd_c_(m,1,1,k,j,i),
                             adm_g_dd_c_(m,1,2,k,j,i),
                             adm_g_dd_c_(m,2,2,k,j,i), g4);
        for (int mu=0; mu<4; ++mu) {
          for (int nu=0; nu<4; ++nu) {
            glower[mu][nu] = g4[4*mu + nu];
          }
        }
      } else {
        ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
      }

      // Compute eta_alpha beta = g_mu nu e^mu_alpha e^nu_beta
      Real test_eta[4][4] = {0.0};
      for (int alpha=0; alpha<4; ++alpha) {
        for (int beta=0; beta<4; ++beta) {
          test_eta[alpha][beta] = 0.0;
          for (int mu=0; mu<4; ++mu) {
            for (int nu=0; nu<4; ++nu) {
              test_eta[alpha][beta] += (glower[mu][nu]*
                                        tet_c_(m,alpha,mu,k,j,i)*tet_c_(m,beta,nu,k,j,i));
            }
          }
        }
      }

      // Check for orthonormality
      for (int alpha=0; alpha<4; ++alpha) {
        for (int beta=0; beta<4; ++beta) {
          Real comp = 1.0;
          if   (alpha != beta) comp =  0.0;
          else if (alpha == 0) comp = -1.0;
          if (fabs(test_eta[alpha][beta] - comp) > 1.0e-13) {
            Kokkos::abort("Tetrad is not orthonormal!\n");
          }
        }
      }
    }
  });

  if (positivity_floor_test) {
    if (pmbp->pdynrad == nullptr) {
      throw std::runtime_error("positivity_floor_test requires <dyn_radiation>");
    }
    auto &i0 = pmbp->pdynrad->i0;
    par_for("dynrad_positivity_seed",DevExeSpace(),0,nmb1,0,nang1,
            0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      const Real intensity = (n == 0) ? -0.25 : 1.0;
      Real norm = sqrt_detg_c_(m,k,j,i);
      if (!(use_adm_geometry_)) {
        Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);
        }
        norm = tet_c_(m,0,0,k,j,i)*n_0;
      }
      i0(m,n,k,j,i) = norm*intensity;
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationCrossingBeams(ParameterInput *pin)
//! \brief Two noninteracting beams crossing in flat spacetime.  The initialized
//! one-sided Gaussian beam profiles use the requested physical axes, and the
//! angular weights are a positive all-angle maximum-entropy projection with exact
//! injected zeroth moment and exact first moment along the requested beam direction,
//! up to the realizable flux factor of the finite angular grid.

void ProblemGenerator::RadiationCrossingBeams(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_bcs_func = CrossingBeamBoundary;

  int nangles = -1;
  DualArray2D<Real> nh_c;
  DualArray1D<Real> solid_angles;
  if (pmbp->prad != nullptr) {
    nangles = pmbp->prad->prgeo->nangles;
    nh_c = pmbp->prad->nh_c;
    solid_angles = pmbp->prad->prgeo->solid_angles;
  } else if (pmbp->pdynrad != nullptr) {
    nangles = pmbp->pdynrad->prgeo->nangles;
    nh_c = pmbp->pdynrad->nh_c;
    solid_angles = pmbp->pdynrad->prgeo->solid_angles;
  } else {
    throw std::runtime_error("rad_crossing_beams requires <radiation> or <dyn_radiation>");
  }

  crossing_beams.enabled = true;
  crossing_beams.amp = pin->GetOrAddReal("problem", "beam_amp", 1.0);
  crossing_beams.sigma = pin->GetOrAddReal("problem", "beam_sigma", 0.055);
  crossing_beams.flux_fraction = pin->GetOrAddReal("problem", "beam_flux_fraction", 0.995);
  crossing_beams.x0 = pin->GetOrAddReal("problem", "beam_x0", 0.12);
  crossing_beams.y_lower = pin->GetOrAddReal("problem", "beam_y_lower", 0.15);
  crossing_beams.y_upper = pin->GetOrAddReal("problem", "beam_y_upper", 0.85);
  const Real x_cross = pin->GetOrAddReal("problem", "beam_x_cross", 0.75);
  const Real y_cross = pin->GetOrAddReal("problem", "beam_y_cross", 0.5);

  Real lower_tx = x_cross - crossing_beams.x0;
  Real lower_ty = y_cross - crossing_beams.y_lower;
  Real upper_tx = x_cross - crossing_beams.x0;
  Real upper_ty = y_cross - crossing_beams.y_upper;
  Real lower_norm = sqrt(SQR(lower_tx) + SQR(lower_ty));
  Real upper_norm = sqrt(SQR(upper_tx) + SQR(upper_ty));
  if (lower_norm <= 0.0 || upper_norm <= 0.0) {
    throw std::runtime_error("rad_crossing_beams requires nonzero beam directions");
  }
  lower_tx /= lower_norm;
  lower_ty /= lower_norm;
  upper_tx /= upper_norm;
  upper_ty /= upper_norm;

  if (crossing_beams.angular_weights == nullptr) {
    crossing_beams.angular_weights = new DvceArray2D<Real>();
  }
  Kokkos::realloc(*(crossing_beams.angular_weights), 2, nangles);
  auto h_weights = Kokkos::create_mirror_view(*(crossing_beams.angular_weights));
  SetAllAngleMomentWeights(nh_c, solid_angles, h_weights,
                           0, nangles, lower_tx, lower_ty, 0.0,
                           crossing_beams.flux_fraction, "rad_crossing_beams");
  SetAllAngleMomentWeights(nh_c, solid_angles, h_weights,
                           1, nangles, upper_tx, upper_ty, 0.0,
                           crossing_beams.flux_fraction, "rad_crossing_beams");
  Kokkos::deep_copy(*(crossing_beams.angular_weights), h_weights);
  crossing_beams.lower_profile_qx = lower_tx;
  crossing_beams.lower_profile_qy = lower_ty;
  crossing_beams.upper_profile_qx = upper_tx;
  crossing_beams.upper_profile_qy = upper_ty;

  if (!(restart)) {
    FillCrossingBeams(pmy_mesh_, false);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationKerrOrbitBeam(ParameterInput *pin)
//! \brief Continuous projected beam source on an equatorial Kerr photon orbit.

void ProblemGenerator::RadiationKerrOrbitBeam(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  user_bcs_func = ZeroIntensity;
  user_srcs_func = KerrOrbitBeamSource;

  int nangles = -1;
  DualArray2D<Real> nh_c;
  if (pmbp->prad != nullptr) {
    nangles = pmbp->prad->prgeo->nangles;
    nh_c = pmbp->prad->nh_c;
    if (!(restart)) { Kokkos::deep_copy(pmbp->prad->i0, 0.0); }
  } else if (pmbp->pdynrad != nullptr) {
    nangles = pmbp->pdynrad->prgeo->nangles;
    nh_c = pmbp->pdynrad->nh_c;
    if (!(restart)) { Kokkos::deep_copy(pmbp->pdynrad->i0, 0.0); }
  } else {
    throw std::runtime_error("rad_kerr_orbit_beam requires <radiation> or <dyn_radiation>");
  }

  const auto &coord = pmbp->pcoord->coord_data;
  const Real spin = coord.bh_spin;
  const bool flat = coord.is_minkowski;
  if (flat) {
    throw std::runtime_error("rad_kerr_orbit_beam requires a Kerr-Schild metric");
  }
  const Real default_r = CounterrotatingPhotonOrbitRadius(spin);
  const Real orbit_r = pin->GetOrAddReal("problem", "orbit_r", default_r);
  const Real orbit_R = sqrt(SQR(orbit_r) + SQR(spin));
  const Real source_phi = pin->GetOrAddReal("problem", "source_phi", 0.0);
  kerr_orbit_beam.enabled = true;
  kerr_orbit_beam.amp = pin->GetOrAddReal("problem", "beam_amp", 1.0);
  kerr_orbit_beam.sigma = pin->GetOrAddReal("problem", "beam_sigma", 0.18);
  kerr_orbit_beam.source_x = orbit_R*cos(source_phi);
  kerr_orbit_beam.source_y = orbit_R*sin(source_phi);
  kerr_orbit_beam.source_z = 0.0;

  const Real tangent_x = -sin(source_phi);
  const Real tangent_y =  cos(source_phi);
  Real ell[3];
  const bool use_adm_geometry = (pmbp->pdynrad != nullptr &&
                                 pmbp->pdynrad->use_adm_geometry);
  if (use_adm_geometry) {
    CoordinateDirectionToADMTetrad(kerr_orbit_beam.source_x, kerr_orbit_beam.source_y,
                                   kerr_orbit_beam.source_z, flat, spin,
                                   tangent_x, tangent_y, 0.0, ell);
  } else {
    CoordinateDirectionToTetrad(kerr_orbit_beam.source_x, kerr_orbit_beam.source_y,
                                kerr_orbit_beam.source_z, flat, spin,
                                tangent_x, tangent_y, 0.0, ell);
  }

  if (kerr_orbit_beam.angular_weights == nullptr) {
    kerr_orbit_beam.angular_weights = new DvceArray2D<Real>();
  }
  Kokkos::realloc(*(kerr_orbit_beam.angular_weights), 1, nangles);
  auto h_weights = Kokkos::create_mirror_view(*(kerr_orbit_beam.angular_weights));
  SetProjectedAngularWeights(nh_c, h_weights, 0, nangles, ell[0], ell[1], ell[2],
                             "rad_kerr_orbit_beam");
  Kokkos::deep_copy(*(kerr_orbit_beam.angular_weights), h_weights);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationFLRWRedshift(ParameterInput *pin)
//! \brief Homogeneous isotropic radiation in an analytic flat FLRW ADM background.

void ProblemGenerator::RadiationFLRWRedshift(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pdynrad == nullptr || !(pmbp->pdynrad->use_adm_geometry) ||
      pmbp->padm == nullptr) {
    throw std::runtime_error("rad_flrw_redshift requires ADM dyn_radiation");
  }
  adm_formal_test.flrw_h = pin->GetOrAddReal("problem", "hubble", 0.2);
  adm_formal_test.flrw_t0 = pin->GetOrAddReal("problem", "t0", 0.0);
  pmbp->padm->SetADMVariables = &SetADMVariablesToFLRWRedshift;
  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdynrad->PrepareADMGeometry();
  pgen_final_func = DynRadFLRWRedshiftCheck;
  if (restart) { return; }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang1 = pmbp->pdynrad->prgeo->nangles - 1;
  const Real erad = pin->GetOrAddReal("problem", "erad", 1.0);
  auto &i0 = pmbp->pdynrad->i0;
  auto &sqrt_detg = pmbp->pdynrad->sqrt_detg_c;
  par_for("flrw_redshift_init", DevExeSpace(), 0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    i0(m,n,k,j,i) = sqrt_detg(m,k,j,i)*erad/(4.0*M_PI);
  });
  Kokkos::fence();
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationLapseGradient(ParameterInput *pin)
//! \brief Uniform anisotropic radiation in a static periodic lapse gradient.

void ProblemGenerator::RadiationLapseGradient(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pdynrad == nullptr || !(pmbp->pdynrad->use_adm_geometry) ||
      pmbp->padm == nullptr) {
    throw std::runtime_error("rad_lapse_gradient requires ADM dyn_radiation");
  }
  adm_formal_test.lapse_amp = pin->GetOrAddReal("problem", "lapse_amp", 0.1);
  adm_formal_test.lapse_k = pin->GetOrAddReal("problem", "lapse_k", 2.0*M_PI);
  pmbp->padm->SetADMVariables = &SetADMVariablesToLapseGradient;
  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdynrad->PrepareADMGeometry();
  pgen_final_func = DynRadLapseGradientCheck;
  if (restart) { return; }

  const int nangles = pmbp->pdynrad->prgeo->nangles;
  const Real flux_fraction = pin->GetOrAddReal("problem", "flux_fraction", 0.7);
  DvceArray2D<Real> weights("lapse_gradient_weights", 1, nangles);
  auto h_weights = Kokkos::create_mirror_view(weights);
  SetAllAngleMomentWeights(pmbp->pdynrad->nh_c, pmbp->pdynrad->prgeo->solid_angles,
                           h_weights, 0, nangles, 1.0, 0.0, 0.0,
                           flux_fraction, "rad_lapse_gradient");
  Kokkos::deep_copy(weights, h_weights);

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang1 = nangles - 1;
  const Real erad = pin->GetOrAddReal("problem", "erad", 1.0);
  auto &i0 = pmbp->pdynrad->i0;
  auto &sqrt_detg = pmbp->pdynrad->sqrt_detg_c;
  auto &solid_angles = pmbp->pdynrad->prgeo->solid_angles;
  par_for("lapse_gradient_init", DevExeSpace(), 0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    i0(m,n,k,j,i) = sqrt_detg(m,k,j,i)*erad*weights(0,n)/solid_angles.d_view(n);
  });
  Kokkos::fence();
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationMomentumSource(ParameterInput *pin)
//! \brief Local ADM Hamiltonian-force versus Valencia momentum-source closure.

void ProblemGenerator::RadiationMomentumSource(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pdynrad == nullptr || !(pmbp->pdynrad->use_adm_geometry) ||
      pmbp->padm == nullptr) {
    throw std::runtime_error("rad_momentum_source requires ADM dyn_radiation");
  }
  adm_formal_test.momentum_alpha_amp =
      pin->GetOrAddReal("problem", "alpha_amp", 0.05);
  adm_formal_test.momentum_beta_amp =
      pin->GetOrAddReal("problem", "beta_amp", 0.04);
  adm_formal_test.momentum_metric_amp =
      pin->GetOrAddReal("problem", "metric_amp", 0.03);
  adm_formal_test.momentum_k = pin->GetOrAddReal("problem", "metric_k", 2.0*M_PI);
  pmbp->padm->SetADMVariables = &SetADMVariablesToMomentumMetric;
  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdynrad->PrepareADMGeometry();
  pgen_final_func = DynRadMomentumSourceCheck;
  if (restart) { return; }

  const int nangles = pmbp->pdynrad->prgeo->nangles;
  const Real flux_fraction = pin->GetOrAddReal("problem", "flux_fraction", 0.45);
  DvceArray2D<Real> weights("momentum_source_weights", 1, nangles);
  auto h_weights = Kokkos::create_mirror_view(weights);
  SetAllAngleMomentWeights(pmbp->pdynrad->nh_c, pmbp->pdynrad->prgeo->solid_angles,
                           h_weights, 0, nangles, 0.7, 0.4, 0.2,
                           flux_fraction, "rad_momentum_source");
  Kokkos::deep_copy(weights, h_weights);

  auto &indcs = pmy_mesh_->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int nang1 = nangles - 1;
  const Real erad = pin->GetOrAddReal("problem", "erad", 1.0);
  auto &i0 = pmbp->pdynrad->i0;
  auto &sqrt_detg = pmbp->pdynrad->sqrt_detg_c;
  auto &solid_angles = pmbp->pdynrad->prgeo->solid_angles;
  par_for("momentum_source_init", DevExeSpace(), 0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    i0(m,n,k,j,i) = sqrt_detg(m,k,j,i)*erad*weights(0,n)/solid_angles.d_view(n);
  });
  Kokkos::fence();
}

//----------------------------------------------------------------------------------------
//! \fn DynRadPositivityFloorCheck
//! \brief Checks the conservative angular positivity limiter used by dyn_radiation.

void DynRadPositivityFloorCheck(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pdynrad == nullptr) {
    throw std::runtime_error("positivity_floor_test requires <dyn_radiation>");
  }

  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  auto &i0 = pmbp->pdynrad->i0;
  auto &sqrt_detg_c = pmbp->pdynrad->sqrt_detg_c;
  auto &tet_c = pmbp->pdynrad->tet_c;
  auto &tetcov_c = pmbp->pdynrad->tetcov_c;
  auto &nh_c = pmbp->pdynrad->nh_c;
  auto &solid_angles = pmbp->pdynrad->prgeo->solid_angles;
  const bool use_adm_geometry = pmbp->pdynrad->use_adm_geometry;
  const int nang1 = pmbp->pdynrad->prgeo->nangles - 1;
  Real min_i = std::numeric_limits<Real>::max();
  Real max_m0_err = 0.0;
  Kokkos::parallel_reduce("dynrad_positivity_check",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &thread_min_i, Real &thread_max_err) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real m0 = 0.0;
    Real expected_m0 = 0.0;
    for (int n=0; n<=nang1; ++n) {
      Real norm = sqrt_detg_c(m,k,j,i);
      if (!(use_adm_geometry)) {
        Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          n_0 += tetcov_c(m,d,0,k,j,i)*nh_c.d_view(n,d);
        }
        norm = tet_c(m,0,0,k,j,i)*n_0;
      }
      const Real intensity = i0(m,n,k,j,i)/norm;
      thread_min_i = fmin(thread_min_i, intensity);
      m0 += intensity*solid_angles.d_view(n);
      expected_m0 += ((n == 0) ? -0.25 : 1.0)*solid_angles.d_view(n);
    }
    thread_max_err = fmax(thread_max_err, fabs(m0 - expected_m0));
  }, Kokkos::Min<Real>(min_i), Kokkos::Max<Real>(max_m0_err));

  if (min_i < -1.0e-13 || max_m0_err > 1.0e-11) {
    std::cout << "### FATAL ERROR in dyn_radiation positivity limiter test"
              << std::endl << "min_i=" << min_i
              << " max_m0_err=" << max_m0_err << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn DynRadFLRWRedshiftCheck
//! \brief Check homogeneous redshift E ~ a^-4 and sqrt(gamma)E ~ a^-1.

void DynRadFLRWRedshiftCheck(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const Real erad = pin->GetOrAddReal("problem", "erad", 1.0);
  const Real tol = pin->GetOrAddReal("problem", "redshift_tolerance", 5.0e-3);
  const Real h = pin->GetOrAddReal("problem", "hubble", adm_formal_test.flrw_h);
  const Real t0 = pin->GetOrAddReal("problem", "t0", adm_formal_test.flrw_t0);
  adm_formal_test.flrw_h = h;
  adm_formal_test.flrw_t0 = t0;
  if (pmbp->padm != nullptr && pmbp->pdynrad != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
    pmbp->pdynrad->PrepareADMGeometry();
  }
  const Real a = 1.0 + h*(pm->time - t0);
  const Real exact_e = erad/std::pow(a, 4);
  const Real exact_u = erad/a;

  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  auto &i0 = pmbp->pdynrad->i0;
  auto &sqrt_detg = pmbp->pdynrad->sqrt_detg_c;
  auto &solid_angles = pmbp->pdynrad->prgeo->solid_angles;
  const int nang1 = pmbp->pdynrad->prgeo->nangles - 1;
  Real sum_e = 0.0;
  Real sum_u = 0.0;
  Real sum_cells = 0.0;
  Kokkos::parallel_reduce("flrw_redshift_check",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &thread_e, Real &thread_u, Real &thread_cells) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real e = 0.0;
    Real u = 0.0;
    for (int n=0; n<=nang1; ++n) {
      u += i0(m,n,k,j,i)*solid_angles.d_view(n);
      e += (i0(m,n,k,j,i)/sqrt_detg(m,k,j,i))*solid_angles.d_view(n);
    }
    thread_e += e;
    thread_u += u;
    thread_cells += 1.0;
  }, Kokkos::Sum<Real>(sum_e), Kokkos::Sum<Real>(sum_u),
     Kokkos::Sum<Real>(sum_cells));

  const Real mean_e = sum_e/sum_cells;
  const Real mean_u = sum_u/sum_cells;
  const Real rel_e = fabs(mean_e - exact_e)/fmax(fabs(exact_e), 1.0e-300);
  const Real rel_u = fabs(mean_u - exact_u)/fmax(fabs(exact_u), 1.0e-300);
  std::cout << std::setprecision(16)
            << "ADM_FORMAL_TEST flrw"
            << " a=" << a
            << " mean_E=" << mean_e
            << " exact_E=" << exact_e
            << " rel_E=" << rel_e
            << " mean_U=" << mean_u
            << " exact_U=" << exact_u
            << " rel_U=" << rel_u << std::endl;
  if (rel_e > tol || rel_u > tol) {
    std::cout << "### FATAL ERROR in FLRW redshift test" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn DynRadLapseGradientCheck
//! \brief Check sign and magnitude of the static lapse-gradient energy source.

void DynRadLapseGradientCheck(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const Real erad = pin->GetOrAddReal("problem", "erad", 1.0);
  const Real flux_fraction = pin->GetOrAddReal("problem", "flux_fraction", 0.7);
  const Real amp = pin->GetOrAddReal("problem", "lapse_amp", adm_formal_test.lapse_amp);
  const Real kwave = pin->GetOrAddReal("problem", "lapse_k", adm_formal_test.lapse_k);
  const Real tol = pin->GetOrAddReal("problem", "lapse_tolerance", 0.35);
  const Real time = pm->time;

  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  auto &size = pmbp->pmb->mb_size;
  auto &i0 = pmbp->pdynrad->i0;
  auto &sqrt_detg = pmbp->pdynrad->sqrt_detg_c;
  auto &solid_angles = pmbp->pdynrad->prgeo->solid_angles;
  const int nang1 = pmbp->pdynrad->prgeo->nangles - 1;
  Real err2 = 0.0;
  Real sig2 = 0.0;
  Real data2 = 0.0;
  Real cross = 0.0;
  Kokkos::parallel_reduce("lapse_gradient_check",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &thread_err2, Real &thread_sig2,
                Real &thread_data2, Real &thread_cross) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real dalpha_dx = amp*kwave*cos(kwave*x);
    const Real expected = erad - 2.0*erad*flux_fraction*dalpha_dx*time;
    Real e = 0.0;
    for (int n=0; n<=nang1; ++n) {
      e += (i0(m,n,k,j,i)/sqrt_detg(m,k,j,i))*solid_angles.d_view(n);
    }
    const Real signal = expected - erad;
    const Real data = e - erad;
    const Real err = e - expected;
    thread_err2 += SQR(err);
    thread_sig2 += SQR(signal);
    thread_data2 += SQR(data);
    thread_cross += signal*data;
  }, Kokkos::Sum<Real>(err2), Kokkos::Sum<Real>(sig2),
     Kokkos::Sum<Real>(data2), Kokkos::Sum<Real>(cross));

  const Real rel_rms = sqrt(err2/fmax(sig2, 1.0e-300));
  const Real corr = cross/sqrt(fmax(sig2*data2, 1.0e-300));
  std::cout << std::setprecision(16)
            << "ADM_FORMAL_TEST lapse_gradient"
            << " rel_rms=" << rel_rms
            << " corr=" << corr
            << " time=" << pm->time << std::endl;
  if (rel_rms > tol || corr < 0.8) {
    std::cout << "### FATAL ERROR in lapse-gradient test" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn DynRadMomentumSourceCheck
//! \brief Compare angular-summed Hamiltonian force with Valencia momentum source.

void DynRadMomentumSourceCheck(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const Real abs_tol = pin->GetOrAddReal("problem", "momentum_abs_tolerance", 1.0e-10);
  const Real rel_tol = pin->GetOrAddReal("problem", "momentum_rel_tolerance", 1.0e-10);

  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  auto &rad = *(pmbp->pdynrad);
  auto &i0 = rad.i0;
  auto &sqrt_detg = rad.sqrt_detg_c;
  auto &solid_angles = rad.prgeo->solid_angles;
  auto &nh_c = rad.nh_c;
  auto &tet_c = rad.tet_c;
  auto &adm_alpha = rad.adm_alpha_c;
  auto &adm_g_dd = rad.adm_g_dd_c;
  auto &adm_grad_alpha = rad.adm_grad_alpha_c;
  auto &adm_grad_beta = rad.adm_grad_beta_u_c;
  auto &adm_grad_g_dd = rad.adm_grad_g_dd_c;
  auto &adm_grad_g_uu = rad.adm_grad_g_uu_c;
  const int nang1 = rad.prgeo->nangles - 1;
  Real max_abs = 0.0;
  Real max_rel = 0.0;
  Kokkos::parallel_reduce("momentum_source_check",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &thread_abs, Real &thread_rel) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real e = 0.0;
    Real s_cov[3] = {0.0, 0.0, 0.0};
    Real p_uu[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    Real ham[3] = {0.0, 0.0, 0.0};
    for (int n=0; n<=nang1; ++n) {
      const Real intensity = i0(m,n,k,j,i)/sqrt_detg(m,k,j,i);
      const Real weight = intensity*solid_angles.d_view(n);
      Real s_con[3] = {0.0, 0.0, 0.0};
      for (int a=0; a<3; ++a) {
        for (int d=0; d<3; ++d) {
          s_con[d] += tet_c(m,a+1,d+1,k,j,i)*nh_c.d_view(n,a+1);
        }
      }
      Real p_cov[3] = {0.0, 0.0, 0.0};
      for (int a=0; a<3; ++a) {
        for (int b=0; b<3; ++b) {
          p_cov[a] += adm_g_dd(m,a,b,k,j,i)*s_con[b];
        }
      }
      e += weight;
      for (int a=0; a<3; ++a) {
        s_cov[a] += weight*p_cov[a];
        for (int b=0; b<3; ++b) {
          p_uu[a][b] += weight*s_con[a]*s_con[b];
        }
      }
      for (int d=0; d<3; ++d) {
        Real a_force = -adm_grad_alpha(m,d,k,j,i);
        for (int a=0; a<3; ++a) {
          a_force += p_cov[a]*adm_grad_beta(m,3*d+a,k,j,i);
          for (int b=0; b<3; ++b) {
            a_force -= 0.5*adm_alpha(m,k,j,i)*p_cov[a]*p_cov[b]*
                       adm_grad_g_uu(m,6*d+LocalSym3Index(a,b),k,j,i);
          }
        }
        ham[d] += weight*a_force;
      }
    }

    for (int d=0; d<3; ++d) {
      Real valencia = -e*adm_grad_alpha(m,d,k,j,i);
      for (int a=0; a<3; ++a) {
        valencia += s_cov[a]*adm_grad_beta(m,3*d+a,k,j,i);
        for (int b=0; b<3; ++b) {
          valencia += 0.5*adm_alpha(m,k,j,i)*p_uu[a][b]*
                      adm_grad_g_dd(m,6*d+LocalSym3Index(a,b),k,j,i);
        }
      }
      const Real diff = fabs(ham[d] - valencia);
      thread_abs = fmax(thread_abs, diff);
      thread_rel = fmax(thread_rel, diff/fmax(fabs(valencia), 1.0e-12));
    }
  }, Kokkos::Max<Real>(max_abs), Kokkos::Max<Real>(max_rel));

  std::cout << std::setprecision(16)
            << "ADM_FORMAL_TEST momentum_source"
            << " max_abs=" << max_abs
            << " max_rel=" << max_rel << std::endl;
  if (max_abs > abs_tol && max_rel > rel_tol) {
    std::cout << "### FATAL ERROR in momentum-source closure test" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn KerrOrbitBeamSource
//! \brief Adds a compact, moment-projected beam source tangent to a Kerr photon orbit.

void KerrOrbitBeamSource(Mesh *pm, const Real bdt) {
  if (!(kerr_orbit_beam.enabled)) { return; }
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = pmbp->nmb_thispack - 1;

  int nang1 = -1;
  bool use_adm_geometry = false;
  DvceArray5D<Real> i0;
  DualArray2D<Real> nh_c;
  DualArray1D<Real> solid_angles;
  DvceArray6D<Real> tet_c;
  DvceArray6D<Real> tetcov_c;
  DvceArray4D<Real> sqrt_detg_c;
  if (pmbp->prad != nullptr) {
    nang1 = pmbp->prad->prgeo->nangles - 1;
    i0 = pmbp->prad->i0;
    nh_c = pmbp->prad->nh_c;
    solid_angles = pmbp->prad->prgeo->solid_angles;
    tet_c = pmbp->prad->tet_c;
    tetcov_c = pmbp->prad->tetcov_c;
  } else if (pmbp->pdynrad != nullptr) {
    nang1 = pmbp->pdynrad->prgeo->nangles - 1;
    use_adm_geometry = pmbp->pdynrad->use_adm_geometry;
    i0 = pmbp->pdynrad->i0;
    nh_c = pmbp->pdynrad->nh_c;
    solid_angles = pmbp->pdynrad->prgeo->solid_angles;
    tet_c = pmbp->pdynrad->tet_c;
    tetcov_c = pmbp->pdynrad->tetcov_c;
    sqrt_detg_c = pmbp->pdynrad->sqrt_detg_c;
  } else {
    throw std::runtime_error("rad_kerr_orbit_beam requires <radiation> or <dyn_radiation>");
  }
  if (kerr_orbit_beam.angular_weights == nullptr) {
    throw std::runtime_error("rad_kerr_orbit_beam angular weights were not initialized");
  }

  auto &size = pmbp->pmb->mb_size;
  auto &excise = pmbp->pcoord->coord_data.bh_excise;
  auto &rad_mask = pmbp->pcoord->excision_floor;
  auto angular_weights = *(kerr_orbit_beam.angular_weights);
  const Real amp = kerr_orbit_beam.amp;
  const Real sigma = kerr_orbit_beam.sigma;
  const Real sx = kerr_orbit_beam.source_x;
  const Real sy = kerr_orbit_beam.source_y;
  const Real sz = kerr_orbit_beam.source_z;

  par_for("kerr_orbit_beam_source",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    if (excise && rad_mask(m,k,j,i)) { return; }

    const Real x = CellCenterX(i-is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const Real dist2 = SQR(x - sx) + SQR(y - sy) + SQR(z - sz);
    const Real primitive_source = amp*bdt*exp(-0.5*dist2/SQR(sigma))*
                                  angular_weights(0,n)/solid_angles.d_view(n);

    Real norm = 1.0;
    if (use_adm_geometry) {
      norm = sqrt_detg_c(m,k,j,i);
    } else {
      Real n_0 = 0.0;
      for (int d=0; d<4; ++d) {
        n_0 += tetcov_c(m,d,0,k,j,i)*nh_c.d_view(n,d);
      }
      norm = tet_c(m,0,0,k,j,i)*n_0;
    }
    i0(m,n,k,j,i) += norm*primitive_source;
  });
}

//----------------------------------------------------------------------------------------
//! \fn ZeroIntensity
//! \brief Sets boundary condition on surfaces of computational domain

void ZeroIntensity(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  // Determine if radiation is enabled
  bool is_radiation_enabled_ = (pm->pmb_pack->prad != nullptr ||
                                pm->pmb_pack->pdynrad != nullptr);
  DvceArray5D<Real> i0_; int nang1;
  if (pm->pmb_pack->prad != nullptr) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  } else if (pm->pmb_pack->pdynrad != nullptr) {
    i0_ = pm->pmb_pack->pdynrad->i0;
    nang1 = pm->pmb_pack->pdynrad->prgeo->nangles - 1;
  }
  int nmb = pm->pmb_pack->nmb_thispack;

  // X1-Boundary
  if (is_radiation_enabled_) {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = 0.0;
        }
      }
    });
  }

  // X2-Boundary
  if (is_radiation_enabled_) {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = 0.0;
        }
      }
    });
  }

  // x3-Boundary
  if (is_radiation_enabled_) {
    // Set x3-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = 0.0;
        }
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn CrossingBeamBoundary
//! \brief Fills physical ghost zones with the analytic crossing-beam inflow/outflow state.

void CrossingBeamBoundary(Mesh *pm) {
  FillCrossingBeams(pm, true);
}
