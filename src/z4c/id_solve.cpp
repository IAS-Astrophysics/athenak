#include "z4c/id_solve.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/nghbr_index.hpp"
#include "parameter_input.hpp"
#include "utils/finite_diff.hpp"
#include "z4c/tmunu.hpp"
#include "z4c/z4c.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {
namespace {

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
Real DxxCenterCoeff(int fd_stencil, Real idx) {
  Real c = -2.0;
  if (fd_stencil == 3) c = -2.5;
  if (fd_stencil == 4) c = -49.0/18.0;
  return c*idx*idx;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
Real FreeSym(const ViewType &free, int base, int m, int a, int b, int k, int j, int i) {
  return free(m, base + SymIdx(a,b), k, j, i);
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
void MetricInverse(const ViewType &free, int m, int k, int j, int i, Real gu[3][3]) {
  Real det = adm::SpatialDet(free(m,ID_FREE_GXX,k,j,i), free(m,ID_FREE_GXY,k,j,i),
                             free(m,ID_FREE_GXZ,k,j,i), free(m,ID_FREE_GYY,k,j,i),
                             free(m,ID_FREE_GYZ,k,j,i), free(m,ID_FREE_GZZ,k,j,i));
  det = (det > 0.0) ? det : 1.0;
  adm::SpatialInv(1.0/det,
                  free(m,ID_FREE_GXX,k,j,i), free(m,ID_FREE_GXY,k,j,i),
                  free(m,ID_FREE_GXZ,k,j,i), free(m,ID_FREE_GYY,k,j,i),
                  free(m,ID_FREE_GYZ,k,j,i), free(m,ID_FREE_GZZ,k,j,i),
                  &gu[0][0], &gu[0][1], &gu[0][2],
                  &gu[1][1], &gu[1][2], &gu[2][2]);
  gu[1][0] = gu[0][1];
  gu[2][0] = gu[0][2];
  gu[2][1] = gu[1][2];
}

template <typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
Real TotalU(const UView &u, const FView &free, int m, int v, int k, int j, int i) {
  int base = (v == ID_CTS_PSI) ? ID_FREE_BASE_PSI : ID_FREE_BASE_BETAX + (v - ID_CTS_BETAX);
  return free(m, base, k, j, i) + u(m, v, k, j, i);
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
Real DxTotal(int dir, const Real idx[], const UView &u, const FView &free, int m, int v,
             int k, int j, int i) {
  int base = (v == ID_CTS_PSI) ? ID_FREE_BASE_PSI : ID_FREE_BASE_BETAX + (v - ID_CTS_BETAX);
  return Dx<NGHOST>(dir, idx, u, m, v, k, j, i)
       + Dx<NGHOST>(dir, idx, free, m, base, k, j, i);
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
Real DxxTotal(int dir, const Real idx[], const UView &u, const FView &free, int m, int v,
              int k, int j, int i) {
  int base = (v == ID_CTS_PSI) ? ID_FREE_BASE_PSI : ID_FREE_BASE_BETAX + (v - ID_CTS_BETAX);
  return Dxx<NGHOST>(dir, idx, u, m, v, k, j, i)
       + Dxx<NGHOST>(dir, idx, free, m, base, k, j, i);
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
Real DxyTotal(int dir1, int dir2, const Real idx[], const UView &u, const FView &free,
              int m, int v, int k, int j, int i) {
  int base = (v == ID_CTS_PSI) ? ID_FREE_BASE_PSI : ID_FREE_BASE_BETAX + (v - ID_CTS_BETAX);
  return Dxy<NGHOST>(dir1, dir2, idx, u, m, v, k, j, i)
       + Dxy<NGHOST>(dir1, dir2, idx, free, m, base, k, j, i);
}

template <int NGHOST, typename ViewType>
KOKKOS_INLINE_FUNCTION
void Christoffel(const ViewType &free, const Real idx[], int m, int k, int j, int i,
                 Real gu[3][3], Real gamma[3][3][3]) {
  Real dg[3][3][3];
  for (int c = 0; c < 3; ++c) {
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        dg[c][a][b] = Dx<NGHOST>(c, idx, free, m, ID_FREE_GXX + SymIdx(a,b), k, j, i);
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        gamma[c][a][b] = 0.0;
        for (int d = 0; d < 3; ++d) {
          gamma[c][a][b] += 0.5*gu[c][d]*(dg[a][b][d] + dg[b][a][d] - dg[d][a][b]);
        }
      }
    }
  }
}

template <int NGHOST, typename ViewType>
KOKKOS_INLINE_FUNCTION
Real RicciScalar(const ViewType &free, const Real idx[], int m, int k, int j, int i,
                 Real gu[3][3], Real gamma[3][3][3]) {
  Real dg[3][3][3];
  Real ddg[3][3][3][3];
  for (int c = 0; c < 3; ++c) {
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        dg[c][a][b] = Dx<NGHOST>(c, idx, free, m, ID_FREE_GXX + SymIdx(a,b), k, j, i);
      }
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          int v = ID_FREE_GXX + SymIdx(c,d);
          ddg[a][b][c][d] = (a == b) ? Dxx<NGHOST>(a, idx, free, m, v, k, j, i)
                                     : Dxy<NGHOST>(a, b, idx, free, m, v, k, j, i);
        }
      }
    }
  }

  Real ricci[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      ricci[a][b] = 0.0;
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          for (int e = 0; e < 3; ++e) {
            Real gamma_ebd = 0.0;
            Real gamma_ecd = 0.0;
            for (int f = 0; f < 3; ++f) {
              gamma_ebd += FreeSym(free, ID_FREE_GXX, m, e, f, k, j, i)*gamma[f][b][d];
              gamma_ecd += FreeSym(free, ID_FREE_GXX, m, e, f, k, j, i)*gamma[f][c][d];
            }
            ricci[a][b] += gu[c][d]*gamma[e][a][c]*gamma_ebd;
            ricci[a][b] -= gu[c][d]*gamma[e][a][b]*gamma_ecd;
          }
          ricci[a][b] += 0.5*gu[c][d]*(-ddg[c][d][a][b] - ddg[a][b][c][d]
                                        + ddg[a][c][b][d] + ddg[b][c][a][d]);
        }
      }
    }
  }

  Real r = 0.0;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      r += gu[a][b]*ricci[a][b];
  return r;
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
Real AHatUU(const UView &u, const FView &free, const Real idx[], int m,
            int k, int j, int i, int a, int b) {
  Real gu[3][3];
  Real gamma[3][3][3];
  MetricInverse(free, m, k, j, i, gu);
  Christoffel<NGHOST>(free, idx, m, k, j, i, gu, gamma);

  Real div_beta = 0.0;
  Real dbeta[3][3];
  for (int c = 0; c < 3; ++c) {
    dbeta[c][c] = DxTotal<NGHOST>(c, idx, u, free, m, ID_CTS_BETAX+c, k, j, i);
    div_beta += dbeta[c][c];
  }
  for (int c = 0; c < 3; ++c) {
    for (int d = 0; d < 3; ++d) {
      if (c != d) dbeta[c][d] = DxTotal<NGHOST>(c, idx, u, free, m,
                                                ID_CTS_BETAX+d, k, j, i);
    }
  }

  Real lbeta = -(2.0/3.0)*gu[a][b]*div_beta;
  for (int c = 0; c < 3; ++c) {
    lbeta += gu[a][c]*dbeta[c][b] + gu[b][c]*dbeta[c][a];
    for (int d = 0; d < 3; ++d) {
      Real beta_d = TotalU(u, free, m, ID_CTS_BETAX+d, k, j, i);
      lbeta += (gu[a][c]*gamma[b][c][d] + gu[b][c]*gamma[a][c][d]
                - (2.0/3.0)*gu[a][b]*gamma[c][c][d])*beta_d;
    }
  }

  Real gdot_trace = 0.0;
  for (int c = 0; c < 3; ++c)
    for (int d = 0; d < 3; ++d)
      gdot_trace += FreeSym(free, ID_FREE_GXX, m, c, d, k, j, i)
                    * FreeSym(free, ID_FREE_GDOTXX, m, c, d, k, j, i);

  Real alpha = std::max(free(m, ID_FREE_ALPHA, k, j, i), static_cast<Real>(1.0e-12));
  // BuildGammaDotAndDK stores a trace-free tensor; this projection keeps the
  // operator robust if roundoff or future coefficient fills introduce drift.
  Real gdot_tf = FreeSym(free, ID_FREE_GDOTXX, m, a, b, k, j, i)
                 - (1.0/3.0)*gdot_trace*gu[a][b];
  return (gdot_tf + lbeta)/(2.0*alpha);
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
Real DxAHatUU(int dir, const Real idx[], const UView &u, const FView &free,
              int m, int k, int j, int i, int a, int b) {
  Real gu[3][3], gamma[3][3][3];
  MetricInverse(free, m, k, j, i, gu);
  Christoffel<NGHOST>(free, idx, m, k, j, i, gu, gamma);

  Real dg[3][3][3];
  Real ddg[3][3][3][3];
  for (int p = 0; p < 3; ++p) {
    for (int c = 0; c < 3; ++c) {
      for (int d = 0; d < 3; ++d) {
        int v = ID_FREE_GXX + SymIdx(c, d);
        dg[p][c][d] = Dx<NGHOST>(p, idx, free, m, v, k, j, i);
      }
    }
  }
  for (int p = 0; p < 3; ++p) {
    for (int q = 0; q < 3; ++q) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          int v = ID_FREE_GXX + SymIdx(c, d);
          ddg[p][q][c][d] = (p == q) ? Dxx<NGHOST>(p, idx, free, m, v, k, j, i)
                                     : Dxy<NGHOST>(p, q, idx, free, m, v, k, j, i);
        }
      }
    }
  }

  Real dgu[3][3][3];
  for (int p = 0; p < 3; ++p) {
    for (int c = 0; c < 3; ++c) {
      for (int d = 0; d < 3; ++d) {
        dgu[p][c][d] = 0.0;
        for (int e = 0; e < 3; ++e) {
          for (int f = 0; f < 3; ++f) {
            dgu[p][c][d] -= gu[c][e]*gu[d][f]*dg[p][e][f];
          }
        }
      }
    }
  }

  Real dgamma[3][3][3][3];
  for (int p = 0; p < 3; ++p) {
    for (int c = 0; c < 3; ++c) {
      for (int e = 0; e < 3; ++e) {
        for (int f = 0; f < 3; ++f) {
          dgamma[p][c][e][f] = 0.0;
          for (int d = 0; d < 3; ++d) {
            Real term = dg[e][f][d] + dg[f][e][d] - dg[d][e][f];
            Real dterm = ddg[p][e][f][d] + ddg[p][f][e][d] - ddg[p][d][e][f];
            dgamma[p][c][e][f] += 0.5*dgu[p][c][d]*term + 0.5*gu[c][d]*dterm;
          }
        }
      }
    }
  }

  Real dbeta[3][3], ddbeta[3][3][3];
  Real div_beta = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int d = 0; d < 3; ++d) {
      dbeta[c][d] = DxTotal<NGHOST>(c, idx, u, free, m, ID_CTS_BETAX+d, k, j, i);
      for (int p = 0; p < 3; ++p) {
        ddbeta[p][c][d] =
            (p == c) ? DxxTotal<NGHOST>(p, idx, u, free, m, ID_CTS_BETAX+d, k, j, i)
                     : DxyTotal<NGHOST>(p, c, idx, u, free, m, ID_CTS_BETAX+d, k, j, i);
      }
    }
    div_beta += dbeta[c][c];
  }

  Real d_div_beta = 0.0;
  for (int c = 0; c < 3; ++c) d_div_beta += ddbeta[dir][c][c];

  Real gdot_trace = 0.0;
  Real dgdot_trace = 0.0;
  for (int c = 0; c < 3; ++c) {
    for (int d = 0; d < 3; ++d) {
      Real gdot_cd = FreeSym(free, ID_FREE_GDOTXX, m, c, d, k, j, i);
      Real dgdot_cd = Dx<NGHOST>(dir, idx, free, m,
                                 ID_FREE_GDOTXX + SymIdx(c, d), k, j, i);
      gdot_trace += FreeSym(free, ID_FREE_GXX, m, c, d, k, j, i)*gdot_cd;
      dgdot_trace += dg[dir][c][d]*gdot_cd
                   + FreeSym(free, ID_FREE_GXX, m, c, d, k, j, i)*dgdot_cd;
    }
  }

  Real gdot_tf = FreeSym(free, ID_FREE_GDOTXX, m, a, b, k, j, i)
                 - (1.0/3.0)*gdot_trace*gu[a][b];
  Real dgdot_tf = Dx<NGHOST>(dir, idx, free, m,
                             ID_FREE_GDOTXX + SymIdx(a, b), k, j, i)
                  - (1.0/3.0)*(dgdot_trace*gu[a][b]
                                + gdot_trace*dgu[dir][a][b]);

  Real lbeta = -(2.0/3.0)*gu[a][b]*div_beta;
  Real dlbeta = -(2.0/3.0)*(dgu[dir][a][b]*div_beta + gu[a][b]*d_div_beta);
  for (int c = 0; c < 3; ++c) {
    lbeta += gu[a][c]*dbeta[c][b] + gu[b][c]*dbeta[c][a];
    dlbeta += dgu[dir][a][c]*dbeta[c][b] + gu[a][c]*ddbeta[dir][c][b]
            + dgu[dir][b][c]*dbeta[c][a] + gu[b][c]*ddbeta[dir][c][a];
    for (int d = 0; d < 3; ++d) {
      Real q = gu[a][c]*gamma[b][c][d] + gu[b][c]*gamma[a][c][d]
             - (2.0/3.0)*gu[a][b]*gamma[c][c][d];
      Real dq = dgu[dir][a][c]*gamma[b][c][d] + gu[a][c]*dgamma[dir][b][c][d]
              + dgu[dir][b][c]*gamma[a][c][d] + gu[b][c]*dgamma[dir][a][c][d]
              - (2.0/3.0)*(dgu[dir][a][b]*gamma[c][c][d]
                            + gu[a][b]*dgamma[dir][c][c][d]);
      Real beta_d = TotalU(u, free, m, ID_CTS_BETAX+d, k, j, i);
      lbeta += beta_d*q;
      dlbeta += dbeta[dir][d]*q + beta_d*dq;
    }
  }

  Real alpha = std::max(free(m, ID_FREE_ALPHA, k, j, i), static_cast<Real>(1.0e-12));
  Real dalpha = Dx<NGHOST>(dir, idx, free, m, ID_FREE_ALPHA, k, j, i);
  Real num = gdot_tf + lbeta;
  Real dnum = dgdot_tf + dlbeta;
  return dnum/(2.0*alpha) - num*dalpha/(2.0*alpha*alpha);
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
void CTSOperator(const UView &u, const FView &free, const Real idx[], int fd_stencil,
                 int m, int k, int j, int i,
                 Real op[ID_CTS_NVAR], Real diag[ID_CTS_NVAR]);

template <typename UView>
struct LocalPerturbedUView {
  UView u;
  int pm;
  int pv;
  int pk;
  int pj;
  int pi;
  Real delta;

  KOKKOS_INLINE_FUNCTION
  Real operator()(int m, int v, int k, int j, int i) const {
    Real value = u(m, v, k, j, i);
    if (m == pm && v == pv && k == pk && j == pj && i == pi) value += delta;
    return value;
  }
};

KOKKOS_INLINE_FUNCTION
bool IsFiniteForDevice(Real x) {
  return (x == x) && (std::abs(x) < 1.0e300);
}

struct SmootherCellStats {
  Real values[ID_CTS_SMOOTH_NSTAT];

  KOKKOS_INLINE_FUNCTION
  void Clear() {
    for (int n = 0; n < ID_CTS_SMOOTH_NSTAT; ++n) values[n] = 0.0;
  }

  KOKKOS_INLINE_FUNCTION
  void Add(const SmootherCellStats &other) {
    for (int n = 0; n < ID_CTS_SMOOTH_MAX_UPDATE; ++n) values[n] += other.values[n];
    if (other.values[ID_CTS_SMOOTH_MAX_UPDATE] > values[ID_CTS_SMOOTH_MAX_UPDATE]) {
      values[ID_CTS_SMOOTH_MAX_UPDATE] = other.values[ID_CTS_SMOOTH_MAX_UPDATE];
    }
  }
};

struct CompositeResidualStats {
  Real valid_sum2 = 0.0;
  Real valid_sum_abs = 0.0;
  Real valid_max = 0.0;
  Real valid_volume = 0.0;
  long long valid_count = 0;
  Real interface_sum2 = 0.0;
  Real interface_volume = 0.0;
  long long interface_count = 0;
  Real covered_sum2 = 0.0;
  Real covered_volume = 0.0;
  long long covered_count = 0;
};

const char *CTSVarName(int n) {
  switch (n) {
    case ID_CTS_PSI: return "psi";
    case ID_CTS_BETAX: return "betax";
    case ID_CTS_BETAY: return "betay";
    case ID_CTS_BETAZ: return "betaz";
    default: return "unknown";
  }
}

enum CompositeRestrictionMode {
  ID_CTS_RESTRICT_HALF_WEIGHT = 0,
  ID_CTS_RESTRICT_AVERAGE = 1
};

struct CompositeRestrictionStats {
  long long dst_count = 0;
  long long valid_child_count = 0;
  long long invalid_child_count = 0;
  long long centered_count = 0;
  long long onesided_xm_count = 0;
  long long onesided_xp_count = 0;
  long long onesided_ym_count = 0;
  long long onesided_yp_count = 0;
  long long onesided_zm_count = 0;
  long long onesided_zp_count = 0;
  long long insufficient_count = 0;
  long long diff_count = 0;
  Real diff_sum2 = 0.0;
  Real diff_max = 0.0;
};

struct CompositeTauStats {
  long long dst_count = 0;
  long long insufficient_count = 0;
  Real rf_sum2 = 0.0;
  Real lhr_sum2 = 0.0;
  Real rlh_sum2 = 0.0;
  Real tau_sum2 = 0.0;
  Real tau_max = 0.0;
  Real consistency_sum2 = 0.0;
  Real consistency_max = 0.0;
};

KOKKOS_INLINE_FUNCTION
void AddRestrictionStats(CompositeRestrictionStats &dst,
                         const CompositeRestrictionStats &src) {
  dst.dst_count += src.dst_count;
  dst.valid_child_count += src.valid_child_count;
  dst.invalid_child_count += src.invalid_child_count;
  dst.centered_count += src.centered_count;
  dst.onesided_xm_count += src.onesided_xm_count;
  dst.onesided_xp_count += src.onesided_xp_count;
  dst.onesided_ym_count += src.onesided_ym_count;
  dst.onesided_yp_count += src.onesided_yp_count;
  dst.onesided_zm_count += src.onesided_zm_count;
  dst.onesided_zp_count += src.onesided_zp_count;
  dst.insufficient_count += src.insufficient_count;
  dst.diff_count += src.diff_count;
  dst.diff_sum2 += src.diff_sum2;
  if (src.diff_max > dst.diff_max) dst.diff_max = src.diff_max;
}

KOKKOS_INLINE_FUNCTION
void AddTauStats(CompositeTauStats &dst, const CompositeTauStats &src) {
  dst.dst_count += src.dst_count;
  dst.insufficient_count += src.insufficient_count;
  dst.rf_sum2 += src.rf_sum2;
  dst.lhr_sum2 += src.lhr_sum2;
  dst.rlh_sum2 += src.rlh_sum2;
  dst.tau_sum2 += src.tau_sum2;
  if (src.tau_max > dst.tau_max) dst.tau_max = src.tau_max;
  dst.consistency_sum2 += src.consistency_sum2;
  if (src.consistency_max > dst.consistency_max) dst.consistency_max = src.consistency_max;
}

template <typename MaskViewType>
KOKKOS_INLINE_FUNCTION
bool CompositeSourceValid(const MaskViewType &mask, int m, int k, int j, int i,
                          int nk, int nj, int ni) {
  return i >= 0 && i < ni && j >= 0 && j < nj && k >= 0 && k < nk
      && mask(m, COMP_VALID, k, j, i) != 0;
}

template <typename ViewType, typename MaskViewType>
KOKKOS_INLINE_FUNCTION
bool CompositeSecondDifference(const ViewType &src, const MaskViewType &mask,
                               int m, int v, int k, int j, int i, int axis,
                               int nk, int nj, int ni, Real &d2, int &kind) {
  auto valid_offset = [&](int offset) {
    const int ii = i + ((axis == 0) ? offset : 0);
    const int jj = j + ((axis == 1) ? offset : 0);
    const int kk = k + ((axis == 2) ? offset : 0);
    return CompositeSourceValid(mask, m, kk, jj, ii, nk, nj, ni);
  };
  auto value_offset = [&](int offset) {
    const int ii = i + ((axis == 0) ? offset : 0);
    const int jj = j + ((axis == 1) ? offset : 0);
    const int kk = k + ((axis == 2) ? offset : 0);
    return src(m, v, kk, jj, ii);
  };

  if (valid_offset(-1) && valid_offset(1)) {
    d2 = value_offset(1) - 2.0*value_offset(0) + value_offset(-1);
    kind = 0;
    return true;
  }
  if (valid_offset(-1) && valid_offset(-2) && valid_offset(-3)) {
    d2 = 2.0*value_offset(0) - 5.0*value_offset(-1)
       + 4.0*value_offset(-2) - value_offset(-3);
    kind = -1;
    return true;
  }
  if (valid_offset(1) && valid_offset(2) && valid_offset(3)) {
    d2 = 2.0*value_offset(0) - 5.0*value_offset(1)
       + 4.0*value_offset(2) - value_offset(3);
    kind = 1;
    return true;
  }
  kind = 2;
  d2 = 0.0;
  return false;
}

KOKKOS_INLINE_FUNCTION
void CountRestrictionStencilKind(CompositeRestrictionStats &stats, int axis, int kind) {
  if (kind == 0) {
    ++stats.centered_count;
  } else if (axis == 0 && kind < 0) {
    ++stats.onesided_xm_count;
  } else if (axis == 0 && kind > 0) {
    ++stats.onesided_xp_count;
  } else if (axis == 1 && kind < 0) {
    ++stats.onesided_ym_count;
  } else if (axis == 1 && kind > 0) {
    ++stats.onesided_yp_count;
  } else if (axis == 2 && kind < 0) {
    ++stats.onesided_zm_count;
  } else if (axis == 2 && kind > 0) {
    ++stats.onesided_zp_count;
  }
}

template <typename ViewType, typename MaskViewType>
CompositeRestrictionStats DiagnoseHalfWeightRestrictionImpl(
    const ViewType &src, const MaskViewType &mask, int nvar,
    int i0, int i1, int j0, int j1, int k0, int k1, int ngh, bool half_weight) {
  using ExeSpace = typename ViewType::execution_space;
  const int nm = src.extent_int(0);
  const int nk = src.extent_int(2);
  const int nj = src.extent_int(3);
  const int ni = src.extent_int(4);
  CompositeRestrictionStats stats;
  Kokkos::parallel_reduce("IDCTS::DiagnoseHalfWeightRestriction",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<5>>(
        {0, 0, k0, j0, i0}, {nm, nvar, k1 + 1, j1 + 1, i1 + 1}),
    KOKKOS_LAMBDA(int m, int v, int k, int j, int i,
                  long long &dst_count, long long &valid_child_count,
                  long long &invalid_child_count, long long &centered_count,
                  long long &onesided_xm_count, long long &onesided_xp_count,
                  long long &onesided_ym_count, long long &onesided_yp_count,
                  long long &onesided_zm_count, long long &onesided_zp_count,
                  long long &insufficient_count, long long &diff_count,
                  Real &diff_sum2, Real &diff_max) {
      ++dst_count;
      const int fk = 2*k - ngh;
      const int fj = 2*j - ngh;
      const int fi = 2*i - ngh;
      Real average = 0.0;
      for (int dk = 0; dk <= 1; ++dk) {
        for (int dj = 0; dj <= 1; ++dj) {
          for (int di = 0; di <= 1; ++di) {
            average += src(m, v, fk + dk, fj + dj, fi + di);
          }
        }
      }
      average *= 0.125;
      if (!half_weight) return;

      Real hw_sum = 0.0;
      int hw_count = 0;
      for (int dk = 0; dk <= 1; ++dk) {
        for (int dj = 0; dj <= 1; ++dj) {
          for (int di = 0; di <= 1; ++di) {
            const int ck = fk + dk;
            const int cj = fj + dj;
            const int ci = fi + di;
            if (!CompositeSourceValid(mask, m, ck, cj, ci, nk, nj, ni)) {
              ++invalid_child_count;
              continue;
            }
            ++valid_child_count;
            Real d2[3] = {0.0, 0.0, 0.0};
            int kind[3] = {0, 0, 0};
            bool ok = true;
            for (int axis = 0; axis < 3; ++axis) {
              Real axis_d2 = 0.0;
              int axis_kind = 2;
              if (!CompositeSecondDifference(src, mask, m, v, ck, cj, ci, axis,
                                             nk, nj, ni, axis_d2, axis_kind)) {
                ok = false;
                break;
              }
              d2[axis] = axis_d2;
              kind[axis] = axis_kind;
            }
            if (!ok) {
              ++insufficient_count;
              continue;
            }
            for (int axis = 0; axis < 3; ++axis) {
              if (kind[axis] == 0) {
                ++centered_count;
              } else if (axis == 0 && kind[axis] < 0) {
                ++onesided_xm_count;
              } else if (axis == 0 && kind[axis] > 0) {
                ++onesided_xp_count;
              } else if (axis == 1 && kind[axis] < 0) {
                ++onesided_ym_count;
              } else if (axis == 1 && kind[axis] > 0) {
                ++onesided_yp_count;
              } else if (axis == 2 && kind[axis] < 0) {
                ++onesided_zm_count;
              } else if (axis == 2 && kind[axis] > 0) {
                ++onesided_zp_count;
              }
            }
            hw_sum += src(m, v, ck, cj, ci) + (d2[0] + d2[1] + d2[2])/12.0;
            ++hw_count;
          }
        }
      }
      if (hw_count > 0) {
        const Real hw_average = hw_sum/static_cast<Real>(hw_count);
        const Real diff = hw_average - average;
        const Real abs_diff = std::abs(diff);
        diff_sum2 += diff*diff;
        ++diff_count;
        if (abs_diff > diff_max) diff_max = abs_diff;
      }
    },
    Kokkos::Sum<long long>(stats.dst_count),
    Kokkos::Sum<long long>(stats.valid_child_count),
    Kokkos::Sum<long long>(stats.invalid_child_count),
    Kokkos::Sum<long long>(stats.centered_count),
    Kokkos::Sum<long long>(stats.onesided_xm_count),
    Kokkos::Sum<long long>(stats.onesided_xp_count),
    Kokkos::Sum<long long>(stats.onesided_ym_count),
    Kokkos::Sum<long long>(stats.onesided_yp_count),
    Kokkos::Sum<long long>(stats.onesided_zm_count),
    Kokkos::Sum<long long>(stats.onesided_zp_count),
    Kokkos::Sum<long long>(stats.insufficient_count),
    Kokkos::Sum<long long>(stats.diff_count),
    Kokkos::Sum<Real>(stats.diff_sum2),
    Kokkos::Max<Real>(stats.diff_max));
  if (!half_weight || stats.diff_count == 0) stats.diff_max = 0.0;
  return stats;
}

template <typename DstViewType, typename SrcViewType, typename MaskViewType>
CompositeRestrictionStats RestrictHalfWeightImpl(
    const DstViewType &dst, const SrcViewType &src, const MaskViewType &mask, int nvar,
    int i0, int i1, int j0, int j1, int k0, int k1, int ngh) {
  using ExeSpace = typename DstViewType::execution_space;
  const int nm = src.extent_int(0);
  const int nk = src.extent_int(2);
  const int nj = src.extent_int(3);
  const int ni = src.extent_int(4);
  CompositeRestrictionStats stats;
  Kokkos::parallel_reduce("IDCTS::RestrictHalfWeight",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<5>>(
        {0, 0, k0, j0, i0}, {nm, nvar, k1 + 1, j1 + 1, i1 + 1}),
    KOKKOS_LAMBDA(int m, int v, int k, int j, int i,
                  long long &dst_count, long long &valid_child_count,
                  long long &invalid_child_count, long long &centered_count,
                  long long &onesided_xm_count, long long &onesided_xp_count,
                  long long &onesided_ym_count, long long &onesided_yp_count,
                  long long &onesided_zm_count, long long &onesided_zp_count,
                  long long &insufficient_count, long long &diff_count,
                  Real &diff_sum2, Real &diff_max) {
      ++dst_count;
      const int fk = 2*k - ngh;
      const int fj = 2*j - ngh;
      const int fi = 2*i - ngh;
      Real average = 0.0;
      for (int dk = 0; dk <= 1; ++dk)
        for (int dj = 0; dj <= 1; ++dj)
          for (int di = 0; di <= 1; ++di)
            average += src(m, v, fk + dk, fj + dj, fi + di);
      average *= 0.125;

      Real hw_sum = 0.0;
      int hw_count = 0;
      for (int dk = 0; dk <= 1; ++dk) {
        for (int dj = 0; dj <= 1; ++dj) {
          for (int di = 0; di <= 1; ++di) {
            const int ck = fk + dk;
            const int cj = fj + dj;
            const int ci = fi + di;
            if (!CompositeSourceValid(mask, m, ck, cj, ci, nk, nj, ni)) {
              ++invalid_child_count;
              continue;
            }
            ++valid_child_count;
            Real d2[3] = {0.0, 0.0, 0.0};
            int kind[3] = {0, 0, 0};
            bool ok = true;
            for (int axis = 0; axis < 3; ++axis) {
              if (!CompositeSecondDifference(src, mask, m, v, ck, cj, ci, axis,
                                             nk, nj, ni, d2[axis], kind[axis])) {
                ok = false;
                break;
              }
            }
            if (!ok) {
              ++insufficient_count;
              continue;
            }
            for (int axis = 0; axis < 3; ++axis) {
              if (kind[axis] == 0) {
                ++centered_count;
              } else if (axis == 0 && kind[axis] < 0) {
                ++onesided_xm_count;
              } else if (axis == 0 && kind[axis] > 0) {
                ++onesided_xp_count;
              } else if (axis == 1 && kind[axis] < 0) {
                ++onesided_ym_count;
              } else if (axis == 1 && kind[axis] > 0) {
                ++onesided_yp_count;
              } else if (axis == 2 && kind[axis] < 0) {
                ++onesided_zm_count;
              } else if (axis == 2 && kind[axis] > 0) {
                ++onesided_zp_count;
              }
            }
            hw_sum += src(m, v, ck, cj, ci) + (d2[0] + d2[1] + d2[2])/12.0;
            ++hw_count;
          }
        }
      }
      if (hw_count > 0) {
        const Real hw_average = hw_sum/static_cast<Real>(hw_count);
        const Real diff = hw_average - average;
        const Real abs_diff = std::abs(diff);
        diff_sum2 += diff*diff;
        ++diff_count;
        if (abs_diff > diff_max) diff_max = abs_diff;
        dst(m, v, k, j, i) = hw_average;
      } else {
        dst(m, v, k, j, i) = average;
      }
    },
    Kokkos::Sum<long long>(stats.dst_count),
    Kokkos::Sum<long long>(stats.valid_child_count),
    Kokkos::Sum<long long>(stats.invalid_child_count),
    Kokkos::Sum<long long>(stats.centered_count),
    Kokkos::Sum<long long>(stats.onesided_xm_count),
    Kokkos::Sum<long long>(stats.onesided_xp_count),
    Kokkos::Sum<long long>(stats.onesided_ym_count),
    Kokkos::Sum<long long>(stats.onesided_yp_count),
    Kokkos::Sum<long long>(stats.onesided_zm_count),
    Kokkos::Sum<long long>(stats.onesided_zp_count),
    Kokkos::Sum<long long>(stats.insufficient_count),
    Kokkos::Sum<long long>(stats.diff_count),
    Kokkos::Sum<Real>(stats.diff_sum2),
    Kokkos::Max<Real>(stats.diff_max));
  if (stats.diff_count == 0) stats.diff_max = 0.0;
  return stats;
}

CompositeRestrictionStats ReduceCompositeRestrictionStats(
    const CompositeRestrictionStats &local) {
  CompositeRestrictionStats global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local.dst_count, &global.dst_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.valid_child_count, &global.valid_child_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.invalid_child_count, &global.invalid_child_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.centered_count, &global.centered_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.onesided_xm_count, &global.onesided_xm_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.onesided_xp_count, &global.onesided_xp_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.onesided_ym_count, &global.onesided_ym_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.onesided_yp_count, &global.onesided_yp_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.onesided_zm_count, &global.onesided_zm_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.onesided_zp_count, &global.onesided_zp_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.insufficient_count, &global.insufficient_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.diff_count, &global.diff_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.diff_sum2, &global.diff_sum2, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.diff_max, &global.diff_max, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
#endif
  return global;
}

CompositeTauStats ReduceCompositeTauStats(const CompositeTauStats &local) {
  CompositeTauStats global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local.dst_count, &global.dst_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.insufficient_count, &global.insufficient_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.rf_sum2, &global.rf_sum2, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.lhr_sum2, &global.lhr_sum2, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.rlh_sum2, &global.rlh_sum2, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.tau_sum2, &global.tau_sum2, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.tau_max, &global.tau_max, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local.consistency_sum2, &global.consistency_sum2, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.consistency_max, &global.consistency_max, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
#endif
  return global;
}

void PrintCompositeRestrictionStats(const char *entity, int level, const char *field,
                                    int mode, const CompositeRestrictionStats &stats) {
  if (global_variable::my_rank != 0) return;
  const Real diff_rms = (stats.diff_count > 0)
      ? std::sqrt(stats.diff_sum2/static_cast<Real>(stats.diff_count)) : 0.0;
  std::cout << "CTS composite restriction: entity=" << entity
            << " level=" << level
            << " field=" << field
            << " mode=" << ((mode == ID_CTS_RESTRICT_HALF_WEIGHT) ? "half_weight" : "average")
            << " dst_count=" << stats.dst_count
            << " valid_children=" << stats.valid_child_count
            << " invalid_children=" << stats.invalid_child_count
            << " centered=" << stats.centered_count
            << " onesided_xm=" << stats.onesided_xm_count
            << " onesided_xp=" << stats.onesided_xp_count
            << " onesided_ym=" << stats.onesided_ym_count
            << " onesided_yp=" << stats.onesided_yp_count
            << " onesided_zm=" << stats.onesided_zm_count
            << " onesided_zp=" << stats.onesided_zp_count
            << " insufficient=" << stats.insufficient_count
            << " avg_diff_max=" << stats.diff_max
            << " avg_diff_rms=" << diff_rms << std::endl;
}

void PrintCompositeTauStats(int fine_level, int coarse_level,
                            const CompositeTauStats &stats) {
  if (global_variable::my_rank != 0) return;
  const Real denom = (stats.dst_count > 0) ? static_cast<Real>(stats.dst_count) : 1.0;
  const Real rf = std::sqrt(stats.rf_sum2/denom);
  const Real lhr = std::sqrt(stats.lhr_sum2/denom);
  const Real rlh = std::sqrt(stats.rlh_sum2/denom);
  const Real tau = std::sqrt(stats.tau_sum2/denom);
  const Real consistency = std::sqrt(stats.consistency_sum2/denom);
  std::cout << "CTS composite tau: entity=meshblock"
            << " fine_level=" << fine_level
            << " coarse_level=" << coarse_level
            << " dst_count=" << stats.dst_count
            << " insufficient=" << stats.insufficient_count
            << " ||R_f||=" << rf
            << " ||L_H_Ru||=" << lhr
            << " ||R_L_h||=" << rlh
            << " ||tau||=" << tau
            << " tau_max=" << stats.tau_max
            << " tau_rms=" << tau
            << " consistency_max=" << stats.consistency_max
            << " consistency_rms=" << consistency << std::endl;
}

Real CompositeRestrictionPolynomial(int i, int j, int k) {
  return 3.0*i*i + 5.0*j*j + 7.0*k*k + 11.0*i - 13.0*j + 17.0*k + 19.0;
}

void FatalCompositeRestrictionSelfCheck(const std::string &msg) {
  std::cout << "### FATAL ERROR in CTS composite restriction self-check" << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

void CheckCloseCompositeRestriction(Real got, Real expected, Real tol,
                                    const std::string &label) {
  if (std::abs(got - expected) > tol) {
    std::ostringstream msg;
    msg << label << " expected " << expected << " but got " << got;
    FatalCompositeRestrictionSelfCheck(msg.str());
  }
}

void RunCompositeRestrictionSelfCheck() {
  constexpr int n = 8;
  constexpr int ngh = 0;
  HostArray5D<Real> src("cts_restrict_self_src", 1, 1, n, n, n);
  HostArray5D<int> mask("cts_restrict_self_mask", 1, COMP_NMASK, n, n, n);
  auto set_all_valid = [&]() {
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          mask(0, COMP_VALID, k, j, i) = 1;
          mask(0, COMP_RELAX, k, j, i) = 1;
          mask(0, COMP_COVERED, k, j, i) = 0;
          mask(0, COMP_INTERFACE, k, j, i) = 0;
        }
      }
    }
  };

  set_all_valid();
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        src(0, 0, k, j, i) = 4.0;
  CompositeRestrictionStats constant =
      DiagnoseHalfWeightRestrictionImpl(src, mask, 1, 0, 3, 0, 3, 0, 3, ngh, true);
  CheckCloseCompositeRestriction(constant.diff_max, 0.0, 1.0e-14, "constant diff");

  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        src(0, 0, k, j, i) = 2.0*i - 3.0*j + 5.0*k + 7.0;
  CompositeRestrictionStats linear =
      DiagnoseHalfWeightRestrictionImpl(src, mask, 1, 0, 3, 0, 3, 0, 3, ngh, true);
  CheckCloseCompositeRestriction(linear.diff_max, 0.0, 1.0e-14, "linear diff");

  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        src(0, 0, k, j, i) = CompositeRestrictionPolynomial(i, j, k);
  CompositeRestrictionStats quadratic =
      DiagnoseHalfWeightRestrictionImpl(src, mask, 1, 0, 3, 0, 3, 0, 3, ngh, true);
  CheckCloseCompositeRestriction(quadratic.diff_max, 2.5, 1.0e-12,
                                 "quadratic half-weight correction");

  Real d2 = 0.0;
  int kind = 2;
  if (!CompositeSecondDifference(src, mask, 0, 0, 4, 4, 4, 0, n, n, n, d2, kind) ||
      kind != 0) {
    FatalCompositeRestrictionSelfCheck("centered x second difference was not selected");
  }
  CheckCloseCompositeRestriction(d2, 6.0, 1.0e-12, "centered Dxx");
  if (!CompositeSecondDifference(src, mask, 0, 0, 4, 4, 4, 1, n, n, n, d2, kind) ||
      kind != 0) {
    FatalCompositeRestrictionSelfCheck("centered y second difference was not selected");
  }
  CheckCloseCompositeRestriction(d2, 10.0, 1.0e-12, "centered Dyy");
  if (!CompositeSecondDifference(src, mask, 0, 0, 4, 4, 4, 2, n, n, n, d2, kind) ||
      kind != 0) {
    FatalCompositeRestrictionSelfCheck("centered z second difference was not selected");
  }
  CheckCloseCompositeRestriction(d2, 14.0, 1.0e-12, "centered Dzz");

  set_all_valid();
  mask(0, COMP_VALID, 4, 4, 1) = 0;
  if (!CompositeSecondDifference(src, mask, 0, 0, 4, 4, 2, 0, n, n, n, d2, kind) ||
      kind != 1) {
    FatalCompositeRestrictionSelfCheck("plus-side one-sided x stencil was not selected");
  }
  CheckCloseCompositeRestriction(d2, 6.0, 1.0e-12, "plus one-sided Dxx");

  set_all_valid();
  mask(0, COMP_VALID, 4, 4, 6) = 0;
  if (!CompositeSecondDifference(src, mask, 0, 0, 4, 4, 5, 0, n, n, n, d2, kind) ||
      kind != -1) {
    FatalCompositeRestrictionSelfCheck("minus-side one-sided x stencil was not selected");
  }
  CheckCloseCompositeRestriction(d2, 6.0, 1.0e-12, "minus one-sided Dxx");
}

const Real &OctetRestrictionValue(const MGOctet &oct, int field, int v,
                                  int k, int j, int i) {
  return (field == 0) ? oct.U(v, k, j, i) : oct.Def(v, k, j, i);
}

bool OctetSourceValid(const MGOctet &oct, int k, int j, int i) {
  return i >= 0 && i < oct.nc && j >= 0 && j < oct.nc && k >= 0 && k < oct.nc
      && oct.Mask(COMP_VALID, k, j, i) != 0;
}

bool OctetSecondDifference(const MGOctet &oct, int field, int v, int k, int j,
                           int i, int axis, Real &d2, int &kind) {
  auto valid_offset = [&](int offset) {
    const int ii = i + ((axis == 0) ? offset : 0);
    const int jj = j + ((axis == 1) ? offset : 0);
    const int kk = k + ((axis == 2) ? offset : 0);
    return OctetSourceValid(oct, kk, jj, ii);
  };
  auto value_offset = [&](int offset) -> Real {
    const int ii = i + ((axis == 0) ? offset : 0);
    const int jj = j + ((axis == 1) ? offset : 0);
    const int kk = k + ((axis == 2) ? offset : 0);
    return OctetRestrictionValue(oct, field, v, kk, jj, ii);
  };
  if (valid_offset(-1) && valid_offset(1)) {
    d2 = value_offset(1) - 2.0*value_offset(0) + value_offset(-1);
    kind = 0;
    return true;
  }
  if (valid_offset(-1) && valid_offset(-2) && valid_offset(-3)) {
    d2 = 2.0*value_offset(0) - 5.0*value_offset(-1)
       + 4.0*value_offset(-2) - value_offset(-3);
    kind = -1;
    return true;
  }
  if (valid_offset(1) && valid_offset(2) && valid_offset(3)) {
    d2 = 2.0*value_offset(0) - 5.0*value_offset(1)
       + 4.0*value_offset(2) - value_offset(3);
    kind = 1;
    return true;
  }
  d2 = 0.0;
  kind = 2;
  return false;
}

CompositeRestrictionStats DiagnoseHalfWeightRestrictionOctet(const MGOctet &oct,
                                                             int nvar, int ngh,
                                                             int field,
                                                             bool half_weight) {
  CompositeRestrictionStats stats;
  for (int v = 0; v < nvar; ++v) {
    ++stats.dst_count;
    Real average = 0.0;
    for (int dk = 0; dk <= 1; ++dk)
      for (int dj = 0; dj <= 1; ++dj)
        for (int di = 0; di <= 1; ++di)
          average += OctetRestrictionValue(oct, field, v, ngh + dk, ngh + dj, ngh + di);
    average *= 0.125;
    if (!half_weight) continue;

    Real hw_sum = 0.0;
    int hw_count = 0;
    for (int dk = 0; dk <= 1; ++dk) {
      for (int dj = 0; dj <= 1; ++dj) {
        for (int di = 0; di <= 1; ++di) {
          const int ck = ngh + dk;
          const int cj = ngh + dj;
          const int ci = ngh + di;
          if (!OctetSourceValid(oct, ck, cj, ci)) {
            ++stats.invalid_child_count;
            continue;
          }
          ++stats.valid_child_count;
          Real d2[3] = {0.0, 0.0, 0.0};
          int kind[3] = {0, 0, 0};
          bool ok = true;
          for (int axis = 0; axis < 3; ++axis) {
            if (!OctetSecondDifference(oct, field, v, ck, cj, ci, axis,
                                       d2[axis], kind[axis])) {
              ok = false;
              break;
            }
          }
          if (!ok) {
            ++stats.insufficient_count;
            continue;
          }
          for (int axis = 0; axis < 3; ++axis) {
            CountRestrictionStencilKind(stats, axis, kind[axis]);
          }
          hw_sum += OctetRestrictionValue(oct, field, v, ck, cj, ci)
                  + (d2[0] + d2[1] + d2[2])/12.0;
          ++hw_count;
        }
      }
    }
    if (hw_count > 0) {
      const Real hw_average = hw_sum/static_cast<Real>(hw_count);
      const Real diff = hw_average - average;
      const Real abs_diff = std::abs(diff);
      stats.diff_sum2 += diff*diff;
      ++stats.diff_count;
      if (abs_diff > stats.diff_max) stats.diff_max = abs_diff;
    }
  }
  return stats;
}

template <typename UView>
struct LocalUpdatedUView {
  UView u;
  int pm;
  int pk;
  int pj;
  int pi;
  Real delta[ID_CTS_NVAR];

  KOKKOS_INLINE_FUNCTION
  Real operator()(int m, int v, int k, int j, int i) const {
    Real value = u(m, v, k, j, i);
    if (m == pm && k == pk && j == pj && i == pi) value += delta[v];
    return value;
  }
};

template <typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
Real TotalUScale(const UView &u, const FView &free, int m, int v, int k, int j, int i) {
  return std::max(static_cast<Real>(1.0),
                  std::abs(TotalU(u, free, m, v, k, j, i)));
}

KOKKOS_INLINE_FUNCTION
Real LimitUpdate(Real update, Real abs_limit, Real frac_limit, Real scale, bool &limited) {
  Real limit = 0.0;
  if (abs_limit > 0.0) limit = abs_limit;
  if (frac_limit > 0.0) {
    Real scaled_limit = frac_limit*std::max(scale, static_cast<Real>(1.0e-30));
    limit = (limit > 0.0) ? std::min(limit, scaled_limit) : scaled_limit;
  }
  if (limit > 0.0 && std::abs(update) > limit) {
    limited = true;
    return (update > 0.0) ? limit : -limit;
  }
  return update;
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
Real LocalResidualNorm2(const UView &u, const FView &src, const FView &free,
                        const Real idx[], int fd_stencil,
                        int m, int k, int j, int i) {
  Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
  CTSOperator<NGHOST>(u, free, idx, fd_stencil, m, k, j, i, op, diag);
  Real norm2 = 0.0;
  for (int v = 0; v < ID_CTS_NVAR; ++v) {
    Real r = op[v] - src(m, v, k, j, i);
    if (!IsFiniteForDevice(r)) return 1.0e300;
    norm2 += r*r;
  }
  return norm2;
}

KOKKOS_INLINE_FUNCTION
bool SolveLinear4x4(Real a[ID_CTS_NVAR][ID_CTS_NVAR], Real b[ID_CTS_NVAR],
                    const Real col_scale[ID_CTS_NVAR], Real x[ID_CTS_NVAR]) {
  Real aug[ID_CTS_NVAR][ID_CTS_NVAR + 1];
  for (int r = 0; r < ID_CTS_NVAR; ++r) {
    Real row_scale = std::max(std::abs(b[r]), static_cast<Real>(1.0));
    for (int c = 0; c < ID_CTS_NVAR; ++c) {
      Real scaled = a[r][c]*std::max(col_scale[c], static_cast<Real>(1.0e-30));
      row_scale = std::max(row_scale, std::abs(scaled));
    }
    row_scale = std::max(row_scale, static_cast<Real>(1.0e-30));
    for (int c = 0; c < ID_CTS_NVAR; ++c) {
      aug[r][c] = a[r][c]*std::max(col_scale[c], static_cast<Real>(1.0e-30))/row_scale;
    }
    aug[r][ID_CTS_NVAR] = b[r]/row_scale;
  }

  for (int p = 0; p < ID_CTS_NVAR; ++p) {
    int pivot = p;
    Real pivot_abs = std::abs(aug[p][p]);
    for (int r = p + 1; r < ID_CTS_NVAR; ++r) {
      Real candidate = std::abs(aug[r][p]);
      if (candidate > pivot_abs) {
        pivot = r;
        pivot_abs = candidate;
      }
    }
    if (!(pivot_abs > 1.0e-12) || !IsFiniteForDevice(pivot_abs)) return false;
    if (pivot != p) {
      for (int c = p; c <= ID_CTS_NVAR; ++c) {
        Real tmp = aug[p][c];
        aug[p][c] = aug[pivot][c];
        aug[pivot][c] = tmp;
      }
    }
    Real inv_pivot = 1.0/aug[p][p];
    for (int c = p; c <= ID_CTS_NVAR; ++c) aug[p][c] *= inv_pivot;
    for (int r = 0; r < ID_CTS_NVAR; ++r) {
      if (r == p) continue;
      Real factor = aug[r][p];
      for (int c = p; c <= ID_CTS_NVAR; ++c) aug[r][c] -= factor*aug[p][c];
    }
  }

  for (int r = 0; r < ID_CTS_NVAR; ++r) {
    x[r] = aug[r][ID_CTS_NVAR]*std::max(col_scale[r], static_cast<Real>(1.0e-30));
    if (!IsFiniteForDevice(x[r])) return false;
  }
  return true;
}

template <int NGHOST, typename ReadView, typename WriteView, typename FView>
KOKKOS_INLINE_FUNCTION
SmootherCellStats ApplyDiagonalCTSUpdate(const ReadView &u_read, WriteView &u_write,
                                         const FView &src, const FView &free,
                                         const Real idx[], int fd_stencil,
                                         int m, int k, int j, int i, Real omega,
                                         Real abs_max_update, Real frac_max_update) {
  SmootherCellStats stats;
  stats.Clear();
  Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
  CTSOperator<NGHOST>(u_read, free, idx, fd_stencil, m, k, j, i, op, diag);
  bool applied = false;
  for (int v = 0; v < ID_CTS_NVAR; ++v) {
    Real residual = op[v] - src(m, v, k, j, i);
    Real scale = TotalUScale(u_read, free, m, v, k, j, i);
    Real d_floor = std::max(static_cast<Real>(1.0e-30),
                            static_cast<Real>(1.0e-14)
                            *std::max(std::abs(residual), static_cast<Real>(1.0))/scale);
    Real d = (std::abs(diag[v]) > d_floor) ? diag[v] :
             ((diag[v] < 0.0) ? -d_floor : d_floor);
    Real update = -omega*residual/d;
    if (!IsFiniteForDevice(residual) || !IsFiniteForDevice(update)) {
      stats.values[ID_CTS_SMOOTH_NONFINITE] += 1.0;
      stats.values[ID_CTS_SMOOTH_REJECTED] += 1.0;
      continue;
    }
    bool limited = false;
    update = LimitUpdate(update, abs_max_update, frac_max_update, scale, limited);
    if (limited) stats.values[ID_CTS_SMOOTH_LIMITED] += 1.0;
    u_write(m, v, k, j, i) += update;
    if (std::abs(update) > stats.values[ID_CTS_SMOOTH_MAX_UPDATE]) {
      stats.values[ID_CTS_SMOOTH_MAX_UPDATE] = std::abs(update);
    }
    applied = true;
  }
  Real psi = TotalU(u_write, free, m, ID_CTS_PSI, k, j, i);
  if (psi < 1.0e-8) {
    u_write(m, ID_CTS_PSI, k, j, i) = 1.0e-8 - free(m, ID_FREE_BASE_PSI, k, j, i);
    stats.values[ID_CTS_SMOOTH_PSI_FLOOR] += 1.0;
  }
  if (applied) stats.values[ID_CTS_SMOOTH_ACCEPTED] += 1.0;
  return stats;
}

template <int NGHOST, typename ReadView, typename WriteView, typename FView>
KOKKOS_INLINE_FUNCTION
SmootherCellStats ApplyNewtonGSCTSUpdate(const ReadView &u_read, WriteView &u_write,
                                         const FView &src, const FView &free,
                                         const Real idx[], int fd_stencil,
                                         int m, int k, int j, int i, Real omega,
                                         int niter, Real jac_eps, Real abs_max_update,
                                         Real frac_max_update, int line_search_steps,
                                         Real line_search_min) {
  SmootherCellStats total_stats;
  total_stats.Clear();
  const int local_iter = niter > 0 ? niter : 1;
  Real accumulated_delta[ID_CTS_NVAR] = {0.0, 0.0, 0.0, 0.0};
  for (int it = 0; it < local_iter; ++it) {
    LocalUpdatedUView<ReadView> u_current{u_read, m, k, j, i,
                                          {accumulated_delta[0],
                                           accumulated_delta[1],
                                           accumulated_delta[2],
                                           accumulated_delta[3]}};
    Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
    CTSOperator<NGHOST>(u_current, free, idx, fd_stencil, m, k, j, i, op, diag);

    Real residual[ID_CTS_NVAR];
    Real old_norm2 = 0.0;
    for (int r = 0; r < ID_CTS_NVAR; ++r) {
      residual[r] = op[r] - src(m, r, k, j, i);
      if (!IsFiniteForDevice(residual[r])) {
        total_stats.values[ID_CTS_SMOOTH_NONFINITE] += 1.0;
        total_stats.values[ID_CTS_SMOOTH_REJECTED] += 1.0;
        return total_stats;
      }
      old_norm2 += residual[r]*residual[r];
    }

    Real jac[ID_CTS_NVAR][ID_CTS_NVAR];
    Real col_scale[ID_CTS_NVAR];
    for (int c = 0; c < ID_CTS_NVAR; ++c) {
      col_scale[c] = TotalUScale(u_current, free, m, c, k, j, i);
      Real eps = jac_eps*col_scale[c];
      if (!(eps > 0.0) || !IsFiniteForDevice(eps)) eps = 1.0e-7;
      LocalPerturbedUView<LocalUpdatedUView<ReadView>> up{u_current, m, c, k, j, i, eps};
      Real op_pert[ID_CTS_NVAR], diag_pert[ID_CTS_NVAR];
      CTSOperator<NGHOST>(up, free, idx, fd_stencil, m, k, j, i,
                          op_pert, diag_pert);
      for (int r = 0; r < ID_CTS_NVAR; ++r) {
        jac[r][c] = (op_pert[r] - op[r])/eps;
      }
    }

    Real rhs[ID_CTS_NVAR];
    Real delta[ID_CTS_NVAR];
    for (int r = 0; r < ID_CTS_NVAR; ++r) rhs[r] = -residual[r];
    bool solved = SolveLinear4x4(jac, rhs, col_scale, delta);
    if (!solved) {
      total_stats.values[ID_CTS_SMOOTH_SINGULAR] += 1.0;
      total_stats.values[ID_CTS_SMOOTH_FALLBACK] += 1.0;
      SmootherCellStats fallback =
          ApplyDiagonalCTSUpdate<NGHOST>(u_current, u_write, src, free, idx,
                                         fd_stencil, m, k, j, i,
                                         omega, abs_max_update, frac_max_update);
      total_stats.Add(fallback);
      for (int v = 0; v < ID_CTS_NVAR; ++v) {
        accumulated_delta[v] =
            u_write(m, v, k, j, i) - u_read(m, v, k, j, i);
      }
      continue;
    }

    bool accepted = false;
    bool any_backtrack = false;
    Real lambda = 1.0;
    const int nls = line_search_steps > 0 ? line_search_steps : 1;
    for (int ls = 0; ls < nls; ++ls) {
      LocalUpdatedUView<LocalUpdatedUView<ReadView>> uc{u_current, m, k, j, i,
                                                        {0.0, 0.0, 0.0, 0.0}};
      bool limited = false;
      bool finite_update = true;
      Real trial_max_update = 0.0;
      for (int v = 0; v < ID_CTS_NVAR; ++v) {
        Real update = omega*lambda*delta[v];
        update = LimitUpdate(update, abs_max_update, frac_max_update, col_scale[v], limited);
        if (!IsFiniteForDevice(update)) finite_update = false;
        uc.delta[v] = update;
        if (std::abs(update) > trial_max_update) trial_max_update = std::abs(update);
      }
      Real trial_psi = TotalU(uc, free, m, ID_CTS_PSI, k, j, i);
      Real new_norm2 = finite_update && trial_psi > 1.0e-8
                       ? LocalResidualNorm2<NGHOST>(uc, src, free, idx, fd_stencil,
                                                    m, k, j, i)
                       : 1.0e300;
      if (IsFiniteForDevice(new_norm2) && new_norm2 <= old_norm2) {
        for (int v = 0; v < ID_CTS_NVAR; ++v) {
          u_write(m, v, k, j, i) += uc.delta[v];
          accumulated_delta[v] += uc.delta[v];
        }
        if (limited) total_stats.values[ID_CTS_SMOOTH_LIMITED] += 1.0;
        if (any_backtrack) total_stats.values[ID_CTS_SMOOTH_BACKTRACKED] += 1.0;
        if (trial_max_update > total_stats.values[ID_CTS_SMOOTH_MAX_UPDATE]) {
          total_stats.values[ID_CTS_SMOOTH_MAX_UPDATE] = trial_max_update;
        }
        total_stats.values[ID_CTS_SMOOTH_ACCEPTED] += 1.0;
        accepted = true;
        break;
      }
      any_backtrack = true;
      lambda *= 0.5;
      if (lambda < line_search_min) break;
    }

    if (!accepted) {
      total_stats.values[ID_CTS_SMOOTH_FALLBACK] += 1.0;
      SmootherCellStats fallback =
          ApplyDiagonalCTSUpdate<NGHOST>(u_current, u_write, src, free, idx,
                                         fd_stencil, m, k, j, i,
                                         omega, abs_max_update, frac_max_update);
      total_stats.Add(fallback);
      for (int v = 0; v < ID_CTS_NVAR; ++v) {
        accumulated_delta[v] =
            u_write(m, v, k, j, i) - u_read(m, v, k, j, i);
      }
    }
  }
  return total_stats;
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
void CTSOperator(const UView &u, const FView &free, const Real idx[], int fd_stencil,
                 int m, int k, int j, int i,
                 Real op[ID_CTS_NVAR], Real diag[ID_CTS_NVAR]) {
  Real gu[3][3];
  Real gamma[3][3][3];
  MetricInverse(free, m, k, j, i, gu);
  Christoffel<NGHOST>(free, idx, m, k, j, i, gu, gamma);
  Real ric = RicciScalar<NGHOST>(free, idx, m, k, j, i, gu, gamma);

  Real psi = std::max(TotalU(u, free, m, ID_CTS_PSI, k, j, i),
                      static_cast<Real>(1.0e-8));
  Real dpsi[3];
  Real ddpsi[3][3];
  for (int a = 0; a < 3; ++a) {
    dpsi[a] = DxTotal<NGHOST>(a, idx, u, free, m, ID_CTS_PSI, k, j, i);
    for (int b = 0; b < 3; ++b) {
      ddpsi[a][b] = (a == b) ? DxxTotal<NGHOST>(a, idx, u, free, m, ID_CTS_PSI, k, j, i)
                             : DxyTotal<NGHOST>(a, b, idx, u, free, m, ID_CTS_PSI, k, j, i);
    }
  }

  Real dsqpsi = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      Real chrs_dpsi = 0.0;
      for (int c = 0; c < 3; ++c) chrs_dpsi += gamma[c][a][b]*dpsi[c];
      dsqpsi += gu[a][b]*(ddpsi[a][b] - chrs_dpsi);
    }
  }

  Real ahat[3][3];
  Real ahat2 = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      ahat[a][b] = AHatUU<NGHOST>(u, free, idx, m, k, j, i, a, b);
    }
  }
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d)
          ahat2 += FreeSym(free, ID_FREE_GXX, m, a, c, k, j, i)
                   * FreeSym(free, ID_FREE_GXX, m, b, d, k, j, i)
                   * ahat[a][b]*ahat[c][d];

  Real K = free(m, ID_FREE_K, k, j, i);
  Real E = free(m, ID_FREE_E, k, j, i);
  Real src = free(m, ID_FREE_SOURCE, k, j, i);
  op[ID_CTS_PSI] = dsqpsi - 0.125*ric*psi + 0.125*ahat2/std::pow(psi, 7.0)
                  + 2.0*3.14159265358979323846*E*std::pow(psi, 5.0)
                  - (K*K)*std::pow(psi, 5.0)/12.0 - src;
  diag[ID_CTS_PSI] = 0.0;
  for (int a = 0; a < 3; ++a) diag[ID_CTS_PSI] += gu[a][a]*DxxCenterCoeff(fd_stencil, idx[a]);
  diag[ID_CTS_PSI] += -0.125*ric - 0.875*ahat2/std::pow(psi, 8.0)
                      + 10.0*3.14159265358979323846*E*std::pow(psi, 4.0)
                      - (5.0/12.0)*(K*K)*std::pow(psi, 4.0);

  for (int a = 0; a < 3; ++a) {
    Real div_a = 0.0;
    for (int b = 0; b < 3; ++b) {
      div_a += DxAHatUU<NGHOST>(b, idx, u, free, m, k, j, i, a, b);
      for (int c = 0; c < 3; ++c) {
        div_a += gamma[a][b][c]*ahat[c][b] + gamma[b][b][c]*ahat[a][c];
      }
    }
    op[ID_CTS_BETAX + a] = 2.0*div_a
      - (4.0/3.0)*std::pow(psi, 6.0)*free(m, ID_FREE_DKX+a, k, j, i)
      - 16.0*3.14159265358979323846*std::pow(psi, 6.0)
         * free(m, ID_FREE_PX+a, k, j, i);

    Real d = 0.0;
    for (int b = 0; b < 3; ++b) d += gu[b][b]*DxxCenterCoeff(fd_stencil, idx[b]);
    d += (1.0/3.0)*gu[a][a]*DxxCenterCoeff(fd_stencil, idx[a]);
    diag[ID_CTS_BETAX + a] = d/std::max(free(m, ID_FREE_ALPHA, k, j, i),
                                         static_cast<Real>(1.0e-12));
  }
}

template <int NGHOST, typename ViewType>
SmootherCellStats SmoothImpl(IDCTSMultigrid *mg, ViewType &u,
                             const ViewType &u_frozen, const ViewType &src,
                             const ViewType &free, int ll, int is, int ie,
                             int js, int je, int ks, int ke, int color,
                             Real omega, int fd_stencil, int smoother_type,
                             int ngs_iterations, Real ngs_jacobian_eps,
                             Real ngs_max_update,
                             Real smoother_max_update_fraction,
                             int ngs_line_search_steps, Real ngs_line_search_min) {
  using ExeSpace = typename ViewType::execution_space;
  auto brdx = [&]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>) return mg->GetBlockDx_h();
    else return mg->GetBlockDx();
  }();
  int nmmb = mg->GetNumMeshBlocks();
  int rlev = -ll;
  SmootherCellStats stats;
  stats.Clear();
  Kokkos::parallel_reduce("IDCTS::Smooth",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<3>>(
        {0, ks, js}, {nmmb, ke + 1, je + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, Real &accepted, Real &limited,
                Real &backtracked, Real &fallback, Real &singular,
                Real &nonfinite, Real &rejected, Real &psi_floor,
                Real &max_update) {
    Real dx = (rlev <= 0) ? brdx(m)*static_cast<Real>(1<<(-rlev))
                          : brdx(m)/static_cast<Real>(1<<rlev);
    Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
    int c = (color + k + j) & 1;
    for (int i = is + c; i <= ie; i += 2) {
      if (free(m, ID_FREE_MASK, k, j, i) < 0.5) continue;
      SmootherCellStats cell;
      if (smoother_type == 1) {
        cell = ApplyNewtonGSCTSUpdate<NGHOST>(u_frozen, u, src, free, idx, fd_stencil,
                                              m, k, j, i, omega, ngs_iterations,
                                              ngs_jacobian_eps, ngs_max_update,
                                              smoother_max_update_fraction,
                                              ngs_line_search_steps,
                                              ngs_line_search_min);
      } else {
        cell = ApplyDiagonalCTSUpdate<NGHOST>(u_frozen, u, src, free, idx, fd_stencil,
                                              m, k, j, i, omega, ngs_max_update,
                                              smoother_max_update_fraction);
      }
      accepted += cell.values[ID_CTS_SMOOTH_ACCEPTED];
      limited += cell.values[ID_CTS_SMOOTH_LIMITED];
      backtracked += cell.values[ID_CTS_SMOOTH_BACKTRACKED];
      fallback += cell.values[ID_CTS_SMOOTH_FALLBACK];
      singular += cell.values[ID_CTS_SMOOTH_SINGULAR];
      nonfinite += cell.values[ID_CTS_SMOOTH_NONFINITE];
      rejected += cell.values[ID_CTS_SMOOTH_REJECTED];
      psi_floor += cell.values[ID_CTS_SMOOTH_PSI_FLOOR];
      if (cell.values[ID_CTS_SMOOTH_MAX_UPDATE] > max_update) {
        max_update = cell.values[ID_CTS_SMOOTH_MAX_UPDATE];
      }
    }
  }, Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_ACCEPTED]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_LIMITED]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_BACKTRACKED]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_FALLBACK]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_SINGULAR]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_NONFINITE]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_REJECTED]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_PSI_FLOOR]),
     Kokkos::Max<Real>(stats.values[ID_CTS_SMOOTH_MAX_UPDATE]));
  return stats;
}

template <int NGHOST, typename ViewType, typename MaskViewType>
SmootherCellStats SmoothImplMasked(IDCTSMultigrid *mg, ViewType &u,
                             const ViewType &u_frozen, const ViewType &src,
                             const ViewType &free, const MaskViewType &mask,
                             int ll, int is, int ie, int js, int je, int ks, int ke,
                             int color, Real omega, int fd_stencil, int smoother_type,
                             int ngs_iterations, Real ngs_jacobian_eps,
                             Real ngs_max_update,
                             Real smoother_max_update_fraction,
                             int ngs_line_search_steps, Real ngs_line_search_min) {
  using ExeSpace = typename ViewType::execution_space;
  auto brdx = [&]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>) return mg->GetBlockDx_h();
    else return mg->GetBlockDx();
  }();
  int nmmb = mg->GetNumMeshBlocks();
  int rlev = -ll;
  SmootherCellStats stats;
  stats.Clear();
  Kokkos::parallel_reduce("IDCTS::SmoothMasked",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<3>>(
        {0, ks, js}, {nmmb, ke + 1, je + 1}),
  KOKKOS_LAMBDA(int m, int k, int j, Real &accepted, Real &limited,
                Real &backtracked, Real &fallback, Real &singular,
                Real &nonfinite, Real &rejected, Real &psi_floor,
                Real &max_update) {
    Real dx = (rlev <= 0) ? brdx(m)*static_cast<Real>(1<<(-rlev))
                          : brdx(m)/static_cast<Real>(1<<rlev);
    Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
    int c = (color + k + j) & 1;
    for (int i = is + c; i <= ie; i += 2) {
      if (mask(m, COMP_RELAX, k, j, i) == 0) continue;
      if (free(m, ID_FREE_MASK, k, j, i) < 0.5) continue;
      SmootherCellStats cell;
      if (smoother_type == 1) {
        cell = ApplyNewtonGSCTSUpdate<NGHOST>(u_frozen, u, src, free, idx, fd_stencil,
                                              m, k, j, i, omega, ngs_iterations,
                                              ngs_jacobian_eps, ngs_max_update,
                                              smoother_max_update_fraction,
                                              ngs_line_search_steps,
                                              ngs_line_search_min);
      } else {
        cell = ApplyDiagonalCTSUpdate<NGHOST>(u_frozen, u, src, free, idx, fd_stencil,
                                              m, k, j, i, omega, ngs_max_update,
                                              smoother_max_update_fraction);
      }
      accepted += cell.values[ID_CTS_SMOOTH_ACCEPTED];
      limited += cell.values[ID_CTS_SMOOTH_LIMITED];
      backtracked += cell.values[ID_CTS_SMOOTH_BACKTRACKED];
      fallback += cell.values[ID_CTS_SMOOTH_FALLBACK];
      singular += cell.values[ID_CTS_SMOOTH_SINGULAR];
      nonfinite += cell.values[ID_CTS_SMOOTH_NONFINITE];
      rejected += cell.values[ID_CTS_SMOOTH_REJECTED];
      psi_floor += cell.values[ID_CTS_SMOOTH_PSI_FLOOR];
      if (cell.values[ID_CTS_SMOOTH_MAX_UPDATE] > max_update) {
        max_update = cell.values[ID_CTS_SMOOTH_MAX_UPDATE];
      }
    }
  }, Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_ACCEPTED]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_LIMITED]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_BACKTRACKED]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_FALLBACK]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_SINGULAR]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_NONFINITE]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_REJECTED]),
     Kokkos::Sum<Real>(stats.values[ID_CTS_SMOOTH_PSI_FLOOR]),
     Kokkos::Max<Real>(stats.values[ID_CTS_SMOOTH_MAX_UPDATE]));
  return stats;
}

template <int NGHOST, typename ViewType>
void DefectImpl(IDCTSMultigrid *mg, ViewType &def, const ViewType &u, const ViewType &src,
                const ViewType &free, int ll, int is, int ie, int js, int je, int ks, int ke,
                int fd_stencil) {
  using ExeSpace = typename ViewType::execution_space;
  auto brdx = [&]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>) return mg->GetBlockDx_h();
    else return mg->GetBlockDx();
  }();
  int nmmb = mg->GetNumMeshBlocks();
  int rlev = -ll;
  par_for("IDCTS::Defect", ExeSpace(), 0, nmmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx = (rlev <= 0) ? brdx(m)*static_cast<Real>(1<<(-rlev))
                          : brdx(m)/static_cast<Real>(1<<rlev);
    Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
    Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
    CTSOperator<NGHOST>(u, free, idx, fd_stencil, m, k, j, i, op, diag);
    for (int v = 0; v < ID_CTS_NVAR; ++v) {
      def(m,v,k,j,i) = (free(m, ID_FREE_MASK, k, j, i) >= 0.5)
                       ? src(m,v,k,j,i) - op[v] : 0.0;
    }
  });
}

template <int NGHOST, typename ViewType, typename MaskViewType>
void DefectImplMasked(IDCTSMultigrid *mg, ViewType &def, const ViewType &u,
                const ViewType &src, const ViewType &free, const MaskViewType &mask,
                int ll, int is, int ie, int js, int je, int ks, int ke,
                int fd_stencil) {
  using ExeSpace = typename ViewType::execution_space;
  auto brdx = [&]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>) return mg->GetBlockDx_h();
    else return mg->GetBlockDx();
  }();
  int nmmb = mg->GetNumMeshBlocks();
  int rlev = -ll;
  par_for("IDCTS::DefectMasked", ExeSpace(), 0, nmmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    if (mask(m, COMP_VALID, k, j, i) == 0 ||
        free(m, ID_FREE_MASK, k, j, i) < 0.5) {
      for (int v = 0; v < ID_CTS_NVAR; ++v) {
        def(m, v, k, j, i) = 0.0;
      }
      return;
    }
    Real dx = (rlev <= 0) ? brdx(m)*static_cast<Real>(1<<(-rlev))
                          : brdx(m)/static_cast<Real>(1<<rlev);
    Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
    Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
    CTSOperator<NGHOST>(u, free, idx, fd_stencil, m, k, j, i, op, diag);
    for (int v = 0; v < ID_CTS_NVAR; ++v) {
      def(m, v, k, j, i) = src(m, v, k, j, i) - op[v];
    }
  });
}

template <int NGHOST, typename ViewType, typename MaskViewType>
CompositeResidualStats CompositeResidualStatsImpl(IDCTSMultigrid *mg,
                const ViewType &u, const ViewType &src, const ViewType &free,
                const MaskViewType &mask, int ll, int is, int ie, int js, int je,
                int ks, int ke, int fd_stencil, int var, bool include_covered) {
  using ExeSpace = typename ViewType::execution_space;
  auto brdx = [&]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>) return mg->GetBlockDx_h();
    else return mg->GetBlockDx();
  }();
  int nmmb = mg->GetNumMeshBlocks();
  int rlev = -ll;
  CompositeResidualStats stats;
  Kokkos::parallel_reduce("IDCTS::CompositeResidualStats",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<4>>(
        {0, ks, js, is}, {nmmb, ke + 1, je + 1, ie + 1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                  Real &valid_sum2, Real &valid_sum_abs, Real &valid_max,
                  Real &valid_volume, long long &valid_count,
                  Real &interface_sum2, Real &interface_volume,
                  long long &interface_count, Real &covered_sum2,
                  Real &covered_volume, long long &covered_count) {
      const bool is_valid = (mask(m, COMP_VALID, k, j, i) != 0);
      const bool is_interface = is_valid && (mask(m, COMP_INTERFACE, k, j, i) != 0);
      const bool is_covered = include_covered && (mask(m, COMP_COVERED, k, j, i) != 0);
      if (!is_valid && !is_covered) return;

      Real dx = (rlev <= 0) ? brdx(m)*static_cast<Real>(1<<(-rlev))
                            : brdx(m)/static_cast<Real>(1<<rlev);
      Real dV = dx*dx*dx;
      Real residual = 0.0;
      if (free(m, ID_FREE_MASK, k, j, i) >= 0.5) {
        Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
        Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
        CTSOperator<NGHOST>(u, free, idx, fd_stencil, m, k, j, i, op, diag);
        residual = src(m, var, k, j, i) - op[var];
      }
      const Real abs_res = std::abs(residual);
      if (is_valid) {
        valid_sum2 += dV*residual*residual;
        valid_sum_abs += dV*abs_res;
        valid_max = std::max(valid_max, abs_res);
        valid_volume += dV;
        ++valid_count;
      }
      if (is_interface) {
        interface_sum2 += dV*residual*residual;
        interface_volume += dV;
        ++interface_count;
      }
      if (is_covered) {
        covered_sum2 += dV*residual*residual;
        covered_volume += dV;
        ++covered_count;
      }
    }, Kokkos::Sum<Real>(stats.valid_sum2),
       Kokkos::Sum<Real>(stats.valid_sum_abs),
       Kokkos::Max<Real>(stats.valid_max),
       Kokkos::Sum<Real>(stats.valid_volume),
       Kokkos::Sum<long long>(stats.valid_count),
       Kokkos::Sum<Real>(stats.interface_sum2),
       Kokkos::Sum<Real>(stats.interface_volume),
       Kokkos::Sum<long long>(stats.interface_count),
       Kokkos::Sum<Real>(stats.covered_sum2),
       Kokkos::Sum<Real>(stats.covered_volume),
       Kokkos::Sum<long long>(stats.covered_count));
  return stats;
}

template <int NGHOST, typename ViewType>
void FASRHSImpl(IDCTSMultigrid *mg, ViewType &src, const ViewType &u, const ViewType &free,
                int ll, int is, int ie, int js, int je, int ks, int ke,
                int fd_stencil) {
  using ExeSpace = typename ViewType::execution_space;
  auto brdx = [&]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>) return mg->GetBlockDx_h();
    else return mg->GetBlockDx();
  }();
  int nmmb = mg->GetNumMeshBlocks();
  int rlev = -ll;
  par_for("IDCTS::FASRHS", ExeSpace(), 0, nmmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx = (rlev <= 0) ? brdx(m)*static_cast<Real>(1<<(-rlev))
                          : brdx(m)/static_cast<Real>(1<<rlev);
    Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
    Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
    CTSOperator<NGHOST>(u, free, idx, fd_stencil, m, k, j, i, op, diag);
    for (int v = 0; v < ID_CTS_NVAR; ++v) src(m,v,k,j,i) += op[v];
  });
}

template <int NGHOST, typename OpViewType, typename UViewType, typename SrcViewType,
          typename FreeViewType, typename MaskViewType>
void ComputeCTSOperatorImpl(IDCTSMultigrid *mg, const OpViewType &opdst,
                const UViewType &u, const SrcViewType &src, const FreeViewType &free,
                const MaskViewType &mask, int ll, int is, int ie, int js, int je,
                int ks, int ke, int fd_stencil) {
  using ExeSpace = typename OpViewType::execution_space;
  auto brdx = [&]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>) return mg->GetBlockDx_h();
    else return mg->GetBlockDx();
  }();
  int nmmb = mg->GetNumMeshBlocks();
  int rlev = -ll;
  par_for("IDCTS::ComputeCTSOperator", ExeSpace(), 0, nmmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    if (mask(m, COMP_VALID, k, j, i) == 0 ||
        free(m, ID_FREE_MASK, k, j, i) < 0.5) {
      for (int v = 0; v < ID_CTS_NVAR; ++v) {
        opdst(m, v, k, j, i) = src(m, v, k, j, i);
      }
      return;
    }
    Real dx = (rlev <= 0) ? brdx(m)*static_cast<Real>(1<<(-rlev))
                          : brdx(m)/static_cast<Real>(1<<rlev);
    Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
    Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
    CTSOperator<NGHOST>(u, free, idx, fd_stencil, m, k, j, i, op, diag);
    for (int v = 0; v < ID_CTS_NVAR; ++v) {
      opdst(m, v, k, j, i) = op[v];
    }
  });
}

template <typename ViewType, typename MaskViewType>
void SetCompositePreFASSourceImpl(const ViewType &dst_src, const ViewType &rf,
                                  const ViewType &rlh, const MaskViewType &mask,
                                  int nvar, int is, int ie, int js, int je,
                                  int ks, int ke) {
  using ExeSpace = typename ViewType::execution_space;
  const int nm = dst_src.extent_int(0);
  par_for("IDCTS::SetCompositePreFASSource", ExeSpace(),
          0, nm-1, 0, nvar-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int v, int k, int j, int i) {
    dst_src(m, v, k, j, i) = (mask(m, COMP_VALID, k, j, i) != 0)
        ? rf(m, v, k, j, i) - rlh(m, v, k, j, i) : 0.0;
  });
}

template <int NGHOST, typename ViewType, typename MaskViewType>
CompositeTauStats CompositeFASRHSImpl(IDCTSMultigrid *mg, const ViewType &src,
                const ViewType &u, const ViewType &free, const ViewType &rf,
                const ViewType &rlh, const MaskViewType &mask, int ll, int is,
                int ie, int js, int je, int ks, int ke, int fd_stencil) {
  using ExeSpace = typename ViewType::execution_space;
  auto brdx = [&]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>) return mg->GetBlockDx_h();
    else return mg->GetBlockDx();
  }();
  int nmmb = mg->GetNumMeshBlocks();
  int rlev = -ll;
  CompositeTauStats stats;
  Kokkos::parallel_reduce("IDCTS::CompositeFASRHS",
    Kokkos::MDRangePolicy<ExeSpace, Kokkos::Rank<4>>(
        {0, ks, js, is}, {nmmb, ke + 1, je + 1, ie + 1}),
    KOKKOS_LAMBDA(int m, int k, int j, int i, long long &dst_count,
                  long long &insufficient_count, Real &rf_sum2, Real &lhr_sum2,
                  Real &rlh_sum2, Real &tau_sum2, Real &tau_max,
                  Real &consistency_sum2, Real &consistency_max) {
      if (mask(m, COMP_VALID, k, j, i) == 0) return;
      ++dst_count;
      Real op[ID_CTS_NVAR];
      for (int v = 0; v < ID_CTS_NVAR; ++v) op[v] = 0.0;
      if (free(m, ID_FREE_MASK, k, j, i) >= 0.5) {
        Real dx = (rlev <= 0) ? brdx(m)*static_cast<Real>(1<<(-rlev))
                              : brdx(m)/static_cast<Real>(1<<rlev);
        Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
        Real diag[ID_CTS_NVAR];
        CTSOperator<NGHOST>(u, free, idx, fd_stencil, m, k, j, i, op, diag);
      }
      for (int v = 0; v < ID_CTS_NVAR; ++v) {
        const Real rf_v = rf(m, v, k, j, i);
        const Real rlh_v = rlh(m, v, k, j, i);
        const Real lhr_v = op[v];
        const Real tau_v = lhr_v - rlh_v;
        src(m, v, k, j, i) += lhr_v;
        const Real consistency = (src(m, v, k, j, i) - lhr_v) - (rf_v - rlh_v);
        rf_sum2 += rf_v*rf_v;
        lhr_sum2 += lhr_v*lhr_v;
        rlh_sum2 += rlh_v*rlh_v;
        tau_sum2 += tau_v*tau_v;
        const Real abs_tau = std::abs(tau_v);
        if (abs_tau > tau_max) tau_max = abs_tau;
        consistency_sum2 += consistency*consistency;
        const Real abs_consistency = std::abs(consistency);
        if (abs_consistency > consistency_max) consistency_max = abs_consistency;
      }
    }, Kokkos::Sum<long long>(stats.dst_count),
       Kokkos::Sum<long long>(stats.insufficient_count),
       Kokkos::Sum<Real>(stats.rf_sum2),
       Kokkos::Sum<Real>(stats.lhr_sum2),
       Kokkos::Sum<Real>(stats.rlh_sum2),
       Kokkos::Sum<Real>(stats.tau_sum2),
       Kokkos::Max<Real>(stats.tau_max),
       Kokkos::Sum<Real>(stats.consistency_sum2),
       Kokkos::Max<Real>(stats.consistency_max));
  return stats;
}

struct OctetArray {
  Real *data;
  int nc;

  KOKKOS_INLINE_FUNCTION
  Real& operator()(int, int v, int k, int j, int i) const {
    return data[((v*nc + k)*nc + j)*nc + i];
  }
};

template <typename Getter>
Real LagrangeExtrapolate(int nsrc, Real x, const Getter &get) {
  Real out = 0.0;
  for (int a = 0; a < nsrc; ++a) {
    Real w = 1.0;
    for (int b = 0; b < nsrc; ++b) {
      if (a != b) w *= (x - static_cast<Real>(b))/static_cast<Real>(a - b);
    }
    out += w*get(a);
  }
  return out;
}

int BoundaryStencilCount(int ngh, bool opposite_boundary) {
  // A boundary octet has two owned cells.  If the opposite side is also a
  // physical boundary, only those two cells are guaranteed to be valid.  Use
  // at most a quadratic extrapolation; wider one-sided fits are too oscillatory
  // for the mixed metric/matter coefficient set at octet corners.
  return opposite_boundary ? 2 : std::min(3, ngh + 2);
}

void ApplyCoefficientPhysicalBoundaries(MultigridDriver *driver, MGOctet &oct,
                                        int ncoeff, int locrootlevel,
                                        int nrbx1, int nrbx2, int nrbx3) {
  (void)driver;
  const int ngh = (oct.nc - 2)/2;
  auto ref = [&](int v, int k, int j, int i) -> Real& {
    return oct.Coeff(v, k, j, i);
  };
  int lev = oct.loc.level - locrootlevel;
  int maxlx1 = nrbx1 << lev;
  int maxlx2 = nrbx2 << lev;
  int maxlx3 = nrbx3 << lev;
  if (oct.loc.lx1 == 0) {
    int nsrc = BoundaryStencilCount(ngh, maxlx1 == 1);
    for (int v = 0; v < ncoeff; ++v)
      for (int k = 0; k < oct.nc; ++k)
        for (int j = 0; j < oct.nc; ++j)
          for (int n = 0; n < ngh; ++n) {
            ref(v, k, j, ngh-1-n) = (v == ID_FREE_MASK)
              ? ref(v, k, j, ngh)
              : LagrangeExtrapolate(nsrc, -1.0 - static_cast<Real>(n),
                                    [&](int s) { return ref(v, k, j, ngh+s); });
          }
  }
  if (oct.loc.lx1 == maxlx1 - 1) {
    int nsrc = BoundaryStencilCount(ngh, maxlx1 == 1);
    for (int v = 0; v < ncoeff; ++v)
      for (int k = 0; k < oct.nc; ++k)
        for (int j = 0; j < oct.nc; ++j)
          for (int n = 0; n < ngh; ++n) {
            ref(v, k, j, ngh+2+n) = (v == ID_FREE_MASK)
              ? ref(v, k, j, ngh+1)
              : LagrangeExtrapolate(nsrc, -1.0 - static_cast<Real>(n),
                                    [&](int s) { return ref(v, k, j, ngh+1-s); });
          }
  }
  if (oct.loc.lx2 == 0) {
    int nsrc = BoundaryStencilCount(ngh, maxlx2 == 1);
    for (int v = 0; v < ncoeff; ++v)
      for (int k = 0; k < oct.nc; ++k)
        for (int i = 0; i < oct.nc; ++i)
          for (int n = 0; n < ngh; ++n) {
            ref(v, k, ngh-1-n, i) = (v == ID_FREE_MASK)
              ? ref(v, k, ngh, i)
              : LagrangeExtrapolate(nsrc, -1.0 - static_cast<Real>(n),
                                    [&](int s) { return ref(v, k, ngh+s, i); });
          }
  }
  if (oct.loc.lx2 == maxlx2 - 1) {
    int nsrc = BoundaryStencilCount(ngh, maxlx2 == 1);
    for (int v = 0; v < ncoeff; ++v)
      for (int k = 0; k < oct.nc; ++k)
        for (int i = 0; i < oct.nc; ++i)
          for (int n = 0; n < ngh; ++n) {
            ref(v, k, ngh+2+n, i) = (v == ID_FREE_MASK)
              ? ref(v, k, ngh+1, i)
              : LagrangeExtrapolate(nsrc, -1.0 - static_cast<Real>(n),
                                    [&](int s) { return ref(v, k, ngh+1-s, i); });
          }
  }
  if (oct.loc.lx3 == 0) {
    int nsrc = BoundaryStencilCount(ngh, maxlx3 == 1);
    for (int v = 0; v < ncoeff; ++v)
      for (int j = 0; j < oct.nc; ++j)
        for (int i = 0; i < oct.nc; ++i)
          for (int n = 0; n < ngh; ++n) {
            ref(v, ngh-1-n, j, i) = (v == ID_FREE_MASK)
              ? ref(v, ngh, j, i)
              : LagrangeExtrapolate(nsrc, -1.0 - static_cast<Real>(n),
                                    [&](int s) { return ref(v, ngh+s, j, i); });
          }
  }
  if (oct.loc.lx3 == maxlx3 - 1) {
    int nsrc = BoundaryStencilCount(ngh, maxlx3 == 1);
    for (int v = 0; v < ncoeff; ++v)
      for (int j = 0; j < oct.nc; ++j)
        for (int i = 0; i < oct.nc; ++i)
          for (int n = 0; n < ngh; ++n) {
            ref(v, ngh+2+n, j, i) = (v == ID_FREE_MASK)
              ? ref(v, ngh+1, j, i)
              : LagrangeExtrapolate(nsrc, -1.0 - static_cast<Real>(n),
                                    [&](int s) { return ref(v, ngh+1-s, j, i); });
          }
  }
}

} // namespace

IDCTSMultigrid::IDCTSMultigrid(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
                               bool on_host)
    : Multigrid(pmd, pmbp, nghost, on_host),
      composite_rf_(nlevel_), composite_rlh_(nlevel_),
      composite_pre_fas_ready_(nlevel_, 0) {
  for (int l = 0; l < nlevel_; ++l) {
    int ll = nlevel_ - 1 - l;
    int ncx = (indcs_.nx1 >> ll) + 2*ngh_;
    int ncy = (indcs_.nx2 >> ll) + 2*ngh_;
    int ncz = (indcs_.nx3 >> ll) + 2*ngh_;
    Kokkos::realloc(composite_rf_[l], nmmb_, nvar_, ncz, ncy, ncx);
    Kokkos::realloc(composite_rlh_[l], nmmb_, nvar_, ncz, ncy, ncx);
  }
}

IDCTSMultigrid::~IDCTSMultigrid() {}

void IDCTSMultigrid::PrepareFrozenView() {
  const auto &u_level = u_[current_level_].d_view;
  Kokkos::realloc(frozen_u_, u_level.extent_int(0), u_level.extent_int(1),
                  u_level.extent_int(2), u_level.extent_int(3),
                  u_level.extent_int(4));
  if (on_host_) {
    Kokkos::deep_copy(HostExeSpace(), frozen_u_.h_view, u_[current_level_].h_view);
  } else {
    Kokkos::deep_copy(DevExeSpace(), frozen_u_.d_view, u_[current_level_].d_view);
  }
}

void IDCTSMultigrid::SmoothPack(int color) {
  auto *driver = static_cast<IDCTSMultigridDriver*>(pmy_driver_);
  color ^= driver->GetCoffset();
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  int fine_fd = driver->owner_->pmy_pack_->pz4c->opt.fd_stencil;
  int fd = (pmy_pack_ != nullptr && current_level_ == nlevel_ - 1)
           ? fine_fd : driver->mg_coarse_fd_stencil_;
  SmootherCellStats stats;
  stats.Clear();
  PrepareFrozenView();
  const bool use_composite_mask =
      driver->composite_fas_ && (pmy_pack_ != nullptr || current_level_ == nlevel_ - 1);
  if (use_composite_mask && !driver->composite_masks_ready_) {
    std::cout << "### FATAL ERROR in IDCTSMultigrid::SmoothPack" << std::endl
              << "Composite FAS smoother requested before composite masks were built."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (on_host_) {
    if (use_composite_mask) {
      auto mask = comp_mask_[current_level_].h_view;
      switch (fd) {
        case 2: stats = SmoothImplMasked<2>(this, u_[current_level_].h_view,
                                      frozen_u_.h_view,
                                      src_[current_level_].h_view,
                                      coeff_[current_level_].h_view, mask, ll, is, ie,
                                      js, je, ks, ke, color,
                                      driver->active_smooth_omega_, fd,
                                      driver->smoother_type_, driver->ngs_iterations_,
                                      driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                      driver->smoother_max_update_fraction_,
                                      driver->ngs_line_search_steps_,
                                      driver->ngs_line_search_min_); break;
        case 3: stats = SmoothImplMasked<3>(this, u_[current_level_].h_view,
                                      frozen_u_.h_view,
                                      src_[current_level_].h_view,
                                      coeff_[current_level_].h_view, mask, ll, is, ie,
                                      js, je, ks, ke, color,
                                      driver->active_smooth_omega_, fd,
                                      driver->smoother_type_, driver->ngs_iterations_,
                                      driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                      driver->smoother_max_update_fraction_,
                                      driver->ngs_line_search_steps_,
                                      driver->ngs_line_search_min_); break;
        default: stats = SmoothImplMasked<4>(this, u_[current_level_].h_view,
                                       frozen_u_.h_view,
                                       src_[current_level_].h_view,
                                       coeff_[current_level_].h_view, mask, ll, is, ie,
                                       js, je, ks, ke, color,
                                       driver->active_smooth_omega_, fd,
                                       driver->smoother_type_, driver->ngs_iterations_,
                                       driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                       driver->smoother_max_update_fraction_,
                                       driver->ngs_line_search_steps_,
                                       driver->ngs_line_search_min_); break;
      }
    } else {
      switch (fd) {
        case 2: stats = SmoothImpl<2>(this, u_[current_level_].h_view,
                                      frozen_u_.h_view,
                                      src_[current_level_].h_view,
                                      coeff_[current_level_].h_view, ll, is, ie, js, je,
                                      ks, ke, color, driver->active_smooth_omega_, fd,
                                      driver->smoother_type_, driver->ngs_iterations_,
                                      driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                      driver->smoother_max_update_fraction_,
                                      driver->ngs_line_search_steps_,
                                      driver->ngs_line_search_min_); break;
        case 3: stats = SmoothImpl<3>(this, u_[current_level_].h_view,
                                      frozen_u_.h_view,
                                      src_[current_level_].h_view,
                                      coeff_[current_level_].h_view, ll, is, ie, js, je,
                                      ks, ke, color, driver->active_smooth_omega_, fd,
                                      driver->smoother_type_, driver->ngs_iterations_,
                                      driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                      driver->smoother_max_update_fraction_,
                                      driver->ngs_line_search_steps_,
                                      driver->ngs_line_search_min_); break;
        default: stats = SmoothImpl<4>(this, u_[current_level_].h_view,
                                       frozen_u_.h_view,
                                       src_[current_level_].h_view,
                                       coeff_[current_level_].h_view, ll, is, ie, js, je,
                                       ks, ke, color, driver->active_smooth_omega_, fd,
                                       driver->smoother_type_, driver->ngs_iterations_,
                                       driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                       driver->smoother_max_update_fraction_,
                                       driver->ngs_line_search_steps_,
                                       driver->ngs_line_search_min_); break;
      }
    }
  } else {
    if (use_composite_mask) {
      auto mask = comp_mask_[current_level_].d_view;
      switch (fd) {
        case 2: stats = SmoothImplMasked<2>(this, u_[current_level_].d_view,
                                      frozen_u_.d_view,
                                      src_[current_level_].d_view,
                                      coeff_[current_level_].d_view, mask, ll, is, ie,
                                      js, je, ks, ke, color,
                                      driver->active_smooth_omega_, fd,
                                      driver->smoother_type_, driver->ngs_iterations_,
                                      driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                      driver->smoother_max_update_fraction_,
                                      driver->ngs_line_search_steps_,
                                      driver->ngs_line_search_min_); break;
        case 3: stats = SmoothImplMasked<3>(this, u_[current_level_].d_view,
                                      frozen_u_.d_view,
                                      src_[current_level_].d_view,
                                      coeff_[current_level_].d_view, mask, ll, is, ie,
                                      js, je, ks, ke, color,
                                      driver->active_smooth_omega_, fd,
                                      driver->smoother_type_, driver->ngs_iterations_,
                                      driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                      driver->smoother_max_update_fraction_,
                                      driver->ngs_line_search_steps_,
                                      driver->ngs_line_search_min_); break;
        default: stats = SmoothImplMasked<4>(this, u_[current_level_].d_view,
                                       frozen_u_.d_view,
                                       src_[current_level_].d_view,
                                       coeff_[current_level_].d_view, mask, ll, is, ie,
                                       js, je, ks, ke, color,
                                       driver->active_smooth_omega_, fd,
                                       driver->smoother_type_, driver->ngs_iterations_,
                                       driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                       driver->smoother_max_update_fraction_,
                                       driver->ngs_line_search_steps_,
                                       driver->ngs_line_search_min_); break;
      }
    } else {
      switch (fd) {
        case 2: stats = SmoothImpl<2>(this, u_[current_level_].d_view,
                                      frozen_u_.d_view,
                                      src_[current_level_].d_view,
                                      coeff_[current_level_].d_view, ll, is, ie, js, je,
                                      ks, ke, color, driver->active_smooth_omega_, fd,
                                      driver->smoother_type_, driver->ngs_iterations_,
                                      driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                      driver->smoother_max_update_fraction_,
                                      driver->ngs_line_search_steps_,
                                      driver->ngs_line_search_min_); break;
        case 3: stats = SmoothImpl<3>(this, u_[current_level_].d_view,
                                      frozen_u_.d_view,
                                      src_[current_level_].d_view,
                                      coeff_[current_level_].d_view, ll, is, ie, js, je,
                                      ks, ke, color, driver->active_smooth_omega_, fd,
                                      driver->smoother_type_, driver->ngs_iterations_,
                                      driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                      driver->smoother_max_update_fraction_,
                                      driver->ngs_line_search_steps_,
                                      driver->ngs_line_search_min_); break;
        default: stats = SmoothImpl<4>(this, u_[current_level_].d_view,
                                       frozen_u_.d_view,
                                       src_[current_level_].d_view,
                                       coeff_[current_level_].d_view, ll, is, ie, js, je,
                                       ks, ke, color, driver->active_smooth_omega_, fd,
                                       driver->smoother_type_, driver->ngs_iterations_,
                                       driver->ngs_jacobian_eps_, driver->ngs_max_update_,
                                       driver->smoother_max_update_fraction_,
                                       driver->ngs_line_search_steps_,
                                       driver->ngs_line_search_min_); break;
      }
    }
  }
  driver->AccumulateSmootherStats(stats.values);
}

void IDCTSMultigrid::CalculateDefectPack() {
  auto *driver = static_cast<IDCTSMultigridDriver*>(pmy_driver_);
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  int fine_fd = driver->owner_->pmy_pack_->pz4c->opt.fd_stencil;
  int fd = (pmy_pack_ != nullptr && current_level_ == nlevel_ - 1)
           ? fine_fd : driver->mg_coarse_fd_stencil_;
  const bool use_composite_mask =
      driver->composite_fas_ && (pmy_pack_ != nullptr || current_level_ == nlevel_ - 1);
  if (use_composite_mask && !driver->composite_masks_ready_) {
    std::cout << "### FATAL ERROR in IDCTSMultigrid::CalculateDefectPack" << std::endl
              << "Composite FAS defect requested before composite masks were built."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (on_host_) {
    if (use_composite_mask) {
      auto mask = comp_mask_[current_level_].h_view;
      switch (fd) {
        case 2: DefectImplMasked<2>(this, def_[current_level_].h_view,
                              u_[current_level_].h_view, src_[current_level_].h_view,
                              coeff_[current_level_].h_view, mask, ll, is, ie, js, je,
                              ks, ke, fd); break;
        case 3: DefectImplMasked<3>(this, def_[current_level_].h_view,
                              u_[current_level_].h_view, src_[current_level_].h_view,
                              coeff_[current_level_].h_view, mask, ll, is, ie, js, je,
                              ks, ke, fd); break;
        default: DefectImplMasked<4>(this, def_[current_level_].h_view,
                               u_[current_level_].h_view, src_[current_level_].h_view,
                               coeff_[current_level_].h_view, mask, ll, is, ie, js, je,
                               ks, ke, fd); break;
      }
    } else {
      switch (fd) {
        case 2: DefectImpl<2>(this, def_[current_level_].h_view, u_[current_level_].h_view,
                              src_[current_level_].h_view, coeff_[current_level_].h_view,
                              ll, is, ie, js, je, ks, ke, fd); break;
        case 3: DefectImpl<3>(this, def_[current_level_].h_view, u_[current_level_].h_view,
                              src_[current_level_].h_view, coeff_[current_level_].h_view,
                              ll, is, ie, js, je, ks, ke, fd); break;
        default: DefectImpl<4>(this, def_[current_level_].h_view, u_[current_level_].h_view,
                               src_[current_level_].h_view, coeff_[current_level_].h_view,
                               ll, is, ie, js, je, ks, ke, fd); break;
      }
    }
  } else {
    if (use_composite_mask) {
      auto mask = comp_mask_[current_level_].d_view;
      switch (fd) {
        case 2: DefectImplMasked<2>(this, def_[current_level_].d_view,
                              u_[current_level_].d_view, src_[current_level_].d_view,
                              coeff_[current_level_].d_view, mask, ll, is, ie, js, je,
                              ks, ke, fd); break;
        case 3: DefectImplMasked<3>(this, def_[current_level_].d_view,
                              u_[current_level_].d_view, src_[current_level_].d_view,
                              coeff_[current_level_].d_view, mask, ll, is, ie, js, je,
                              ks, ke, fd); break;
        default: DefectImplMasked<4>(this, def_[current_level_].d_view,
                               u_[current_level_].d_view, src_[current_level_].d_view,
                               coeff_[current_level_].d_view, mask, ll, is, ie, js, je,
                               ks, ke, fd); break;
      }
    } else {
      switch (fd) {
        case 2: DefectImpl<2>(this, def_[current_level_].d_view, u_[current_level_].d_view,
                              src_[current_level_].d_view, coeff_[current_level_].d_view,
                              ll, is, ie, js, je, ks, ke, fd); break;
        case 3: DefectImpl<3>(this, def_[current_level_].d_view, u_[current_level_].d_view,
                              src_[current_level_].d_view, coeff_[current_level_].d_view,
                              ll, is, ie, js, je, ks, ke, fd); break;
        default: DefectImpl<4>(this, def_[current_level_].d_view, u_[current_level_].d_view,
                               src_[current_level_].d_view, coeff_[current_level_].d_view,
                               ll, is, ie, js, je, ks, ke, fd); break;
      }
    }
  }
}

void IDCTSMultigrid::DiagnosticRestrictPack() {
  auto *driver = static_cast<IDCTSMultigridDriver*>(pmy_driver_);
  if (!driver->composite_fas_ || !driver->debug_composite_restriction_) return;
  if (!driver->composite_masks_ready_) {
    std::cout << "### FATAL ERROR in IDCTSMultigrid::DiagnosticRestrictPack"
              << std::endl
              << "Composite restriction diagnostics requested before masks were built."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!driver->composite_restriction_self_check_done_) {
    RunCompositeRestrictionSelfCheck();
    driver->composite_restriction_self_check_done_ = true;
    if (global_variable::my_rank == 0) {
      std::cout << "CTS composite restriction self-check: passed" << std::endl;
    }
  }

  // Commit 1 only builds meaningful root bridge masks on the finest root level.
  if (pmy_pack_ == nullptr && current_level_ != nlevel_ - 1) return;

  const int ll = nlevel_ - current_level_;
  const int is = ngh_;
  const int js = ngh_;
  const int ks = ngh_;
  const int ie = is + (indcs_.nx1 >> ll) - 1;
  const int je = js + (indcs_.nx2 >> ll) - 1;
  const int ke = ks + (indcs_.nx3 >> ll) - 1;
  if (ie < is || je < js || ke < ks) return;

  const bool half_weight =
      (driver->composite_restriction_ == ID_CTS_RESTRICT_HALF_WEIGHT);
  const char *entity = (pmy_pack_ == nullptr) ? "root" : "meshblock";
  const bool compute_on_this_rank =
      (pmy_pack_ != nullptr || global_variable::my_rank == 0);

  CompositeRestrictionStats local_u;
  CompositeRestrictionStats local_def;
  if (!compute_on_this_rank) {
    // Root grids are replicated; only rank 0 contributes root diagnostic counts.
  } else if (on_host_) {
    auto mask = comp_mask_[current_level_].h_view;
    local_u = DiagnoseHalfWeightRestrictionImpl(u_[current_level_].h_view, mask,
                                                nvar_, is, ie, js, je, ks, ke,
                                                ngh_, half_weight);
    local_def = DiagnoseHalfWeightRestrictionImpl(def_[current_level_].h_view, mask,
                                                  nvar_, is, ie, js, je, ks, ke,
                                                  ngh_, half_weight);
  } else {
    auto mask = comp_mask_[current_level_].d_view;
    local_u = DiagnoseHalfWeightRestrictionImpl(u_[current_level_].d_view, mask,
                                                nvar_, is, ie, js, je, ks, ke,
                                                ngh_, half_weight);
    local_def = DiagnoseHalfWeightRestrictionImpl(def_[current_level_].d_view, mask,
                                                  nvar_, is, ie, js, je, ks, ke,
                                                  ngh_, half_weight);
  }

  CompositeRestrictionStats global_u = ReduceCompositeRestrictionStats(local_u);
  CompositeRestrictionStats global_def = ReduceCompositeRestrictionStats(local_def);
  PrintCompositeRestrictionStats(entity, current_level_, "u",
                                 driver->composite_restriction_, global_u);
  PrintCompositeRestrictionStats(entity, current_level_, "def",
                                 driver->composite_restriction_, global_def);
}

bool IDCTSMultigrid::CompositeRestrictPack() {
  auto *driver = static_cast<IDCTSMultigridDriver*>(pmy_driver_);
  if (!driver->composite_fas_) return false;
  if (pmy_pack_ == nullptr) return false;
  if (current_level_ - 1 <= driver->MeshBlockTransferLevel()) return false;
  if (driver->composite_restriction_ != ID_CTS_RESTRICT_HALF_WEIGHT) {
    std::cout << "### FATAL ERROR in IDCTSMultigrid::CompositeRestrictPack"
              << std::endl
              << "Production composite FAS restriction requires "
              << "id_solve/composite_restriction=half_weight." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!driver->composite_masks_ready_) {
    std::cout << "### FATAL ERROR in IDCTSMultigrid::CompositeRestrictPack"
              << std::endl
              << "Composite FAS restriction requested before masks were built."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int fine_level = current_level_;
  const int coarse_level = current_level_ - 1;
  const int fine_shift = nlevel_ - 1 - fine_level;
  const int coarse_shift = nlevel_ - fine_level;
  const int fis = ngh_;
  const int fjs = ngh_;
  const int fks = ngh_;
  const int fie = fis + (indcs_.nx1 >> fine_shift) - 1;
  const int fje = fjs + (indcs_.nx2 >> fine_shift) - 1;
  const int fke = fks + (indcs_.nx3 >> fine_shift) - 1;
  const int cis = ngh_;
  const int cjs = ngh_;
  const int cks = ngh_;
  const int cie = cis + (indcs_.nx1 >> coarse_shift) - 1;
  const int cje = cjs + (indcs_.nx2 >> coarse_shift) - 1;
  const int cke = cks + (indcs_.nx3 >> coarse_shift) - 1;

  DualArray5D<Real> fine_op;
  Kokkos::realloc(fine_op, nmmb_, nvar_, def_[fine_level].d_view.extent_int(2),
                  def_[fine_level].d_view.extent_int(3),
                  def_[fine_level].d_view.extent_int(4));
  int fine_fd = driver->owner_->pmy_pack_->pz4c->opt.fd_stencil;
  int fd = (fine_level == nlevel_ - 1) ? fine_fd : driver->mg_coarse_fd_stencil_;
  if (on_host_) {
    auto fine_mask = comp_mask_[fine_level].h_view;
    switch (fd) {
      case 2:
        ComputeCTSOperatorImpl<2>(this, fine_op.h_view, u_[fine_level].h_view,
                                  src_[fine_level].h_view, coeff_[fine_level].h_view,
                                  fine_mask, fine_shift, fis, fie, fjs, fje, fks, fke, fd);
        break;
      case 3:
        ComputeCTSOperatorImpl<3>(this, fine_op.h_view, u_[fine_level].h_view,
                                  src_[fine_level].h_view, coeff_[fine_level].h_view,
                                  fine_mask, fine_shift, fis, fie, fjs, fje, fks, fke, fd);
        break;
      default:
        ComputeCTSOperatorImpl<4>(this, fine_op.h_view, u_[fine_level].h_view,
                                  src_[fine_level].h_view, coeff_[fine_level].h_view,
                                  fine_mask, fine_shift, fis, fie, fjs, fje, fks, fke, fd);
        break;
    }
    CompositeRestrictionStats u_stats =
        RestrictHalfWeightImpl(u_[coarse_level].h_view, u_[fine_level].h_view,
                               fine_mask, nvar_, cis, cie, cjs, cje, cks, cke, ngh_);
    CompositeRestrictionStats rf_stats =
        RestrictHalfWeightImpl(composite_rf_[coarse_level].h_view, src_[fine_level].h_view,
                               fine_mask, nvar_, cis, cie, cjs, cje, cks, cke, ngh_);
    CompositeRestrictionStats rlh_stats =
        RestrictHalfWeightImpl(composite_rlh_[coarse_level].h_view, fine_op.h_view,
                               fine_mask, nvar_, cis, cie, cjs, cje, cks, cke, ngh_);
    CompositeRestrictionStats gu = ReduceCompositeRestrictionStats(u_stats);
    CompositeRestrictionStats grf = ReduceCompositeRestrictionStats(rf_stats);
    CompositeRestrictionStats grlh = ReduceCompositeRestrictionStats(rlh_stats);
    const long long insufficient =
        gu.insufficient_count + grf.insufficient_count + grlh.insufficient_count;
    if (insufficient != 0) {
      std::cout << "### FATAL ERROR in IDCTSMultigrid::CompositeRestrictPack"
                << std::endl
                << "MeshBlock composite half-weight restriction has insufficient "
                << "stencil support: " << insufficient << "." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto coarse_mask = comp_mask_[coarse_level].h_view;
    SetCompositePreFASSourceImpl(src_[coarse_level].h_view,
                                 composite_rf_[coarse_level].h_view,
                                 composite_rlh_[coarse_level].h_view,
                                 coarse_mask, nvar_, cis, cie, cjs, cje, cks, cke);
  } else {
    auto fine_mask = comp_mask_[fine_level].d_view;
    switch (fd) {
      case 2:
        ComputeCTSOperatorImpl<2>(this, fine_op.d_view, u_[fine_level].d_view,
                                  src_[fine_level].d_view, coeff_[fine_level].d_view,
                                  fine_mask, fine_shift, fis, fie, fjs, fje, fks, fke, fd);
        break;
      case 3:
        ComputeCTSOperatorImpl<3>(this, fine_op.d_view, u_[fine_level].d_view,
                                  src_[fine_level].d_view, coeff_[fine_level].d_view,
                                  fine_mask, fine_shift, fis, fie, fjs, fje, fks, fke, fd);
        break;
      default:
        ComputeCTSOperatorImpl<4>(this, fine_op.d_view, u_[fine_level].d_view,
                                  src_[fine_level].d_view, coeff_[fine_level].d_view,
                                  fine_mask, fine_shift, fis, fie, fjs, fje, fks, fke, fd);
        break;
    }
    CompositeRestrictionStats u_stats =
        RestrictHalfWeightImpl(u_[coarse_level].d_view, u_[fine_level].d_view,
                               fine_mask, nvar_, cis, cie, cjs, cje, cks, cke, ngh_);
    CompositeRestrictionStats rf_stats =
        RestrictHalfWeightImpl(composite_rf_[coarse_level].d_view, src_[fine_level].d_view,
                               fine_mask, nvar_, cis, cie, cjs, cje, cks, cke, ngh_);
    CompositeRestrictionStats rlh_stats =
        RestrictHalfWeightImpl(composite_rlh_[coarse_level].d_view, fine_op.d_view,
                               fine_mask, nvar_, cis, cie, cjs, cje, cks, cke, ngh_);
    CompositeRestrictionStats gu = ReduceCompositeRestrictionStats(u_stats);
    CompositeRestrictionStats grf = ReduceCompositeRestrictionStats(rf_stats);
    CompositeRestrictionStats grlh = ReduceCompositeRestrictionStats(rlh_stats);
    const long long insufficient =
        gu.insufficient_count + grf.insufficient_count + grlh.insufficient_count;
    if (insufficient != 0) {
      std::cout << "### FATAL ERROR in IDCTSMultigrid::CompositeRestrictPack"
                << std::endl
                << "MeshBlock composite half-weight restriction has insufficient "
                << "stencil support: " << insufficient << "." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto coarse_mask = comp_mask_[coarse_level].d_view;
    SetCompositePreFASSourceImpl(src_[coarse_level].d_view,
                                 composite_rf_[coarse_level].d_view,
                                 composite_rlh_[coarse_level].d_view,
                                 coarse_mask, nvar_, cis, cie, cjs, cje, cks, cke);
  }
  composite_pre_fas_ready_[coarse_level] = 1;
  return true;
}

void IDCTSMultigrid::CalculateFASRHSPack() {
  auto *driver = static_cast<IDCTSMultigridDriver*>(pmy_driver_);
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  int fine_fd = driver->owner_->pmy_pack_->pz4c->opt.fd_stencil;
  int fd = (pmy_pack_ != nullptr && current_level_ == nlevel_ - 1)
           ? fine_fd : driver->mg_coarse_fd_stencil_;
  if (driver->composite_fas_ && composite_pre_fas_ready_[current_level_]) {
    // CTS uses def = src - L(u).  CompositeRestrictPack stores
    // src_H = R(src_h - L_h(u_h)); this step adds L_H(Ru_h), giving
    // the FAS RHS f_H = L_H(Ru_h) + R(f_h - L_h(u_h)).
    CompositeTauStats local_stats;
    if (on_host_) {
      auto mask = comp_mask_[current_level_].h_view;
      switch (fd) {
        case 2:
          local_stats = CompositeFASRHSImpl<2>(
              this, src_[current_level_].h_view, u_[current_level_].h_view,
              coeff_[current_level_].h_view, composite_rf_[current_level_].h_view,
              composite_rlh_[current_level_].h_view, mask, ll, is, ie, js, je, ks, ke, fd);
          break;
        case 3:
          local_stats = CompositeFASRHSImpl<3>(
              this, src_[current_level_].h_view, u_[current_level_].h_view,
              coeff_[current_level_].h_view, composite_rf_[current_level_].h_view,
              composite_rlh_[current_level_].h_view, mask, ll, is, ie, js, je, ks, ke, fd);
          break;
        default:
          local_stats = CompositeFASRHSImpl<4>(
              this, src_[current_level_].h_view, u_[current_level_].h_view,
              coeff_[current_level_].h_view, composite_rf_[current_level_].h_view,
              composite_rlh_[current_level_].h_view, mask, ll, is, ie, js, je, ks, ke, fd);
          break;
      }
    } else {
      auto mask = comp_mask_[current_level_].d_view;
      switch (fd) {
        case 2:
          local_stats = CompositeFASRHSImpl<2>(
              this, src_[current_level_].d_view, u_[current_level_].d_view,
              coeff_[current_level_].d_view, composite_rf_[current_level_].d_view,
              composite_rlh_[current_level_].d_view, mask, ll, is, ie, js, je, ks, ke, fd);
          break;
        case 3:
          local_stats = CompositeFASRHSImpl<3>(
              this, src_[current_level_].d_view, u_[current_level_].d_view,
              coeff_[current_level_].d_view, composite_rf_[current_level_].d_view,
              composite_rlh_[current_level_].d_view, mask, ll, is, ie, js, je, ks, ke, fd);
          break;
        default:
          local_stats = CompositeFASRHSImpl<4>(
              this, src_[current_level_].d_view, u_[current_level_].d_view,
              coeff_[current_level_].d_view, composite_rf_[current_level_].d_view,
              composite_rlh_[current_level_].d_view, mask, ll, is, ie, js, je, ks, ke, fd);
          break;
      }
    }
    if (driver->debug_composite_tau_) {
      CompositeTauStats global_stats = ReduceCompositeTauStats(local_stats);
      PrintCompositeTauStats(current_level_ + 1, current_level_, global_stats);
    }
    composite_pre_fas_ready_[current_level_] = 0;
    return;
  }
  if (on_host_) {
    switch (fd) {
      case 2: FASRHSImpl<2>(this, src_[current_level_].h_view, u_[current_level_].h_view,
                            coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                            fd); break;
      case 3: FASRHSImpl<3>(this, src_[current_level_].h_view, u_[current_level_].h_view,
                            coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                            fd); break;
      default: FASRHSImpl<4>(this, src_[current_level_].h_view, u_[current_level_].h_view,
                             coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                             fd); break;
    }
  } else {
    switch (fd) {
      case 2: FASRHSImpl<2>(this, src_[current_level_].d_view, u_[current_level_].d_view,
                            coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                            fd); break;
      case 3: FASRHSImpl<3>(this, src_[current_level_].d_view, u_[current_level_].d_view,
                            coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                            fd); break;
      default: FASRHSImpl<4>(this, src_[current_level_].d_view, u_[current_level_].d_view,
                             coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                             fd); break;
    }
  }
}

void IDCTSMultigridDriver::ValidateCompositeFASOptions() const {
  if (!composite_fas_) return;
  auto *mesh = pmy_pack_->pmesh;
  auto fail = [](const std::string &msg) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl << msg << std::endl;
    std::exit(EXIT_FAILURE);
  };
  if (!mesh->multilevel || mesh->adaptive) {
    fail("<id_solve>/composite_fas=true currently requires static SMR "
         "(<mesh_refinement>/refinement=static), not unigrid or adaptive AMR.");
  }
  if (nreflevel_ <= 0) {
    fail("<id_solve>/composite_fas=true requires at least one static refinement "
         "level (nreflevel_ > 0).");
  }
  if (mg_coarse_fd_stencil_ != 2) {
    fail("<id_solve>/composite_fas=true currently requires "
         "<id_solve>/mg_coarse_fd_stencil=2.");
  }
  if (owner_->pmy_pack_->pz4c->opt.spatial_order != 2) {
    std::ostringstream msg;
    msg << "<id_solve>/composite_fas=true currently supports only "
        << "<z4c>/spatial_order=2. High-order composite FAS is reserved "
        << "for later milestones; requested spatial_order="
        << owner_->pmy_pack_->pz4c->opt.spatial_order << ".";
    fail(msg.str());
  }
  if (!composite_second_order_only_) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in IDCTSMultigridDriver::IDCTSMultigridDriver"
                << std::endl
                << "<id_solve>/composite_second_order_only=false is parsed, but "
                << "Commit 1 still restricts composite_fas to second-order SMR."
                << std::endl;
    }
  }
}

IDCTSMultigridDriver::IDCTSMultigridDriver(IDConformalThinSandwich *owner,
                                           MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, ID_CTS_NVAR), owner_(owner) {
  ncoeff_ = ID_FREE_NVAR;
  locrootlevel_ = pmbp->pmesh->root_level;
  nreflevel_ = 0;
  if (pmbp->pmesh->multilevel) {
    for (int n = 0; n < nbtotal_; ++n) {
      int lev = pmbp->pmesh->lloc_eachmb[n].level - locrootlevel_;
      nreflevel_ = std::max(nreflevel_, lev);
    }
  }
  composite_fas_ = pin->GetOrAddBoolean("id_solve", "composite_fas", false);
  composite_second_order_only_ =
      pin->GetOrAddBoolean("id_solve", "composite_second_order_only", true);
  debug_composite_masks_ =
      pin->GetOrAddBoolean("id_solve", "debug_composite_masks", false);
  debug_composite_residual_ =
      pin->GetOrAddBoolean("id_solve", "debug_composite_residual", false);
  composite_masks_ready_ = false;
  std::string composite_restriction =
      pin->GetOrAddString("id_solve", "composite_restriction", "half_weight");
  if (composite_restriction == "half_weight") {
    composite_restriction_ = ID_CTS_RESTRICT_HALF_WEIGHT;
  } else if (composite_restriction == "average") {
    composite_restriction_ = ID_CTS_RESTRICT_AVERAGE;
  } else {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "id_solve/composite_restriction must be half_weight or average."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  debug_composite_restriction_ =
      pin->GetOrAddBoolean("id_solve", "debug_composite_restriction", false);
  composite_restriction_self_check_done_ = false;
  debug_composite_tau_ =
      pin->GetOrAddBoolean("id_solve", "debug_composite_tau", false);
  composite_tau_deferred_note_printed_ = false;
  omega_ = pin->GetOrAddReal("id_solve", "omega", 1.0);
  default_smooth_omega_ = omega_;
  active_smooth_omega_ = omega_;
  post_smooth_omega_ = pin->GetOrAddReal("id_solve", "post_smooth_omega", omega_);
  if (post_smooth_omega_ < 0.0) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/post_smooth_omega must be non-negative." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string post_smooth_mode =
      pin->GetOrAddString("id_solve", "post_smooth_mode", "both");
  if (post_smooth_mode == "both") {
    post_smooth_mode_ = 0;
  } else if (post_smooth_mode == "red_only") {
    post_smooth_mode_ = 1;
  } else if (post_smooth_mode == "black_only") {
    post_smooth_mode_ = 2;
  } else if (post_smooth_mode == "red_boundary") {
    post_smooth_mode_ = 3;
  } else if (post_smooth_mode == "no_fc_after_prolongation") {
    post_smooth_mode_ = 4;
  } else if (post_smooth_mode == "none" ||
             post_smooth_mode == "none_at_transfer_level") {
    post_smooth_mode_ = 5;
  } else {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/post_smooth_mode must be 'both', 'red_only', "
              << "'black_only', 'red_boundary', 'no_fc_after_prolongation', "
              << "'none', or 'none_at_transfer_level', but is "
              << post_smooth_mode << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string smoother = pin->GetOrAddString("id_solve", "smoother", "diagonal");
  if (smoother == "diagonal" || smoother == "diag") {
    smoother_type_ = 0;
  } else if (smoother == "newton_gs" || smoother == "newton-gauss-seidel" ||
             smoother == "ngs") {
    smoother_type_ = 1;
  } else {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/smoother must be 'diagonal' or 'newton_gs', but is "
              << smoother << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ngs_iterations_ = pin->GetOrAddInteger("id_solve", "ngs_iterations", 1);
  std::string smoother_update =
      pin->GetOrAddString("id_solve", "smoother_update", "frozen_view");
  if (smoother_update != "frozen_view") {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/smoother_update currently supports only 'frozen_view', "
              << "but is " << smoother_update << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ngs_jacobian_eps_ = pin->GetOrAddReal("id_solve", "ngs_jacobian_eps", 1.0e-7);
  ngs_max_update_ = pin->GetOrAddReal("id_solve", "ngs_max_update", 0.0);
  smoother_max_update_fraction_ =
      pin->GetOrAddReal("id_solve", "smoother_max_update_fraction", 0.25);
  ngs_line_search_steps_ = pin->GetOrAddInteger("id_solve", "ngs_line_search_steps", 8);
  ngs_line_search_min_ = pin->GetOrAddReal("id_solve", "ngs_line_search_min", 1.0e-4);
  show_smoother_stats_ = pin->GetOrAddBoolean("id_solve", "show_smoother_stats", true);
  if (ngs_iterations_ < 1 || !(ngs_jacobian_eps_ > 0.0) || ngs_max_update_ < 0.0 ||
      smoother_max_update_fraction_ < 0.0 || ngs_line_search_steps_ < 1 ||
      !(ngs_line_search_min_ > 0.0) || ngs_line_search_min_ > 1.0) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/ngs_iterations must be >=1, ngs_jacobian_eps must be >0, "
              << "ngs_max_update and smoother_max_update_fraction must be >=0, "
              << "ngs_line_search_steps must be >=1, and ngs_line_search_min must be "
              << "in (0,1]." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  eps_ = pin->GetOrAddReal("id_solve", "threshold", -1.0);
  niter_ = pin->GetOrAddInteger("id_solve", "niteration", 1);
  max_iter_ = pin->GetOrAddInteger("id_solve", "max_iterations", 40);
  npresmooth_ = pin->GetOrAddInteger("id_solve", "npresmooth", npresmooth_);
  npostsmooth_ = pin->GetOrAddInteger("id_solve", "npostsmooth", npostsmooth_);
  full_multigrid_ = pin->GetOrAddBoolean("id_solve", "full_multigrid", false);
  fmg_ncycle_ = pin->GetOrAddInteger("id_solve", "fmg_ncycle", 1);
  fshowdef_ = pin->GetOrAddBoolean("id_solve", "show_defect", true);
  reject_worse_ = pin->GetOrAddBoolean("id_solve", "reject_worse", true);
  keep_best_solution_ = pin->GetOrAddBoolean("id_solve", "keep_best_solution", true);
  stop_on_defect_increase_ =
      pin->GetOrAddBoolean("id_solve", "stop_on_defect_increase", false);
  defect_increase_tol_ =
      pin->GetOrAddReal("id_solve", "defect_increase_tolerance", 1.0e-3);
  allow_incomplete_amr_ = pin->GetOrAddBoolean("id_solve", "allow_incomplete_amr", false);
  solution_applied_ = false;
  ResetSmootherStats();
  fsubtract_average_ = false;
  octet_correction_omega_ =
      pin->GetOrAddReal("id_solve", "octet_correction_omega", 1.0);
  disable_octet_correction_ =
      pin->GetOrAddBoolean("id_solve", "disable_octet_correction", false);
  check_octet_coefficients_ =
      pin->GetOrAddBoolean("id_solve", "check_octet_coefficients", false);
  if (disable_octet_correction_) octet_correction_omega_ = 0.0;
  if (octet_correction_omega_ < 0.0 || octet_correction_omega_ > 1.0) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/octet_correction_omega must be in [0,1]."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string mg_bc_str = pin->GetOrAddString("id_solve", "mg_bc", "zerograd");
  BoundaryFlag id_mg_bc;
  if (mg_bc_str == "zerograd") {
    id_mg_bc = BoundaryFlag::mg_zerograd;
  } else if (mg_bc_str == "zerofixed") {
    id_mg_bc = BoundaryFlag::mg_zerofixed;
  } else {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/mg_bc must be 'zerograd' or 'zerofixed', but is "
              << mg_bc_str << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int f = 0; f < 6; ++f) {
    mg_mesh_bcs_[f] = (pmbp->pmesh->mesh_bcs[f] == BoundaryFlag::periodic)
                      ? BoundaryFlag::periodic : id_mg_bc;
  }

  int mesh_nghost = pmbp->pmesh->mb_indcs.ng;
  int nghost = pin->GetOrAddInteger("id_solve", "mg_nghost", mesh_nghost);
  if (nghost > mesh_nghost) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/mg_nghost cannot exceed <mesh>/nghost because the CTS "
              << "free-data arrays are allocated on the mesh ghost halo." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int fd_stencil = pmbp->pz4c->opt.fd_stencil;
  int free_data_reach = 2*(fd_stencil - 1);
  if (mesh_nghost < free_data_reach) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "Native CTS with <z4c>/spatial_order=" << pmbp->pz4c->opt.spatial_order
              << " requires <mesh>/nghost >= " << free_data_reach
              << " for the gamma-dot/DK free-data prepass, but <mesh>/nghost="
              << mesh_nghost << ".  This CTS-specific prepass requirement can "
              << "be stricter than the multigrid correction-variable halo."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nghost < fd_stencil) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/mg_nghost=" << nghost << " is too small for the CTS "
              << "operator selected by <z4c>/spatial_order="
              << pmbp->pz4c->opt.spatial_order << ".  The compact "
              << "D_j Ahat^{ij} operator requires at least " << fd_stencil
              << " ghost cells." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int requested_coarse_fd = pin->GetOrAddInteger("id_solve", "mg_coarse_fd_stencil", 0);
  mg_coarse_fd_stencil_ = (requested_coarse_fd > 0) ? requested_coarse_fd :
                          ((fd_stencil > 2) ? 2 : fd_stencil);
  if (mg_coarse_fd_stencil_ < 2 || mg_coarse_fd_stencil_ > 4 ||
      mg_coarse_fd_stencil_ > fd_stencil) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/mg_coarse_fd_stencil must be 2, 3, or 4 and cannot "
              << "exceed the fine Z4c stencil " << fd_stencil << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nghost < mg_coarse_fd_stencil_) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/mg_nghost=" << nghost << " is too small for "
              << "<id_solve>/mg_coarse_fd_stencil=" << mg_coarse_fd_stencil_ << "."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mg_coarse_fd_stencil_ < fd_stencil && global_variable::my_rank == 0) {
    std::cout << "### WARNING in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "Using CTS coarse MG stencil fd=" << mg_coarse_fd_stencil_
              << " below the finest MeshBlock level." << std::endl;
  }
  ValidateCompositeFASOptions();
  std::string coarse_prolongation =
      pin->GetOrAddString("id_solve", "coarse_prolongation", "auto");
  if (coarse_prolongation == "auto") {
    fprolongation_ = (nreflevel_ > 0 && mg_coarse_fd_stencil_ == 2) ? 0 : 1;
  } else if (coarse_prolongation == "linear") {
    fprolongation_ = 0;
  } else if (coarse_prolongation == "cubic") {
    fprolongation_ = 1;
  } else {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/coarse_prolongation must be 'auto', 'linear', or "
              << "'cubic', but is " << coarse_prolongation << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool explicit_coarse_omega =
      pin->DoesParameterExist("id_solve", "coarse_correction_omega");
  Real coarse_default =
      (!explicit_coarse_omega && nreflevel_ > 0 && mg_coarse_fd_stencil_ == 2) ? 0.0 : 1.0;
  coarse_correction_omega_ =
      pin->GetOrAddReal("id_solve", "coarse_correction_omega", coarse_default);
  if (coarse_correction_omega_ < 0.0 || coarse_correction_omega_ > 1.0) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/coarse_correction_omega must be in [0,1]."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  root_correction_omega_ =
      pin->GetOrAddReal("id_solve", "root_correction_omega", coarse_correction_omega_);
  meshblock_correction_omega_ =
      pin->GetOrAddReal("id_solve", "meshblock_correction_omega", coarse_correction_omega_);
  std::string meshblock_correction_mode =
      pin->GetOrAddString("id_solve", "meshblock_correction_mode", "linear");
  if (meshblock_correction_mode == "none") {
    meshblock_correction_mode_ = 0;
  } else if (meshblock_correction_mode == "injection") {
    meshblock_correction_mode_ = 1;
  } else if (meshblock_correction_mode == "linear") {
    meshblock_correction_mode_ = 2;
  } else if (meshblock_correction_mode == "linear_active_only") {
    meshblock_correction_mode_ = 3;
  } else {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/meshblock_correction_mode must be 'none', 'injection', "
              << "'linear', or 'linear_active_only', but is "
              << meshblock_correction_mode << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  meshblock_correction_sign_ =
      pin->GetOrAddInteger("id_solve", "meshblock_correction_sign", 1);
  if (meshblock_correction_sign_ != 1 && meshblock_correction_sign_ != -1) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/meshblock_correction_sign must be 1 or -1."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  debug_meshblock_correction_ =
      pin->GetOrAddBoolean("id_solve", "debug_meshblock_correction", false);
  if (root_correction_omega_ < 0.0 || root_correction_omega_ > 1.0 ||
      meshblock_correction_omega_ < 0.0 || meshblock_correction_omega_ > 1.0) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/root_correction_omega and "
              << "<id_solve>/meshblock_correction_omega must be in [0,1]."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (full_multigrid_ && nreflevel_ > 0 && mg_coarse_fd_stencil_ == 2
      && fprolongation_ == 1 && global_variable::my_rank == 0) {
    std::cout << "### WARNING in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "Using tricubic coarse_prolongation with second-order SMR FMG; "
              << "linear is the safer bridge policy." << std::endl;
  }
  int requested_octet_fd = pin->GetOrAddInteger("id_solve", "octet_fd_stencil", 0);
  octet_fd_stencil_ = (requested_octet_fd > 0) ? requested_octet_fd
                                               : mg_coarse_fd_stencil_;
  if (octet_fd_stencil_ < 2 || octet_fd_stencil_ > 4 || octet_fd_stencil_ > fd_stencil) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/octet_fd_stencil must be 2, 3, or 4 and cannot exceed "
              << "the Z4c stencil " << fd_stencil << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nghost < octet_fd_stencil_) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/mg_nghost=" << nghost << " is too small for "
              << "<id_solve>/octet_fd_stencil=" << octet_fd_stencil_ << "."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (octet_fd_stencil_ < fd_stencil && global_variable::my_rank == 0) {
    std::cout << "### WARNING in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "Using CTS octet stencil fd=" << octet_fd_stencil_
              << " for the halo-limited SMR bridge." << std::endl;
  }
  std::string octet_prolongation =
      pin->GetOrAddString("id_solve", "octet_prolongation", "auto");
  if (octet_prolongation == "auto") {
    octet_prolongation_ = (octet_fd_stencil_ == 2) ? 0 : 1;
  } else if (octet_prolongation == "linear") {
    octet_prolongation_ = 0;
  } else if (octet_prolongation == "cubic") {
    octet_prolongation_ = 1;
  } else {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/octet_prolongation must be 'auto', 'linear', or "
              << "'cubic', but is " << octet_prolongation << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool root_on_host = pin->GetOrAddBoolean("id_solve", "root_on_host", false);
  mgroot_ = new IDCTSMultigrid(this, nullptr, nghost, root_on_host);
  mglevels_ = new IDCTSMultigrid(this, pmbp, nghost);
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers(std::max(nvar_, ncoeff_));
  mglevels_->pbval->RemapIndicesForMG();
}

IDCTSMultigridDriver::~IDCTSMultigridDriver() {
  delete mgroot_;
  delete mglevels_;
}

void IDCTSMultigridDriver::ResetSmootherStats() {
  for (int n = 0; n < ID_CTS_SMOOTH_NSTAT; ++n) smoother_stats_[n] = 0.0;
}

void IDCTSMultigridDriver::AccumulateSmootherStats(
    const Real stats[ID_CTS_SMOOTH_NSTAT]) {
  for (int n = 0; n < ID_CTS_SMOOTH_MAX_UPDATE; ++n) smoother_stats_[n] += stats[n];
  if (stats[ID_CTS_SMOOTH_MAX_UPDATE] > smoother_stats_[ID_CTS_SMOOTH_MAX_UPDATE]) {
    smoother_stats_[ID_CTS_SMOOTH_MAX_UPDATE] = stats[ID_CTS_SMOOTH_MAX_UPDATE];
  }
}

void IDCTSMultigridDriver::PrintSmootherStats(int iter) const {
  if (!show_smoother_stats_) return;
  Real sums[ID_CTS_SMOOTH_MAX_UPDATE];
  for (int n = 0; n < ID_CTS_SMOOTH_MAX_UPDATE; ++n) sums[n] = smoother_stats_[n];
  Real max_update = smoother_stats_[ID_CTS_SMOOTH_MAX_UPDATE];
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums, ID_CTS_SMOOTH_MAX_UPDATE, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_update, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "  CTS smoother stats";
    if (iter >= 0) std::cout << " iteration " << iter;
    std::cout << ": accepted=" << sums[ID_CTS_SMOOTH_ACCEPTED]
              << ", limited=" << sums[ID_CTS_SMOOTH_LIMITED]
              << ", backtracked=" << sums[ID_CTS_SMOOTH_BACKTRACKED]
              << ", fallback=" << sums[ID_CTS_SMOOTH_FALLBACK]
              << ", singular=" << sums[ID_CTS_SMOOTH_SINGULAR]
              << ", nonfinite=" << sums[ID_CTS_SMOOTH_NONFINITE]
              << ", rejected=" << sums[ID_CTS_SMOOTH_REJECTED]
              << ", psi_floor=" << sums[ID_CTS_SMOOTH_PSI_FLOOR]
              << ", max_update=" << max_update << std::endl;
  }
}

void IDCTSMultigridDriver::TransferCoefficientsFromBlocksToRoot() {
  if (ncoeff_ <= 0) return;
  int transfer_level = MeshBlockTransferLevel();
  mglevels_->SyncCoefficientLevelToHost(transfer_level);
  auto root_coeff_h = mgroot_->GetCoefficientLevel_h(nrootlevel_ - 1);
  auto block_coeff_h = mglevels_->GetCoefficientLevel_h(transfer_level);
  int ngh = mgroot_->GetGhostCells();
  int bngh = mglevels_->GetGhostCells();
  int bncells = mglevels_->GetLevelActiveCells(transfer_level);
  Real inv_vol = 1.0/static_cast<Real>(bncells*bncells*bncells);
  int padding = nslist_[global_variable::my_rank];
  std::vector<Real> block_coeff_global(static_cast<std::size_t>(ncoeff_) * nbtotal_,
                                       0.0);
  for (int m = 0; m < mglevels_->GetNumMeshBlocks(); ++m) {
    int gid = m + padding;
    for (int v = 0; v < ncoeff_; ++v) {
      Real sum = 0.0;
      for (int k = bngh; k < bngh + bncells; ++k) {
        for (int j = bngh; j < bngh + bncells; ++j) {
          for (int i = bngh; i < bngh + bncells; ++i) {
            sum += block_coeff_h(m, v, k, j, i);
          }
        }
      }
      block_coeff_global[static_cast<std::size_t>(v)*nbtotal_ + gid] =
          sum * inv_vol;
    }
  }
#if MPI_PARALLEL_ENABLED
  std::vector<Real> local_coeff(mglevels_->GetNumMeshBlocks());
  for (int v = 0; v < ncoeff_; ++v) {
    for (int m = 0; m < mglevels_->GetNumMeshBlocks(); ++m) {
      Real sum = 0.0;
      for (int k = bngh; k < bngh + bncells; ++k) {
        for (int j = bngh; j < bngh + bncells; ++j) {
          for (int i = bngh; i < bngh + bncells; ++i) {
            sum += block_coeff_h(m, v, k, j, i);
          }
        }
      }
      local_coeff[m] = sum * inv_vol;
    }
    MPI_Allgatherv(local_coeff.data(), nblist_[global_variable::my_rank],
                   MPI_ATHENA_REAL,
                   block_coeff_global.data() + static_cast<std::size_t>(v)*nbtotal_,
                   nblist_, nslist_, MPI_ATHENA_REAL, MPI_COMM_WORLD);
  }
#endif

  for (int n = 0; n < nbtotal_; ++n) {
    LogicalLocation loc = pmy_mesh_->lloc_eachmb[n];
    if (loc.level == locrootlevel_) {
      int ri = static_cast<int>(loc.lx1) + ngh;
      int rj = static_cast<int>(loc.lx2) + ngh;
      int rk = static_cast<int>(loc.lx3) + ngh;
      for (int v = 0; v < ncoeff_; ++v) {
        root_coeff_h(0, v, rk, rj, ri) =
            block_coeff_global[static_cast<std::size_t>(v)*nbtotal_ + n];
      }
    } else {
      LogicalLocation oloc;
      oloc.lx1 = loc.lx1 >> 1;
      oloc.lx2 = loc.lx2 >> 1;
      oloc.lx3 = loc.lx3 >> 1;
      oloc.level = loc.level - 1;
      int olev = oloc.level - locrootlevel_;
      auto oit = octetmap_[olev].find(oloc);
      if (oit == octetmap_[olev].end()) {
        std::cout << "### FATAL ERROR in IDCTSMultigridDriver::TransferCoefficientsFromBlocksToRoot"
                  << std::endl
                  << "Missing parent octet for refined coefficient block." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      int oid = oit->second;
      int oi = (static_cast<int>(loc.lx1) & 1) + ngh;
      int oj = (static_cast<int>(loc.lx2) & 1) + ngh;
      int ok = (static_cast<int>(loc.lx3) & 1) + ngh;
      MGOctet &oct = octets_[olev][oid];
      for (int v = 0; v < ncoeff_; ++v) {
        oct.Coeff(v, ok, oj, oi) =
            block_coeff_global[static_cast<std::size_t>(v)*nbtotal_ + n];
      }
    }
  }

  for (int l = nreflevel_ - 1; l >= 1; --l) {
    for (int o = 0; o < noctets_[l]; ++o) {
      MGOctet &foct = octets_[l][o];
      const LogicalLocation &floc = foct.loc;
      LogicalLocation cloc;
      cloc.lx1 = floc.lx1 >> 1;
      cloc.lx2 = floc.lx2 >> 1;
      cloc.lx3 = floc.lx3 >> 1;
      cloc.level = floc.level - 1;
      auto oit = octetmap_[l-1].find(cloc);
      if (oit == octetmap_[l-1].end()) {
        std::cout << "### FATAL ERROR in IDCTSMultigridDriver::TransferCoefficientsFromBlocksToRoot"
                  << std::endl
                  << "Missing parent octet while restricting coefficient octets." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      int oid = oit->second;
      int oi = (static_cast<int>(floc.lx1) & 1) + ngh;
      int oj = (static_cast<int>(floc.lx2) & 1) + ngh;
      int ok = (static_cast<int>(floc.lx3) & 1) + ngh;
      MGOctet &coct = octets_[l-1][oid];
      for (int v = 0; v < ncoeff_; ++v) {
        coct.Coeff(v, ok, oj, oi) = RestrictOneCoeff(foct, v, ngh, ngh, ngh);
      }
    }
  }
  if (nreflevel_ > 0) {
    for (int o = 0; o < noctets_[0]; ++o) {
      MGOctet &oct = octets_[0][o];
      const LogicalLocation &oloc = oct.loc;
      for (int v = 0; v < ncoeff_; ++v) {
        root_coeff_h(0, v, static_cast<int>(oloc.lx3)+ngh,
                           static_cast<int>(oloc.lx2)+ngh,
                           static_cast<int>(oloc.lx1)+ngh) =
            RestrictOneCoeff(oct, v, ngh, ngh, ngh);
      }
    }
  }
  mgroot_->ModifyCoefficientLevelOnHost(nrootlevel_ - 1);
  mgroot_->SyncCoefficientLevelToDevice(nrootlevel_ - 1);
  mgroot_->SetCurrentLevel(nrootlevel_ - 1);
  mgroot_->FillCoefficientBoundaries(nrootlevel_ - 1);
  mgroot_->RestrictCoefficients();

  if (nreflevel_ > 0) {
    std::vector<Real> root_coeff_buf;
    int root_coeff_nc = 0;
    {
      auto root_coeff = mgroot_->GetCoefficientLevel_h(nrootlevel_ - 1);
      int rnx = root_coeff.extent_int(4);
      int rny = root_coeff.extent_int(3);
      int rnz = root_coeff.extent_int(2);
      root_coeff_nc = std::max({rnx, rny, rnz});
      root_coeff_buf.assign(static_cast<std::size_t>(ncoeff_) * root_coeff_nc
                            * root_coeff_nc * root_coeff_nc, 0.0);
      for (int v = 0; v < ncoeff_; ++v)
        for (int k = 0; k < rnz; ++k)
          for (int j = 0; j < rny; ++j)
            for (int i = 0; i < rnx; ++i)
              BufRef(root_coeff_buf, root_coeff_nc, v, k, j, i) = root_coeff(0,v,k,j,i);
    }

    for (int lev = 0; lev < nreflevel_; ++lev) {
      for (int o = 0; o < noctets_[lev]; ++o) {
        MGOctet &oct = octets_[lev][o];
        MGOctet coeff_oct = oct;
        coeff_oct.nvar = ncoeff_;
        coeff_oct.u = oct.coeff;
        std::fill(ncoarse_.begin(), ncoarse_.end(), false);
        std::fill(cbuf_.begin(), cbuf_.end(), 0.0);
        std::fill(cbufold_.begin(), cbufold_.end(), 0.0);
        for (int ox3 = -1; ox3 <= 1; ++ox3) {
          for (int ox2 = -1; ox2 <= 1; ++ox2) {
            for (int ox1 = -1; ox1 <= 1; ++ox1) {
              if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
              int dir = (ox3+1)*9 + (ox2+1)*3 + (ox1+1);
              const OctetNeighborInfo &nb = oct.neighbors[dir];
              if (nb.same_id == -2) {
                continue;
              } else if (nb.same_id >= 0) {
                MGOctet coeff_noct = octets_[lev][nb.same_id];
                coeff_noct.nvar = ncoeff_;
                coeff_noct.u = coeff_noct.coeff;
                SetOctetBoundarySameLevel(coeff_oct, coeff_noct, cbuf_, cbufold_,
                                          ncoeff_, ox1, ox2, ox3, false);
              } else {
                ncoarse_[dir] = true;
                if (lev > 0 && nb.coarse_id >= 0) {
                  MGOctet &coct = octets_[lev-1][nb.coarse_id];
                  SetOctetBoundaryFromCoarser(coct.coeff, nullptr, cbuf_, cbufold_,
                                              ncoeff_, coct.nc, oct.loc,
                                              ox1, ox2, ox3, false);
                } else if (lev == 0) {
                  LogicalLocation nloc;
                  nloc.level = oct.loc.level;
                  nloc.lx1 = oct.loc.lx1 + ox1;
                  nloc.lx2 = oct.loc.lx2 + ox2;
                  nloc.lx3 = oct.loc.lx3 + ox3;
                  if (nloc.lx1 < 0)       nloc.lx1 = nrbx1_ - 1;
                  if (nloc.lx1 >= nrbx1_) nloc.lx1 = 0;
                  if (nloc.lx2 < 0)       nloc.lx2 = nrbx2_ - 1;
                  if (nloc.lx2 >= nrbx2_) nloc.lx2 = 0;
                  if (nloc.lx3 < 0)       nloc.lx3 = nrbx3_ - 1;
                  if (nloc.lx3 >= nrbx3_) nloc.lx3 = 0;
                  SetOctetBoundaryFromCoarser(root_coeff_buf.data(), nullptr,
                                              cbuf_, cbufold_, ncoeff_, root_coeff_nc,
                                              nloc, ox1, ox2, ox3, false);
                }
              }
            }
          }
        }
        ProlongateOctetBoundaries(coeff_oct, cbuf_, cbufold_, ncoeff_, ncoarse_, false);
        ApplyCoefficientPhysicalBoundaries(this, oct, ncoeff_, locrootlevel_,
                                           nrbx1_, nrbx2_, nrbx3_);
      }
    }
  }
  if (check_octet_coefficients_) {
    mgroot_->SyncCoefficientLevelToHost(nrootlevel_ - 1);
    auto root_coeff_check = mgroot_->GetCoefficientLevel_h(nrootlevel_ - 1);
    Real local_l2 = 0.0;
    Real local_max = 0.0;
    long long local_count = 0;
    for (int n = 0; n < nbtotal_; ++n) {
      LogicalLocation loc = pmy_mesh_->lloc_eachmb[n];
      for (int v = 0; v < ncoeff_; ++v) {
        Real expected = block_coeff_global[static_cast<std::size_t>(v)*nbtotal_ + n];
        Real actual;
        if (loc.level == locrootlevel_) {
          int ri = static_cast<int>(loc.lx1) + ngh;
          int rj = static_cast<int>(loc.lx2) + ngh;
          int rk = static_cast<int>(loc.lx3) + ngh;
          actual = root_coeff_check(0, v, rk, rj, ri);
        } else {
          LogicalLocation oloc;
          oloc.lx1 = loc.lx1 >> 1;
          oloc.lx2 = loc.lx2 >> 1;
          oloc.lx3 = loc.lx3 >> 1;
          oloc.level = loc.level - 1;
          int olev = oloc.level - locrootlevel_;
          int oid = FindOctetIdOrDie(olev, oloc,
                                     "IDCTSMultigridDriver::TransferCoefficientsFromBlocksToRoot");
          int oi = (static_cast<int>(loc.lx1) & 1) + ngh;
          int oj = (static_cast<int>(loc.lx2) & 1) + ngh;
          int ok = (static_cast<int>(loc.lx3) & 1) + ngh;
          actual = octets_[olev][oid].Coeff(v, ok, oj, oi);
        }
        Real diff = actual - expected;
        local_l2 += diff * diff;
        local_max = std::max(local_max, std::abs(diff));
        ++local_count;
      }
    }
    Real global_l2 = local_l2;
    Real global_max = local_max;
    long long global_count = local_count;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&local_l2, &global_l2, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      Real rms = (global_count > 0) ? std::sqrt(global_l2/static_cast<Real>(global_count)) : 0.0;
      std::cout << "CTS root/octet coefficient payload gather-scatter check: max=" << global_max
                << " rms=" << rms << " count=" << global_count << std::endl;
    }
  }
}

void IDCTSMultigridDriver::BuildCompositeMasks() {
  if (!composite_fas_) return;
  composite_masks_ready_ = false;
  mgroot_->ClearCompositeMasks();
  mglevels_->ClearCompositeMasks();
  for (int l = 0; l < nreflevel_; ++l) {
    for (int o = 0; o < noctets_[l]; ++o) {
      octets_[l][o].ZeroClearMask();
    }
  }
  BuildCompositeMeshBlockMasks();
  BuildCompositeRootAndOctetMasks();
  composite_masks_ready_ = true;
  if (debug_composite_masks_) PrintCompositeMaskDiagnostics();
}

void IDCTSMultigridDriver::BuildCompositeMeshBlockMasks() {
  auto nghbr_h = pmy_pack_->pmb->nghbr.h_view;
  auto mblev_h = pmy_pack_->pmb->mb_lev.h_view;
  const int nnghbr = pmy_pack_->pmb->nnghbr;
  const int ngh = mglevels_->GetGhostCells();

  for (int level = 0; level < mglevels_->GetNumberOfLevels(); ++level) {
    auto mask = mglevels_->GetCompositeMaskLevel_h(level);
    const int nx = mask.extent_int(4);
    const int ny = mask.extent_int(3);
    const int nz = mask.extent_int(2);
    const int ncells = mglevels_->GetLevelActiveCells(level);

    for (int m = 0; m < mglevels_->GetNumMeshBlocks(); ++m) {
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            mask(m, COMP_VALID, k, j, i) = 0;
            mask(m, COMP_RELAX, k, j, i) = 0;
            mask(m, COMP_COVERED, k, j, i) = 1;
            mask(m, COMP_INTERFACE, k, j, i) = 0;
          }
        }
      }

      for (int k = ngh; k < ngh + ncells; ++k) {
        for (int j = ngh; j < ngh + ncells; ++j) {
          for (int i = ngh; i < ngh + ncells; ++i) {
            mask(m, COMP_VALID, k, j, i) = 1;
            mask(m, COMP_RELAX, k, j, i) = 1;
            mask(m, COMP_COVERED, k, j, i) = 0;
          }
        }
      }

      const int mlev = mblev_h(m);
      auto has_different_level_neighbor = [&](int ox1, int ox2, int ox3) {
        for (int f2 = 0; f2 <= 1; ++f2) {
          for (int f1 = 0; f1 <= 1; ++f1) {
            const int n = NeighborIndex(ox1, ox2, ox3, f1, f2);
            if (n < 0 || n >= nnghbr) continue;
            if (nghbr_h(m, n).gid >= 0 && nghbr_h(m, n).lev != mlev) return true;
          }
        }
        return false;
      };
      auto mark_face = [&](int ox1, int ox2, int ox3) {
        if (!has_different_level_neighbor(ox1, ox2, ox3)) return;
        const int il = (ox1 < 0) ? ngh : ngh + ncells - 1;
        const int jl = (ox2 < 0) ? ngh : ngh + ncells - 1;
        const int kl = (ox3 < 0) ? ngh : ngh + ncells - 1;
        const int is = (ox1 == 0) ? ngh : il;
        const int ie = (ox1 == 0) ? ngh + ncells - 1 : il;
        const int js = (ox2 == 0) ? ngh : jl;
        const int je = (ox2 == 0) ? ngh + ncells - 1 : jl;
        const int ks = (ox3 == 0) ? ngh : kl;
        const int ke = (ox3 == 0) ? ngh + ncells - 1 : kl;
        for (int k = ks; k <= ke; ++k)
          for (int j = js; j <= je; ++j)
            for (int i = is; i <= ie; ++i)
              if (mask(m, COMP_VALID, k, j, i) != 0)
                mask(m, COMP_INTERFACE, k, j, i) = 1;
      };
      mark_face(-1, 0, 0);
      mark_face( 1, 0, 0);
      mark_face(0, -1, 0);
      mark_face(0,  1, 0);
      mark_face(0, 0, -1);
      mark_face(0, 0,  1);
    }
    mglevels_->ModifyCompositeMaskLevelOnHost(level);
    mglevels_->SyncCompositeMaskLevelToDevice(level);
  }
}

void IDCTSMultigridDriver::BuildCompositeRootAndOctetMasks() {
  const int ngh = mgroot_->GetGhostCells();
  const int root_level = nrootlevel_ - 1;
  auto root_mask = mgroot_->GetCompositeMaskLevel_h(root_level);
  const int rnx = root_mask.extent_int(4);
  const int rny = root_mask.extent_int(3);
  const int rnz = root_mask.extent_int(2);

  for (int k = 0; k < rnz; ++k) {
    for (int j = 0; j < rny; ++j) {
      for (int i = 0; i < rnx; ++i) {
        root_mask(0, COMP_VALID, k, j, i) = 0;
        root_mask(0, COMP_RELAX, k, j, i) = 0;
        root_mask(0, COMP_COVERED, k, j, i) = 1;
        root_mask(0, COMP_INTERFACE, k, j, i) = 0;
      }
    }
  }

  auto wrap_index = [&](int &idx, int max_idx, BoundaryFace inner, BoundaryFace outer) {
    if (idx < 0) {
      if (mg_mesh_bcs_[inner] == BoundaryFlag::periodic) {
        idx = max_idx - 1;
        return true;
      }
      return false;
    }
    if (idx >= max_idx) {
      if (mg_mesh_bcs_[outer] == BoundaryFlag::periodic) {
        idx = 0;
        return true;
      }
      return false;
    }
    return true;
  };
  auto root_covered = [&](int i, int j, int k) {
    if (!wrap_index(i, nrbx1_, BoundaryFace::inner_x1, BoundaryFace::outer_x1) ||
        !wrap_index(j, nrbx2_, BoundaryFace::inner_x2, BoundaryFace::outer_x2) ||
        !wrap_index(k, nrbx3_, BoundaryFace::inner_x3, BoundaryFace::outer_x3)) {
      return false;
    }
    LogicalLocation loc;
    loc.level = locrootlevel_;
    loc.lx1 = i;
    loc.lx2 = j;
    loc.lx3 = k;
    return nreflevel_ > 0 && octetmap_[0].find(loc) != octetmap_[0].end();
  };

  for (int k = 0; k < nrbx3_; ++k) {
    for (int j = 0; j < nrbx2_; ++j) {
      for (int i = 0; i < nrbx1_; ++i) {
        const bool covered = root_covered(i, j, k);
        root_mask(0, COMP_VALID, k + ngh, j + ngh, i + ngh) = covered ? 0 : 1;
        root_mask(0, COMP_RELAX, k + ngh, j + ngh, i + ngh) = covered ? 0 : 1;
        root_mask(0, COMP_COVERED, k + ngh, j + ngh, i + ngh) = covered ? 1 : 0;
      }
    }
  }
  const int dirs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  for (int k = 0; k < nrbx3_; ++k) {
    for (int j = 0; j < nrbx2_; ++j) {
      for (int i = 0; i < nrbx1_; ++i) {
        if (root_mask(0, COMP_VALID, k + ngh, j + ngh, i + ngh) == 0) continue;
        for (const auto &dir : dirs) {
          if (root_covered(i + dir[0], j + dir[1], k + dir[2])) {
            root_mask(0, COMP_INTERFACE, k + ngh, j + ngh, i + ngh) = 1;
            break;
          }
        }
      }
    }
  }
  mgroot_->ModifyCompositeMaskLevelOnHost(root_level);
  mgroot_->SyncCompositeMaskLevelToDevice(root_level);

  for (int l = 0; l < nreflevel_; ++l) {
    const int maxx = nrbx1_ << (l + 1);
    const int maxy = nrbx2_ << (l + 1);
    const int maxz = nrbx3_ << (l + 1);
    auto child_covered = [&](LogicalLocation loc) {
      int lx1 = static_cast<int>(loc.lx1);
      int lx2 = static_cast<int>(loc.lx2);
      int lx3 = static_cast<int>(loc.lx3);
      if (!wrap_index(lx1, maxx, BoundaryFace::inner_x1, BoundaryFace::outer_x1) ||
          !wrap_index(lx2, maxy, BoundaryFace::inner_x2, BoundaryFace::outer_x2) ||
          !wrap_index(lx3, maxz, BoundaryFace::inner_x3, BoundaryFace::outer_x3)) {
        return false;
      }
      loc.lx1 = lx1;
      loc.lx2 = lx2;
      loc.lx3 = lx3;
      return (l + 1 < nreflevel_) &&
             (octetmap_[l + 1].find(loc) != octetmap_[l + 1].end());
    };
    for (int o = 0; o < noctets_[l]; ++o) {
      MGOctet &oct = octets_[l][o];
      for (int k = 0; k < oct.nc; ++k) {
        for (int j = 0; j < oct.nc; ++j) {
          for (int i = 0; i < oct.nc; ++i) {
            oct.Mask(COMP_VALID, k, j, i) = 0;
            oct.Mask(COMP_RELAX, k, j, i) = 0;
            oct.Mask(COMP_COVERED, k, j, i) = 1;
            oct.Mask(COMP_INTERFACE, k, j, i) = 0;
          }
        }
      }
      for (int ck = 0; ck <= 1; ++ck) {
        for (int cj = 0; cj <= 1; ++cj) {
          for (int ci = 0; ci <= 1; ++ci) {
            LogicalLocation child;
            child.level = oct.loc.level + 1;
            child.lx1 = (oct.loc.lx1 << 1) + ci;
            child.lx2 = (oct.loc.lx2 << 1) + cj;
            child.lx3 = (oct.loc.lx3 << 1) + ck;
            const bool covered = child_covered(child);
            const int i = ngh + ci;
            const int j = ngh + cj;
            const int k = ngh + ck;
            oct.Mask(COMP_VALID, k, j, i) = covered ? 0 : 1;
            oct.Mask(COMP_RELAX, k, j, i) = covered ? 0 : 1;
            oct.Mask(COMP_COVERED, k, j, i) = covered ? 1 : 0;
          }
        }
      }
      for (int ck = 0; ck <= 1; ++ck) {
        for (int cj = 0; cj <= 1; ++cj) {
          for (int ci = 0; ci <= 1; ++ci) {
            const int i = ngh + ci;
            const int j = ngh + cj;
            const int k = ngh + ck;
            if (oct.Mask(COMP_VALID, k, j, i) == 0) continue;
            for (const auto &dir : dirs) {
              LogicalLocation nloc;
              nloc.level = oct.loc.level + 1;
              nloc.lx1 = (oct.loc.lx1 << 1) + ci + dir[0];
              nloc.lx2 = (oct.loc.lx2 << 1) + cj + dir[1];
              nloc.lx3 = (oct.loc.lx3 << 1) + ck + dir[2];
              if (child_covered(nloc)) {
                oct.Mask(COMP_INTERFACE, k, j, i) = 1;
                break;
              }
            }
          }
        }
      }
    }
  }
}

void IDCTSMultigridDriver::PrintCompositeMaskDiagnostics() const {
  auto reduce_counts = [](CompositeMaskCounts local) {
    CompositeMaskCounts global = local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&local.valid, &global.valid, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local.relax, &global.relax, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local.covered, &global.covered, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local.interface, &global.interface, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    return global;
  };
  auto print_counts = [](const std::string &prefix, const CompositeMaskCounts &counts) {
    if (global_variable::my_rank == 0) {
      std::cout << prefix
                << " valid=" << counts.valid
                << " relax=" << counts.relax
                << " covered=" << counts.covered
                << " interface=" << counts.interface << std::endl;
    }
  };

  for (int level = 0; level < mglevels_->GetNumberOfLevels(); ++level) {
    CompositeMaskCounts counts = reduce_counts(mglevels_->CountCompositeMasks(level, true));
    std::ostringstream prefix;
    prefix << "CTS composite masks: entity=meshblock level=" << level
           << " transfer=" << (level == MeshBlockTransferLevel() ? 1 : 0)
           << " blocks=" << nbtotal_ << " region=active";
    print_counts(prefix.str(), counts);
  }

  CompositeMaskCounts root_local;
  if (global_variable::my_rank == 0) {
    root_local = mgroot_->CountCompositeMasks(nrootlevel_ - 1, true);
  }
  CompositeMaskCounts root_counts = reduce_counts(root_local);
  {
    std::ostringstream prefix;
    prefix << "CTS composite masks: entity=root level=" << (nrootlevel_ - 1)
           << " region=active";
    print_counts(prefix.str(), root_counts);
  }

  const int ngh = mgroot_->GetGhostCells();
  for (int level = 0; level < nreflevel_; ++level) {
    CompositeMaskCounts active_local;
    CompositeMaskCounts staging_local;
    if (global_variable::my_rank == 0) {
      for (int o = 0; o < noctets_[level]; ++o) {
        const MGOctet &oct = octets_[level][o];
        for (int k = 0; k < oct.nc; ++k) {
          for (int j = 0; j < oct.nc; ++j) {
            for (int i = 0; i < oct.nc; ++i) {
              const bool active = (i >= ngh && i <= ngh + 1 &&
                                   j >= ngh && j <= ngh + 1 &&
                                   k >= ngh && k <= ngh + 1);
              CompositeMaskCounts &counts = active ? active_local : staging_local;
              counts.valid += oct.Mask(COMP_VALID, k, j, i);
              counts.relax += oct.Mask(COMP_RELAX, k, j, i);
              counts.covered += oct.Mask(COMP_COVERED, k, j, i);
              counts.interface += oct.Mask(COMP_INTERFACE, k, j, i);
            }
          }
        }
      }
    }
    CompositeMaskCounts active_counts = reduce_counts(active_local);
    CompositeMaskCounts staging_counts = reduce_counts(staging_local);
    std::ostringstream active_prefix;
    active_prefix << "CTS composite masks: entity=octet level=" << level
                  << " octets=" << noctets_[level] << " region=active";
    print_counts(active_prefix.str(), active_counts);
    std::ostringstream staging_prefix;
    staging_prefix << "CTS composite masks: entity=octet level=" << level
                   << " octets=" << noctets_[level] << " region=staging";
    print_counts(staging_prefix.str(), staging_counts);
  }
}

Real IDCTSMultigridDriver::CalculateCompositeDefectNorm(MGNormType nrm, int n) {
  if (!composite_fas_) return CalculateDefectNorm(nrm, n);
  if (!composite_masks_ready_) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::CalculateCompositeDefectNorm"
              << std::endl
              << "Composite FAS residual requested before composite masks were built."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  IDCTSMultigrid *mg = static_cast<IDCTSMultigrid*>(mglevels_);
  const int level = mglevels_->GetCurrentLevel();
  const int ll = mglevels_->GetLevelShift();
  const int is = mglevels_->GetGhostCells();
  const int js = is;
  const int ks = is;
  const int ncells = mglevels_->GetLevelActiveCells(level);
  const int ie = is + ncells - 1;
  const int je = js + ncells - 1;
  const int ke = ks + ncells - 1;
  const int fine_fd = owner_->pmy_pack_->pz4c->opt.fd_stencil;
  const int fd = (level == mglevels_->GetNumberOfLevels() - 1)
                 ? fine_fd : mg_coarse_fd_stencil_;

  {
    auto mask = mglevels_->GetCompositeMaskLevel_h(level);
    auto u = mglevels_->GetCurrentData_h();
    if (mask.extent_int(0) != u.extent_int(0) ||
        mask.extent_int(2) != u.extent_int(2) ||
        mask.extent_int(3) != u.extent_int(3) ||
        mask.extent_int(4) != u.extent_int(4)) {
      std::cout << "### FATAL ERROR in IDCTSMultigridDriver::CalculateCompositeDefectNorm"
                << std::endl
                << "Composite mask extents do not match the current CTS MG level."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  mglevels_->CalculateDefectPack();
  CompositeResidualStats local;
  const bool include_covered = debug_composite_residual_;
  if (mglevels_->OnHost()) {
    auto u = mglevels_->GetCurrentData_h();
    auto src = mglevels_->GetCurrentSource_h();
    auto free = mglevels_->GetCoefficientLevel_h(level);
    auto mask = mglevels_->GetCompositeMaskLevel_h(level);
    switch (fd) {
      case 2:
        local = CompositeResidualStatsImpl<2>(mg, u, src, free, mask, ll, is, ie,
                                              js, je, ks, ke, fd, n, include_covered);
        break;
      case 3:
        local = CompositeResidualStatsImpl<3>(mg, u, src, free, mask, ll, is, ie,
                                              js, je, ks, ke, fd, n, include_covered);
        break;
      default:
        local = CompositeResidualStatsImpl<4>(mg, u, src, free, mask, ll, is, ie,
                                              js, je, ks, ke, fd, n, include_covered);
        break;
    }
  } else {
    auto u = mglevels_->GetCurrentData();
    auto src = mglevels_->GetCurrentSource();
    auto free = mglevels_->GetCurrentCoefficient();
    auto mask = mglevels_->GetCompositeMaskLevel(level);
    switch (fd) {
      case 2:
        local = CompositeResidualStatsImpl<2>(mg, u, src, free, mask, ll, is, ie,
                                              js, je, ks, ke, fd, n, include_covered);
        break;
      case 3:
        local = CompositeResidualStatsImpl<3>(mg, u, src, free, mask, ll, is, ie,
                                              js, je, ks, ke, fd, n, include_covered);
        break;
      default:
        local = CompositeResidualStatsImpl<4>(mg, u, src, free, mask, ll, is, ie,
                                              js, je, ks, ke, fd, n, include_covered);
        break;
    }
  }

  CompositeResidualStats global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local.valid_sum2, &global.valid_sum2, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.valid_sum_abs, &global.valid_sum_abs, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.valid_max, &global.valid_max, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local.valid_volume, &global.valid_volume, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.valid_count, &global.valid_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.interface_sum2, &global.interface_sum2, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.interface_volume, &global.interface_volume, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.interface_count, &global.interface_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.covered_sum2, &global.covered_sum2, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.covered_volume, &global.covered_volume, 1, MPI_ATHENA_REAL,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local.covered_count, &global.covered_count, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global.valid_count == 0 || !(global.valid_volume > 0.0)) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::CalculateCompositeDefectNorm"
              << std::endl
              << "Composite residual norm has zero valid cells on level " << level
              << "." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real accepted = 0.0;
  if (nrm == MGNormType::max) {
    accepted = global.valid_max;
  } else if (nrm == MGNormType::l1) {
    accepted = global.valid_sum_abs / global.valid_volume;
  } else {
    accepted = std::sqrt(global.valid_sum2 / global.valid_volume);
  }

  if (debug_composite_residual_ && global_variable::my_rank == 0) {
    const Real valid_l2 = std::sqrt(global.valid_sum2 / global.valid_volume);
    const Real interface_l2 = (global.interface_count > 0 && global.interface_volume > 0.0)
        ? std::sqrt(global.interface_sum2 / global.interface_volume) : 0.0;
    const Real covered_l2 = (global.covered_count > 0 && global.covered_volume > 0.0)
        ? std::sqrt(global.covered_sum2 / global.covered_volume) : 0.0;
    std::cout << "CTS composite residual: level=" << level
              << " var=" << CTSVarName(n)
              << " valid_count=" << global.valid_count
              << " valid_l2=" << valid_l2
              << " interface_count=" << global.interface_count
              << " interface_l2=" << interface_l2
              << " covered_count=" << global.covered_count
              << " covered_l2=" << covered_l2
              << " accepted_l2=" << valid_l2 << std::endl;
  }
  return accepted;
}

void IDCTSMultigridDriver::Solve(Driver *pdriver, int stage, Real dt) {
  solution_applied_ = false;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  PrepareForAMR();
  if (pmy_pack_->pmesh->adaptive && nreflevel_ > 0) {
      std::cout << "### FATAL ERROR in IDCTSMultigridDriver::Solve" << std::endl
                << "Native CTS currently supports unigrid and static mesh refinement. "
                << "Adaptive AMR would require rebuilding CTS free data, octet "
                << "coefficients, masks, and horizon-junk fields after refinement."
                << std::endl;
      if (allow_incomplete_amr_) {
        std::cout << "<id_solve>/allow_incomplete_amr is no longer allowed to "
                  << "bypass this guard." << std::endl;
      }
      std::exit(EXIT_FAILURE);
  }
  if (nreflevel_ > 0 && pmy_pack_->pz4c->opt.spatial_order > 4) {
      std::cout << "### FATAL ERROR in IDCTSMultigridDriver::Solve" << std::endl
                << "Sixth-order native CTS on SMR is not supported by the current "
                << "octet coarse/fine boundary interpolation. Use unigrid for "
                << "sixth-order CTS or set <z4c>/spatial_order=4 for SMR until "
                << "the octet bridge is widened beyond its 3-point coarse stencil."
                << std::endl;
      std::exit(EXIT_FAILURE);
  }
  owner_->BuildFreeData();

  mglevels_->LoadFinestData(owner_->u_cts, 0, indcs.ng);
  mglevels_->LoadSource(owner_->u_defect, 0, indcs.ng, 0.0);
  mglevels_->LoadCoefficients(owner_->u_free, indcs.ng);
  mglevels_->RestrictCoefficients();

  SetupMultigrid(dt, false);
  BuildCompositeMasks();
  if (composite_fas_ && debug_composite_tau_ && !composite_tau_deferred_note_printed_
      && global_variable::my_rank == 0) {
    std::cout << "CTS composite tau: root/octet composite tau deferred because "
              << "Commit 3 reported insufficient stencil support there." << std::endl;
    composite_tau_deferred_note_printed_ = true;
  }
  TransferCoefficientsFromBlocksToRoot();

  auto calculate_defects = [&](Real defects[ID_CTS_NVAR]) {
    Real sumsq = 0.0;
    for (int v = 0; v < ID_CTS_NVAR; ++v) {
      defects[v] = composite_fas_ ? CalculateCompositeDefectNorm(MGNormType::l2, v)
                                  : CalculateDefectNorm(MGNormType::l2, v);
      sumsq += defects[v]*defects[v];
    }
    return std::sqrt(sumsq);
  };

  Real initial_defects[ID_CTS_NVAR];
  Real mg_defect = calculate_defects(initial_defects);
  Real initial = mg_defect;
  owner_->RecordConstraintHistory(0, initial_defects);
  if (fshowdef_) {
    std::cout << "IDSolve initial CTS defect = " << initial_defects[ID_CTS_PSI]
              << " (total = " << initial << ")" << std::endl;
  }

  Real final_defects[ID_CTS_NVAR];
  for (int v = 0; v < ID_CTS_NVAR; ++v) final_defects[v] = initial_defects[v];
  if (full_multigrid_) {
    ResetSmootherStats();
    SolveFMG(pdriver);
    mg_defect = calculate_defects(final_defects);
    owner_->RecordConstraintHistory(eps_ >= 0.0 ? -1 : niter_, final_defects);
    if (fshowdef_) PrintSmootherStats(eps_ >= 0.0 ? -1 : niter_);
  } else if (eps_ >= 0.0) {
    if (fshowdef_) std::cout << "MG initial defect = " << mg_defect << std::endl;
    int n = 0;
    while (mg_defect > eps_) {
      ResetSmootherStats();
      SolveVCycle(pdriver, npresmooth_, npostsmooth_);
      Real olddef = mg_defect;
      mg_defect = calculate_defects(final_defects);
      owner_->RecordConstraintHistory(n + 1, final_defects);
      if (fshowdef_) {
        std::cout << "  MG iteration " << n << ": defect = " << mg_defect << std::endl;
        PrintSmootherStats(n);
      }
      if (mg_defect/olddef > 0.9) {
        if (eps_ == 0.0) break;
        if (fshowdef_) {
          std::cout << "### WARNING in IDCTSMultigridDriver::Solve" << std::endl
                    << "Slow convergence: defect ratio = " << mg_defect/olddef
                    << std::endl;
        }
      }
      ++n;
      if (n >= max_iter_) {
        std::cout << "### FATAL ERROR in IDCTSMultigridDriver::Solve" << std::endl
                  << "Failed to converge after " << n << " iterations (defect = "
                  << mg_defect << ", threshold = " << eps_ << ")" << std::endl;
        pdriver->nlim = pmy_mesh_->ncycle;
        break;
      }
    }
  } else {
    if (fshowdef_) std::cout << "MG initial defect = " << mg_defect << std::endl;
    Real best_defect = mg_defect;
    Real best_defects[ID_CTS_NVAR];
    for (int v = 0; v < ID_CTS_NVAR; ++v) best_defects[v] = initial_defects[v];
    if (keep_best_solution_) {
      mglevels_->RetrieveResult(owner_->u_cts, 0, indcs.ng);
    }
    for (int n = 0; n < niter_; ++n) {
      ResetSmootherStats();
      SolveVCycle(pdriver, npresmooth_, npostsmooth_);
      mg_defect = calculate_defects(final_defects);
      owner_->RecordConstraintHistory(n + 1, final_defects);
      if (fshowdef_) {
        std::cout << "  MG iteration " << n << ": defect = " << mg_defect << std::endl;
        PrintSmootherStats(n);
      }
      if (keep_best_solution_ && std::isfinite(mg_defect) && mg_defect < best_defect) {
        best_defect = mg_defect;
        for (int v = 0; v < ID_CTS_NVAR; ++v) best_defects[v] = final_defects[v];
        mglevels_->RetrieveResult(owner_->u_cts, 0, indcs.ng);
      }
      if (stop_on_defect_increase_ &&
          (!std::isfinite(mg_defect) ||
           mg_defect > best_defect*(1.0 + defect_increase_tol_))) {
        if (fshowdef_) {
          std::cout << "Stopping native CTS V-cycles after defect increase; "
                    << "best defect = " << best_defect << std::endl;
        }
        break;
      }
    }
    if (keep_best_solution_ && best_defect < mg_defect) {
      mg_defect = best_defect;
      for (int v = 0; v < ID_CTS_NVAR; ++v) final_defects[v] = best_defects[v];
      mglevels_->LoadFinestData(owner_->u_cts, 0, indcs.ng);
    }
  }
  Kokkos::fence();
  Real final = mg_defect;
  if (fshowdef_) {
    std::cout << "IDSolve final CTS defect = " << final_defects[ID_CTS_PSI]
              << " (total = " << final << ")" << std::endl;
  }
  if (reject_worse_ && (!std::isfinite(final) || final > initial)) {
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in IDCTSMultigridDriver::Solve" << std::endl
                << "Rejecting native CTS update because the defect did not decrease."
                << std::endl;
    }
    return;
  }

  mglevels_->RetrieveResult(owner_->u_cts, 0, indcs.ng);
  owner_->ApplySolution();
  solution_applied_ = true;
}

void IDCTSMultigridDriver::DiagnosticRestrictOctets(int lev) {
  if (!composite_fas_ || !debug_composite_restriction_) return;
  if (lev < 0 || lev >= nreflevel_) return;
  if (!composite_masks_ready_) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::DiagnosticRestrictOctets"
              << std::endl
              << "Composite restriction diagnostics requested before masks were built."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!composite_restriction_self_check_done_) {
    RunCompositeRestrictionSelfCheck();
    composite_restriction_self_check_done_ = true;
    if (global_variable::my_rank == 0) {
      std::cout << "CTS composite restriction self-check: passed" << std::endl;
    }
  }

  CompositeRestrictionStats local_u;
  CompositeRestrictionStats local_def;
  const bool half_weight = (composite_restriction_ == ID_CTS_RESTRICT_HALF_WEIGHT);
  const int ngh = mgroot_->GetGhostCells();
  if (global_variable::my_rank == 0 && lev >= 0 && lev < nreflevel_) {
    for (int o = 0; o < noctets_[lev]; ++o) {
      MGOctet &oct = octets_[lev][o];
      CalculateDefectOctet(oct, lev + 1);
      CompositeRestrictionStats u_stats =
          DiagnoseHalfWeightRestrictionOctet(oct, nvar_, ngh, 0, half_weight);
      CompositeRestrictionStats def_stats =
          DiagnoseHalfWeightRestrictionOctet(oct, nvar_, ngh, 1, half_weight);
      AddRestrictionStats(local_u, u_stats);
      AddRestrictionStats(local_def, def_stats);
    }
  }
  CompositeRestrictionStats global_u = ReduceCompositeRestrictionStats(local_u);
  CompositeRestrictionStats global_def = ReduceCompositeRestrictionStats(local_def);
  PrintCompositeRestrictionStats("octet", lev, "u", composite_restriction_, global_u);
  PrintCompositeRestrictionStats("octet", lev, "def", composite_restriction_, global_def);
}

void IDCTSMultigridDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx/static_cast<Real>(1 << rlev);
  Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
  OctetArray u{oct.u, oct.nc};
  std::vector<Real> frozen_buf(static_cast<std::size_t>(oct.size()));
  std::copy(oct.u, oct.u + oct.size(), frozen_buf.begin());
  OctetArray u_frozen{frozen_buf.data(), oct.nc};
  OctetArray src{oct.src, oct.nc};
  OctetArray free{oct.coeff, oct.nc};
  SmootherCellStats stats;
  stats.Clear();
  color ^= coffset_;
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      int c = (color + k + j) & 1;
      for (int i = ngh + c; i <= ngh+1; i += 2) {
        if (composite_fas_ && oct.Mask(COMP_RELAX, k, j, i) == 0) continue;
        if (free(0, ID_FREE_MASK, k, j, i) < 0.5) continue;
        SmootherCellStats cell;
        switch (octet_fd_stencil_) {
          case 2:
            if (smoother_type_ == 1) {
              cell = ApplyNewtonGSCTSUpdate<2>(u_frozen, u, src, free,
                                               idx, octet_fd_stencil_,
                                               0, k, j, i, omega_, ngs_iterations_,
                                               ngs_jacobian_eps_, ngs_max_update_,
                                               smoother_max_update_fraction_,
                                               ngs_line_search_steps_,
                                               ngs_line_search_min_);
            } else {
              cell = ApplyDiagonalCTSUpdate<2>(u_frozen, u, src, free,
                                               idx, octet_fd_stencil_,
                                               0, k, j, i, omega_, ngs_max_update_,
                                               smoother_max_update_fraction_);
            }
            break;
          case 3:
            if (smoother_type_ == 1) {
              cell = ApplyNewtonGSCTSUpdate<3>(u_frozen, u, src, free,
                                               idx, octet_fd_stencil_,
                                               0, k, j, i, omega_, ngs_iterations_,
                                               ngs_jacobian_eps_, ngs_max_update_,
                                               smoother_max_update_fraction_,
                                               ngs_line_search_steps_,
                                               ngs_line_search_min_);
            } else {
              cell = ApplyDiagonalCTSUpdate<3>(u_frozen, u, src, free,
                                               idx, octet_fd_stencil_,
                                               0, k, j, i, omega_, ngs_max_update_,
                                               smoother_max_update_fraction_);
            }
            break;
          default:
            if (smoother_type_ == 1) {
              cell = ApplyNewtonGSCTSUpdate<4>(u_frozen, u, src, free,
                                               idx, octet_fd_stencil_,
                                               0, k, j, i, omega_, ngs_iterations_,
                                               ngs_jacobian_eps_, ngs_max_update_,
                                               smoother_max_update_fraction_,
                                               ngs_line_search_steps_,
                                               ngs_line_search_min_);
            } else {
              cell = ApplyDiagonalCTSUpdate<4>(u_frozen, u, src, free,
                                               idx, octet_fd_stencil_,
                                               0, k, j, i, omega_, ngs_max_update_,
                                               smoother_max_update_fraction_);
            }
            break;
        }
        stats.Add(cell);
      }
    }
  }
  AccumulateSmootherStats(stats.values);
}

void IDCTSMultigridDriver::CalculateDefectOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx/static_cast<Real>(1 << rlev);
  Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
  OctetArray u{oct.u, oct.nc};
  OctetArray src{oct.src, oct.nc};
  OctetArray free{oct.coeff, oct.nc};
  OctetArray def{oct.def, oct.nc};
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        if (composite_fas_ && oct.Mask(COMP_VALID, k, j, i) == 0) {
          for (int v = 0; v < ID_CTS_NVAR; ++v) {
            def(0, v, k, j, i) = 0.0;
          }
          continue;
        }
        Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
        switch (octet_fd_stencil_) {
          case 2:
            CTSOperator<2>(u, free, idx, octet_fd_stencil_, 0, k, j, i, op, diag);
            break;
          case 3:
            CTSOperator<3>(u, free, idx, octet_fd_stencil_, 0, k, j, i, op, diag);
            break;
          default:
            CTSOperator<4>(u, free, idx, octet_fd_stencil_, 0, k, j, i, op, diag);
            break;
        }
        for (int v = 0; v < ID_CTS_NVAR; ++v) {
          def(0, v, k, j, i) = (free(0, ID_FREE_MASK, k, j, i) >= 0.5)
                               ? src(0, v, k, j, i) - op[v] : 0.0;
        }
      }
    }
  }
}

void IDCTSMultigridDriver::CalculateFASRHSOctet(MGOctet &oct, int rlev) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx/static_cast<Real>(1 << rlev);
  Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
  OctetArray u{oct.u, oct.nc};
  OctetArray src{oct.src, oct.nc};
  OctetArray free{oct.coeff, oct.nc};
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      for (int i = ngh; i <= ngh+1; ++i) {
        Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
        switch (octet_fd_stencil_) {
          case 2:
            CTSOperator<2>(u, free, idx, octet_fd_stencil_, 0, k, j, i, op, diag);
            break;
          case 3:
            CTSOperator<3>(u, free, idx, octet_fd_stencil_, 0, k, j, i, op, diag);
            break;
          default:
            CTSOperator<4>(u, free, idx, octet_fd_stencil_, 0, k, j, i, op, diag);
            break;
        }
        for (int v = 0; v < ID_CTS_NVAR; ++v) {
          src(0, v, k, j, i) += op[v];
        }
      }
    }
  }
}

IDConformalThinSandwich::IDConformalThinSandwich(MeshBlockPack *pmbp, ParameterInput *pin)
    : pmy_pack_(pmbp), pmgd_(nullptr), enabled_(true), solved_(false),
      history_file_(nullptr) {
  enabled_ = pin->GetOrAddBoolean("id_solve", "enable", true);
  solve_once_ = pin->GetOrAddBoolean("id_solve", "solve_once", true);
  run_on_restart_ = pin->GetOrAddBoolean("id_solve", "run_on_restart", false);
  stop_after_solve_ = pin->GetOrAddBoolean("id_solve", "stop_after_solve", false);
  skip_initial_output_ = pin->GetOrAddBoolean("id_solve", "skip_initial_output",
                                              stop_after_solve_);
  full_multigrid_ = pin->GetOrAddBoolean("id_solve", "full_multigrid", false);
  fill_horizon_junk_ = pin->GetOrAddBoolean("id_solve", "fill_horizon_junk", false);
  mask_horizon_defect_ = pin->GetOrAddBoolean("id_solve", "mask_horizon_defect", false);
  dump_constraint_diagnostics_ =
      pin->GetOrAddBoolean("id_solve", "dump_constraint_diagnostics", false);
  history_every_ = pin->GetOrAddInteger("id_solve", "history_every", 1);
  history_name_ = pin->GetString("job", "basename") + ".id_solve.hst";
  horizon_radius_ = pin->GetOrAddReal("id_solve", "horizon_fill_radius", -1.0);
  horizon_mask_radius_ = pin->GetOrAddReal("id_solve", "horizon_mask_radius",
                                           horizon_radius_);
  horizon_center_[0] = pin->GetOrAddReal("id_solve", "horizon_center_x1", 0.0);
  horizon_center_[1] = pin->GetOrAddReal("id_solve", "horizon_center_x2", 0.0);
  horizon_center_[2] = pin->GetOrAddReal("id_solve", "horizon_center_x3", 0.0);
  diagnostic_slice_z_ =
      pin->GetOrAddReal("id_solve", "diagnostic_slice_z", horizon_center_[2]);

  int nmb = std::max(pmbp->nmb_thispack, pmbp->pmesh->nmb_maxperrank);
  auto &indcs = pmbp->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;
  int ncells3 = indcs.nx3 + 2*indcs.ng;
  Kokkos::realloc(u_cts, nmb, ID_CTS_NVAR, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_free, nmb, ID_FREE_NVAR, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_defect, nmb, ID_CTS_NVAR, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(u_defect, 0.0);

  if (enabled_) pmgd_ = new IDCTSMultigridDriver(this, pmbp, pin);
}

IDConformalThinSandwich::~IDConformalThinSandwich() {
  if (history_file_ != nullptr) std::fclose(history_file_);
  delete pmgd_;
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
  WriteConstraintDiagnostics("before");
  pmgd_->Solve(pdriver, stage, 0.0);
  if (pmgd_->SolutionApplied()) {
    RefreshZ4cBoundariesAfterSolve(pdriver);
    RecomputeConstraintsAfterSolve();
  }
  WriteConstraintDiagnostics("after");
  solved_ = true;
  return TaskStatus::complete;
}

void IDConformalThinSandwich::RefreshZ4cBoundariesAfterSolve(Driver *pdriver) {
  Z4c *pz4c = pmy_pack_->pz4c;
  if (pz4c == nullptr) return;

  // Z4c_Recv has already posted receives for the upcoming RK boundary exchange.
  // Use that receive once to refresh the just-solved initial data, then re-post it.
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
  switch (fd) {
    case 2:
      pz4c->ADMConstraints<2>(pmy_pack_);
      break;
    case 3:
      pz4c->ADMConstraints<3>(pmy_pack_);
      break;
    default:
      pz4c->ADMConstraints<4>(pmy_pack_);
      break;
  }
}

void IDConformalThinSandwich::WriteConstraintDiagnostics(const char *stage) {
  if (!dump_constraint_diagnostics_) return;

  MeshBlockPack *pmbp = pmy_pack_;
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &gid = pmbp->pmb->mb_gid;
  auto con_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pz4c->u_con);
  size.sync<HostMemSpace>();
  gid.sync<HostMemSpace>();

  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;

  Real local[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (int m = 0; m < nmb; ++m) {
    const Real vol = size.h_view(m).dx1*size.h_view(m).dx2*size.h_view(m).dx3;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          local[0] += vol*std::max(con_h(m, Z4c::I_CON_C, k, j, i), 0.0);
          local[1] += vol*SQR(con_h(m, Z4c::I_CON_H, k, j, i));
          local[2] += vol*std::max(con_h(m, Z4c::I_CON_M, k, j, i), 0.0);
          local[3] += vol*std::max(con_h(m, Z4c::I_CON_Z, k, j, i), 0.0);
          local[4] += vol;
        }
      }
    }
  }

  Real global[5];
  for (int n = 0; n < 5; ++n) global[n] = local[n];
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, global, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (global_variable::my_rank == 0) {
    const std::string norm_name = history_name_ + ".constraints.tsv";
    const bool write_header = (std::string(stage) == "before");
    std::ofstream norm_file(norm_name, write_header ? std::ios::out : std::ios::app);
    if (write_header) {
      norm_file << "# stage\tC_l2\tH_l2\tM_l2\tZ_l2\tvolume\n";
    }
    const Real inv_vol = (global[4] > 0.0) ? 1.0/global[4] : 0.0;
    norm_file << stage << '\t'
              << std::setprecision(16) << std::sqrt(global[0]*inv_vol) << '\t'
              << std::sqrt(global[1]*inv_vol) << '\t'
              << std::sqrt(global[2]*inv_vol) << '\t'
              << std::sqrt(global[3]*inv_vol) << '\t'
              << global[4] << '\n';
  }

  std::ostringstream rank;
  rank << std::setw(6) << std::setfill('0') << global_variable::my_rank;
  const std::string slice_name =
      history_name_ + ".constraints." + stage + ".rank" + rank.str() + ".tsv";
  std::ofstream slice_file(slice_name);
  slice_file << "# gid\tx\ty\tz\tsqrt_C\tabs_H\tsqrt_M\tsqrt_Z\n";
  for (int m = 0; m < nmb; ++m) {
    const Real half_dz = 0.5*size.h_view(m).dx3;
    for (int k = ks; k <= ke; ++k) {
      const Real z = CellCenterX(k - ks, indcs.nx3, size.h_view(m).x3min,
                                 size.h_view(m).x3max);
      if (std::abs(z - diagnostic_slice_z_) > half_dz) continue;
      for (int j = js; j <= je; ++j) {
        const Real y = CellCenterX(j - js, indcs.nx2, size.h_view(m).x2min,
                                   size.h_view(m).x2max);
        for (int i = is; i <= ie; ++i) {
          const Real x = CellCenterX(i - is, indcs.nx1, size.h_view(m).x1min,
                                     size.h_view(m).x1max);
          slice_file << gid.h_view(m) << '\t'
                     << std::setprecision(16) << x << '\t' << y << '\t' << z << '\t'
                     << std::sqrt(std::max(con_h(m, Z4c::I_CON_C, k, j, i), 0.0)) << '\t'
                     << std::abs(con_h(m, Z4c::I_CON_H, k, j, i)) << '\t'
                     << std::sqrt(std::max(con_h(m, Z4c::I_CON_M, k, j, i), 0.0)) << '\t'
                     << std::sqrt(std::max(con_h(m, Z4c::I_CON_Z, k, j, i), 0.0)) << '\n';
        }
      }
    }
  }
}

void IDConformalThinSandwich::BuildFreeData() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  int isg = indcs.is - indcs.ng, ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng, jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng, keg = indcs.ke + indcs.ng;
  int nmb = pmy_pack_->nmb_thispack;
  auto &admvars = pmy_pack_->padm->adm;
  auto free = u_free;
  auto cts = u_cts;
  bool has_tmunu = (pmy_pack_->ptmunu != nullptr);
  Tmunu::Tmunu_vars tmunu;
  if (has_tmunu) tmunu = pmy_pack_->ptmunu->tmunu;
  bool fill_junk = fill_horizon_junk_;
  bool mask_horizon = mask_horizon_defect_;
  Real rfill = horizon_radius_;
  Real rmask = horizon_mask_radius_;
  Real cx = horizon_center_[0], cy = horizon_center_[1], cz = horizon_center_[2];

  par_for("IDCTS::BuildFreeData", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b)
        free(m, ID_FREE_GXX + SymIdx(a,b), k,j,i) = admvars.g_dd(m,a,b,k,j,i);

    Real gt_det = adm::SpatialDet(free(m,ID_FREE_GXX,k,j,i), free(m,ID_FREE_GXY,k,j,i),
                                  free(m,ID_FREE_GXZ,k,j,i), free(m,ID_FREE_GYY,k,j,i),
                                  free(m,ID_FREE_GYZ,k,j,i), free(m,ID_FREE_GZZ,k,j,i));
    Real gu[3][3];
    adm::SpatialInv(1.0/gt_det,
                    free(m,ID_FREE_GXX,k,j,i), free(m,ID_FREE_GXY,k,j,i),
                    free(m,ID_FREE_GXZ,k,j,i), free(m,ID_FREE_GYY,k,j,i),
                    free(m,ID_FREE_GYZ,k,j,i), free(m,ID_FREE_GZZ,k,j,i),
                    &gu[0][0], &gu[0][1], &gu[0][2],
                    &gu[1][1], &gu[1][2], &gu[2][2]);
    gu[1][0]=gu[0][1]; gu[2][0]=gu[0][2]; gu[2][1]=gu[1][2];

    Real K = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        K += gu[a][b]*admvars.vK_dd(m,a,b,k,j,i);
    free(m, ID_FREE_K, k,j,i) = K;
    free(m, ID_FREE_ALPHA, k,j,i) = admvars.alpha(m,k,j,i);
    free(m, ID_FREE_SOURCE, k,j,i) = 0.0;
    free(m, ID_FREE_MASK, k,j,i) = 1.0;
    free(m, ID_FREE_BASE_PSI, k,j,i) = 1.0;
    cts(m, ID_CTS_PSI, k,j,i) = 0.0;
    for (int a = 0; a < 6; ++a) free(m, ID_FREE_GDOTXX+a, k,j,i) = 0.0;
    for (int a = 0; a < 3; ++a) {
      free(m, ID_FREE_BASE_BETAX+a, k,j,i) = admvars.beta_u(m,a,k,j,i);
      cts(m, ID_CTS_BETAX+a, k,j,i) = 0.0;
    }

    Real E = 0.0;
    Real p[3] = {0.0, 0.0, 0.0};
    if (has_tmunu) {
      E = tmunu.E(m,k,j,i);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          p[a] += gu[a][b]*tmunu.S_d(m,b,k,j,i);
    }
    Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    Real r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz);
    if (mask_horizon && rmask > 0.0 && r2 < rmask*rmask) {
      free(m, ID_FREE_MASK, k,j,i) = 0.0;
    }
    if (fill_junk && rfill > 0.0 && r2 < rfill*rfill) {
      E = 0.0;
      p[0] = p[1] = p[2] = 0.0;
    }
    free(m, ID_FREE_E, k,j,i) = E;
    for (int a = 0; a < 3; ++a) free(m, ID_FREE_PX+a, k,j,i) = p[a];
  });

  int fd = pmy_pack_->pz4c->opt.fd_stencil;
  switch (fd) {
    case 2:
      BuildGammaDotAndDK<2>();
      FillHorizonJunk<2>();
      break;
    case 3:
      BuildGammaDotAndDK<3>();
      FillHorizonJunk<3>();
      break;
    default:
      BuildGammaDotAndDK<4>();
      FillHorizonJunk<4>();
      break;
  }
}

template <int NGHOST>
void BuildGammaDotAndDKImpl(MeshBlockPack *pmbp, DvceArray5D<Real> u_cts,
                            DvceArray5D<Real> u_free) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ahat_halo = NGHOST - 1;
  int total_reach = (NGHOST - 1) + ahat_halo;
  if (total_reach > indcs.ng) {
    std::cout << "### FATAL ERROR in IDCTS::BuildGammaDotAndDK" << std::endl
              << "Need " << total_reach << " ghost cells for the selected "
              << "gamma-dot/DK halo, but <mesh>/nghost=" << indcs.ng << "."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int isg = is - ahat_halo, ieg = ie + ahat_halo;
  int jsg = js - ahat_halo, jeg = je + ahat_halo;
  int ksg = ks - ahat_halo, keg = ke + ahat_halo;
  int nmb = pmbp->nmb_thispack;
  auto &admvars = pmbp->padm->adm;
  par_for("IDCTS::BuildGammaDotAndDK", DevExeSpace(), 0, nmb-1,
          ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2, 1.0/size.d_view(m).dx3};
    Real gu[3][3];
    Real gamma[3][3][3];
    MetricInverse(u_free, m, k, j, i, gu);
    Christoffel<NGHOST>(u_free, idx, m, k, j, i, gu, gamma);

    for (int a = 0; a < 3; ++a) {
      Real dK = 0.0;
      for (int b = 0; b < 3; ++b) {
        dK += gu[a][b]*Dx<NGHOST>(b, idx, u_free, m, ID_FREE_K, k, j, i);
      }
      u_free(m, ID_FREE_DKX+a, k,j,i) = dK;
    }

    Real K = u_free(m, ID_FREE_K, k,j,i);
    Real Ahat_uu[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        Ahat_uu[a][b] = 0.0;
        for (int c = 0; c < 3; ++c)
          for (int d = 0; d < 3; ++d) {
            Real A_cov = (admvars.vK_dd(m,c,d,k,j,i)
                          - (1.0/3.0)*admvars.g_dd(m,c,d,k,j,i)*K);
            Ahat_uu[a][b] += gu[a][c]*gu[b][d]*A_cov;
          }
      }
    }

    Real dbeta[3][3];
    Real div_beta = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        dbeta[a][b] = DxTotal<NGHOST>(a, idx, u_cts, u_free, m,
                                      ID_CTS_BETAX+b, k, j, i);
      }
      div_beta += dbeta[a][a];
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        Real lbeta = -(2.0/3.0)*gu[a][b]*div_beta;
        for (int c = 0; c < 3; ++c) {
          lbeta += gu[a][c]*dbeta[c][b] + gu[b][c]*dbeta[c][a];
          for (int d = 0; d < 3; ++d) {
            Real beta_d = TotalU(u_cts, u_free, m, ID_CTS_BETAX+d, k, j, i);
            lbeta += (gu[a][c]*gamma[b][c][d] + gu[b][c]*gamma[a][c][d]
                      - (2.0/3.0)*gu[a][b]*gamma[c][c][d])*beta_d;
          }
        }
        Real alpha = std::max(u_free(m, ID_FREE_ALPHA, k,j,i), static_cast<Real>(1.0e-12));
        u_free(m, ID_FREE_GDOTXX + SymIdx(a,b), k,j,i) = 2.0*alpha*Ahat_uu[a][b] - lbeta;
      }
    }
  });
}

template <int NGHOST>
void IDConformalThinSandwich::BuildGammaDotAndDK() {
  BuildGammaDotAndDKImpl<NGHOST>(pmy_pack_, u_cts, u_free);
}

template <int NGHOST>
void FillHorizonJunkImpl(MeshBlockPack *pmbp, DvceArray5D<Real> u_cts,
                         DvceArray5D<Real> u_free, Real rfill,
                         Real cx, Real cy, Real cz) {
  if (rfill <= 0.0) return;
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int fd = pmbp->pz4c->opt.fd_stencil;
  const Real rfill2 = rfill*rfill;
  par_for("IDCTS::FillHorizonJunk", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                         size.d_view(m).x1max);
    Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                         size.d_view(m).x2max);
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                         size.d_view(m).x3max);
    Real r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz);
    if (r2 >= rfill2) return;

    u_free(m, ID_FREE_E, k,j,i) = 0.0;
    for (int a = 0; a < 3; ++a) u_free(m, ID_FREE_PX+a, k,j,i) = 0.0;

    Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
    CTSOperator<NGHOST>(u_cts, u_free, idx, fd, m, k, j, i, op, diag);

    Real psi = std::max(TotalU(u_cts, u_free, m, ID_CTS_PSI, k,j,i),
                        static_cast<Real>(1.0e-8));
    constexpr Real inv_two_pi = 1.0/(2.0*3.14159265358979323846);
    constexpr Real inv_sixteen_pi = 1.0/(16.0*3.14159265358979323846);
    u_free(m, ID_FREE_E, k,j,i) = -op[ID_CTS_PSI]*inv_two_pi/std::pow(psi, 5.0);
    for (int a = 0; a < 3; ++a) {
      u_free(m, ID_FREE_PX+a, k,j,i) =
          op[ID_CTS_BETAX+a]*inv_sixteen_pi/std::pow(psi, 6.0);
    }
  });
}

template <int NGHOST>
void IDConformalThinSandwich::FillHorizonJunk() {
  if (!fill_horizon_junk_) return;
  FillHorizonJunkImpl<NGHOST>(pmy_pack_, u_cts, u_free, horizon_radius_,
                              horizon_center_[0], horizon_center_[1], horizon_center_[2]);
}

void IDConformalThinSandwich::ApplySolution() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  auto &size = pmy_pack_->pmb->mb_size;
  auto cts = u_cts;
  auto free = u_free;
  auto &admvars = pmy_pack_->padm->adm;
  int fd = pmy_pack_->pz4c->opt.fd_stencil;
  auto apply_kernel = [&](auto stencil_tag) {
    constexpr int NGHOST = decltype(stencil_tag)::value;
    par_for("IDCTS::ApplySolution", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                     1.0/size.d_view(m).dx3};
      Real psi = std::max(TotalU(cts, free, m, ID_CTS_PSI, k,j,i),
                          static_cast<Real>(1.0e-8));
      Real psi2 = psi*psi;
      Real psi4 = psi2*psi2;
      Real free_det = adm::SpatialDet(free(m,ID_FREE_GXX,k,j,i),
                                      free(m,ID_FREE_GXY,k,j,i),
                                      free(m,ID_FREE_GXZ,k,j,i),
                                      free(m,ID_FREE_GYY,k,j,i),
                                      free(m,ID_FREE_GYZ,k,j,i),
                                      free(m,ID_FREE_GZZ,k,j,i));
      for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b)
          admvars.g_dd(m,a,b,k,j,i) = psi4*FreeSym(free, ID_FREE_GXX, m, a, b, k,j,i);
      admvars.psi4(m,k,j,i) = psi4*std::pow(free_det, 1.0/3.0);
      admvars.alpha(m,k,j,i) = free(m, ID_FREE_ALPHA, k,j,i);
      for (int a = 0; a < 3; ++a) {
        admvars.beta_u(m,a,k,j,i) = TotalU(cts, free, m, ID_CTS_BETAX+a,k,j,i);
      }

      Real ahat_uu[3][3];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          ahat_uu[a][b] = AHatUU<NGHOST>(cts, free, idx, m, k, j, i, a, b);

      Real ahat_dd[3][3];
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          ahat_dd[a][b] = 0.0;
          for (int c = 0; c < 3; ++c)
            for (int d = 0; d < 3; ++d)
              ahat_dd[a][b] += FreeSym(free, ID_FREE_GXX, m, a, c, k,j,i)
                               * FreeSym(free, ID_FREE_GXX, m, b, d, k,j,i)
                               * ahat_uu[c][d];
        }
      }
      Real invpsi2 = 1.0/psi2;
      Real K = free(m, ID_FREE_K, k,j,i);
      for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b)
          admvars.vK_dd(m,a,b,k,j,i) = invpsi2*ahat_dd[a][b]
            + (1.0/3.0)*admvars.g_dd(m,a,b,k,j,i)*K;
    });
  };
  if (fd == 2) apply_kernel(std::integral_constant<int,2>{});
  else if (fd == 3) apply_kernel(std::integral_constant<int,3>{});
  else apply_kernel(std::integral_constant<int,4>{});
  switch (fd) {
    case 2:
      pmy_pack_->pz4c->ADMToZ4c<2>(pmy_pack_, nullptr);
      break;
    case 3:
      pmy_pack_->pz4c->ADMToZ4c<3>(pmy_pack_, nullptr);
      break;
    default:
      pmy_pack_->pz4c->ADMToZ4c<4>(pmy_pack_, nullptr);
      break;
  }
  pmy_pack_->pz4c->Z4cToADM(pmy_pack_);
  switch (fd) {
    case 2:
      pmy_pack_->pz4c->ADMConstraints<2>(pmy_pack_);
      break;
    case 3:
      pmy_pack_->pz4c->ADMConstraints<3>(pmy_pack_);
      break;
    default:
      pmy_pack_->pz4c->ADMConstraints<4>(pmy_pack_);
      break;
  }
}

void IDConformalThinSandwich::RecordConstraintHistory(int iter,
                                                      const Real defects[ID_CTS_NVAR]) {
  if (history_every_ <= 0) return;
  if (history_file_ == nullptr) {
    history_file_ = std::fopen(history_name_.c_str(), "w");
    if (history_file_ != nullptr) {
      std::fprintf(history_file_, "# AthenaK native id_solve CTS history\n");
      std::fprintf(history_file_,
                   "# [1]=iter [2]=psi_l2 [3]=betax_l2 [4]=betay_l2 [5]=betaz_l2 "
                   "[6]=total_l2\n");
    }
  }
  if (history_file_ != nullptr) {
    Real total = 0.0;
    for (int v = 0; v < ID_CTS_NVAR; ++v) total += defects[v]*defects[v];
    total = std::sqrt(total);
    std::fprintf(history_file_, "%d", iter);
    for (int v = 0; v < ID_CTS_NVAR; ++v) {
      std::fprintf(history_file_, " %.16e", static_cast<double>(defects[v]));
    }
    std::fprintf(history_file_, " %.16e\n", static_cast<double>(total));
    std::fflush(history_file_);
  }
}

} // namespace z4c
