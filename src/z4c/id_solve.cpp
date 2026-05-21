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

KOKKOS_INLINE_FUNCTION
int DivAHatStencil(int fd_stencil, int nghost_available) {
  int inner_radius = fd_stencil - 1;
  int outer_radius = nghost_available - inner_radius;
  if (outer_radius >= 3 && fd_stencil >= 4) return 4;
  if (outer_radius >= 2 && fd_stencil >= 3) return 3;
  return 2;
}

KOKKOS_INLINE_FUNCTION
int DivAHatRequiredGhosts(int fd_stencil, int div_ahat_stencil) {
  // D_j Ahat^{ij} is currently formed by differencing Ahat, while Ahat itself
  // contains derivatives of beta and the conformal metric.  The halo reach is
  // therefore the sum of the inner Ahat radius and the outer divergence radius.
  return (fd_stencil - 1) + (div_ahat_stencil - 1);
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
              int div_ahat_stencil, int m, int k, int j, int i, int a, int b) {
  int const shiftk = dir == 2;
  int const shiftj = dir == 1;
  int const shifti = dir == 0;
  auto val = [&](int s) {
    return AHatUU<NGHOST>(u, free, idx, m, k + s*shiftk, j + s*shiftj,
                          i + s*shifti, a, b);
  };
  Real out = -0.5*val(-1) + 0.5*val(1);
  if (div_ahat_stencil >= 3) {
    out = (1.0/12.0)*val(-2) - (2.0/3.0)*val(-1)
        + (2.0/3.0)*val(1) - (1.0/12.0)*val(2);
  }
  if (div_ahat_stencil >= 4) {
    out = (-1.0/60.0)*val(-3) + (3.0/20.0)*val(-2) - (3.0/4.0)*val(-1)
        + (3.0/4.0)*val(1) - (3.0/20.0)*val(2) + (1.0/60.0)*val(3);
  }
  return out*idx[dir];
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
void CTSOperator(const UView &u, const FView &free, const Real idx[], int fd_stencil,
                 int div_ahat_stencil, int m, int k, int j, int i,
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

KOKKOS_INLINE_FUNCTION
bool SolveLinear4x4(Real a[ID_CTS_NVAR][ID_CTS_NVAR], Real b[ID_CTS_NVAR],
                    Real x[ID_CTS_NVAR]) {
  Real aug[ID_CTS_NVAR][ID_CTS_NVAR + 1];
  for (int r = 0; r < ID_CTS_NVAR; ++r) {
    for (int c = 0; c < ID_CTS_NVAR; ++c) aug[r][c] = a[r][c];
    aug[r][ID_CTS_NVAR] = b[r];
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
    if (!(pivot_abs > 1.0e-30) || !IsFiniteForDevice(pivot_abs)) return false;
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
    x[r] = aug[r][ID_CTS_NVAR];
    if (!IsFiniteForDevice(x[r])) return false;
  }
  return true;
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
void ApplyDiagonalCTSUpdate(UView &u, const FView &src, const FView &free,
                            const Real idx[], int fd_stencil, int div_ahat_stencil,
                            int m, int k, int j, int i, Real omega) {
  Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
  CTSOperator<NGHOST>(u, free, idx, fd_stencil, div_ahat_stencil,
                      m, k, j, i, op, diag);
  for (int v = 0; v < ID_CTS_NVAR; ++v) {
    Real d = (std::abs(diag[v]) > 1.0e-30) ? diag[v] :
             ((diag[v] < 0.0) ? -1.0e-30 : 1.0e-30);
    u(m, v, k, j, i) -= omega*(op[v] - src(m, v, k, j, i))/d;
  }
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
void ApplyNewtonGSCTSUpdate(UView &u, const FView &src, const FView &free,
                            const Real idx[], int fd_stencil, int div_ahat_stencil,
                            int m, int k, int j, int i, Real omega,
                            int niter, Real jac_eps, Real max_update) {
  const int local_iter = niter > 0 ? niter : 1;
  for (int it = 0; it < local_iter; ++it) {
    Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
    CTSOperator<NGHOST>(u, free, idx, fd_stencil, div_ahat_stencil,
                        m, k, j, i, op, diag);

    Real residual[ID_CTS_NVAR];
    for (int r = 0; r < ID_CTS_NVAR; ++r) {
      residual[r] = op[r] - src(m, r, k, j, i);
    }

    Real jac[ID_CTS_NVAR][ID_CTS_NVAR];
    for (int c = 0; c < ID_CTS_NVAR; ++c) {
      Real uval = u(m, c, k, j, i);
      Real eps = jac_eps*std::max(static_cast<Real>(1.0), std::abs(uval));
      if (!(eps > 0.0) || !IsFiniteForDevice(eps)) eps = 1.0e-7;
      LocalPerturbedUView<UView> up{u, m, c, k, j, i, eps};
      Real op_pert[ID_CTS_NVAR], diag_pert[ID_CTS_NVAR];
      CTSOperator<NGHOST>(up, free, idx, fd_stencil, div_ahat_stencil,
                          m, k, j, i, op_pert, diag_pert);
      for (int r = 0; r < ID_CTS_NVAR; ++r) {
        jac[r][c] = (op_pert[r] - op[r])/eps;
      }
    }

    Real rhs[ID_CTS_NVAR];
    Real delta[ID_CTS_NVAR];
    for (int r = 0; r < ID_CTS_NVAR; ++r) rhs[r] = -residual[r];
    bool solved = SolveLinear4x4(jac, rhs, delta);
    if (!solved) {
      for (int v = 0; v < ID_CTS_NVAR; ++v) {
        Real d = (std::abs(diag[v]) > 1.0e-30) ? diag[v] :
                 ((diag[v] < 0.0) ? -1.0e-30 : 1.0e-30);
        delta[v] = -residual[v]/d;
      }
    }

    for (int v = 0; v < ID_CTS_NVAR; ++v) {
      Real update = omega*delta[v];
      if (max_update > 0.0 && std::abs(update) > max_update) {
        update = (update > 0.0 ? max_update : -max_update);
      }
      u(m, v, k, j, i) += update;
    }

    Real psi = TotalU(u, free, m, ID_CTS_PSI, k, j, i);
    if (psi < 1.0e-8) {
      u(m, ID_CTS_PSI, k, j, i) = 1.0e-8 - free(m, ID_FREE_BASE_PSI, k, j, i);
    }
  }
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
void CTSOperator(const UView &u, const FView &free, const Real idx[], int fd_stencil,
                 int div_ahat_stencil, int m, int k, int j, int i,
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
      div_a += DxAHatUU<NGHOST>(b, idx, u, free, div_ahat_stencil, m, k, j, i, a, b);
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
void SmoothImpl(IDCTSMultigrid *mg, ViewType &u, const ViewType &src, const ViewType &free,
                int ll, int is, int ie, int js, int je, int ks, int ke, int color,
                Real omega, int fd_stencil, int div_ahat_stencil, int smoother_type,
                int ngs_iterations, Real ngs_jacobian_eps, Real ngs_max_update) {
  using ExeSpace = typename ViewType::execution_space;
  auto brdx = [&]() {
    if constexpr (std::is_same_v<ExeSpace, HostExeSpace>) return mg->GetBlockDx_h();
    else return mg->GetBlockDx();
  }();
  int nmmb = mg->GetNumMeshBlocks();
  int rlev = -ll;
  par_for("IDCTS::Smooth", ExeSpace(), 0, nmmb-1, ks, ke, js, je,
  KOKKOS_LAMBDA(int m, int k, int j) {
    Real dx = (rlev <= 0) ? brdx(m)*static_cast<Real>(1<<(-rlev))
                          : brdx(m)/static_cast<Real>(1<<rlev);
    Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
    int c = (color + k + j) & 1;
    for (int i = is + c; i <= ie; i += 2) {
      if (free(m, ID_FREE_MASK, k, j, i) < 0.5) continue;
      if (smoother_type == 1) {
        ApplyNewtonGSCTSUpdate<NGHOST>(u, src, free, idx, fd_stencil, div_ahat_stencil,
                                       m, k, j, i, omega, ngs_iterations,
                                       ngs_jacobian_eps, ngs_max_update);
      } else {
        ApplyDiagonalCTSUpdate<NGHOST>(u, src, free, idx, fd_stencil,
                                       div_ahat_stencil, m, k, j, i, omega);
      }
      Real psi = TotalU(u, free, m, ID_CTS_PSI, k, j, i);
      if (psi < 1.0e-8) {
        u(m,ID_CTS_PSI,k,j,i) = 1.0e-8 - free(m, ID_FREE_BASE_PSI, k, j, i);
      }
    }
  });
}

template <int NGHOST, typename ViewType>
void DefectImpl(IDCTSMultigrid *mg, ViewType &def, const ViewType &u, const ViewType &src,
                const ViewType &free, int ll, int is, int ie, int js, int je, int ks, int ke,
                int fd_stencil, int div_ahat_stencil) {
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
    CTSOperator<NGHOST>(u, free, idx, fd_stencil, div_ahat_stencil,
                        m, k, j, i, op, diag);
    for (int v = 0; v < ID_CTS_NVAR; ++v) {
      def(m,v,k,j,i) = (free(m, ID_FREE_MASK, k, j, i) >= 0.5)
                       ? src(m,v,k,j,i) - op[v] : 0.0;
    }
  });
}

template <int NGHOST, typename ViewType>
void FASRHSImpl(IDCTSMultigrid *mg, ViewType &src, const ViewType &u, const ViewType &free,
                int ll, int is, int ie, int js, int je, int ks, int ke, int fd_stencil,
                int div_ahat_stencil) {
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
    CTSOperator<NGHOST>(u, free, idx, fd_stencil, div_ahat_stencil,
                        m, k, j, i, op, diag);
    for (int v = 0; v < ID_CTS_NVAR; ++v) src(m,v,k,j,i) += op[v];
  });
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
    : Multigrid(pmd, pmbp, nghost, on_host) {}

IDCTSMultigrid::~IDCTSMultigrid() {}

void IDCTSMultigrid::SmoothPack(int color) {
  auto *driver = static_cast<IDCTSMultigridDriver*>(pmy_driver_);
  color ^= driver->GetCoffset();
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  int fd = driver->owner_->pmy_pack_->pz4c->opt.fd_stencil;
  int da = driver->div_ahat_stencil_;
  if (on_host_) {
    switch (fd) {
      case 2: SmoothImpl<2>(this, u_[current_level_].h_view, src_[current_level_].h_view,
                            coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                            color, driver->omega_, fd, da, driver->smoother_type_,
                            driver->ngs_iterations_, driver->ngs_jacobian_eps_,
                            driver->ngs_max_update_); break;
      case 3: SmoothImpl<3>(this, u_[current_level_].h_view, src_[current_level_].h_view,
                            coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                            color, driver->omega_, fd, da, driver->smoother_type_,
                            driver->ngs_iterations_, driver->ngs_jacobian_eps_,
                            driver->ngs_max_update_); break;
      default: SmoothImpl<4>(this, u_[current_level_].h_view, src_[current_level_].h_view,
                             coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                             color, driver->omega_, fd, da, driver->smoother_type_,
                             driver->ngs_iterations_, driver->ngs_jacobian_eps_,
                             driver->ngs_max_update_); break;
    }
  } else {
    switch (fd) {
      case 2: SmoothImpl<2>(this, u_[current_level_].d_view, src_[current_level_].d_view,
                            coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                            color, driver->omega_, fd, da, driver->smoother_type_,
                            driver->ngs_iterations_, driver->ngs_jacobian_eps_,
                            driver->ngs_max_update_); break;
      case 3: SmoothImpl<3>(this, u_[current_level_].d_view, src_[current_level_].d_view,
                            coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                            color, driver->omega_, fd, da, driver->smoother_type_,
                            driver->ngs_iterations_, driver->ngs_jacobian_eps_,
                            driver->ngs_max_update_); break;
      default: SmoothImpl<4>(this, u_[current_level_].d_view, src_[current_level_].d_view,
                             coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                             color, driver->omega_, fd, da, driver->smoother_type_,
                             driver->ngs_iterations_, driver->ngs_jacobian_eps_,
                             driver->ngs_max_update_); break;
    }
  }
}

void IDCTSMultigrid::CalculateDefectPack() {
  auto *driver = static_cast<IDCTSMultigridDriver*>(pmy_driver_);
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  int fd = driver->owner_->pmy_pack_->pz4c->opt.fd_stencil;
  int da = driver->div_ahat_stencil_;
  if (on_host_) {
    switch (fd) {
      case 2: DefectImpl<2>(this, def_[current_level_].h_view, u_[current_level_].h_view,
                            src_[current_level_].h_view, coeff_[current_level_].h_view,
                            ll, is, ie, js, je, ks, ke, fd, da); break;
      case 3: DefectImpl<3>(this, def_[current_level_].h_view, u_[current_level_].h_view,
                            src_[current_level_].h_view, coeff_[current_level_].h_view,
                            ll, is, ie, js, je, ks, ke, fd, da); break;
      default: DefectImpl<4>(this, def_[current_level_].h_view, u_[current_level_].h_view,
                             src_[current_level_].h_view, coeff_[current_level_].h_view,
                             ll, is, ie, js, je, ks, ke, fd, da); break;
    }
  } else {
    switch (fd) {
      case 2: DefectImpl<2>(this, def_[current_level_].d_view, u_[current_level_].d_view,
                            src_[current_level_].d_view, coeff_[current_level_].d_view,
                            ll, is, ie, js, je, ks, ke, fd, da); break;
      case 3: DefectImpl<3>(this, def_[current_level_].d_view, u_[current_level_].d_view,
                            src_[current_level_].d_view, coeff_[current_level_].d_view,
                            ll, is, ie, js, je, ks, ke, fd, da); break;
      default: DefectImpl<4>(this, def_[current_level_].d_view, u_[current_level_].d_view,
                             src_[current_level_].d_view, coeff_[current_level_].d_view,
                             ll, is, ie, js, je, ks, ke, fd, da); break;
    }
  }
}

void IDCTSMultigrid::CalculateFASRHSPack() {
  auto *driver = static_cast<IDCTSMultigridDriver*>(pmy_driver_);
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  int fd = driver->owner_->pmy_pack_->pz4c->opt.fd_stencil;
  int da = driver->div_ahat_stencil_;
  if (on_host_) {
    switch (fd) {
      case 2: FASRHSImpl<2>(this, src_[current_level_].h_view, u_[current_level_].h_view,
                            coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                            fd, da); break;
      case 3: FASRHSImpl<3>(this, src_[current_level_].h_view, u_[current_level_].h_view,
                            coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                            fd, da); break;
      default: FASRHSImpl<4>(this, src_[current_level_].h_view, u_[current_level_].h_view,
                             coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                             fd, da); break;
    }
  } else {
    switch (fd) {
      case 2: FASRHSImpl<2>(this, src_[current_level_].d_view, u_[current_level_].d_view,
                            coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                            fd, da); break;
      case 3: FASRHSImpl<3>(this, src_[current_level_].d_view, u_[current_level_].d_view,
                            coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                            fd, da); break;
      default: FASRHSImpl<4>(this, src_[current_level_].d_view, u_[current_level_].d_view,
                             coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                             fd, da); break;
    }
  }
}

IDCTSMultigridDriver::IDCTSMultigridDriver(IDConformalThinSandwich *owner,
                                           MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, ID_CTS_NVAR), owner_(owner) {
  ncoeff_ = ID_FREE_NVAR;
  omega_ = pin->GetOrAddReal("id_solve", "omega", 0.02);
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
  ngs_jacobian_eps_ = pin->GetOrAddReal("id_solve", "ngs_jacobian_eps", 1.0e-7);
  ngs_max_update_ = pin->GetOrAddReal("id_solve", "ngs_max_update", 0.0);
  if (ngs_iterations_ < 1 || !(ngs_jacobian_eps_ > 0.0) || ngs_max_update_ < 0.0) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/ngs_iterations must be >=1, ngs_jacobian_eps must be >0, "
              << "and ngs_max_update must be >=0." << std::endl;
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
  fsubtract_average_ = false;
  fprolongation_ = 1;

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
  if (nghost < fd_stencil) {
    std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "<id_solve>/mg_nghost=" << nghost << " is too small for the CTS "
              << "operator selected by <z4c>/spatial_order="
              << pmbp->pz4c->opt.spatial_order << ".  The composed "
              << "D_j Ahat^{ij} operator requires at least " << fd_stencil
              << " ghost cells." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int available_div_halo = std::min(nghost, mesh_nghost);
  int max_div_stencil = DivAHatStencil(fd_stencil, available_div_halo);
  int requested_div_stencil = pin->GetOrAddInteger("id_solve", "div_ahat_stencil", 0);
  if (requested_div_stencil > 0) {
    if (requested_div_stencil > max_div_stencil) {
      std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
                << std::endl
                << "Requested <id_solve>/div_ahat_stencil=" << requested_div_stencil
                << " exceeds the available halo-limited maximum " << max_div_stencil
                << ".  The nested D_j Ahat^{ij} operator needs "
                << DivAHatRequiredGhosts(fd_stencil, requested_div_stencil)
                << " ghost cells for this combination of stencils, but only "
                << available_div_halo << " are available." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    div_ahat_stencil_ = requested_div_stencil;
  } else {
    div_ahat_stencil_ = fd_stencil;
    if (div_ahat_stencil_ > max_div_stencil) {
      std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
                << std::endl
                << "The native CTS operator would silently lower "
                << "D_j Ahat^{ij} from stencil " << div_ahat_stencil_
                << " to " << max_div_stencil << " with the current halo.  "
                << "Use <mesh>/nghost and <id_solve>/mg_nghost >= "
                << DivAHatRequiredGhosts(fd_stencil, div_ahat_stencil_)
                << " for a fully order-matched CTS operator, or explicitly set "
                << "<id_solve>/div_ahat_stencil=" << max_div_stencil
                << " to request the lower-order fallback." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (div_ahat_stencil_ < pmbp->pz4c->opt.fd_stencil && global_variable::my_rank == 0) {
    std::cout << "### WARNING in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "Using lower-order Ahat divergence stencil " << div_ahat_stencil_
              << " because the composed CTS operator needs a wider halo than the base "
              << "Z4c stencil." << std::endl;
  }
  int requested_octet_fd = pin->GetOrAddInteger("id_solve", "octet_fd_stencil", 0);
  octet_fd_stencil_ = (requested_octet_fd > 0) ? requested_octet_fd : fd_stencil;
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
  int max_octet_div_stencil = DivAHatStencil(octet_fd_stencil_, nghost);
  int requested_octet_div_stencil =
      pin->GetOrAddInteger("id_solve", "octet_div_ahat_stencil", 0);
  if (requested_octet_div_stencil > 0) {
    if (requested_octet_div_stencil > max_octet_div_stencil) {
      std::cout << "### FATAL ERROR in IDCTSMultigridDriver::IDCTSMultigridDriver"
                << std::endl
                << "Requested <id_solve>/octet_div_ahat_stencil="
                << requested_octet_div_stencil
                << " exceeds the available octet halo-limited maximum "
                << max_octet_div_stencil << ".  This stencil needs "
                << DivAHatRequiredGhosts(octet_fd_stencil_, requested_octet_div_stencil)
                << " ghost cells for the nested octet operator." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    octet_div_ahat_stencil_ = requested_octet_div_stencil;
  } else {
    octet_div_ahat_stencil_ = std::min(div_ahat_stencil_, max_octet_div_stencil);
  }
  if ((octet_fd_stencil_ < fd_stencil || octet_div_ahat_stencil_ < div_ahat_stencil_) &&
      global_variable::my_rank == 0) {
    std::cout << "### WARNING in IDCTSMultigridDriver::IDCTSMultigridDriver"
              << std::endl
              << "Using CTS octet stencil fd=" << octet_fd_stencil_
              << ", div_ahat=" << octet_div_ahat_stencil_
              << " for the halo-limited SMR bridge." << std::endl;
  }
  bool root_on_host = pin->GetOrAddBoolean("id_solve", "root_on_host", false);
  mgroot_ = new IDCTSMultigrid(this, nullptr, nghost, root_on_host);
  mglevels_ = new IDCTSMultigrid(this, pmbp, nghost);
  mglevels_->pbval = new MultigridBoundaryValues(pmbp, pin, false, mglevels_);
  mglevels_->pbval->InitializeBuffers(nvar_);
  mglevels_->pbval->RemapIndicesForMG();
}

IDCTSMultigridDriver::~IDCTSMultigridDriver() {
  delete mgroot_;
  delete mglevels_;
}

void IDCTSMultigridDriver::TransferCoefficientsFromBlocksToRoot() {
  if (ncoeff_ <= 0) return;
  mglevels_->SyncCoefficientLevelToHost(0);
  auto root_coeff_h = mgroot_->GetCoefficientLevel_h(nrootlevel_ - 1);
  auto block_coeff_h = mglevels_->GetCoefficientLevel_h(0);
  int ngh = mgroot_->GetGhostCells();
  int bngh = mglevels_->GetGhostCells();
  int padding = nslist_[global_variable::my_rank];
  for (int m = 0; m < mglevels_->GetNumMeshBlocks(); ++m) {
    LogicalLocation loc = pmy_mesh_->lloc_eachmb[m + padding];
    if (loc.level == locrootlevel_) {
      int ri = static_cast<int>(loc.lx1) + ngh;
      int rj = static_cast<int>(loc.lx2) + ngh;
      int rk = static_cast<int>(loc.lx3) + ngh;
      for (int v = 0; v < ncoeff_; ++v) {
        root_coeff_h(0, v, rk, rj, ri) = block_coeff_h(m, v, bngh, bngh, bngh);
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
        oct.Coeff(v, ok, oj, oi) = block_coeff_h(m, v, bngh, bngh, bngh);
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
  TransferCoefficientsFromBlocksToRoot();

  auto calculate_defects = [&](Real defects[ID_CTS_NVAR]) {
    Real sumsq = 0.0;
    for (int v = 0; v < ID_CTS_NVAR; ++v) {
      defects[v] = CalculateDefectNorm(MGNormType::l2, v);
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
    SolveFMG(pdriver);
    mg_defect = calculate_defects(final_defects);
    owner_->RecordConstraintHistory(eps_ >= 0.0 ? -1 : niter_, final_defects);
  } else if (eps_ >= 0.0) {
    if (fshowdef_) std::cout << "MG initial defect = " << mg_defect << std::endl;
    int n = 0;
    while (mg_defect > eps_) {
      SolveVCycle(pdriver, npresmooth_, npostsmooth_);
      Real olddef = mg_defect;
      mg_defect = calculate_defects(final_defects);
      owner_->RecordConstraintHistory(n + 1, final_defects);
      if (fshowdef_) {
        std::cout << "  MG iteration " << n << ": defect = " << mg_defect << std::endl;
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
      SolveVCycle(pdriver, npresmooth_, npostsmooth_);
      mg_defect = calculate_defects(final_defects);
      owner_->RecordConstraintHistory(n + 1, final_defects);
      if (fshowdef_) {
        std::cout << "  MG iteration " << n << ": defect = " << mg_defect << std::endl;
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

void IDCTSMultigridDriver::SmoothOctet(MGOctet &oct, int rlev, int color) {
  int ngh = mgroot_->GetGhostCells();
  Real root_dx = mgroot_->GetRootDx();
  Real dx = root_dx/static_cast<Real>(1 << rlev);
  Real idx[3] = {1.0/dx, 1.0/dx, 1.0/dx};
  OctetArray u{oct.u, oct.nc};
  OctetArray src{oct.src, oct.nc};
  OctetArray free{oct.coeff, oct.nc};
  color ^= coffset_;
  for (int k = ngh; k <= ngh+1; ++k) {
    for (int j = ngh; j <= ngh+1; ++j) {
      int c = (color + k + j) & 1;
      for (int i = ngh + c; i <= ngh+1; i += 2) {
        if (free(0, ID_FREE_MASK, k, j, i) < 0.5) continue;
        switch (octet_fd_stencil_) {
          case 2:
            if (smoother_type_ == 1) {
              ApplyNewtonGSCTSUpdate<2>(u, src, free, idx, octet_fd_stencil_,
                                        octet_div_ahat_stencil_, 0, k, j, i,
                                        omega_, ngs_iterations_, ngs_jacobian_eps_,
                                        ngs_max_update_);
            } else {
              ApplyDiagonalCTSUpdate<2>(u, src, free, idx, octet_fd_stencil_,
                                        octet_div_ahat_stencil_, 0, k, j, i, omega_);
            }
            break;
          case 3:
            if (smoother_type_ == 1) {
              ApplyNewtonGSCTSUpdate<3>(u, src, free, idx, octet_fd_stencil_,
                                        octet_div_ahat_stencil_, 0, k, j, i,
                                        omega_, ngs_iterations_, ngs_jacobian_eps_,
                                        ngs_max_update_);
            } else {
              ApplyDiagonalCTSUpdate<3>(u, src, free, idx, octet_fd_stencil_,
                                        octet_div_ahat_stencil_, 0, k, j, i, omega_);
            }
            break;
          default:
            if (smoother_type_ == 1) {
              ApplyNewtonGSCTSUpdate<4>(u, src, free, idx, octet_fd_stencil_,
                                        octet_div_ahat_stencil_, 0, k, j, i,
                                        omega_, ngs_iterations_, ngs_jacobian_eps_,
                                        ngs_max_update_);
            } else {
              ApplyDiagonalCTSUpdate<4>(u, src, free, idx, octet_fd_stencil_,
                                        octet_div_ahat_stencil_, 0, k, j, i, omega_);
            }
            break;
        }
        Real psi = TotalU(u, free, 0, ID_CTS_PSI, k, j, i);
        if (psi < 1.0e-8) {
          u(0, ID_CTS_PSI, k, j, i) =
              1.0e-8 - free(0, ID_FREE_BASE_PSI, k, j, i);
        }
      }
    }
  }
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
        Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
        switch (octet_fd_stencil_) {
          case 2:
            CTSOperator<2>(u, free, idx, octet_fd_stencil_, octet_div_ahat_stencil_,
                           0, k, j, i, op, diag);
            break;
          case 3:
            CTSOperator<3>(u, free, idx, octet_fd_stencil_, octet_div_ahat_stencil_,
                           0, k, j, i, op, diag);
            break;
          default:
            CTSOperator<4>(u, free, idx, octet_fd_stencil_, octet_div_ahat_stencil_,
                           0, k, j, i, op, diag);
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
            CTSOperator<2>(u, free, idx, octet_fd_stencil_, octet_div_ahat_stencil_,
                           0, k, j, i, op, diag);
            break;
          case 3:
            CTSOperator<3>(u, free, idx, octet_fd_stencil_, octet_div_ahat_stencil_,
                           0, k, j, i, op, diag);
            break;
          default:
            CTSOperator<4>(u, free, idx, octet_fd_stencil_, octet_div_ahat_stencil_,
                           0, k, j, i, op, diag);
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
  int ahat_halo = DivAHatStencil(NGHOST, indcs.ng) - 1;
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
  int div_ahat_stencil = DivAHatStencil(fd, indcs.ng);
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
    CTSOperator<NGHOST>(u_cts, u_free, idx, fd, div_ahat_stencil, m, k, j, i, op, diag);

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
