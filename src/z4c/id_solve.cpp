#include "z4c/id_solve.hpp"

#include <algorithm>
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
constexpr Real kPunctureR2Floor = 1.0e-24;
constexpr Real kPsiFloor = 1.0e-10;

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

IDConstraintFormulation ParseIDFormulation(const std::string &raw,
                                           std::string *canonical) {
  if (raw == "ctt" || raw == "ctt_bowen_york" || raw == "bowen_york") {
    if (canonical != nullptr) *canonical = "ctt_bowen_york";
    return IDConstraintFormulation::CTTBowenYork;
  }
  if (raw == "cts") {
    if (canonical != nullptr) *canonical = "cts";
    return IDConstraintFormulation::CTS;
  }
  std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
            << "Unsupported <id_solve>/formulation='" << raw
            << "'. Supported values are ctt_bowen_york, ctt, and cts."
            << std::endl;
  std::exit(EXIT_FAILURE);
}

// SYCL/PVC portability: take centre, momentum, spin as scalar arguments rather
// than as small captured arrays.  Some SYCL+icpx revisions on Aurora reject
// passing fixed-size Real[3] arrays from a KOKKOS_LAMBDA into an inlined
// device function.  Scalarising the signature avoids that risk while leaving
// the algebra unchanged.
KOKKOS_INLINE_FUNCTION
void AddBowenYorkAhat(Real xx, Real yy, Real zz,
                      Real xp0, Real xp1, Real xp2,
                      Real p0, Real p1, Real p2,
                      Real s0, Real s1, Real s2,
                      AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> &a) {
  Real rx = xx - xp0;
  Real ry = yy - xp1;
  Real rz = zz - xp2;
  Real r2 = rx*rx + ry*ry + rz*rz;
  Real r2_safe = fmax(r2, kPunctureR2Floor);
  Real r  = sqrt(r2_safe);
  Real nx = rx/r;
  Real ny = ry/r;
  Real nz = rz/r;
  Real pdotn = p0*nx + p1*ny + p2*nz;
  // (s x n)
  Real sxn0 = s1*nz - s2*ny;
  Real sxn1 = s2*nx - s0*nz;
  Real sxn2 = s0*ny - s1*nx;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> n_arr;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> p_arr;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> sxn_arr;
  n_arr(0) = nx; n_arr(1) = ny; n_arr(2) = nz;
  p_arr(0) = p0; p_arr(1) = p1; p_arr(2) = p2;
  sxn_arr(0) = sxn0; sxn_arr(1) = sxn1; sxn_arr(2) = sxn2;
  Real inv_r2  = 1.0/r2_safe;
  Real inv_r3  = inv_r2/r;
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      Real delta = (i == j) ? 1.0 : 0.0;
      a(i,j) += 1.5*inv_r2*
          (p_arr(i)*n_arr(j) + p_arr(j)*n_arr(i) - (delta - n_arr(i)*n_arr(j))*pdotn);
      a(i,j) += 3.0*inv_r3*(n_arr(i)*sxn_arr(j) + n_arr(j)*sxn_arr(i));
    }
  }
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real HamiltonianResidual(const IDConformalThinSandwich::RelaxVars &relax,
                         const IDConformalThinSandwich::FreeVars &free,
                         Real idx0, Real idx1, Real idx2,
                         int m, int k, int j, int i) {
  Real idx[3] = {idx0, idx1, idx2};
  Real lap = Dxx<NGHOST>(0, idx, relax.u, m, k, j, i)
           + Dxx<NGHOST>(1, idx, relax.u, m, k, j, i)
           + Dxx<NGHOST>(2, idx, relax.u, m, k, j, i);
  Real psi = fmax(free.psi_singular(m,k,j,i) + relax.u(m,k,j,i),
                  kPsiFloor);
  // Replace pow(psi, 7) with explicit multiplies (SYCL/PVC: avoid std::pow on
  // device, integer exponent is faster as repeated multiplications).
  Real psi2 = psi*psi;
  Real psi4 = psi2*psi2;
  Real psi7 = psi4*psi2*psi;
  return lap + 0.125*free.ahat2(m,k,j,i)/psi7;
}

KOKKOS_INLINE_FUNCTION
void MetricInverse(const IDConformalThinSandwich::FreeVars &free,
                   int m, int k, int j, int i,
                   AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> &g_uu) {
  Real det = adm::SpatialDet(free.cts_g_dd(m,0,0,k,j,i),
                             free.cts_g_dd(m,0,1,k,j,i),
                             free.cts_g_dd(m,0,2,k,j,i),
                             free.cts_g_dd(m,1,1,k,j,i),
                             free.cts_g_dd(m,1,2,k,j,i),
                             free.cts_g_dd(m,2,2,k,j,i));
  Real det_safe = (fabs(det) < static_cast<Real>(1.0e-30)) ?
                  ((det < 0.0) ? static_cast<Real>(-1.0e-30)
                               : static_cast<Real>(1.0e-30)) : det;
  Real detinv = 1.0/det_safe;
  adm::SpatialInv(detinv,
                  free.cts_g_dd(m,0,0,k,j,i),
                  free.cts_g_dd(m,0,1,k,j,i),
                  free.cts_g_dd(m,0,2,k,j,i),
                  free.cts_g_dd(m,1,1,k,j,i),
                  free.cts_g_dd(m,1,2,k,j,i),
                  free.cts_g_dd(m,2,2,k,j,i),
                  &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
                  &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
void Christoffel(const IDConformalThinSandwich::FreeVars &free,
                 const Real idx[3], int m, int k, int j, int i,
                 AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> &gamma) {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
  MetricInverse(free, m, k, j, i, g_uu);
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg;
  for (int c = 0; c < 3; ++c) {
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        dg(c,a,b) = Dx<NGHOST>(c, idx, free.cts_g_dd, m, a, b, k, j, i);
      }
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        gamma(c,a,b) = 0.0;
        for (int d = 0; d < 3; ++d) {
          gamma(c,a,b) += 0.5*g_uu(c,d)*
              (dg(a,b,d) + dg(b,a,d) - dg(d,a,b));
        }
      }
    }
  }
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real RicciScalar(const IDConformalThinSandwich::FreeVars &free,
                 const Real idx[3], int m, int k, int j, int i) {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
  MetricInverse(free, m, k, j, i, g_uu);
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> gamma_d;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> gamma_u;
  AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg;
  for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b)
        dg(c,a,b) = Dx<NGHOST>(c, idx, free.cts_g_dd, m, a, b, k, j, i);
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
      for (int c = 0; c < 3; ++c)
        for (int d = c; d < 3; ++d)
          ddg(a,b,c,d) = (a == b)
              ? Dxx<NGHOST>(a, idx, free.cts_g_dd, m, c, d, k, j, i)
              : Dxy<NGHOST>(a, b, idx, free.cts_g_dd, m, c, d, k, j, i);
  for (int c = 0; c < 3; ++c)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        gamma_d(c,a,b) = 0.5*(dg(a,b,c) + dg(b,a,c) - dg(c,a,b));
        gamma_u(c,a,b) = 0.0;
        for (int d = 0; d < 3; ++d) gamma_u(c,a,b) += g_uu(c,d)*gamma_d(d,a,b);
      }
  Real R = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      Real R_ab = 0.0;
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          for (int e = 0; e < 3; ++e) {
            R_ab += g_uu(c,d)*gamma_u(e,a,c)*gamma_d(e,b,d);
            R_ab -= g_uu(c,d)*gamma_u(e,a,b)*gamma_d(e,c,d);
          }
          R_ab += 0.5*g_uu(c,d)*
              (-ddg(c,d,a,b) - ddg(a,b,c,d) +
                ddg(a,c,b,d) + ddg(b,c,a,d));
        }
      }
      R += g_uu(a,b)*R_ab;
    }
  }
  return R;
}

KOKKOS_INLINE_FUNCTION
Real TotalBeta(const IDConformalThinSandwich::RelaxVars &relax,
               const IDConformalThinSandwich::FreeVars &free,
               int a, int m, int k, int j, int i) {
  return free.cts_base_beta_u(m,a,k,j,i) + relax.corr(m,a+1,k,j,i);
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
void AhatUUAt(const IDConformalThinSandwich::RelaxVars &relax,
              const IDConformalThinSandwich::FreeVars &free,
              const Real idx[3], int m, int k, int j, int i,
              AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> &ahat) {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> gamma;
  MetricInverse(free, m, k, j, i, g_uu);
  Christoffel<NGHOST>(free, idx, m, k, j, i, gamma);
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> beta;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta;
  for (int a = 0; a < 3; ++a) {
    beta(a) = TotalBeta(relax, free, a, m, k, j, i);
    for (int d = 0; d < 3; ++d) {
      dbeta(d,a) = Dx<NGHOST>(d, idx, free.cts_base_beta_u, m, a, k, j, i) +
                    Dx<NGHOST>(d, idx, relax.corr, m, a+1, k, j, i);
    }
  }
  Real div_beta = 0.0;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> covd_beta;
  for (int d = 0; d < 3; ++d) {
    for (int a = 0; a < 3; ++a) {
      covd_beta(d,a) = dbeta(d,a);
      for (int e = 0; e < 3; ++e) covd_beta(d,a) += gamma(a,d,e)*beta(e);
    }
    div_beta += covd_beta(d,d);
  }
  Real alpha = fmax(free.cts_alpha(m,k,j,i), static_cast<Real>(1.0e-12));
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      Real Daup_b = 0.0;
      Real Dbup_a = 0.0;
      for (int c = 0; c < 3; ++c) {
        Daup_b += g_uu(a,c)*covd_beta(c,b);
        Dbup_a += g_uu(b,c)*covd_beta(c,a);
      }
      Real Lbeta = Daup_b + Dbup_a - (2.0/3.0)*g_uu(a,b)*div_beta;
      ahat(a,b) = (Lbeta - free.cts_udot_uu(m,a,b,k,j,i))/(2.0*alpha);
    }
  }
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real D2Beta(const IDConformalThinSandwich::RelaxVars &relax,
            const IDConformalThinSandwich::FreeVars &free,
            const Real idx[3], int dir0, int dir1, int a,
            int m, int k, int j, int i) {
  Real out = 0.0;
  if (dir0 == dir1) {
    out = Dxx<NGHOST>(dir0, idx, free.cts_base_beta_u, m, a, k, j, i) +
          Dxx<NGHOST>(dir0, idx, relax.corr, m, a+1, k, j, i);
  } else {
    out = Dxy<NGHOST>(dir0, dir1, idx, free.cts_base_beta_u, m, a, k, j, i) +
          Dxy<NGHOST>(dir0, dir1, idx, relax.corr, m, a+1, k, j, i);
  }
  return out;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real D2Metric(const IDConformalThinSandwich::FreeVars &free,
              const Real idx[3], int dir0, int dir1, int a, int b,
              int m, int k, int j, int i) {
  return (dir0 == dir1)
      ? Dxx<NGHOST>(dir0, idx, free.cts_g_dd, m, a, b, k, j, i)
      : Dxy<NGHOST>(dir0, dir1, idx, free.cts_g_dd, m, a, b, k, j, i);
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real PartialDivergenceAhatUU(const IDConformalThinSandwich::RelaxVars &relax,
                             const IDConformalThinSandwich::FreeVars &free,
                             const Real idx[3], int a,
                             int m, int k, int j, int i) {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> gamma;
  MetricInverse(free, m, k, j, i, g_uu);
  Christoffel<NGHOST>(free, idx, m, k, j, i, gamma);

  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> beta;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta;
  AthenaPointTensor<Real, TensorSymm::ISYM2, 3, 3> d2beta;
  for (int c = 0; c < 3; ++c) {
    beta(c) = TotalBeta(relax, free, c, m, k, j, i);
    for (int d = 0; d < 3; ++d) {
      dbeta(d,c) = Dx<NGHOST>(d, idx, free.cts_base_beta_u, m, c, k, j, i) +
                    Dx<NGHOST>(d, idx, relax.corr, m, c+1, k, j, i);
      for (int e = d; e < 3; ++e) {
        d2beta(d,e,c) = D2Beta<NGHOST>(relax, free, idx, d, e, c,
                                         m, k, j, i);
      }
    }
  }

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg;
  AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> d2g;
  for (int d = 0; d < 3; ++d) {
    for (int p = 0; p < 3; ++p) {
      for (int q = p; q < 3; ++q) {
        dg(d,p,q) = Dx<NGHOST>(d, idx, free.cts_g_dd, m, p, q, k, j, i);
        for (int e = d; e < 3; ++e) {
          d2g(d,e,p,q) = D2Metric<NGHOST>(free, idx, d, e, p, q,
                                             m, k, j, i);
        }
      }
    }
  }

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_uu;
  for (int d = 0; d < 3; ++d) {
    for (int p = 0; p < 3; ++p) {
      for (int q = p; q < 3; ++q) {
        dg_uu(d,p,q) = 0.0;
        for (int r = 0; r < 3; ++r)
          for (int s = 0; s < 3; ++s)
            dg_uu(d,p,q) -= g_uu(p,r)*g_uu(q,s)*dg(d,r,s);
      }
    }
  }

  AthenaPointTensor<Real, TensorSymm::NONE, 3, 4> dgamma;
  for (int d = 0; d < 3; ++d) {
    for (int up = 0; up < 3; ++up) {
      for (int p = 0; p < 3; ++p) {
        for (int q = 0; q < 3; ++q) {
          dgamma(d,up,p,q) = 0.0;
          for (int r = 0; r < 3; ++r) {
            Real s0 = dg(p,q,r) + dg(q,p,r) - dg(r,p,q);
            Real ds0 = d2g(d,p,q,r) + d2g(d,q,p,r)
                     - d2g(d,r,p,q);
            dgamma(d,up,p,q) +=
                0.5*(dg_uu(d,up,r)*s0 + g_uu(up,r)*ds0);
          }
        }
      }
    }
  }

  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> covd_beta;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 3> dcovd_beta;
  for (int d = 0; d < 3; ++d) {
    for (int c = 0; c < 3; ++c) {
      covd_beta(d,c) = dbeta(d,c);
      for (int e = 0; e < 3; ++e) {
        covd_beta(d,c) += gamma(c,d,e)*beta(e);
      }
      for (int b = 0; b < 3; ++b) {
        dcovd_beta(b,d,c) = d2beta(b,d,c);
        for (int e = 0; e < 3; ++e) {
          dcovd_beta(b,d,c) += dgamma(b,c,d,e)*beta(e)
                               + gamma(c,d,e)*dbeta(b,e);
        }
      }
    }
  }

  Real div_beta = 0.0;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> ddiv_beta;
  ddiv_beta.ZeroClear();
  for (int d = 0; d < 3; ++d) {
    div_beta += covd_beta(d,d);
    for (int b = 0; b < 3; ++b) ddiv_beta(b) += dcovd_beta(b,d,d);
  }

  Real alpha = fmax(free.cts_alpha(m,k,j,i), static_cast<Real>(1.0e-12));
  Real inv_two_alpha = 0.5/alpha;
  Real divA = 0.0;
  for (int b = 0; b < 3; ++b) {
    Real s_ab = -free.cts_udot_uu(m,a,b,k,j,i);
    Real ds_ab = -Dx<NGHOST>(b, idx, free.cts_udot_uu, m, a, b, k, j, i);
    for (int c = 0; c < 3; ++c) {
      s_ab += g_uu(a,c)*covd_beta(c,b)
            + g_uu(b,c)*covd_beta(c,a);
      ds_ab += dg_uu(b,a,c)*covd_beta(c,b)
             + g_uu(a,c)*dcovd_beta(b,c,b)
             + dg_uu(b,b,c)*covd_beta(c,a)
             + g_uu(b,c)*dcovd_beta(b,c,a);
    }
    s_ab -= (2.0/3.0)*g_uu(a,b)*div_beta;
    ds_ab -= (2.0/3.0)*(dg_uu(b,a,b)*div_beta
                       + g_uu(a,b)*ddiv_beta(b));
    Real dalpha = Dx<NGHOST>(b, idx, free.cts_alpha, m, k, j, i);
    divA += inv_two_alpha*ds_ab
          - inv_two_alpha*s_ab*dalpha/alpha;
  }
  return divA;
}

template <typename ViewType>
void DeepCopy5D(ViewType dst, ViewType src) {
  Kokkos::deep_copy(DevExeSpace(), dst, src);
}

KOKKOS_INLINE_FUNCTION
bool IsRadiationBoundary(BoundaryFlag flag) {
  return flag == BoundaryFlag::outflow || flag == BoundaryFlag::diode ||
         flag == BoundaryFlag::vacuum || flag == BoundaryFlag::user;
}

KOKKOS_INLINE_FUNCTION
bool IsReflectBoundary(BoundaryFlag flag) {
  return flag == BoundaryFlag::reflect;
}

KOKKOS_INLINE_FUNCTION
bool IsFiniteReal(Real x) {
  return (x == x) && fabs(x) < static_cast<Real>(1.0e30);
}

KOKKOS_INLINE_FUNCTION
int AdjacentInteriorIndex(int idx, bool lower, bool upper, int lo, int hi) {
  if (lower && idx < hi) return idx + 1;
  if (upper && idx > lo) return idx - 1;
  return idx;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
Real GradComponent(ViewType u, int dir, int m, int n, int k, int j, int i,
                   int is, int ie, int js, int je, int ks, int ke,
                   Real idx0, Real idx1, Real idx2) {
  if (dir == 0) {
    if (is == ie) return 0.0;
    if (i <= is) return (u(m,n,k,j,i+1) - u(m,n,k,j,i))*idx0;
    if (i >= ie) return (u(m,n,k,j,i) - u(m,n,k,j,i-1))*idx0;
    return 0.5*(u(m,n,k,j,i+1) - u(m,n,k,j,i-1))*idx0;
  }
  if (dir == 1) {
    if (js == je) return 0.0;
    if (j <= js) return (u(m,n,k,j+1,i) - u(m,n,k,j,i))*idx1;
    if (j >= je) return (u(m,n,k,j,i) - u(m,n,k,j-1,i))*idx1;
    return 0.5*(u(m,n,k,j+1,i) - u(m,n,k,j-1,i))*idx1;
  }
  if (ks == ke) return 0.0;
  if (k <= ks) return (u(m,n,k+1,j,i) - u(m,n,k,j,i))*idx2;
  if (k >= ke) return (u(m,n,k,j,i) - u(m,n,k-1,j,i))*idx2;
  return 0.5*(u(m,n,k+1,j,i) - u(m,n,k-1,j,i))*idx2;
}

template <typename ViewType>
KOKKOS_INLINE_FUNCTION
Real AdvectiveSommerfeldRHS(ViewType u, int m, int n, int k, int j, int i,
                            Real x, Real y, Real z, Real c,
                            int is, int ie, int js, int je, int ks, int ke,
                            Real idx0, Real idx1, Real idx2) {
  Real r2 = fmax(x*x + y*y + z*z, static_cast<Real>(1.0e-24));
  Real r = sqrt(r2);
  Real dfdx = GradComponent(u, 0, m, n, k, j, i, is, ie, js, je, ks, ke,
                            idx0, idx1, idx2);
  Real dfdy = GradComponent(u, 1, m, n, k, j, i, is, ie, js, je, ks, ke,
                            idx0, idx1, idx2);
  Real dfdz = GradComponent(u, 2, m, n, k, j, i, is, ie, js, je, ks, ke,
                            idx0, idx1, idx2);
  Real ngrad = (x*dfdx + y*dfdy + z*dfdz)/r;
  return -c*(ngrad + u(m,n,k,j,i)/r);
}

KOKKOS_INLINE_FUNCTION
Real RadiusFromCenter(Real x, Real y, Real z,
                      Real cx, Real cy, Real cz) {
  Real dx = x - cx;
  Real dy = y - cy;
  Real dz = z - cz;
  return sqrt(fmax(dx*dx + dy*dy + dz*dz, kPunctureR2Floor));
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
      history_file_(nullptr), formulation_(IDConstraintFormulation::CTTBowenYork),
      formulation_name_("ctt_bowen_york"), nactive_vars_(1) {
  enabled_ = pin->GetOrAddBoolean("id_solve", "enable", true);
  std::string method = pin->GetOrAddString("id_solve", "method", "hyperbolic_relaxation");
  std::string formulation = pin->GetOrAddString("id_solve", "formulation", "ctt");
  if (method != "hyperbolic_relaxation") {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "This branch supports only <id_solve>/method=hyperbolic_relaxation."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  formulation_ = ParseIDFormulation(formulation, &formulation_name_);
  if (pin->DoesParameterExist("problem", "constraint_formulation")) {
    std::string pgen_formulation_name;
    (void) ParseIDFormulation(pin->GetString("problem", "constraint_formulation"),
                              &pgen_formulation_name);
    if (pgen_formulation_name != formulation_name_) {
      std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
                << "<problem>/constraint_formulation=" << pgen_formulation_name
                << " does not match <id_solve>/formulation="
                << formulation_name_ << "." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  nactive_vars_ = (formulation_ == IDConstraintFormulation::CTS) ? 4 : 1;

  solve_once_ = pin->GetOrAddBoolean("id_solve", "solve_once", true);
  run_on_restart_ = pin->GetOrAddBoolean("id_solve", "run_on_restart", false);
  stop_after_solve_ = pin->GetOrAddBoolean("id_solve", "stop_after_solve", false);
  skip_initial_output_ = pin->GetOrAddBoolean("id_solve", "skip_initial_output",
                                              stop_after_solve_);
  abort_on_reject_ = pin->GetOrAddBoolean("id_solve", "abort_on_reject",
                                          formulation_ == IDConstraintFormulation::CTS);
  stop_on_growth_ = pin->GetOrAddBoolean("id_solve", "stop_on_growth", true);
  growth_window_ = std::max(1, pin->GetOrAddInteger("id_solve", "growth_window", 10));
  growth_start_iter_ = std::max(0, pin->GetOrAddInteger("id_solve", "growth_start_iter",
                                                        2*growth_window_));
  eta_control_max_scans_ =
      std::max(1, pin->GetOrAddInteger("id_solve", "eta_control_max_scans", 3));
  eta_control_trigger_count_ =
      std::max(1, pin->GetOrAddInteger("id_solve", "eta_control_trigger_count", 2));
  max_steps_ = pin->GetOrAddInteger("id_solve", "max_steps", 2000);
  history_every_ = std::max(1, pin->GetOrAddInteger("id_solve", "history_every", 10));
  tolerance_ = pin->GetOrAddReal("id_solve", "tolerance", 1.0e-8);
  growth_tolerance_ = pin->GetOrAddReal("id_solve", "growth_tolerance", 0.01);
  Real default_relax_cfl = (formulation_ == IDConstraintFormulation::CTS) ? 0.01 : 0.7;
  relax_cfl_ = pin->GetOrAddReal("id_solve", "relax_cfl", default_relax_cfl);
  if (!(relax_cfl_ > 0.0)) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/relax_cfl must be positive." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  eta_auto_ = !pin->DoesParameterExist("id_solve", "eta");
  eta_ = eta_auto_ ? static_cast<Real>(0.0) : pin->GetReal("id_solve", "eta");
  if (eta_ < 0.0) {
    eta_auto_ = true;
    eta_ = 0.0;
  }
  if (!eta_auto_ && eta_ < 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta must be non-negative." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  eta_schedule_ = pin->GetOrAddString("id_solve", "eta_schedule", "constant");
  eta_initial_auto_ = !pin->DoesParameterExist("id_solve", "eta_initial");
  eta_final_auto_ = !pin->DoesParameterExist("id_solve", "eta_final");
  eta_initial_ = pin->GetOrAddReal("id_solve", "eta_initial", -1.0);
  eta_final_ = pin->GetOrAddReal("id_solve", "eta_final", -1.0);
  eta_decay_tau_ = pin->GetOrAddReal("id_solve", "eta_decay_tau", -1.0);
  eta_period_tau_ = pin->GetOrAddReal("id_solve", "eta_period_tau", -1.0);
  eta_control_sweep_tau_ =
      pin->GetOrAddReal("id_solve", "eta_control_sweep_tau", -1.0);
  eta_control_tau_ = pin->GetOrAddReal("id_solve", "eta_control_tau", -1.0);
  eta_control_rate_fraction_ =
      pin->GetOrAddReal("id_solve", "eta_control_rate_fraction", 0.25);
  eta_control_power_ = pin->GetOrAddReal("id_solve", "eta_control_power", 2.0);
  eta_control_smooth_alpha_ =
      pin->GetOrAddReal("id_solve", "eta_control_smooth_alpha", 0.25);
  eta_control_curvature_target_ =
      pin->GetOrAddReal("id_solve", "eta_control_curvature_target", 0.25);
  eta_control_velocity_safeguard_ =
      pin->GetOrAddBoolean("id_solve", "eta_control_velocity_safeguard", false);
  eta_control_velocity_threshold_ =
      pin->GetOrAddReal("id_solve", "eta_control_velocity_threshold", 0.0);
  eta_control_velocity_width_ =
      pin->GetOrAddReal("id_solve", "eta_control_velocity_width", -1.0);
  eta_control_curvature_tol_ =
      pin->GetOrAddReal("id_solve", "eta_control_curvature_tol", 0.0);
  eta_long_wavelength_fraction_ =
      pin->GetOrAddReal("id_solve", "eta_long_wavelength_fraction", 1.0);
  if (eta_schedule_ != "constant" && eta_schedule_ != "exp_decay" &&
      eta_schedule_ != "periodic_sine" && eta_schedule_ != "adaptive_curvature" &&
      eta_schedule_ != "adaptive_rate" && eta_schedule_ != "adaptive_slope" &&
      eta_schedule_ != "adaptive_probe" && eta_schedule_ != "adaptive_hill" &&
      eta_schedule_ != "adaptive_scan") {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "Supported <id_solve>/eta_schedule values are constant "
              << "exp_decay, periodic_sine, adaptive_curvature, and "
              << "adaptive_rate, adaptive_slope, adaptive_probe, adaptive_hill, "
              << "and adaptive_scan."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!eta_initial_auto_ && eta_initial_ < 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_initial must be non-negative." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!eta_final_auto_ && eta_final_ < 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_final must be non-negative." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eta_decay_tau_ == 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_decay_tau must be positive, or negative "
              << "for the default." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eta_period_tau_ == 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_period_tau must be positive, or negative "
              << "for the default." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eta_control_sweep_tau_ == 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_control_sweep_tau must be positive, or "
              << "negative for the default." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eta_control_tau_ == 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_control_tau must be positive, or negative "
              << "for the default." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(eta_control_rate_fraction_ > 0.0)) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_control_rate_fraction must be positive."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(eta_control_power_ > 0.0)) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_control_power must be positive." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(eta_control_smooth_alpha_ > 0.0 && eta_control_smooth_alpha_ <= 1.0)) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_control_smooth_alpha must be in (0, 1]."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eta_control_curvature_target_ < 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_control_curvature_target must be non-negative."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eta_control_velocity_width_ == 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_control_velocity_width must be positive, or "
              << "negative for the default." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eta_control_curvature_tol_ < 0.0) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_control_curvature_tol must be non-negative."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(eta_long_wavelength_fraction_ > 0.0)) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/eta_long_wavelength_fraction must be positive."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  eta_current_ = eta_;
  damping_stability_limit_ =
      pin->GetOrAddReal("id_solve", "damping_stability_limit", 2.0);
  if (!(damping_stability_limit_ > 0.0)) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/damping_stability_limit must be positive."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string damping_form =
      pin->GetOrAddString("id_solve", "damping_form", "paper");
  if (damping_form == "paper") {
    damp_velocity_ = true;
  } else if (damping_form == "nrpy") {
    damp_velocity_ = false;
  } else {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "Supported <id_solve>/damping_form values are paper and nrpy."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string block = "problem";
  Real b = pin->GetOrAddReal(block, "par_b", 1.0);
  bare_mass_[0] = pin->GetOrAddReal(block, "par_m_plus", 1.0);
  bare_mass_[1] = pin->GetOrAddReal(block, "par_m_minus", 1.0);
  Real default_excision_radius = (formulation_ == IDConstraintFormulation::CTS) ?
                                 0.0 : 0.5*std::min(bare_mass_[0], bare_mass_[1]);
  residual_excision_radius_ = pin->GetOrAddReal("id_solve", "residual_excision_radius",
                                                default_excision_radius);
  wavespeed_scale_ = pin->GetOrAddReal("id_solve", "wavespeed_scale", 1.0);
  if (!(wavespeed_scale_ > 0.0)) {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "<id_solve>/wavespeed_scale must be positive." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string default_wavespeed_mode = "smooth_box";
  wavespeed_mode_ = pin->GetOrAddString("id_solve", "wavespeed_mode",
                                        default_wavespeed_mode);
  if (wavespeed_mode_ != "local_dx" && wavespeed_mode_ != "smooth_box") {
    std::cout << "### FATAL ERROR in IDConformalThinSandwich" << std::endl
              << "Supported <id_solve>/wavespeed_mode values are local_dx and "
              << "smooth_box." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  dtau_ = 0.0;  // set by SolveRelaxation() before the per-stage loop runs
  history_name_ = pin->GetString("job", "basename") + ".id_solve.hst";

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
  Kokkos::realloc(u_relax_best, nmb, ID_RELAX_NVAR, ncells3, ncells2, ncells1);
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

  relax_.u.InitWithShallowSlice(u_relax, ID_RELAX_DPSI);
  relax_.v.InitWithShallowSlice(u_relax, ID_RELAX_VDPSI);
  relax_.corr.InitWithShallowSlice(u_relax, ID_RELAX_DPSI, ID_RELAX_BETAZ);
  relax_.vel.InitWithShallowSlice(u_relax, ID_RELAX_VDPSI, ID_RELAX_VBETAZ);
  rhs_.u.InitWithShallowSlice(u_rhs, ID_RELAX_DPSI);
  rhs_.v.InitWithShallowSlice(u_rhs, ID_RELAX_VDPSI);
  rhs_.corr.InitWithShallowSlice(u_rhs, ID_RELAX_DPSI, ID_RELAX_BETAZ);
  rhs_.vel.InitWithShallowSlice(u_rhs, ID_RELAX_VDPSI, ID_RELAX_VBETAZ);
  free_.psi_singular.InitWithShallowSlice(u_free, ID_RELAX_PSI_SINGULAR);
  free_.ahat2.InitWithShallowSlice(u_free, ID_RELAX_AHAT2);
  free_.residual.InitWithShallowSlice(u_free, ID_RELAX_RESIDUAL_PSI);
  free_.residual_u.InitWithShallowSlice(u_free, ID_RELAX_RESIDUAL_PSI,
                                        ID_RELAX_RESIDUAL_BETAZ);
  free_.wavespeed.InitWithShallowSlice(u_free, ID_RELAX_WAVESPEED);
  free_.cts_g_dd.InitWithShallowSlice(u_free, ID_RELAX_CTS_GXX,
                                      ID_RELAX_CTS_GZZ);
  free_.cts_udot_uu.InitWithShallowSlice(u_free, ID_RELAX_CTS_UDOTXX,
                                         ID_RELAX_CTS_UDOTZZ);
  free_.cts_ahat_uu.InitWithShallowSlice(u_free, ID_RELAX_CTS_AHATXX,
                                         ID_RELAX_CTS_AHATZZ);
  free_.cts_alpha.InitWithShallowSlice(u_free, ID_RELAX_CTS_ALPHA);
  free_.cts_K.InitWithShallowSlice(u_free, ID_RELAX_CTS_K);
  free_.cts_DK_u.InitWithShallowSlice(u_free, ID_RELAX_CTS_DKX,
                                      ID_RELAX_CTS_DKZ);
  free_.cts_base_psi.InitWithShallowSlice(u_free, ID_RELAX_CTS_BASE_PSI);
  free_.cts_base_beta_u.InitWithShallowSlice(u_free, ID_RELAX_CTS_BASE_BETAX,
                                             ID_RELAX_CTS_BASE_BETAZ);

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

// ============================================================================
// Per-stage task methods.  These are direct counterparts of z4c::Z4c::{InitRecv,
// ClearRecv, ClearSend, CopyU, CalcRHS, ExpRKUpdate, RestrictU, SendU, RecvU,
// ApplyPhysicalBCs, Prolongate} but acting on (u_relax, u_relax_tmp, u_rhs,
// coarse_u_relax) and with Z4c::CalcRHS replaced by the hyperbolic-relaxation
// RHS for the velocity-damped paper form, or Eq. 21 of NRPyElliptic
// (arXiv:2111.02424), selected by <id_solve>/damping_form.
// Paper:
//     d/dt u = v,
//     d/dt v = c^2 * Hamiltonian_residual(u) - eta*v.
// NRPy:
//     d/dt u = v - eta*u,
//     d/dt v = c^2 * Hamiltonian_residual(u).
// Inter-block exchange and AMR restriction/prolongation are the same
// infrastructure used by z4c::Z4c.  Physical boundaries are ID-specific:
// CalcRHS() overwrites boundary-cell RHS values with a Cartesian Sommerfeld
// condition, and ApplyPhysicalBCs() fills scalar ghost zones for the next
// centered-stencil residual evaluation without calling Z4cBCs().
// ============================================================================

TaskStatus IDConformalThinSandwich::InitRecv(Driver *pdrive, int stage) {
  (void) pdrive; (void) stage;
  return pbval_relax_->InitRecv(ID_RELAX_NVAR);
}

TaskStatus IDConformalThinSandwich::ClearRecv(Driver *pdrive, int stage) {
  (void) pdrive; (void) stage;
  return pbval_relax_->ClearRecv();
}

TaskStatus IDConformalThinSandwich::ClearSend(Driver *pdrive, int stage) {
  (void) pdrive; (void) stage;
  return pbval_relax_->ClearSend();
}

// Mirrors Z4c::CopyU: at stage 1, u1 := u0; for the rk4 (Ketcheson 2010)
// 2-register scheme an additional u1 += delta * u0 update fires for stages
// 2..nexp_stages.  We keep u_relax_tmp playing the role of Z4c's u1.
TaskStatus IDConformalThinSandwich::CopyU(Driver *pdrive, int stage) {
  auto integrator = pdrive->integrator;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int nvar = ID_RELAX_NVAR;
  auto &u0 = u_relax;
  auto &u1 = u_relax_tmp;
  if (integrator == "rk4") {
    Real &delta = pdrive->delta[stage-1];
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      par_for("IDCTT::CopyU", DevExeSpace(),
              0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        u1(m,n,k,j,i) += delta*u0(m,n,k,j,i);
      });
    }
  } else {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    }
  }
  return TaskStatus::complete;
}

// Mirrors Z4c::CalcRHS<NGHOST>: writes the (du, dv) RHS into u_rhs from the
// current u_relax state and the cached free data (psi_singular, ahat2,
// wavespeed).  We compute the Hamiltonian-constraint residual into
// free_.residual first via the helper ComputeResidual<NGHOST>(), then
// assemble the (du, dv) pair point-wise.
template <int NGHOST>
TaskStatus IDConformalThinSandwich::CalcRHS(Driver *pdrive, int stage) {
  (void) pdrive; (void) stage;
  ComputeResidual<NGHOST>();

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  auto &u0 = u_relax;
  auto &rhs = u_rhs;
  auto free = free_;
  Real eta = eta_current_;
  int nactive = nactive_vars_;
  bool damp_velocity = damp_velocity_;
  par_for("IDCTT::CalcRHS", DevExeSpace(),
          0, nmb1, 0, ID_RELAX_NVAR-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    Real c   = free.wavespeed(m,k,j,i);
    if (n < nactive) {
      Real u_t = u0(m, n, k,j,i);
      Real v_t = u0(m, ID_RELAX_VDPSI + n, k,j,i);
      rhs(m, n, k,j,i) = damp_velocity ? v_t : v_t - eta*u_t;
    } else if (n >= ID_RELAX_VDPSI && n < ID_RELAX_VDPSI + nactive) {
      int q = n - ID_RELAX_VDPSI;
      Real residual_drive = c*c*free.residual_u(m,q,k,j,i);
      rhs(m, n, k,j,i) =
          damp_velocity ? residual_drive - eta*u0(m,n,k,j,i) : residual_drive;
    } else {
      rhs(m, n, k,j,i) = 0.0;
    }
  });
  ApplyKODissipation<NGHOST>();
  ApplySommerfeldRHS();
  return TaskStatus::complete;
}

template <int NGHOST>
void IDConformalThinSandwich::ApplyKODissipation() {
  Real diss = pmy_pack_->pz4c->diss;
  if (diss == 0.0) return;
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  auto &u0 = u_relax;
  auto &rhs = u_rhs;
  par_for("IDCTT::KODissipation", DevExeSpace(),
          0, nmb1, 0, ID_RELAX_NVAR-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    Real idx[] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                  1.0/size.d_view(m).dx3};
    for (int a = 0; a < 3; ++a) {
      rhs(m,n,k,j,i) += Diss<NGHOST>(a, idx, u0, m, n, k, j, i)*diss;
    }
  });
}

void IDConformalThinSandwich::ApplySommerfeldRHS() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  auto &mb_bcs = pmy_pack_->pmb->mb_bcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  auto &u0 = u_relax;
  auto &rhs = u_rhs;
  auto free = free_;
  Real cx = wavespeed_center_[0];
  Real cy = wavespeed_center_[1];
  Real cz = wavespeed_center_[2];
  bool multi_d = pmy_pack_->pmesh->multi_d;
  bool three_d = pmy_pack_->pmesh->three_d;

  par_for("IDCTT::SommerfeldRHS", DevExeSpace(),
          0, nmb1, 0, ID_RELAX_NVAR-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    BoundaryFlag bix1 = mb_bcs.d_view(m, BoundaryFace::inner_x1);
    BoundaryFlag box1 = mb_bcs.d_view(m, BoundaryFace::outer_x1);
    BoundaryFlag bix2 = mb_bcs.d_view(m, BoundaryFace::inner_x2);
    BoundaryFlag box2 = mb_bcs.d_view(m, BoundaryFace::outer_x2);
    BoundaryFlag bix3 = mb_bcs.d_view(m, BoundaryFace::inner_x3);
    BoundaryFlag box3 = mb_bcs.d_view(m, BoundaryFace::outer_x3);

    bool x1_in = (i == is) && IsRadiationBoundary(bix1);
    bool x1_out = (i == ie) && IsRadiationBoundary(box1);
    bool x2_in = multi_d && (j == js) && IsRadiationBoundary(bix2);
    bool x2_out = multi_d && (j == je) && IsRadiationBoundary(box2);
    bool x3_in = three_d && (k == ks) && IsRadiationBoundary(bix3);
    bool x3_out = three_d && (k == ke) && IsRadiationBoundary(box3);
    if (!(x1_in || x1_out || x2_in || x2_out || x3_in || x3_out)) return;

    Real idx0 = 1.0/size.d_view(m).dx1;
    Real idx1 = 1.0/size.d_view(m).dx2;
    Real idx2 = 1.0/size.d_view(m).dx3;

    Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                         size.d_view(m).x1max) - cx;
    Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                         size.d_view(m).x2max) - cy;
    Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                         size.d_view(m).x3max) - cz;
    Real adv_b = AdvectiveSommerfeldRHS(u0, m, n, k, j, i, x, y, z,
                                        free.wavespeed(m,k,j,i),
                                        is, ie, js, je, ks, ke,
                                        idx0, idx1, idx2);

    int ii = AdjacentInteriorIndex(i, x1_in, x1_out, is, ie);
    int jj = AdjacentInteriorIndex(j, x2_in, x2_out, js, je);
    int kk = AdjacentInteriorIndex(k, x3_in, x3_out, ks, ke);

    Real xi = CellCenterX(ii - indcs.is, indcs.nx1, size.d_view(m).x1min,
                          size.d_view(m).x1max) - cx;
    Real yi = CellCenterX(jj - indcs.js, indcs.nx2, size.d_view(m).x2min,
                          size.d_view(m).x2max) - cy;
    Real zi = CellCenterX(kk - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                          size.d_view(m).x3max) - cz;
    Real adv_i = AdvectiveSommerfeldRHS(u0, m, n, kk, jj, ii, xi, yi, zi,
                                        free.wavespeed(m,kk,jj,ii),
                                        is, ie, js, je, ks, ke,
                                        idx0, idx1, idx2);
    Real ri = sqrt(fmax(xi*xi + yi*yi + zi*zi, static_cast<Real>(1.0e-24)));
    Real rb = sqrt(fmax(x*x + y*y + z*z, static_cast<Real>(1.0e-24)));
    Real kcoef = ri*ri*ri*(rhs(m,n,kk,jj,ii) - adv_i);
    rhs(m,n,k,j,i) = adv_b + kcoef/(rb*rb*rb);
  });
}

// Mirrors Z4c::ExpRKUpdate:
//     u0 = gam0 * u0 + gam1 * u1 + (beta * dtau) * u_rhs.
// Identical body and gam0/gam1/beta access pattern as z4c_update.cpp; the
// only difference is that we replace pmesh->dt by the relaxation pseudo-time
// step dtau_ that SolveRelaxation() set.
TaskStatus IDConformalThinSandwich::ExpRKUpdate(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;

  Real &gam0 = pdrive->gam0[stage-1];
  Real &gam1 = pdrive->gam1[stage-1];
  Real beta_dt = (pdrive->beta[stage-1])*dtau_;
  auto &u0 = u_relax;
  auto &u1 = u_relax_tmp;
  auto &rhs = u_rhs;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  int nvar = ID_RELAX_NVAR;
  par_for("IDCTT::ExpRKUpdate", DevExeSpace(),
          0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    u0(m,n,k,j,i) = gam0*u0(m,n,k,j,i) + gam1*u1(m,n,k,j,i) + beta_dt*rhs(m,n,k,j,i);
  });
  return TaskStatus::complete;
}

TaskStatus IDConformalThinSandwich::RestrictU(Driver *pdrive, int stage) {
  (void) pdrive; (void) stage;
  if (pmy_pack_->pmesh->multilevel) {
    pmy_pack_->pmesh->pmr->RestrictCC(u_relax, coarse_u_relax, true);
  }
  return TaskStatus::complete;
}

TaskStatus IDConformalThinSandwich::SendU(Driver *pdrive, int stage) {
  (void) pdrive; (void) stage;
  return pbval_relax_->PackAndSendCC(u_relax, coarse_u_relax);
}

TaskStatus IDConformalThinSandwich::RecvU(Driver *pdrive, int stage) {
  (void) pdrive; (void) stage;
  return pbval_relax_->RecvAndUnpackCC(u_relax, coarse_u_relax);
}

TaskStatus IDConformalThinSandwich::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  (void) pdrive; (void) stage;
  if (pmy_pack_->pmesh->strictly_periodic) return TaskStatus::complete;

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  auto &mb_bcs = pmy_pack_->pmb->mb_bcs;
  int ng = indcs.ng;
  int nmb1 = pmy_pack_->nmb_thispack - 1;
  Real cx = wavespeed_center_[0];
  Real cy = wavespeed_center_[1];
  Real cz = wavespeed_center_[2];

  // The actual radiation boundary condition is applied to u_rhs in
  // ApplySommerfeldRHS().  This routine only fills ghost cells after each RK
  // update so the next centered-stencil residual evaluation has scalar,
  // asymptotically-compatible exterior data.
  auto fill_boundaries = [&](DvceArray5D<Real> u, int is, int ie, int js, int je,
                             int ks, int ke, int nx1, int nx2, int nx3) {
    par_for("IDCTT::RadiationBC_x1i", DevExeSpace(),
            0, nmb1, 0, ID_RELAX_NVAR-1, ks, ke, js, je, 0, ng-1,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int g) {
      BoundaryFlag flag = mb_bcs.d_view(m, BoundaryFace::inner_x1);
      int ig = is - g - 1;
      if (IsReflectBoundary(flag)) {
        u(m,n,k,j,ig) = u(m,n,k,j,is+g);
      } else if (IsRadiationBoundary(flag)) {
        Real xg = CellCenterX(ig - is, nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
        Real xb = CellCenterX(0, nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
        Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                             size.d_view(m).x2max);
        Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                             size.d_view(m).x3max);
        Real rg = RadiusFromCenter(xg, y, z, cx, cy, cz);
        Real rb = RadiusFromCenter(xb, y, z, cx, cy, cz);
        u(m,n,k,j,ig) = u(m,n,k,j,is)*(rb/rg);
      }
    });

    par_for("IDCTT::RadiationBC_x1o", DevExeSpace(),
            0, nmb1, 0, ID_RELAX_NVAR-1, ks, ke, js, je, 0, ng-1,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int g) {
      BoundaryFlag flag = mb_bcs.d_view(m, BoundaryFace::outer_x1);
      int ig = ie + g + 1;
      if (IsReflectBoundary(flag)) {
        u(m,n,k,j,ig) = u(m,n,k,j,ie-g);
      } else if (IsRadiationBoundary(flag)) {
        Real xg = CellCenterX(ig - is, nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
        Real xb = CellCenterX(ie - is, nx1, size.d_view(m).x1min,
                              size.d_view(m).x1max);
        Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                             size.d_view(m).x2max);
        Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                             size.d_view(m).x3max);
        Real rg = RadiusFromCenter(xg, y, z, cx, cy, cz);
        Real rb = RadiusFromCenter(xb, y, z, cx, cy, cz);
        u(m,n,k,j,ig) = u(m,n,k,j,ie)*(rb/rg);
      }
    });

    if (nx2 > 1) {
      par_for("IDCTT::RadiationBC_x2i", DevExeSpace(),
              0, nmb1, 0, ID_RELAX_NVAR-1, ks, ke, is, ie, 0, ng-1,
      KOKKOS_LAMBDA(int m, int n, int k, int i, int g) {
        BoundaryFlag flag = mb_bcs.d_view(m, BoundaryFace::inner_x2);
        int jg = js - g - 1;
        if (IsReflectBoundary(flag)) {
          u(m,n,k,jg,i) = u(m,n,k,js+g,i);
        } else if (IsRadiationBoundary(flag)) {
          Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
          Real yg = CellCenterX(jg - js, nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
          Real yb = CellCenterX(0, nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
          Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
          Real rg = RadiusFromCenter(x, yg, z, cx, cy, cz);
          Real rb = RadiusFromCenter(x, yb, z, cx, cy, cz);
          u(m,n,k,jg,i) = u(m,n,k,js,i)*(rb/rg);
        }
      });

      par_for("IDCTT::RadiationBC_x2o", DevExeSpace(),
              0, nmb1, 0, ID_RELAX_NVAR-1, ks, ke, is, ie, 0, ng-1,
      KOKKOS_LAMBDA(int m, int n, int k, int i, int g) {
        BoundaryFlag flag = mb_bcs.d_view(m, BoundaryFace::outer_x2);
        int jg = je + g + 1;
        if (IsReflectBoundary(flag)) {
          u(m,n,k,jg,i) = u(m,n,k,je-g,i);
        } else if (IsRadiationBoundary(flag)) {
          Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
          Real yg = CellCenterX(jg - js, nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
          Real yb = CellCenterX(je - js, nx2, size.d_view(m).x2min,
                                size.d_view(m).x2max);
          Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
          Real rg = RadiusFromCenter(x, yg, z, cx, cy, cz);
          Real rb = RadiusFromCenter(x, yb, z, cx, cy, cz);
          u(m,n,k,jg,i) = u(m,n,k,je,i)*(rb/rg);
        }
      });
    }

    if (nx3 > 1) {
      par_for("IDCTT::RadiationBC_x3i", DevExeSpace(),
              0, nmb1, 0, ID_RELAX_NVAR-1, js, je, is, ie, 0, ng-1,
      KOKKOS_LAMBDA(int m, int n, int j, int i, int g) {
        BoundaryFlag flag = mb_bcs.d_view(m, BoundaryFace::inner_x3);
        int kg = ks - g - 1;
        if (IsReflectBoundary(flag)) {
          u(m,n,kg,j,i) = u(m,n,ks+g,j,i);
        } else if (IsRadiationBoundary(flag)) {
          Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
          Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
          Real zg = CellCenterX(kg - ks, nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
          Real zb = CellCenterX(0, nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
          Real rg = RadiusFromCenter(x, y, zg, cx, cy, cz);
          Real rb = RadiusFromCenter(x, y, zb, cx, cy, cz);
          u(m,n,kg,j,i) = u(m,n,ks,j,i)*(rb/rg);
        }
      });

      par_for("IDCTT::RadiationBC_x3o", DevExeSpace(),
              0, nmb1, 0, ID_RELAX_NVAR-1, js, je, is, ie, 0, ng-1,
      KOKKOS_LAMBDA(int m, int n, int j, int i, int g) {
        BoundaryFlag flag = mb_bcs.d_view(m, BoundaryFace::outer_x3);
        int kg = ke + g + 1;
        if (IsReflectBoundary(flag)) {
          u(m,n,kg,j,i) = u(m,n,ke-g,j,i);
        } else if (IsRadiationBoundary(flag)) {
          Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
          Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
          Real zg = CellCenterX(kg - ks, nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
          Real zb = CellCenterX(ke - ks, nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max);
          Real rg = RadiusFromCenter(x, y, zg, cx, cy, cz);
          Real rb = RadiusFromCenter(x, y, zb, cx, cy, cz);
          u(m,n,kg,j,i) = u(m,n,ke,j,i)*(rb/rg);
        }
      });
    }
  };

  fill_boundaries(u_relax, indcs.is, indcs.ie, indcs.js, indcs.je, indcs.ks,
                  indcs.ke, indcs.nx1, indcs.nx2, indcs.nx3);
  if (pmy_pack_->pmesh->multilevel) {
    fill_boundaries(coarse_u_relax, indcs.cis, indcs.cie, indcs.cjs, indcs.cje,
                    indcs.cks, indcs.cke, indcs.cnx1, indcs.cnx2, indcs.cnx3);
  }
  return TaskStatus::complete;
}

TaskStatus IDConformalThinSandwich::Prolongate(Driver *pdrive, int stage) {
  (void) pdrive; (void) stage;
  if (pmy_pack_->pmesh->multilevel) {
    pbval_relax_->ProlongateCC(u_relax, coarse_u_relax, true);
  }
  return TaskStatus::complete;
}

void IDConformalThinSandwich::BuildFreeData() {
  if (formulation_ == IDConstraintFormulation::CTS) {
    int fd = pmy_pack_->pz4c->opt.fd_stencil;
    if (fd == 2) BuildCTSFreeData<2>();
    else if (fd == 3) BuildCTSFreeData<3>();
    else BuildCTSFreeData<4>();
  } else {
    BuildCTTBowenYorkFreeData();
  }
}

void IDConformalThinSandwich::BuildCTTBowenYorkFreeData() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  int isg = indcs.is - indcs.ng, ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng, jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng, keg = indcs.ke + indcs.ng;
  int nmb = pmy_pack_->nmb_thispack;
  auto free = free_;
  Real m0 = bare_mass_[0], m1 = bare_mass_[1];
  // Capture per-puncture parameters as scalars so the lambda capture clause
  // does not include any small Real[3] arrays.  This avoids SYCL/PVC issues
  // with capturing fixed-size local arrays into device kernels.
  Real x0_0 = pos_[0][0], x0_1 = pos_[0][1], x0_2 = pos_[0][2];
  Real x1_0 = pos_[1][0], x1_1 = pos_[1][1], x1_2 = pos_[1][2];
  Real p0_0 = mom_[0][0], p0_1 = mom_[0][1], p0_2 = mom_[0][2];
  Real p1_0 = mom_[1][0], p1_1 = mom_[1][1], p1_2 = mom_[1][2];
  Real s0_0 = spin_[0][0], s0_1 = spin_[0][1], s0_2 = spin_[0][2];
  Real s1_0 = spin_[1][0], s1_1 = spin_[1][1], s1_2 = spin_[1][2];

  par_for("IDCTT/BY::BuildFreeData", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real xx = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                          size.d_view(m).x1max);
    Real yy = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                          size.d_view(m).x2max);
    Real zz = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                          size.d_view(m).x3max);
    Real r0 = sqrt(fmax(SQR(xx - x0_0) + SQR(yy - x0_1) + SQR(zz - x0_2),
                        static_cast<Real>(1.0e-24)));
    Real r1 = sqrt(fmax(SQR(xx - x1_0) + SQR(yy - x1_1) + SQR(zz - x1_2),
                        static_cast<Real>(1.0e-24)));
    free.psi_singular(m,k,j,i) = 1.0 + 0.5*m0/r0 + 0.5*m1/r1;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> a;
    a.ZeroClear();
    AddBowenYorkAhat(xx, yy, zz, x0_0, x0_1, x0_2,
                     p0_0, p0_1, p0_2, s0_0, s0_1, s0_2, a);
    AddBowenYorkAhat(xx, yy, zz, x1_0, x1_1, x1_2,
                     p1_0, p1_1, p1_2, s1_0, s1_1, s1_2, a);
    Real ahat2 = 0.0;
    for (int q = 0; q < 3; ++q)
      for (int r = 0; r < 3; ++r) ahat2 += a(q,r)*a(q,r);
    free.ahat2(m,k,j,i) = ahat2;
    free.residual(m,k,j,i) = 0.0;
    for (int q = 0; q < 4; ++q) free.residual_u(m,q,k,j,i) = 0.0;
    for (int q = 0; q < 3; ++q)
      for (int r = q; r < 3; ++r) free.cts_ahat_uu(m,q,r,k,j,i) = 0.0;
  });
}

template <int NGHOST>
void IDConformalThinSandwich::BuildCTSFreeData() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  (void) size;
  int isg = indcs.is - indcs.ng, ieg = indcs.ie + indcs.ng;
  int jsg = indcs.js - indcs.ng, jeg = indcs.je + indcs.ng;
  int ksg = indcs.ks - indcs.ng, keg = indcs.ke + indcs.ng;
  int nmb = pmy_pack_->nmb_thispack;
  auto free = free_;
  auto &admvars = pmy_pack_->padm->adm;

  par_for("IDCTS::BuildFreeData", DevExeSpace(), 0, nmb-1,
          ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        g(a,b) = admvars.g_dd(m,a,b,k,j,i);
        free.cts_g_dd(m,a,b,k,j,i) = g(a,b);
      }
    }
    Real det = adm::SpatialDet(g(0,0), g(0,1), g(0,2), g(1,1), g(1,2), g(2,2));
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gi;
    Real det_safe = (fabs(det) < static_cast<Real>(1.0e-30)) ?
                    ((det < 0.0) ? static_cast<Real>(-1.0e-30)
                                 : static_cast<Real>(1.0e-30)) : det;
    adm::SpatialInv(1.0/det_safe,
                    g(0,0), g(0,1), g(0,2), g(1,1), g(1,2), g(2,2),
                    &gi(0,0), &gi(0,1), &gi(0,2),
                    &gi(1,1), &gi(1,2), &gi(2,2));

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> udot_cov;
    Real tr = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        udot_cov(a,b) = admvars.vK_dd(m,a,b,k,j,i);
        tr += gi(a,b)*udot_cov(a,b);
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        udot_cov(a,b) -= (1.0/3.0)*g(a,b)*tr;
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        Real raised = 0.0;
        for (int c = 0; c < 3; ++c)
          for (int d = 0; d < 3; ++d) raised += gi(a,c)*gi(b,d)*udot_cov(c,d);
        free.cts_udot_uu(m,a,b,k,j,i) = raised;
      }
    }
    free.cts_alpha(m,k,j,i) = fmax(admvars.alpha(m,k,j,i), static_cast<Real>(1.0e-12));
    free.cts_K(m,k,j,i) = 0.0;
    for (int a = 0; a < 3; ++a) {
      free.cts_DK_u(m,a,k,j,i) = 0.0;
      free.cts_base_beta_u(m,a,k,j,i) = admvars.beta_u(m,a,k,j,i);
    }
    for (int q = 0; q < 4; ++q) free.residual_u(m,q,k,j,i) = 0.0;
    free.cts_base_psi(m,k,j,i) =
        Kokkos::pow(fmax(admvars.psi4(m,k,j,i), static_cast<Real>(1.0e-40)),
                    static_cast<Real>(0.25));
    free.psi_singular(m,k,j,i) = free.cts_base_psi(m,k,j,i);
    free.ahat2(m,k,j,i) = 0.0;
    free.residual(m,k,j,i) = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) free.cts_ahat_uu(m,a,b,k,j,i) = 0.0;
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
    std::cout << "ID " << formulation_name_ << " relaxation wave speed mode = "
              << wavespeed_mode_
              << ", levels = " << max_phys_level + 1 << ", radii:";
    for (int lev = 0; lev <= max_phys_level; ++lev) {
      std::cout << " L" << lev << "=" << radii[lev];
    }
    std::cout << std::endl;
  }
}

Real IDConformalThinSandwich::EstimateDefaultEta(Real dx_min) const {
  int fd = pmy_pack_->pz4c->opt.fd_stencil;
  Real max_symbol_1d = 4.0;
  if (fd == 3) {
    max_symbol_1d = static_cast<Real>(16.0/3.0);
  } else if (fd >= 4) {
    max_symbol_1d = static_cast<Real>(272.0/45.0);
  }
  Real principal_factor =
      (formulation_ == IDConstraintFormulation::CTS) ? static_cast<Real>(4.0/3.0)
                                                     : static_cast<Real>(1.0);
  Real c_over_dx = wavespeed_scale_/std::max(dx_min, static_cast<Real>(1.0e-30));
  return static_cast<Real>(2.0)*std::sqrt(principal_factor*max_symbol_1d)*
         c_over_dx;
}

Real IDConformalThinSandwich::EstimateBoxEta() const {
  return EstimateEtaForLength(EstimateBoxLength());
}

Real IDConformalThinSandwich::EstimateBoxLength() const {
  const RegionSize &ms = pmy_pack_->pmesh->mesh_size;
  Real lx = ms.x1max - ms.x1min;
  Real ly = ms.x2max - ms.x2min;
  Real lz = ms.x3max - ms.x3min;
  return std::max(lx, std::max(ly, lz));
}

Real IDConformalThinSandwich::EstimateEtaForLength(Real length) const {
  Real principal_factor =
      (formulation_ == IDConstraintFormulation::CTS) ? static_cast<Real>(4.0/3.0)
                                                     : static_cast<Real>(1.0);
  return static_cast<Real>(4.0)*wavespeed_scale_*std::sqrt(principal_factor)*
         kPi/std::max(length, static_cast<Real>(1.0e-30));
}

Real IDConformalThinSandwich::EtaAtTau(Real tau) const {
  if (eta_schedule_ == "exp_decay") {
    Real decay_tau = std::max(eta_decay_tau_, static_cast<Real>(1.0e-30));
    return eta_final_ + (eta_initial_ - eta_final_)*std::exp(-tau/decay_tau);
  }
  if (eta_schedule_ == "periodic_sine") {
    Real period = std::max(eta_period_tau_, static_cast<Real>(1.0e-30));
    Real eta_mid = static_cast<Real>(0.5)*(eta_initial_ + eta_final_);
    Real eta_amp = static_cast<Real>(0.5)*std::fabs(eta_final_ - eta_initial_);
    return eta_mid - eta_amp*std::cos(static_cast<Real>(2.0)*kPi*tau/period);
  }
  return eta_;
}

template <int NGHOST>
void IDConformalThinSandwich::ComputeResidual() {
  if (formulation_ == IDConstraintFormulation::CTS) {
    ComputeCTSResidual<NGHOST>();
  } else {
    ComputeCTTBowenYorkResidual<NGHOST>();
  }
}

template <int NGHOST>
void IDConformalThinSandwich::ComputeCTTBowenYorkResidual() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  // Restrict the residual evaluation to the interior cells.  Earlier code
  // looped over [is-ng, ie+ng] etc., but Dxx<NGHOST> reads indices i ± k for
  // k>=1 and walks off the end of the View at the very corners of the
  // (ng-deep) ghost band.  On Aurora SYCL/PVC this turned into hard PTE-Read
  // GPU faults as soon as we exercised refinement levels in a grid where
  // the surrounding pages were no longer mapped.  The residual is only
  // physically meaningful in the interior anyway; ghost-zone values come
  // back through MeshBoundaryValuesCC after the RK update.
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  auto relax = relax_;
  auto free = free_;
  par_for("IDCTT::ComputeResidual", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real idx0 = 1.0/size.d_view(m).dx1;
    Real idx1 = 1.0/size.d_view(m).dx2;
    Real idx2 = 1.0/size.d_view(m).dx3;
    free.residual(m,k,j,i) = HamiltonianResidual<NGHOST>(relax, free,
                                                         idx0, idx1, idx2,
                                                         m, k, j, i);
    free.residual_u(m,0,k,j,i) = free.residual(m,k,j,i);
    free.residual_u(m,1,k,j,i) = 0.0;
    free.residual_u(m,2,k,j,i) = 0.0;
    free.residual_u(m,3,k,j,i) = 0.0;
  });
}

template <int NGHOST>
void IDConformalThinSandwich::ComputeCTSResidual() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  auto &size = pmy_pack_->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  auto relax = relax_;
  auto free = free_;
  par_for("IDCTS::ComputeResidual", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real idx[3] = {1.0/size.d_view(m).dx1,
                   1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> gamma;
    MetricInverse(free, m, k, j, i, g_uu);
    Christoffel<NGHOST>(free, idx, m, k, j, i, gamma);

    Real psi = fmax(free.cts_base_psi(m,k,j,i) + relax.corr(m,0,k,j,i),
                    kPsiFloor);
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dpsi;
    for (int a = 0; a < 3; ++a) {
      dpsi(a) = Dx<NGHOST>(a, idx, relax.corr, m, 0, k, j, i);
    }
    Real lap = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        Real dd = (a == b) ? Dxx<NGHOST>(a, idx, relax.corr, m, 0, k, j, i)
                           : Dxy<NGHOST>(a, b, idx, relax.corr, m, 0, k, j, i);
        for (int c = 0; c < 3; ++c) dd -= gamma(c,a,b)*dpsi(c);
        lap += g_uu(a,b)*dd;
      }
    }

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ahat;
    AhatUUAt<NGHOST>(relax, free, idx, m, k, j, i, ahat);
    Real ahat2 = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        free.cts_ahat_uu(m,a,b,k,j,i) = ahat(a,b);
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        Real ahat_dd = 0.0;
        for (int c = 0; c < 3; ++c)
          for (int d = 0; d < 3; ++d)
            ahat_dd += free.cts_g_dd(m,a,c,k,j,i)*free.cts_g_dd(m,b,d,k,j,i)*
                        free.cts_ahat_uu(m,c,d,k,j,i);
        ahat2 += ahat_dd*free.cts_ahat_uu(m,a,b,k,j,i);
      }
    }
    free.ahat2(m,k,j,i) = ahat2;
    Real psi2 = psi*psi;
    Real psi4 = psi2*psi2;
    Real psi5 = psi4*psi;
    Real psi6 = psi4*psi2;
    Real psi7 = psi6*psi;
    Real K = free.cts_K(m,k,j,i);
    Real R = RicciScalar<NGHOST>(free, idx, m, k, j, i);
    Real ham = lap - 0.125*R*psi + 0.125*ahat2/psi7
             - (1.0/12.0)*K*K*psi5;
    free.residual_u(m,0,k,j,i) = ham;

    for (int a = 0; a < 3; ++a) {
      Real divA = PartialDivergenceAhatUU<NGHOST>(relax, free, idx, a,
                                                  m, k, j, i);
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          divA += gamma(a,b,c)*free.cts_ahat_uu(m,c,b,k,j,i);
          divA += gamma(b,b,c)*free.cts_ahat_uu(m,a,c,k,j,i);
        }
      }
      free.residual_u(m,a+1,k,j,i) =
          2.0*divA - (4.0/3.0)*psi6*free.cts_DK_u(m,a,k,j,i);
    }
    Real r2 = 0.0;
    for (int q = 0; q < 4; ++q) r2 += free.residual_u(m,q,k,j,i)*free.residual_u(m,q,k,j,i);
    free.residual(m,k,j,i) = sqrt(r2);
  });
}

void IDConformalThinSandwich::OpenHistory() {
  if (history_file_ != nullptr || global_variable::my_rank != 0) return;
  history_file_ = std::fopen(history_name_.c_str(), "w");
  if (history_file_ != nullptr) {
    std::fprintf(history_file_, "# iter tau dtau eta residual_l2 residual_rel_l2 "
                               "residual_max residual_excised_l2 "
                               "residual_excised_rel_l2 residual_excised_max "
                               "u_l2 v_l2 v_max volume excised_volume ncell "
                               "excised_ncell finite residual_excision_radius\n");
  }
}

void IDConformalThinSandwich::RecordHistory(int iter, Real tau,
                                            const Diagnostics &diag) {
  if (global_variable::my_rank != 0) return;
  OpenHistory();
  if (history_file_ == nullptr) return;
  std::fprintf(history_file_,
               "%d %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e "
               "%.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e\n",
               iter, static_cast<double>(tau), static_cast<double>(dtau_),
               static_cast<double>(eta_current_),
               static_cast<double>(diag.residual_l2),
               static_cast<double>(diag.residual_rel_l2),
               static_cast<double>(diag.residual_max),
               static_cast<double>(diag.residual_excised_l2),
               static_cast<double>(diag.residual_excised_rel_l2),
               static_cast<double>(diag.residual_excised_max),
               static_cast<double>(diag.u_l2),
               static_cast<double>(diag.v_l2),
               static_cast<double>(diag.v_max),
               static_cast<double>(diag.volume),
               static_cast<double>(diag.excised_volume),
               static_cast<double>(diag.ncell),
               static_cast<double>(diag.excised_ncell),
               static_cast<double>(diag.finite),
               static_cast<double>(residual_excision_radius_));
  std::fflush(history_file_);
}

IDConformalThinSandwich::Diagnostics
IDConformalThinSandwich::ReduceDiagnostics(Real initial_residual_l2,
                                           Real initial_residual_excised_l2) {
  Diagnostics diag;
  diag.residual_l2 = 0.0;
  diag.residual_rel_l2 = 0.0;
  diag.residual_max = 0.0;
  diag.residual_excised_l2 = 0.0;
  diag.residual_excised_rel_l2 = 0.0;
  diag.residual_excised_max = 0.0;
  diag.u_l2 = 0.0;
  diag.v_l2 = 0.0;
  diag.v_max = 0.0;
  diag.volume = 0.0;
  diag.excised_volume = 0.0;
  diag.ncell = 0.0;
  diag.excised_ncell = 0.0;
  diag.finite = 1.0;

  Real res2 = 0.0;
  Real res2_excised = 0.0;
  Real u2 = 0.0;
  Real v2 = 0.0;
  Real volume = 0.0;
  Real volume_excised = 0.0;
  Real ncell_sum = 0.0;
  Real ncell_excised = 0.0;
  Real maxres = 0.0;
  Real maxres_excised = 0.0;
  Real maxv = 0.0;
  Real finite = 1.0;

  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  int nmb = pmy_pack_->nmb_thispack;
  int nkji = nx3*nx2*nx1;
  int nji = nx2*nx1;
  int nmkji = nmb*nkji;
  auto free = free_;
  auto u = u_relax;
  auto &size = pmy_pack_->pmb->mb_size;
  int nactive = nactive_vars_;
  Real x0_0 = pos_[0][0], x0_1 = pos_[0][1], x0_2 = pos_[0][2];
  Real x1_0 = pos_[1][0], x1_1 = pos_[1][1], x1_2 = pos_[1][2];
  Real rex2 = residual_excision_radius_*residual_excision_radius_;

  Kokkos::parallel_reduce("IDCTT::Diagnostics",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &lres2, Real &lres2_excised,
                  Real &lu2, Real &lv2, Real &lvol, Real &lvol_excised,
                  Real &lncell, Real &lncell_excised, Real &lmaxres,
                  Real &lmaxres_excised, Real &lmaxv, Real &lfinite) {
      int m = idx/nkji;
      int rem = idx - m*nkji;
      int k = rem/nji;
      rem -= k*nji;
      int j = rem/nx1;
      int i = rem - j*nx1;
      k += ks;
      j += js;
      i += is;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      Real res_cell2 = 0.0;
      bool all_finite = true;
      for (int q = 0; q < nactive; ++q) {
        Real rq = free.residual_u(m,q,k,j,i);
        res_cell2 += rq*rq;
        all_finite = all_finite && IsFiniteReal(rq);
      }
      Real ar = sqrt(res_cell2);
      Real uu2 = 0.0;
      Real vv2 = 0.0;
      Real av = 0.0;
      for (int q = 0; q < nactive; ++q) {
        Real uq = u(m, q, k,j,i);
        Real vq = u(m, ID_RELAX_VDPSI + q, k,j,i);
        uu2 += uq*uq;
        vv2 += vq*vq;
        av = fmax(av, fabs(vq));
        all_finite = all_finite && IsFiniteReal(uq) && IsFiniteReal(vq);
      }
      Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                           size.d_view(m).x1max);
      Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                           size.d_view(m).x2max);
      Real z = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                           size.d_view(m).x3max);
      Real r0_2 = SQR(x - x0_0) + SQR(y - x0_1) + SQR(z - x0_2);
      Real r1_2 = SQR(x - x1_0) + SQR(y - x1_1) + SQR(z - x1_2);
      bool keep = (rex2 <= 0.0) || (r0_2 > rex2 && r1_2 > rex2);

      lres2 += vol*res_cell2;
      lu2 += vol*uu2;
      lv2 += vol*vv2;
      lvol += vol;
      lncell += 1.0;
      lmaxres = fmax(lmaxres, ar);
      if (keep) {
        lres2_excised += vol*res_cell2;
        lvol_excised += vol;
        lncell_excised += 1.0;
        lmaxres_excised = fmax(lmaxres_excised, ar);
      }
      lmaxv = fmax(lmaxv, av);
      lfinite = fmin(lfinite, all_finite ?
                     static_cast<Real>(1.0) : static_cast<Real>(0.0));
    },
    Kokkos::Sum<Real>(res2), Kokkos::Sum<Real>(res2_excised),
    Kokkos::Sum<Real>(u2), Kokkos::Sum<Real>(v2),
    Kokkos::Sum<Real>(volume), Kokkos::Sum<Real>(volume_excised),
    Kokkos::Sum<Real>(ncell_sum), Kokkos::Sum<Real>(ncell_excised),
    Kokkos::Max<Real>(maxres), Kokkos::Max<Real>(maxres_excised),
    Kokkos::Max<Real>(maxv), Kokkos::Min<Real>(finite));

  Real sums[8] = {res2, res2_excised, u2, v2, volume, volume_excised,
                  ncell_sum, ncell_excised};
  Real maxes[3] = {maxres, maxres_excised, maxv};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &sums[0], 8, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maxes[0], 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &finite, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif

  Real norm_floor = std::numeric_limits<Real>::min();
  Real norm_volume = std::max(sums[4], norm_floor);
  Real norm_excised_volume = std::max(sums[5], norm_floor);
  diag.residual_l2 = std::sqrt(sums[0]/norm_volume);
  diag.residual_excised_l2 = std::sqrt(sums[1]/norm_excised_volume);
  diag.u_l2 = std::sqrt(sums[2]/norm_volume);
  diag.v_l2 = std::sqrt(sums[3]/norm_volume);
  diag.residual_max = maxes[0];
  diag.residual_excised_max = maxes[1];
  diag.v_max = maxes[2];
  diag.volume = sums[4];
  diag.excised_volume = sums[5];
  diag.ncell = sums[6];
  diag.excised_ncell = sums[7];
  diag.finite = finite;
  Real rel_floor = std::max(initial_residual_l2, norm_floor);
  Real rel_excised_floor = std::max(initial_residual_excised_l2, norm_floor);
  diag.residual_rel_l2 = diag.residual_l2/rel_floor;
  diag.residual_excised_rel_l2 = diag.residual_excised_l2/rel_excised_floor;
  return diag;
}

void IDConformalThinSandwich::SolveRelaxation(Driver *pdriver) {
  BuildFreeData();
  Kokkos::deep_copy(u_relax, 0.0);
  // Pre-step boundary exchange so the very first ComputeResidual<NGHOST>()
  // sees correctly-filled ghost cells of the (zero) initial guess.
  InitRecv(pdriver, 0);
  RestrictU(pdriver, 0);
  SendU(pdriver, 0);
  ClearSend(pdriver, 0);
  ClearRecv(pdriver, 0);
  RecvU(pdriver, 0);
  ApplyPhysicalBCs(pdriver, 0);
  Prolongate(pdriver, 0);

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
  BuildWaveSpeedProfile(dx_min);
  if (eta_auto_) {
    eta_ = EstimateDefaultEta(dx_min);
    if (global_variable::my_rank == 0) {
      std::cout << "ID " << formulation_name_
                << " relaxation auto eta = " << eta_
                << " from the smooth wave-speed envelope and discrete "
                << "finite-difference symbol." << std::endl;
    }
  }
  Real wave_dtau = relax_cfl_*dx_min/std::max(wavespeed_scale_,
                                             static_cast<Real>(1.0e-12));
  if (eta_schedule_ == "exp_decay") {
    if (eta_initial_auto_) eta_initial_ = eta_;
    if (eta_final_auto_) eta_final_ = EstimateBoxEta();
  } else if (eta_schedule_ == "periodic_sine" ||
             eta_schedule_ == "adaptive_curvature" ||
             eta_schedule_ == "adaptive_slope" ||
             eta_schedule_ == "adaptive_probe" ||
             eta_schedule_ == "adaptive_hill" ||
             eta_schedule_ == "adaptive_scan") {
    Real long_length = eta_long_wavelength_fraction_*EstimateBoxLength();
    Real eta_long = EstimateEtaForLength(long_length);
    Real eta_nyquist = EstimateDefaultEta(dx_min);
    if (eta_initial_auto_) eta_initial_ = std::min(eta_long, eta_nyquist);
    if (eta_final_auto_) eta_final_ = std::max(eta_long, eta_nyquist);
  } else if (eta_schedule_ == "adaptive_rate") {
    Real long_length = eta_long_wavelength_fraction_*EstimateBoxLength();
    Real eta_long = EstimateEtaForLength(long_length);
    Real eta_nyquist = EstimateDefaultEta(dx_min);
    Real eta_stability_half = static_cast<Real>(0.5)*damping_stability_limit_/
        std::max(wave_dtau, std::numeric_limits<Real>::min());
    if (eta_initial_auto_) eta_initial_ = std::min(eta_long, eta_nyquist);
    if (eta_final_auto_) eta_final_ = eta_stability_half;
  } else {
    eta_initial_ = eta_;
    eta_final_ = eta_;
  }

  // Cache dtau on the class so ExpRKUpdate() picks it up at every stage.
  // The wave CFL alone is not sufficient when eta is large: the damping
  // term is integrated explicitly by the same RK scheme, so eta*dtau must
  // also remain on the stable real-axis part of the RK stability region.
  dtau_ = wave_dtau;
  Real eta_stability = eta_;
  if (eta_schedule_ == "exp_decay" || eta_schedule_ == "periodic_sine" ||
      eta_schedule_ == "adaptive_curvature" || eta_schedule_ == "adaptive_rate" ||
      eta_schedule_ == "adaptive_slope" || eta_schedule_ == "adaptive_probe" ||
      eta_schedule_ == "adaptive_hill" || eta_schedule_ == "adaptive_scan") {
    eta_stability = std::max(eta_initial_, eta_final_);
  }
  if (eta_stability > 0.0) {
    Real damp_dtau = damping_stability_limit_/eta_stability;
    if (dtau_ > damp_dtau) {
      dtau_ = damp_dtau;
      if (global_variable::my_rank == 0) {
        std::cout << "ID " << formulation_name_
                  << " relaxation damping stability limited dtau from "
                  << wave_dtau << " to " << dtau_
                  << " because eta*dtau would exceed "
                  << damping_stability_limit_ << "." << std::endl;
      }
    }
  }
  if (eta_schedule_ == "exp_decay" && eta_decay_tau_ < 0.0) {
    eta_decay_tau_ = static_cast<Real>(0.25)*static_cast<Real>(max_steps_)*dtau_;
  }
  if (eta_schedule_ == "periodic_sine" && eta_period_tau_ < 0.0) {
    eta_period_tau_ = static_cast<Real>(20.0)*dtau_;
  }
  if ((eta_schedule_ == "adaptive_curvature" || eta_schedule_ == "adaptive_probe") &&
      eta_control_sweep_tau_ < 0.0) {
    eta_control_sweep_tau_ = static_cast<Real>(10.0)*dtau_;
  }
  if (eta_schedule_ == "adaptive_scan" && eta_control_sweep_tau_ < 0.0) {
    eta_control_sweep_tau_ = static_cast<Real>(40.0)*dtau_;
  }
  if ((eta_schedule_ == "adaptive_rate" || eta_schedule_ == "adaptive_slope" ||
       eta_schedule_ == "adaptive_probe" || eta_schedule_ == "adaptive_hill" ||
       eta_schedule_ == "adaptive_scan") &&
      eta_control_tau_ < 0.0) {
    eta_control_tau_ = static_cast<Real>(8.0)*dtau_;
  }
  if (eta_control_velocity_width_ < 0.0) {
    eta_control_velocity_width_ =
        static_cast<Real>(0.1)/std::max(dtau_, std::numeric_limits<Real>::min());
  }
  if (eta_schedule_ == "adaptive_slope" || eta_schedule_ == "adaptive_probe" ||
      eta_schedule_ == "adaptive_hill") {
    eta_current_ = static_cast<Real>(0.5)*(std::min(eta_initial_, eta_final_) +
                                           std::max(eta_initial_, eta_final_));
  } else if (eta_schedule_ == "adaptive_scan") {
    eta_current_ = std::min(eta_initial_, eta_final_);
  } else {
    eta_current_ = (eta_schedule_ == "adaptive_curvature" ||
                    eta_schedule_ == "adaptive_rate") ?
                   std::min(eta_initial_, eta_final_) : EtaAtTau(0.0);
  }
  Real dtau = dtau_;
  (void) dtau;
  if (global_variable::my_rank == 0) {
    if (eta_schedule_ == "exp_decay") {
      std::cout << "ID " << formulation_name_
                << " relaxation eta schedule = exp_decay, eta_initial = "
                << eta_initial_ << ", eta_final = " << eta_final_
                << ", eta_decay_tau = " << eta_decay_tau_ << std::endl;
    } else if (eta_schedule_ == "periodic_sine") {
      std::cout << "ID " << formulation_name_
                << " relaxation eta schedule = periodic_sine, eta_min = "
                << std::min(eta_initial_, eta_final_)
                << ", eta_max = " << std::max(eta_initial_, eta_final_)
                << ", eta_period_tau = " << eta_period_tau_
                << ", eta_max source = Nyquist critical damping, long wavelength = "
                << eta_long_wavelength_fraction_ << " box lengths."
                << std::endl;
    } else if (eta_schedule_ == "adaptive_curvature") {
      std::cout << "ID " << formulation_name_
                << " relaxation eta schedule = adaptive_curvature, eta_min = "
                << std::min(eta_initial_, eta_final_)
                << ", eta_max = " << std::max(eta_initial_, eta_final_)
                << ", eta_control_sweep_tau = " << eta_control_sweep_tau_
                << ", eta_control_curvature_tol = "
                << eta_control_curvature_tol_
                << ", eta_max source = Nyquist critical damping, long wavelength = "
                << eta_long_wavelength_fraction_ << " box lengths."
                << std::endl;
    } else if (eta_schedule_ == "adaptive_rate") {
      std::cout << "ID " << formulation_name_
                << " relaxation eta schedule = adaptive_rate, eta_min = "
                << std::min(eta_initial_, eta_final_)
                << ", eta_max = " << std::max(eta_initial_, eta_final_)
                << ", eta_control_tau = " << eta_control_tau_
                << ", eta_control_rate_fraction = "
                << eta_control_rate_fraction_
                << ", eta_control_power = " << eta_control_power_
                << ", eta_control_velocity_safeguard = "
                << (eta_control_velocity_safeguard_ ? "true" : "false")
                << ", eta_control_velocity_threshold = "
                << eta_control_velocity_threshold_
                << ", eta_control_velocity_width = "
                << eta_control_velocity_width_
                << ", eta_max source = half explicit damping stability ceiling, "
                << "long wavelength = "
                << eta_long_wavelength_fraction_ << " box lengths."
                << std::endl;
    } else if (eta_schedule_ == "adaptive_slope" ||
               eta_schedule_ == "adaptive_probe" ||
               eta_schedule_ == "adaptive_hill") {
      std::cout << "ID " << formulation_name_
                << " relaxation eta schedule = " << eta_schedule_
                << ", eta_min = " << std::min(eta_initial_, eta_final_)
                << ", eta_max = " << std::max(eta_initial_, eta_final_)
                << ", eta_control_tau = " << eta_control_tau_
                << ", eta_control_sweep_tau = " << eta_control_sweep_tau_
                << ", eta_control_rate_fraction = "
                << eta_control_rate_fraction_
                << ", eta_control_power = " << eta_control_power_
                << ", eta_max source = Nyquist critical damping, long wavelength = "
                << eta_long_wavelength_fraction_ << " box lengths."
                << std::endl;
    } else if (eta_schedule_ == "adaptive_scan") {
      std::cout << "ID " << formulation_name_
                << " relaxation eta schedule = adaptive_scan, eta_min = "
                << std::min(eta_initial_, eta_final_)
                << ", eta_max = " << std::max(eta_initial_, eta_final_)
                << ", eta_control_tau = " << eta_control_tau_
                << ", eta_control_sweep_tau = " << eta_control_sweep_tau_
                << ", eta_control_smooth_alpha = "
                << eta_control_smooth_alpha_
                << ", eta_control_curvature_target = "
                << eta_control_curvature_target_
                << ", eta_control_trigger_count = "
                << eta_control_trigger_count_
                << ", eta_control_max_scans = " << eta_control_max_scans_
                << ", eta_max source = Nyquist critical damping, long wavelength = "
                << eta_long_wavelength_fraction_ << " box lengths."
                << std::endl;
    }
    std::cout << "ID " << formulation_name_ << " residual excision radius = "
              << residual_excision_radius_ << std::endl;
  }

  Real initial = std::numeric_limits<Real>::max();
  Real initial_excised = std::numeric_limits<Real>::max();
  Real best_combined_metric = std::numeric_limits<Real>::max();
  int best_iter = 0;
  Real best_smoothed_all = std::numeric_limits<Real>::max();
  Real best_smoothed_excised = std::numeric_limits<Real>::max();
  int best_smoothed_all_iter = 0;
  int best_smoothed_excised_iter = 0;
  Real all_window_sum = 0.0;
  Real excised_window_sum = 0.0;
  std::vector<Real> all_window;
  std::vector<Real> excised_window;
  all_window.reserve(static_cast<std::size_t>(growth_window_));
  excised_window.reserve(static_cast<std::size_t>(growth_window_));
  int fd = pmy_pack_->pz4c->opt.fd_stencil;
  bool adaptive_curvature_eta = (eta_schedule_ == "adaptive_curvature");
  bool adaptive_rate_eta = (eta_schedule_ == "adaptive_rate");
  bool adaptive_slope_eta = (eta_schedule_ == "adaptive_slope");
  bool adaptive_probe_eta = (eta_schedule_ == "adaptive_probe");
  bool adaptive_hill_eta = (eta_schedule_ == "adaptive_hill");
  bool adaptive_scan_eta = (eta_schedule_ == "adaptive_scan");
  bool adaptive_reversible_eta =
      adaptive_slope_eta || adaptive_probe_eta || adaptive_hill_eta;
  Real eta_control_min = std::min(eta_initial_, eta_final_);
  Real eta_control_max = std::max(eta_initial_, eta_final_);
  Real eta_control_range = eta_control_max - eta_control_min;
  Real eta_control_direction = 0.0;
  Real eta_residual_nm2 = 0.0;
  Real eta_residual_nm1 = 0.0;
  Real eta_curvature_prev = 0.0;
  bool eta_have_residual_nm2 = false;
  bool eta_have_curvature_prev = false;
  Real eta_rate_residual_prev = 0.0;
  Real eta_rate_v_prev = 0.0;
  Real eta_rate_best = 0.0;
  bool eta_rate_have_prev = false;
  Real eta_opt_residual_prev = 0.0;
  Real eta_opt_q_prev = 0.0;
  Real eta_opt_eta_prev = eta_current_;
  Real eta_opt_center = static_cast<Real>(0.5)*(eta_control_min + eta_control_max);
  Real eta_opt_direction = 1.0;
  bool eta_opt_have_residual = false;
  bool eta_opt_have_q = false;
  Real eta_scan_log_res_nm2 = 0.0;
  Real eta_scan_log_res_nm1 = 0.0;
  Real eta_scan_d1 = 0.0;
  Real eta_scan_d2 = 0.0;
  bool eta_scan_have_nm1 = false;
  bool eta_scan_have_nm2 = false;
  bool eta_scan_smooth_init = false;
  bool eta_scan_increase = false;
  bool eta_scan_stop = false;
  int eta_scan_trigger_count = 0;
  int eta_scan_count = 0;
  auto zero_relax_velocities = [&]() {
    auto &indcs = pmy_pack_->pmesh->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb1 = pmy_pack_->nmb_thispack - 1;
    int nactive = nactive_vars_;
    auto &u0 = u_relax;
    auto &u1 = u_relax_tmp;
    par_for("IDCTT::ZeroRelaxVelocities", DevExeSpace(),
            0, nmb1, 0, nactive-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int q, int k, int j, int i) {
      u0(m, ID_RELAX_VDPSI + q, k,j,i) = 0.0;
      u1(m, ID_RELAX_VDPSI + q, k,j,i) = 0.0;
    });
  };
  auto reset_eta_scan_history = [&]() {
    eta_scan_have_nm1 = false;
    eta_scan_have_nm2 = false;
    eta_scan_smooth_init = false;
    eta_scan_increase = false;
    eta_scan_trigger_count = 0;
    eta_current_ = eta_control_min;
  };

  // Drive the relaxation pseudo-time integration the same way Driver::Execute
  // drives the Z4c evolution: for each pseudo-step, run stages 1..nexp_stages
  // of the AthenaK low-storage RK integrator the user picked via
  // <time>integrator (rk1 / rk2 / rk3 / rk4 etc.); each stage executes the
  // identical-ordering task list that Z4c's QueueZ4cTasks builds, on the
  // (u_relax, u_relax_tmp, u_rhs, coarse_u_relax) arrays managed here.
  int const nstages = pdriver->nexp_stages;
  for (int iter = 0; iter <= max_steps_; ++iter) {
    Real tau = static_cast<Real>(iter)*dtau_;
    if (!adaptive_curvature_eta && !adaptive_rate_eta && !adaptive_reversible_eta &&
        !adaptive_scan_eta) {
      eta_current_ = EtaAtTau(tau);
    }
    // Convergence diagnostic: the per-iteration residual_l2 reported here
    // is the residual_l2 *at the start* of iteration iter (i.e. after
    // iter-1 pseudo-steps have completed).
    if (fd == 2) ComputeResidual<2>();
    else if (fd == 3) ComputeResidual<3>();
    else ComputeResidual<4>();
    auto diag = ReduceDiagnostics((iter == 0) ? static_cast<Real>(1.0) : initial,
                                  (iter == 0) ? static_cast<Real>(1.0) : initial_excised);
    if (iter == 0) {
      initial = diag.residual_l2;
      initial_excised = diag.residual_excised_l2;
      diag.residual_rel_l2 = 1.0;
      diag.residual_excised_rel_l2 = 1.0;
    }
    Real all_metric = diag.residual_l2;
    Real excised_metric = (diag.excised_ncell > 0.0) ? diag.residual_excised_l2
                                                     : diag.residual_l2;
    bool diag_finite = (diag.finite >= 0.5) && std::isfinite(diag.residual_l2) &&
                       std::isfinite(diag.residual_max) &&
                       std::isfinite(diag.u_l2) && std::isfinite(diag.v_l2) &&
                       std::isfinite(diag.v_max);
    if (!diag_finite) {
      if (global_variable::my_rank == 0) {
        std::cout << "### FATAL ERROR in IDConformalThinSandwich::SolveRelaxation"
                  << std::endl
                  << "Non-finite " << formulation_name_
                  << " relaxation diagnostics at iter " << iter
                  << ": residual_l2 = " << diag.residual_l2
                  << ", residual_max = " << diag.residual_max
                  << ", u_l2 = " << diag.u_l2
                  << ", v_l2 = " << diag.v_l2
                  << ", v_max = " << diag.v_max << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
    if (adaptive_curvature_eta && eta_control_range > 0.0) {
      Real curvature = 0.0;
      bool curvature_hit = false;
      if (eta_have_residual_nm2) {
        curvature = (diag.residual_l2 -
                     static_cast<Real>(2.0)*eta_residual_nm1 +
                     eta_residual_nm2)/(dtau_*dtau_);
        if (eta_have_curvature_prev) {
          Real curvature_scale =
              std::max(std::max(std::fabs(curvature),
                                std::fabs(eta_curvature_prev)),
                       std::numeric_limits<Real>::min());
          curvature_hit = (curvature*eta_curvature_prev <= 0.0) ||
                          (std::fabs(curvature) <=
                           eta_control_curvature_tol_*curvature_scale);
        }
        eta_curvature_prev = curvature;
        eta_have_curvature_prev = true;
      }
      if (eta_control_direction == 0.0 && curvature_hit) {
        eta_control_direction = 1.0;
      }
      if (eta_control_direction != 0.0) {
        Real control_rate = eta_control_range/
            std::max(eta_control_sweep_tau_, std::numeric_limits<Real>::min());
        eta_current_ += eta_control_direction*control_rate*dtau_;
        if (eta_current_ >= eta_control_max) {
          eta_current_ = eta_control_max;
          eta_control_direction = -1.0;
        } else if (eta_current_ <= eta_control_min) {
          eta_current_ = eta_control_min;
          eta_control_direction = 0.0;
          eta_have_curvature_prev = false;
        }
      }
      eta_residual_nm2 = eta_residual_nm1;
      eta_residual_nm1 = diag.residual_l2;
      if (iter >= 1) eta_have_residual_nm2 = true;
    }
    if (iter % history_every_ == 0 || iter == 0 || diag.residual_l2 <= tolerance_) {
      RecordHistory(iter, tau, diag);
      if (global_variable::my_rank == 0) {
        std::cout << "ID " << formulation_name_ << " relaxation iter " << iter
                  << ": residual_l2 = " << diag.residual_l2
                  << ", residual_rel_l2 = " << diag.residual_rel_l2
                  << ", residual_excised_l2 = " << diag.residual_excised_l2
                  << ", residual_max = " << diag.residual_max
                  << ", eta = " << eta_current_
                  << ", v_l2 = " << diag.v_l2
                  << ", max_v = " << diag.v_max << std::endl;
      }
    }
    if (adaptive_rate_eta && eta_control_range > 0.0) {
      if (eta_rate_have_prev) {
        Real residual_floor = std::numeric_limits<Real>::min();
        Real v_floor = std::numeric_limits<Real>::min();
        Real q_r = -std::log(std::max(diag.residual_l2, residual_floor)/
                             std::max(eta_rate_residual_prev, residual_floor))/dtau_;
        Real q_v = std::log(std::max(diag.v_l2, v_floor)/
                            std::max(eta_rate_v_prev, v_floor))/dtau_;
        Real q_r_pos = std::max(q_r, static_cast<Real>(0.0));
        eta_rate_best = std::max(eta_rate_best, q_r_pos);
        Real q_ref = eta_control_rate_fraction_*eta_rate_best +
                     std::numeric_limits<Real>::min();
        Real ratio = q_r_pos/q_ref;
        Real residual_switch =
            static_cast<Real>(1.0)/(static_cast<Real>(1.0) +
                                    std::pow(ratio, eta_control_power_));
        Real velocity_switch = 0.0;
        if (eta_control_velocity_safeguard_) {
          Real arg = -(q_v - eta_control_velocity_threshold_)/
                     eta_control_velocity_width_;
          arg = std::max(static_cast<Real>(-60.0),
                         std::min(static_cast<Real>(60.0), arg));
          velocity_switch =
              static_cast<Real>(1.0)/(static_cast<Real>(1.0) + std::exp(arg));
        }
        Real eta_target = eta_control_min +
            eta_control_range*std::max(residual_switch, velocity_switch);
        Real response = std::min(static_cast<Real>(1.0),
            dtau_/std::max(eta_control_tau_, std::numeric_limits<Real>::min()));
        eta_current_ += response*(eta_target - eta_current_);
        eta_current_ = std::max(eta_control_min,
                                std::min(eta_control_max, eta_current_));
      }
      eta_rate_residual_prev = diag.residual_l2;
      eta_rate_v_prev = diag.v_l2;
      eta_rate_have_prev = true;
    }
    if (adaptive_reversible_eta && eta_control_range > 0.0) {
      Real eta_used = eta_current_;
      if (eta_opt_have_residual) {
        Real residual_floor = std::numeric_limits<Real>::min();
        Real q_r = -std::log(std::max(diag.residual_l2, residual_floor)/
                             std::max(eta_opt_residual_prev, residual_floor))/dtau_;
        Real q_scale = std::max(std::fabs(q_r), std::fabs(eta_opt_q_prev));
        q_scale = std::max(q_scale, static_cast<Real>(1.0e-12)/
                                    std::max(dtau_, std::numeric_limits<Real>::min()));
        Real response = std::min(static_cast<Real>(1.0),
            dtau_/std::max(eta_control_tau_, std::numeric_limits<Real>::min()));
        Real step_limit = response*eta_control_range;
        Real eta_next = eta_current_;
        auto clamp_unit = [](Real x) {
          return std::max(static_cast<Real>(-1.0),
                          std::min(static_cast<Real>(1.0), x));
        };
        if (adaptive_slope_eta) {
          if (eta_opt_have_q) {
            Real deta = eta_used - eta_opt_eta_prev;
            if (std::fabs(deta) > static_cast<Real>(1.0e-12)*eta_control_range) {
              Real slope = (q_r - eta_opt_q_prev)/deta;
              eta_next = eta_current_ +
                         step_limit*clamp_unit(slope*eta_control_range/q_scale);
            } else {
              eta_next = eta_current_ +
                         eta_opt_direction*static_cast<Real>(0.25)*step_limit;
            }
          } else {
            eta_next = eta_current_ + eta_opt_direction*step_limit;
          }
        } else if (adaptive_probe_eta) {
          constexpr Real two_pi = static_cast<Real>(6.2831853071795864769);
          Real sweep_tau = std::max(eta_control_sweep_tau_,
                                    std::numeric_limits<Real>::min());
          Real phase = two_pi*tau/sweep_tau;
          Real probe = std::sin(phase);
          if (eta_opt_have_q) {
            eta_opt_center += step_limit*
                clamp_unit((q_r - eta_opt_q_prev)*probe/q_scale);
          }
          Real amp_fraction =
              std::min(static_cast<Real>(0.45),
                       std::max(static_cast<Real>(0.0), eta_control_rate_fraction_));
          Real amplitude = amp_fraction*eta_control_range;
          eta_opt_center = std::max(eta_control_min + amplitude,
              std::min(eta_control_max - amplitude, eta_opt_center));
          Real next_phase = two_pi*(tau + dtau_)/sweep_tau;
          eta_next = eta_opt_center + amplitude*std::sin(next_phase);
        } else if (adaptive_hill_eta) {
          if (eta_opt_have_q && q_r < eta_opt_q_prev) {
            eta_opt_direction = -eta_opt_direction;
          }
          eta_next = eta_current_ + eta_opt_direction*step_limit;
        }
        if (eta_next >= eta_control_max) {
          eta_next = eta_control_max;
          if (adaptive_hill_eta || adaptive_slope_eta) eta_opt_direction = -1.0;
        } else if (eta_next <= eta_control_min) {
          eta_next = eta_control_min;
          if (adaptive_hill_eta || adaptive_slope_eta) eta_opt_direction = 1.0;
        }
        eta_current_ = eta_next;
        eta_opt_q_prev = q_r;
        eta_opt_eta_prev = eta_used;
        eta_opt_have_q = true;
      }
      eta_opt_residual_prev = diag.residual_l2;
      eta_opt_have_residual = true;
    }
    if (adaptive_scan_eta && eta_control_range > 0.0) {
      Real residual_floor = std::numeric_limits<Real>::min();
      Real y = std::log(std::max(diag.residual_l2, residual_floor));
      bool reset_scan_history = false;
      if (eta_scan_have_nm2) {
        Real d1 = (y - eta_scan_log_res_nm1)/dtau_;
        Real d2 = (y - static_cast<Real>(2.0)*eta_scan_log_res_nm1 +
                   eta_scan_log_res_nm2)/(dtau_*dtau_);
        if (eta_scan_smooth_init) {
          eta_scan_d1 += eta_control_smooth_alpha_*(d1 - eta_scan_d1);
          eta_scan_d2 += eta_control_smooth_alpha_*(d2 - eta_scan_d2);
        } else {
          eta_scan_d1 = d1;
          eta_scan_d2 = d2;
          eta_scan_smooth_init = true;
        }
        Real curvature_scale = std::max(std::fabs(eta_scan_d2),
            std::fabs(eta_scan_d1)/
            std::max(eta_control_tau_, std::numeric_limits<Real>::min()));
        curvature_scale =
            std::max(curvature_scale, std::numeric_limits<Real>::min());
        Real curvature_tol = eta_control_curvature_tol_*curvature_scale;
        if (!eta_scan_increase) {
          eta_current_ = eta_control_min;
          if (eta_scan_d2 >= -curvature_tol) {
            ++eta_scan_trigger_count;
          } else {
            eta_scan_trigger_count = 0;
          }
          if (eta_scan_trigger_count >= eta_control_trigger_count_) {
            eta_scan_increase = true;
            eta_scan_trigger_count = 0;
            if (global_variable::my_rank == 0) {
              std::cout << "ID " << formulation_name_
                        << " adaptive_scan curvature trigger at iter " << iter
                        << "; zeroing v before eta scan." << std::endl;
            }
            zero_relax_velocities();
          }
        }
        if (eta_scan_increase) {
          Real target = -eta_control_curvature_target_*std::fabs(eta_scan_d1)/
              std::max(eta_control_tau_, std::numeric_limits<Real>::min());
          Real width = std::max(std::fabs(target),
              static_cast<Real>(0.1)*std::fabs(eta_scan_d1)/
              std::max(eta_control_tau_, std::numeric_limits<Real>::min()));
          width = std::max(width, std::numeric_limits<Real>::min());
          Real drive = (eta_scan_d2 - target)/width;
          Real fraction = std::max(static_cast<Real>(0.0),
              std::min(static_cast<Real>(1.0), drive));
          Real sweep_tau = std::max(eta_control_sweep_tau_,
                                    std::numeric_limits<Real>::min());
          eta_current_ += eta_control_range*dtau_*fraction/sweep_tau;
          if (eta_current_ >= eta_control_max) {
            eta_current_ = eta_control_max;
            ++eta_scan_count;
            if (global_variable::my_rank == 0) {
              std::cout << "ID " << formulation_name_
                        << " adaptive_scan reached eta_max at iter " << iter
                        << " after scan " << eta_scan_count
                        << "; resetting eta to eta_min and zeroing v."
                        << std::endl;
            }
            zero_relax_velocities();
            reset_scan_history = true;
            if (eta_scan_count >= eta_control_max_scans_) {
              eta_scan_stop = true;
            } else {
              reset_eta_scan_history();
            }
          }
        }
      } else {
        eta_current_ = eta_control_min;
      }
      if (reset_scan_history) {
        reset_eta_scan_history();
      }
      if (!eta_scan_stop) {
        if (eta_scan_have_nm1) {
          eta_scan_log_res_nm2 = eta_scan_log_res_nm1;
          eta_scan_have_nm2 = true;
        }
        eta_scan_log_res_nm1 = y;
        eta_scan_have_nm1 = true;
      }
    }
    Real combined_metric = fmax(diag.residual_rel_l2,
                                diag.residual_excised_rel_l2);
    if (combined_metric <= best_combined_metric*(1.0 + static_cast<Real>(1.0e-12))) {
      best_combined_metric = combined_metric;
      best_iter = iter;
      Kokkos::deep_copy(DevExeSpace(), u_relax_best, u_relax);
    }
    if (adaptive_scan_eta && eta_scan_increase && !eta_scan_stop &&
        iter > best_iter &&
        combined_metric > best_combined_metric*(1.0 + growth_tolerance_)) {
      ++eta_scan_count;
      if (global_variable::my_rank == 0) {
        std::cout << "ID " << formulation_name_
                  << " adaptive_scan restarting at iter " << iter
                  << " because the residual grew beyond the trust region after scan "
                  << eta_scan_count
                  << ". Restoring best raw state from iter " << best_iter
                  << " and zeroing v." << std::endl;
      }
      Kokkos::deep_copy(DevExeSpace(), u_relax, u_relax_best);
      zero_relax_velocities();
      reset_eta_scan_history();
      if (eta_scan_count >= eta_control_max_scans_) eta_scan_stop = true;
    }
    if (eta_scan_stop) {
      if (global_variable::my_rank == 0) {
        std::cout << "ID " << formulation_name_
                  << " adaptive_scan stopping after "
                  << eta_control_max_scans_
                  << " completed eta scans. Restoring best raw state from iter "
                  << best_iter << " with combined relative residual = "
                  << best_combined_metric << "." << std::endl;
      }
      Kokkos::deep_copy(DevExeSpace(), u_relax, u_relax_best);
      break;
    }

    all_window.push_back(all_metric);
    excised_window.push_back(excised_metric);
    all_window_sum += all_metric;
    excised_window_sum += excised_metric;
    if (static_cast<int>(all_window.size()) > growth_window_) {
      all_window_sum -= all_window.front();
      excised_window_sum -= excised_window.front();
      all_window.erase(all_window.begin());
      excised_window.erase(excised_window.begin());
    }
    if (static_cast<int>(all_window.size()) == growth_window_) {
      Real smoothed_all = all_window_sum/static_cast<Real>(growth_window_);
      Real smoothed_excised = excised_window_sum/static_cast<Real>(growth_window_);
      if (smoothed_all <= best_smoothed_all) {
        best_smoothed_all = smoothed_all;
        best_smoothed_all_iter = iter;
      }
      if (smoothed_excised <= best_smoothed_excised) {
        best_smoothed_excised = smoothed_excised;
        best_smoothed_excised_iter = iter;
      }
      bool all_growing = smoothed_all >
          best_smoothed_all*(1.0 + growth_tolerance_);
      bool excised_growing = smoothed_excised >
          best_smoothed_excised*(1.0 + growth_tolerance_);
      if (stop_on_growth_ && !adaptive_scan_eta && iter >= growth_start_iter_ &&
          all_growing && excised_growing) {
        if (global_variable::my_rank == 0) {
          std::cout << "ID " << formulation_name_ << " relaxation stopping at iter " << iter
                    << " because both " << growth_window_
                    << "-step mean residuals grew. all: best = "
                    << best_smoothed_all << " at iter " << best_smoothed_all_iter
                    << ", current = " << smoothed_all
                    << "; excised: best = " << best_smoothed_excised
                    << " at iter " << best_smoothed_excised_iter
                    << ", current = " << smoothed_excised
                    << ". Restoring best raw state from iter " << best_iter
                    << " with combined relative residual = "
                    << best_combined_metric << "." << std::endl;
        }
        Kokkos::deep_copy(DevExeSpace(), u_relax, u_relax_best);
        break;
      }
    }
    if (diag.residual_l2 <= tolerance_) break;
    if (iter == max_steps_) break;

    // One pseudo-step = nstages of the same per-stage update sequence that
    // z4c::Z4c::QueueZ4cTasks builds (CopyU -> CalcRHS -> ExpRKUpdate ->
    // RestrictU -> SendU -> RecvU -> ApplyPhysicalBCs -> Prolongate), with
    // the relaxation-specific CalcRHS<NGHOST> instead of Z4c::CalcRHS, and
    // dtau (set above on the class) instead of pmesh->dt inside ExpRKUpdate.
    for (int stage = 1; stage <= nstages; ++stage) {
      InitRecv(pdriver, stage);
      CopyU(pdriver, stage);
      if (fd == 2) CalcRHS<2>(pdriver, stage);
      else if (fd == 3) CalcRHS<3>(pdriver, stage);
      else CalcRHS<4>(pdriver, stage);
      ExpRKUpdate(pdriver, stage);
      RestrictU(pdriver, stage);
      SendU(pdriver, stage);
      ClearSend(pdriver, stage);
      ClearRecv(pdriver, stage);
      RecvU(pdriver, stage);
      ApplyPhysicalBCs(pdriver, stage);
      Prolongate(pdriver, stage);
    }
  }

  if (adaptive_scan_eta) {
    if (global_variable::my_rank == 0) {
      std::cout << "ID " << formulation_name_
                << " adaptive_scan finished; restoring best raw state from iter "
                << best_iter << " with combined relative residual = "
                << best_combined_metric << "." << std::endl;
    }
    Kokkos::deep_copy(DevExeSpace(), u_relax, u_relax_best);
  }

  if (fd == 2) ComputeResidual<2>();
  else if (fd == 3) ComputeResidual<3>();
  else ComputeResidual<4>();
  auto final_diag = ReduceDiagnostics(initial, initial_excised);
  if (!std::isfinite(final_diag.residual_l2) || final_diag.residual_l2 > initial) {
    if (abort_on_reject_) {
      if (global_variable::my_rank == 0) {
        std::cout << "### FATAL ERROR in IDConformalThinSandwich::SolveRelaxation"
                  << std::endl
                  << "Rejecting " << formulation_name_
                  << " relaxation result because the residual did not "
                  << "decrease; abort_on_reject=true prevents silently "
                  << "continuing with the unsolved seed data." << std::endl;
      }
      std::exit(EXIT_FAILURE);
    } else {
      if (global_variable::my_rank == 0) {
        std::cout << "### WARNING in IDConformalThinSandwich::SolveRelaxation"
                  << std::endl
                  << "Rejecting " << formulation_name_
                  << " relaxation result because the residual did not "
                  << "decrease." << std::endl;
      }
      Kokkos::deep_copy(u_relax, 0.0);
    }
  }
  ApplySolution();
}

void IDConformalThinSandwich::ApplySolution() {
  if (formulation_ == IDConstraintFormulation::CTS) {
    ApplyCTSSolution();
  } else {
    ApplyCTTBowenYorkSolution();
  }
}

void IDConformalThinSandwich::ApplyCTTBowenYorkSolution() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  auto rel = relax_;
  auto free = free_;
  auto &admvars = pmy_pack_->padm->adm;
  Real x0_0 = pos_[0][0], x0_1 = pos_[0][1], x0_2 = pos_[0][2];
  Real x1_0 = pos_[1][0], x1_1 = pos_[1][1], x1_2 = pos_[1][2];
  Real p0_0 = mom_[0][0], p0_1 = mom_[0][1], p0_2 = mom_[0][2];
  Real p1_0 = mom_[1][0], p1_1 = mom_[1][1], p1_2 = mom_[1][2];
  Real s0_0 = spin_[0][0], s0_1 = spin_[0][1], s0_2 = spin_[0][2];
  Real s1_0 = spin_[1][0], s1_1 = spin_[1][1], s1_2 = spin_[1][2];
  auto &size = pmy_pack_->pmb->mb_size;

  par_for("IDCTT/BY::ApplySolution", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real xx = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                          size.d_view(m).x1max);
    Real yy = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                          size.d_view(m).x2max);
    Real zz = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                          size.d_view(m).x3max);
    Real psi = fmax(free.psi_singular(m,k,j,i) + rel.u(m,k,j,i),
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

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ahat;
    ahat.ZeroClear();
    AddBowenYorkAhat(xx, yy, zz, x0_0, x0_1, x0_2,
                     p0_0, p0_1, p0_2, s0_0, s0_1, s0_2, ahat);
    AddBowenYorkAhat(xx, yy, zz, x1_0, x1_1, x1_2,
                     p1_0, p1_1, p1_2, s1_0, s1_1, s1_2, ahat);
    Real invpsi2 = 1.0/psi2;
    for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b)
        admvars.vK_dd(m,a,b,k,j,i) = invpsi2*ahat(a,b);
  });

  int fd = pmy_pack_->pz4c->opt.fd_stencil;
  if (fd == 2) pmy_pack_->pz4c->ADMToZ4c<2>(pmy_pack_, nullptr);
  else if (fd == 3) pmy_pack_->pz4c->ADMToZ4c<3>(pmy_pack_, nullptr);
  else pmy_pack_->pz4c->ADMToZ4c<4>(pmy_pack_, nullptr);
  auto &z4cvars = pmy_pack_->pz4c->z4c;
  par_for("IDCTT/BY::SyncGaugeToZ4c", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    z4cvars.alpha(m,k,j,i) = admvars.alpha(m,k,j,i);
    for (int a = 0; a < 3; ++a) {
      z4cvars.beta_u(m,a,k,j,i) = admvars.beta_u(m,a,k,j,i);
      z4cvars.vB_d(m,a,k,j,i) = 0.0;
    }
  });
  pmy_pack_->pz4c->Z4cToADM(pmy_pack_);
}

void IDConformalThinSandwich::ApplyCTSSolution() {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack_->nmb_thispack;
  auto rel = relax_;
  auto free = free_;
  auto &admvars = pmy_pack_->padm->adm;
  int fd = pmy_pack_->pz4c->opt.fd_stencil;

  if (fd == 2) ComputeCTSResidual<2>();
  else if (fd == 3) ComputeCTSResidual<3>();
  else ComputeCTSResidual<4>();

  par_for("IDCTS::ApplySolution", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real psi = fmax(free.cts_base_psi(m,k,j,i) + rel.corr(m,0,k,j,i),
                    static_cast<Real>(1.0e-10));
    Real psi2 = psi*psi;
    Real psi4 = psi2*psi2;
    Real psi6 = psi4*psi2;
    Real detg = adm::SpatialDet(free.cts_g_dd(m,0,0,k,j,i),
                                free.cts_g_dd(m,0,1,k,j,i),
                                free.cts_g_dd(m,0,2,k,j,i),
                                free.cts_g_dd(m,1,1,k,j,i),
                                free.cts_g_dd(m,1,2,k,j,i),
                                free.cts_g_dd(m,2,2,k,j,i));
    admvars.psi4(m,k,j,i) = psi4*Kokkos::pow(fmax(fabs(detg), static_cast<Real>(1.0e-30)),
                                             static_cast<Real>(1.0/3.0));
    admvars.alpha(m,k,j,i) = psi6*free.cts_alpha(m,k,j,i);
    for (int a = 0; a < 3; ++a) {
      admvars.beta_u(m,a,k,j,i) = free.cts_base_beta_u(m,a,k,j,i) + rel.corr(m,a+1,k,j,i);
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        admvars.g_dd(m,a,b,k,j,i) = psi4*free.cts_g_dd(m,a,b,k,j,i);
      }
    }

    Real invpsi2 = 1.0/psi2;
    Real K = free.cts_K(m,k,j,i);
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        Real ahat_dd = 0.0;
        for (int c = 0; c < 3; ++c)
          for (int d = 0; d < 3; ++d)
            ahat_dd += free.cts_g_dd(m,a,c,k,j,i)*free.cts_g_dd(m,b,d,k,j,i)*
                        free.cts_ahat_uu(m,c,d,k,j,i);
        admvars.vK_dd(m,a,b,k,j,i) =
            invpsi2*ahat_dd + (1.0/3.0)*admvars.g_dd(m,a,b,k,j,i)*K;
      }
    }
  });

  if (fd == 2) pmy_pack_->pz4c->ADMToZ4c<2>(pmy_pack_, nullptr);
  else if (fd == 3) pmy_pack_->pz4c->ADMToZ4c<3>(pmy_pack_, nullptr);
  else pmy_pack_->pz4c->ADMToZ4c<4>(pmy_pack_, nullptr);
  auto &z4cvars = pmy_pack_->pz4c->z4c;
  par_for("IDCTS::SyncGaugeToZ4c", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    z4cvars.alpha(m,k,j,i) = admvars.alpha(m,k,j,i);
    for (int a = 0; a < 3; ++a) {
      z4cvars.beta_u(m,a,k,j,i) = admvars.beta_u(m,a,k,j,i);
      z4cvars.vB_d(m,a,k,j,i) = 0.0;
    }
  });
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
template void IDConformalThinSandwich::ComputeCTTBowenYorkResidual<2>();
template void IDConformalThinSandwich::ComputeCTTBowenYorkResidual<3>();
template void IDConformalThinSandwich::ComputeCTTBowenYorkResidual<4>();
template void IDConformalThinSandwich::ComputeCTSResidual<2>();
template void IDConformalThinSandwich::ComputeCTSResidual<3>();
template void IDConformalThinSandwich::ComputeCTSResidual<4>();
template void IDConformalThinSandwich::BuildCTSFreeData<2>();
template void IDConformalThinSandwich::BuildCTSFreeData<3>();
template void IDConformalThinSandwich::BuildCTSFreeData<4>();
template TaskStatus IDConformalThinSandwich::CalcRHS<2>(Driver*, int);
template TaskStatus IDConformalThinSandwich::CalcRHS<3>(Driver*, int);
template TaskStatus IDConformalThinSandwich::CalcRHS<4>(Driver*, int);

} // namespace z4c
