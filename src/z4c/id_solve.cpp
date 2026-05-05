#include "z4c/id_solve.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>

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
    dbeta[c][c] = Dx<NGHOST>(c, idx, u, m, ID_CTS_BETAX+c, k, j, i);
    div_beta += dbeta[c][c];
  }
  for (int c = 0; c < 3; ++c) {
    for (int d = 0; d < 3; ++d) {
      if (c != d) dbeta[c][d] = Dx<NGHOST>(c, idx, u, m, ID_CTS_BETAX+d, k, j, i);
    }
  }

  Real lbeta = -(2.0/3.0)*gu[a][b]*div_beta;
  for (int c = 0; c < 3; ++c) {
    lbeta += gu[a][c]*dbeta[b][c] + gu[b][c]*dbeta[a][c];
    for (int d = 0; d < 3; ++d) {
      Real beta_d = u(m, ID_CTS_BETAX+d, k, j, i);
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
  Real gdot_tf = FreeSym(free, ID_FREE_GDOTXX, m, a, b, k, j, i)
                 - (1.0/3.0)*gdot_trace*gu[a][b];
  return (gdot_tf + lbeta)/(2.0*alpha);
}

template <int NGHOST, typename UView, typename FView>
KOKKOS_INLINE_FUNCTION
void CTSOperator(const UView &u, const FView &free, const Real idx[], int fd_stencil,
                 int m, int k, int j, int i, Real op[ID_CTS_NVAR],
                 Real diag[ID_CTS_NVAR]) {
  Real gu[3][3];
  Real gamma[3][3][3];
  MetricInverse(free, m, k, j, i, gu);
  Christoffel<NGHOST>(free, idx, m, k, j, i, gu, gamma);
  Real ric = RicciScalar<NGHOST>(free, idx, m, k, j, i, gu, gamma);

  Real psi = std::max(u(m, ID_CTS_PSI, k, j, i), static_cast<Real>(1.0e-8));
  Real dpsi[3];
  Real ddpsi[3][3];
  for (int a = 0; a < 3; ++a) {
    dpsi[a] = Dx<NGHOST>(a, idx, u, m, ID_CTS_PSI, k, j, i);
    for (int b = 0; b < 3; ++b) {
      ddpsi[a][b] = (a == b) ? Dxx<NGHOST>(a, idx, u, m, ID_CTS_PSI, k, j, i)
                             : Dxy<NGHOST>(a, b, idx, u, m, ID_CTS_PSI, k, j, i);
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
                  + 2.0*3.14159265358979323846*E/std::pow(psi, 3.0)
                  - (K*K)*std::pow(psi, 5.0)/12.0 - src;
  diag[ID_CTS_PSI] = 0.0;
  for (int a = 0; a < 3; ++a) diag[ID_CTS_PSI] += gu[a][a]*DxxCenterCoeff(fd_stencil, idx[a]);
  diag[ID_CTS_PSI] += -0.125*ric - 0.875*ahat2/std::pow(psi, 8.0)
                      - 6.0*3.14159265358979323846*E/std::pow(psi, 4.0)
                      - (5.0/12.0)*(K*K)*std::pow(psi, 4.0);

  for (int a = 0; a < 3; ++a) {
    Real div_a = 0.0;
    for (int b = 0; b < 3; ++b) {
      Real dA = 0.5*idx[b]*
        (AHatUU<NGHOST>(u, free, idx, m, k+(b==2), j+(b==1), i+(b==0), a, b)
        -AHatUU<NGHOST>(u, free, idx, m, k-(b==2), j-(b==1), i-(b==0), a, b));
      div_a += dA;
      for (int c = 0; c < 3; ++c) {
        div_a += gamma[a][b][c]*ahat[c][b] + gamma[b][b][c]*ahat[a][c];
      }
    }
    op[ID_CTS_BETAX + a] = 2.0*div_a
      - (4.0/3.0)*std::pow(psi, 6.0)*free(m, ID_FREE_DKX+a, k, j, i)
      - 16.0*3.14159265358979323846*free(m, ID_FREE_PX+a, k, j, i);

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
                Real omega, int fd_stencil) {
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
      Real op[ID_CTS_NVAR], diag[ID_CTS_NVAR];
      CTSOperator<NGHOST>(u, free, idx, fd_stencil, m, k, j, i, op, diag);
      for (int v = 0; v < ID_CTS_NVAR; ++v) {
        Real d = (std::abs(diag[v]) > 1.0e-30) ? diag[v] : ((diag[v] < 0.0) ? -1.0e-30 : 1.0e-30);
        u(m,v,k,j,i) -= omega*(op[v] - src(m,v,k,j,i))/d;
      }
      if (u(m,ID_CTS_PSI,k,j,i) < 1.0e-8) u(m,ID_CTS_PSI,k,j,i) = 1.0e-8;
    }
  });
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

template <int NGHOST, typename ViewType>
void FASRHSImpl(IDCTSMultigrid *mg, ViewType &src, const ViewType &u, const ViewType &free,
                int ll, int is, int ie, int js, int je, int ks, int ke, int fd_stencil) {
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
  if (on_host_) {
    switch (fd) {
      case 2: SmoothImpl<2>(this, u_[current_level_].h_view, src_[current_level_].h_view,
                            coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                            color, driver->omega_, fd); break;
      case 3: SmoothImpl<3>(this, u_[current_level_].h_view, src_[current_level_].h_view,
                            coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                            color, driver->omega_, fd); break;
      default: SmoothImpl<4>(this, u_[current_level_].h_view, src_[current_level_].h_view,
                             coeff_[current_level_].h_view, ll, is, ie, js, je, ks, ke,
                             color, driver->omega_, fd); break;
    }
  } else {
    switch (fd) {
      case 2: SmoothImpl<2>(this, u_[current_level_].d_view, src_[current_level_].d_view,
                            coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                            color, driver->omega_, fd); break;
      case 3: SmoothImpl<3>(this, u_[current_level_].d_view, src_[current_level_].d_view,
                            coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                            color, driver->omega_, fd); break;
      default: SmoothImpl<4>(this, u_[current_level_].d_view, src_[current_level_].d_view,
                             coeff_[current_level_].d_view, ll, is, ie, js, je, ks, ke,
                             color, driver->omega_, fd); break;
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
  if (on_host_) {
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

void IDCTSMultigrid::CalculateFASRHSPack() {
  auto *driver = static_cast<IDCTSMultigridDriver*>(pmy_driver_);
  int ll = nlevel_ - 1 - current_level_;
  int is = ngh_, ie = is + (indcs_.nx1 >> ll) - 1;
  int js = ngh_, je = js + (indcs_.nx2 >> ll) - 1;
  int ks = ngh_, ke = ks + (indcs_.nx3 >> ll) - 1;
  int fd = driver->owner_->pmy_pack_->pz4c->opt.fd_stencil;
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

IDCTSMultigridDriver::IDCTSMultigridDriver(IDConformalThinSandwich *owner,
                                           MeshBlockPack *pmbp, ParameterInput *pin)
    : MultigridDriver(pmbp, ID_CTS_NVAR), owner_(owner) {
  ncoeff_ = ID_FREE_NVAR;
  omega_ = pin->GetOrAddReal("id_solve", "omega", 0.75);
  eps_ = pin->GetOrAddReal("id_solve", "threshold", -1.0);
  niter_ = pin->GetOrAddInteger("id_solve", "niteration", 1);
  npresmooth_ = pin->GetOrAddInteger("id_solve", "npresmooth", npresmooth_);
  npostsmooth_ = pin->GetOrAddInteger("id_solve", "npostsmooth", npostsmooth_);
  full_multigrid_ = pin->GetOrAddBoolean("id_solve", "full_multigrid", false);
  fmg_ncycle_ = pin->GetOrAddInteger("id_solve", "fmg_ncycle", 1);
  fshowdef_ = pin->GetOrAddBoolean("id_solve", "show_defect", true);
  reject_worse_ = pin->GetOrAddBoolean("id_solve", "reject_worse", true);
  fsubtract_average_ = false;
  fprolongation_ = 1;

  int nghost = pin->GetOrAddInteger("id_solve", "mg_nghost", pmbp->pmesh->mb_indcs.ng);
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
    if (loc.level != locrootlevel_) continue;
    int ri = static_cast<int>(loc.lx1) + ngh;
    int rj = static_cast<int>(loc.lx2) + ngh;
    int rk = static_cast<int>(loc.lx3) + ngh;
    for (int v = 0; v < ncoeff_; ++v) {
      root_coeff_h(0, v, rk, rj, ri) = block_coeff_h(m, v, bngh, bngh, bngh);
    }
  }
  mgroot_->ModifyCoefficientLevelOnHost(nrootlevel_ - 1);
  mgroot_->SyncCoefficientLevelToDevice(nrootlevel_ - 1);
  mgroot_->SetCurrentLevel(nrootlevel_ - 1);
  mgroot_->RestrictCoefficients();
}

void IDCTSMultigridDriver::Solve(Driver *pdriver, int stage, Real dt) {
  auto &indcs = pmy_pack_->pmesh->mb_indcs;
  if (pmy_pack_->pmesh->multilevel && nreflevel_ > 0 && global_variable::my_rank == 0) {
    std::cout << "### WARNING in IDCTSMultigridDriver::Solve" << std::endl
              << "Native CTS AMR coefficient octets are not complete yet; "
              << "this run uses MeshBlock/root multigrid coefficients only." << std::endl;
  }
  PrepareForAMR();
  owner_->BuildFreeData();

  mglevels_->LoadFinestData(owner_->u_cts, 0, indcs.ng);
  mglevels_->LoadSource(owner_->u_defect, 0, indcs.ng, 0.0);
  mglevels_->LoadCoefficients(owner_->u_free, indcs.ng);
  mglevels_->RestrictCoefficients();

  SetupMultigrid(dt, false);
  TransferCoefficientsFromBlocksToRoot();

  Real initial = CalculateDefectNorm(MGNormType::l2, 0);
  owner_->RecordConstraintHistory(0, initial);
  if (fshowdef_) std::cout << "IDSolve initial CTS defect = " << initial << std::endl;

  if (full_multigrid_) SolveFMG(pdriver);
  else SolveMG(pdriver);

  Real final = CalculateDefectNorm(MGNormType::l2, 0);
  owner_->RecordConstraintHistory(niter_ >= 0 ? niter_ : -1, final);
  if (fshowdef_) std::cout << "IDSolve final CTS defect = " << final << std::endl;
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
}

void IDCTSMultigridDriver::SmoothOctet(MGOctet&, int, int) {
  // AMR octet CTS free-data support is intentionally not approximated here.
}
void IDCTSMultigridDriver::CalculateDefectOctet(MGOctet&, int) {}
void IDCTSMultigridDriver::CalculateFASRHSOctet(MGOctet&, int) {}

IDConformalThinSandwich::IDConformalThinSandwich(MeshBlockPack *pmbp, ParameterInput *pin)
    : pmy_pack_(pmbp), pmgd_(nullptr), enabled_(true), solved_(false),
      history_file_(nullptr) {
  enabled_ = pin->GetOrAddBoolean("id_solve", "enable", true);
  solve_once_ = pin->GetOrAddBoolean("id_solve", "solve_once", true);
  full_multigrid_ = pin->GetOrAddBoolean("id_solve", "full_multigrid", false);
  fill_horizon_junk_ = pin->GetOrAddBoolean("id_solve", "fill_horizon_junk", false);
  history_every_ = pin->GetOrAddInteger("id_solve", "history_every", 1);
  history_name_ = pin->GetString("job", "basename") + ".id_solve.hst";
  horizon_radius_ = pin->GetOrAddReal("id_solve", "horizon_fill_radius", -1.0);
  horizon_center_[0] = pin->GetOrAddReal("id_solve", "horizon_center_x1", 0.0);
  horizon_center_[1] = pin->GetOrAddReal("id_solve", "horizon_center_x2", 0.0);
  horizon_center_[2] = pin->GetOrAddReal("id_solve", "horizon_center_x3", 0.0);

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

TaskStatus IDConformalThinSandwich::SolveTask(Driver *pdriver, int stage) {
  if (!enabled_) return TaskStatus::complete;
  if (solve_once_ && solved_) return TaskStatus::complete;
  pmgd_->Solve(pdriver, stage, 0.0);
  solved_ = true;
  return TaskStatus::complete;
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
  Real rfill = horizon_radius_;
  Real cx = horizon_center_[0], cy = horizon_center_[1], cz = horizon_center_[2];

  par_for("IDCTS::BuildFreeData", DevExeSpace(), 0, nmb-1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real detg = adm::SpatialDet(admvars.g_dd(m,0,0,k,j,i), admvars.g_dd(m,0,1,k,j,i),
                                admvars.g_dd(m,0,2,k,j,i), admvars.g_dd(m,1,1,k,j,i),
                                admvars.g_dd(m,1,2,k,j,i), admvars.g_dd(m,2,2,k,j,i));
    detg = (detg > 0.0) ? detg : 1.0;
    Real psi4 = std::cbrt(detg);
    Real oopsi4 = 1.0/psi4;
    for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b)
        free(m, ID_FREE_GXX + SymIdx(a,b), k,j,i) = oopsi4*admvars.g_dd(m,a,b,k,j,i);

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
        K += gu[a][b]*oopsi4*admvars.vK_dd(m,a,b,k,j,i);
    free(m, ID_FREE_K, k,j,i) = K;
    free(m, ID_FREE_ALPHA, k,j,i) = admvars.alpha(m,k,j,i);
    free(m, ID_FREE_SOURCE, k,j,i) = 0.0;
    free(m, ID_FREE_MASK, k,j,i) = 1.0;
    cts(m, ID_CTS_PSI, k,j,i) = 1.0;
    for (int a = 0; a < 3; ++a) cts(m, ID_CTS_BETAX+a, k,j,i) = admvars.beta_u(m,a,k,j,i);

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
      break;
    case 3:
      BuildGammaDotAndDK<3>();
      break;
    default:
      BuildGammaDotAndDK<4>();
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
  int nmb = pmbp->nmb_thispack;
  auto &admvars = pmbp->padm->adm;
  par_for("IDCTS::BuildGammaDotAndDK", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
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
        dbeta[a][b] = Dx<NGHOST>(a, idx, u_cts, m, ID_CTS_BETAX+b, k, j, i);
      }
      div_beta += dbeta[a][a];
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        Real lbeta = -(2.0/3.0)*gu[a][b]*div_beta;
        for (int c = 0; c < 3; ++c) {
          lbeta += gu[a][c]*dbeta[b][c] + gu[b][c]*dbeta[a][c];
          for (int d = 0; d < 3; ++d) {
            Real beta_d = u_cts(m, ID_CTS_BETAX+d, k, j, i);
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
      Real psi = std::max(cts(m, ID_CTS_PSI, k,j,i), static_cast<Real>(1.0e-8));
      Real psi2 = psi*psi;
      Real psi4 = psi2*psi2;
      for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b)
          admvars.g_dd(m,a,b,k,j,i) = psi4*FreeSym(free, ID_FREE_GXX, m, a, b, k,j,i);
      admvars.psi4(m,k,j,i) = psi4;
      admvars.alpha(m,k,j,i) = free(m, ID_FREE_ALPHA, k,j,i);
      for (int a = 0; a < 3; ++a) admvars.beta_u(m,a,k,j,i) = cts(m, ID_CTS_BETAX+a,k,j,i);

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

void IDConformalThinSandwich::RecordConstraintHistory(int iter, Real defect) {
  if (history_every_ <= 0) return;
  if (history_file_ == nullptr) {
    history_file_ = std::fopen(history_name_.c_str(), "w");
    if (history_file_ != nullptr) {
      std::fprintf(history_file_, "# AthenaK native id_solve CTS history\n");
      std::fprintf(history_file_, "# [1]=iter [2]=cts_defect_l2\n");
    }
  }
  if (history_file_ != nullptr) {
    std::fprintf(history_file_, "%d %.16e\n", iter, static_cast<double>(defect));
    std::fflush(history_file_);
  }
}

} // namespace z4c
