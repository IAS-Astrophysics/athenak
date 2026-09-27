//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2026 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef OUTPUTS_Z4C_DIAGNOSTICS_HPP_
#define OUTPUTS_Z4C_DIAGNOSTICS_HPP_

#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "utils/finite_diff.hpp"

namespace z4c_diagnostics {
// Vacuum, Eulerian-frame curvature. epsilon^{123}=+1/sqrt(det(gamma)).
// P^i=-epsilon^{ijk} E_jl B_k^l, preserving the branch's normalization.
template<int NGHOST>
void Compute(Mesh *pm, DvceArray5D<Real> dv, int offset, int selected) {
  auto ind = pm->mb_indcs;
  auto size = pm->pmb_pack->pmb->mb_size;
  auto adm = pm->pmb_pack->padm->adm;
  int count = selected<0 ? 17 : 1;
  par_for("vacuum curvature diagnostics",DevExeSpace(),0,pm->pmb_pack->nmb_thispack-1,
      ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real idx[] = {1.0 / size.d_view(m).dx1, 1.0 / size.d_view(m).dx2,
                  1.0 / size.d_view(m).dx3};

    // Scalars
    Real detg = 0.0;
    Real K = 0.0;

    // Symmetric tensors
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> K_ud;

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dK_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_ddd;

    AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;

    // Weyl and Output Tensors
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> E_dd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> B_dd;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> P_u;

    // Initialize tensors
    for (int a = 0; a < 3; ++a) {
      P_u(a) = 0.0;
      for (int b = 0; b < 3; ++b) {
        K_ud(a, b) = 0.0;
        for (int c = 0; c < 3; ++c) {
          dg_ddd(c, a, b) = 0.0;
          dK_ddd(c, a, b) = 0.0;
          Gamma_ddd(c, a, b) = 0.0;
          Gamma_udd(c, a, b) = 0.0;
          DK_ddd(c, a, b) = 0.0;
          for (int d = c; d < 3; ++d) {
            ddg_dddd(c, d, a, b) = 0.0;
          }
        }
      }
      for (int b = a; b < 3; ++b) {
        g_uu(a, b) = 0.0;
        R_dd(a, b) = 0.0;
        E_dd(a, b) = 0.0;
        B_dd(a, b) = 0.0;
      }
    }

    // Inverse metric and Guard
    detg = adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                           adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                           adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));

    // Guard the inverse metric properly. If degenerate, set to NAN and return.
    if (!(detg > 0.0) || !Kokkos::isfinite(detg)) {
      for (int v=0; v<count; ++v) {
        dv(m, offset+v, k, j, i) = NAN;
      }
      return;
    }

    adm::SpatialInv(1.0 / detg, adm.g_dd(m, 0, 0, k, j, i),
                    adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                    adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i),
                    adm.g_dd(m, 2, 2, k, j, i), &g_uu(0, 0), &g_uu(0, 1),
                    &g_uu(0, 2), &g_uu(1, 1), &g_uu(1, 2), &g_uu(2, 2));

    // Derivatives of g and K
    for (int c = 0; c < 3; ++c) {
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          dg_ddd(c, a, b) = Dx<NGHOST>(c, idx, adm.g_dd, m, a, b, k, j, i);
          dK_ddd(c, a, b) = Dx<NGHOST>(c, idx, adm.vK_dd, m, a, b, k, j, i);
        }
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          for (int d = c; d < 3; ++d) {
            if (a == b) {
              ddg_dddd(a, b, c, d) =
                  Dxx<NGHOST>(a, idx, adm.g_dd, m, c, d, k, j, i);
            } else {
              ddg_dddd(a, b, c, d) =
                  Dxy<NGHOST>(a, b, idx, adm.g_dd, m, c, d, k, j, i);
            }
          }
        }
      }
    }

    // Christoffel symbols
    for (int c = 0; c < 3; ++c) {
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          Gamma_ddd(c, a, b) =
              0.5 * (dg_ddd(a, b, c) + dg_ddd(b, a, c) - dg_ddd(c, a, b));
        }
      }
    }

    for (int c = 0; c < 3; ++c) {
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          for (int d = 0; d < 3; ++d) {
            Gamma_udd(c, a, b) += g_uu(c, d) * Gamma_ddd(d, a, b);
          }
        }
      }
    }

    // Ricci tensor
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          for (int d = 0; d < 3; ++d) {
            for (int e = 0; e < 3; ++e) {
              R_dd(a, b) += g_uu(c, d) * Gamma_udd(e, a, c) * Gamma_ddd(e, b, d);
              R_dd(a, b) -= g_uu(c, d) * Gamma_udd(e, a, b) * Gamma_ddd(e, c, d);
            }
            R_dd(a, b) += 0.5 * g_uu(c, d) *
                          (-ddg_dddd(c, d, a, b) - ddg_dddd(a, b, c, d) +
                           ddg_dddd(a, c, b, d) + ddg_dddd(b, c, a, d));
          }
        }
      }
    }

    // Extrinsic curvature traces & Covariant Derivative
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          K_ud(a, b) += g_uu(a, c) * adm.vK_dd(m, c, b, k, j, i);
        }
      }
      K += K_ud(a, a);
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          DK_ddd(a, b, c) = dK_ddd(a, b, c);
          for (int d = 0; d < 3; ++d) {
            DK_ddd(a, b, c) -= Gamma_udd(d, a, b) * adm.vK_dd(m, d, c, k, j, i);
            DK_ddd(a, b, c) -= Gamma_udd(d, a, c) * adm.vK_dd(m, b, d, k, j, i);
          }
        }
      }
    }

    // Electric and Magnetic Weyl, Kretschmann, and Super-Poynting Flux

    // E_ij
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        E_dd(a, b) = R_dd(a, b) + K * adm.vK_dd(m, a, b, k, j, i);
        for (int c = 0; c < 3; ++c) {
          E_dd(a, b) -= adm.vK_dd(m, a, c, k, j, i) * K_ud(c, b);
        }
      }
    }

    // Levi-Civita Construction
    Real sqrt_g = std::sqrt(detg);
    Real LC[3][3][3] = {};
    LC[0][1][2] = LC[1][2][0] = LC[2][0][1] = 1.0;
    LC[0][2][1] = LC[2][1][0] = LC[1][0][2] = -1.0;

    Real eps_duu[3][3][3] = {};
    for (int a=0; a<3; ++a)
    for (int c=0; c<3; ++c)
    for (int d=0; d<3; ++d)
    for (int e=0; e<3; ++e) {
      eps_duu[a][c][d] += adm.g_dd(m,a,e,k,j,i)*LC[e][c][d]/sqrt_g;
    }

    // B_ij
    Real B_tmp[3][3] = {};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          for (int d = 0; d < 3; ++d) {
            B_tmp[a][b] += eps_duu[a][c][d] * DK_ddd(c, d, b);
          }
        }
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        B_dd(a, b) = 0.5 * (B_tmp[a][b] + B_tmp[b][a]);
      }
    }

    // Project out constraint/roundoff traces; Weyl tensors are trace-free.
    Real traceE=0.0, traceB=0.0;
    for (int a=0; a<3; ++a)
    for (int b=0; b<3; ++b) {
      traceE += g_uu(a,b)*E_dd(a,b);
      traceB += g_uu(a,b)*B_dd(a,b);
    }
    for (int a=0; a<3; ++a)
    for (int b=a; b<3; ++b) {
      E_dd(a,b) -= adm.g_dd(m,a,b,k,j,i)*traceE/3.0;
      B_dd(a,b) -= adm.g_dd(m,a,b,k,j,i)*traceB/3.0;
    }
    Real invariant=0.0;
    for (int a=0; a<3; ++a)
    for (int b=0; b<3; ++b)
    for (int c=0; c<3; ++c)
    for (int d=0; d<3; ++d) {
      invariant += 8.0*g_uu(a,c)*g_uu(b,d)*
                   (E_dd(a,b)*E_dd(c,d)-B_dd(a,b)*B_dd(c,d));
    }
    for (int a=0; a<3; ++a)
    for (int b=0; b<3; ++b)
    for (int c=0; c<3; ++c)
    for (int d=0; d<3; ++d)
    for (int e=0; e<3; ++e) {
      P_u(a) -= LC[a][b][c]/sqrt_g*E_dd(b,d)*g_uu(d,e)*B_dd(c,e);
    }
    Real norm2=0.0;
    for (int a=0; a<3; ++a)
    for (int b=0; b<3; ++b) {
      norm2 += adm.g_dd(m,a,b,k,j,i)*P_u(a)*P_u(b);
    }
    Real values[17] = {invariant,
        E_dd(0,0),E_dd(0,1),E_dd(0,2),E_dd(1,1),E_dd(1,2),E_dd(2,2),
        B_dd(0,0),B_dd(0,1),B_dd(0,2),B_dd(1,1),B_dd(1,2),B_dd(2,2),
        P_u(0),P_u(1),P_u(2),sqrt(fmax(0.0,norm2))};
    for (int v=0; v<count; ++v) {
      dv(m,offset+v,k,j,i) = values[selected<0 ? v : selected];
    }
  });
}
} // namespace z4c_diagnostics
#endif // OUTPUTS_Z4C_DIAGNOSTICS_HPP_
