#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {
//---------------------------------------------------------------------------------------
//! \fn TaskStatus Z4c::Z4cBoundaryRHS
//! \brief Implement the Sommerfield Boundary conditions for z4c 
TaskStatus Z4c::Z4cBoundaryRHS(Driver *pdriver, int stage)
{
#if 0
  // Ideas:
  // 1) Implement this with 2 nested multirange. This requires definining spatial 0-D tensors.
  // 2) Implement as it is but copy code in every if (generate with python)
  printf("In Z4cBoundaryRHS\n");
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is_ = indcs.is, ie_ = indcs.ie;
  int js_ = indcs.js, je_ = indcs.je;
  int ks_ = indcs.ks, ke_ = indcs.ke;
    //For GLOOPS
  int nmb1 = pmy_pack->nmb_thispack - 1;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pm = pmy_pack->pmesh;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  int ncells1 = indcs.nx1+indcs.ng; // Align scratch buffers with variables
  auto &size = pmy_pack->pmb->mb_size;
  auto &z4c = pmy_pack->pz4c->z4c;
  auto &rhs = pmy_pack->pz4c->rhs;
  int &NDIM = pmy_pack->pz4c->NDIM;
  // 2 1D scratch array and 1 2D scratch array
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)   // 0 tensors
                  + ScrArray2D<Real>::shmem_size(3,ncells1)*3 // vectors
                  + ScrArray2D<Real>::shmem_size(9,ncells1)  // 2D tensor with no symm
                  + ScrArray2D<Real>::shmem_size(18,ncells1); // 3D tensor with symm
  int scr_level = 1;
  //par_for_outer("Sommerfeld loop",DevExeSpace(),scr_size,scr_level,0,nmb1,ks_,ke_,js_,je_,
  par_for("Sommerfeld loop",DevExeSpace(),0,nmb1,ks_,ke_,js_,je_,
  KOKKOS_LAMBDA(const int m, const int k, const int j) 
    {
    //int is, ie, js, je, ks, ke, p;
      Real idx[] = {size.d_view(m).idx1, size.d_view(m).idx2, size.d_view(m).idx3};
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
/*
      AthenaTensor<Real, TensorSymm::NONE, 1, 0> r;
      
             r.NewAthenaTensor(member, scr_level, nx1);    

      AthenaTensor<Real, TensorSymm::NONE, 1, 1> dKhat_d;
      AthenaTensor<Real, TensorSymm::NONE, 1, 1> dTheta_d;
      AthenaTensor<Real, TensorSymm::NONE, 1, 1> s_u;
      
       dKhat_d.NewAthenaTensor(member, scr_level, nx1);
      dTheta_d.NewAthenaTensor(member, scr_level, nx1);
           s_u.NewAthenaTensor(member, scr_level, nx1);
      
      AthenaTensor<Real, TensorSymm::NONE, 1, 2> dGam_du;
      
      dGam_du.NewAthenaTensor(member, scr_level, nx1);
      
      AthenaTensor<Real, TensorSymm::SYM2, 1, 3> dA_ddd;
      
       dA_ddd.NewAthenaTensor(member, scr_level, nx1);
    //bool Boundary = false;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::outflow) {
      is = is_, ie = is_;
      js = js_, je = je_;
      ks = ks_, ke = ke_;
      p = +1;
      
      // -----------------------------------------------------------------------------------
      // 1st derivatives
      //
      // Scalars
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          dKhat_d (a,i)  = p*Dx(a, 2, idx, z4c.Khat, m, k, j, i);
          dTheta_d(a,i)  = p*Dx(a, 2, idx, z4c.Theta, m, k, j, i);
        });
      }
      //par_for_inner(member, is, ie, [&](const int i) {
      //  int a = 0;
      //  if (j==11 && k ==11) printf("dKhat_d(a=%d,i=%d) = %.9g, stage = %d\n", a, i, dKhat_d(a,i), stage);
      //});
      par_for_inner(member, is, ie, [&](const int i) {
        if (j==11 && k ==11) printf("z4c.Khat(i=%d) = %.9g, stage = %d\n", i, z4c.Khat(m,k,j,i), stage);
      });
      // Vectors
      for(int a = 0; a < NDIM; ++a)
      for(int b = 0; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          dGam_du(b,a,i) = p*Dx(b, 2, idx, z4c.Gam_u, m,a,k,j,i);
        });
      }
      // Tensors
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          dA_ddd(c,a,b,i) = p*Dx(c, 2, idx, z4c.A_dd, m,a,b,k,j,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Compute pseudo-radial vector
      //
      // NOTE: this will need to be changed if the Z4c variables become vertex center
      par_for_inner(member, is, ie, [&](const int i) {
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        r(i) = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        s_u(0,i) = x1v/r(i);
        s_u(1,i) = x2v/r(i);
        s_u(2,i) = x3v/r(i);
      });

      // -----------------------------------------------------------------------------------
      // Boundary RHS for scalars
      //
      par_for_inner(member, is, ie, [&](const int i) {
        rhs.Theta(m,k,j,i) = - z4c.Theta(m,k,j,i)/r(i);
        rhs.Khat(m,k,j,i) = - SQRT2 * z4c.Khat(m,k,j,i)/r(i);
      });
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Theta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
          rhs.Khat(m,k,j,i) -= SQRT2 * s_u(a,i) * dKhat_d(a,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the Gamma's
      //
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Gam_u(m,a,k,j,i) = - z4c.Gam_u(m,a,k,j,i)/r(i);
        });
        for(int b = 0; b < NDIM; ++b) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.Gam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
          });
        }
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the A_ab
      //
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.A_dd(m,a,b,k,j,i) = - z4c.A_dd(m,a,b,k,j,i)/r(i);
        });
        for(int c = 0; c < NDIM; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.A_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
          });
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::outflow) {
      is = ie_, ie = ie_;
      js = js_, je = je_;
      ks = ks_, ke = ke_;
      p = -1;
      
      // -----------------------------------------------------------------------------------
      // 1st derivatives
      //
      // Scalars
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          dKhat_d (a,i)  = p*Dx(a, 2, idx, z4c.Khat, m, k, j, i);
          dTheta_d(a,i)  = p*Dx(a, 2, idx, z4c.Theta, m, k, j, i);
        });
      }
      //par_for_inner(member, is, ie, [&](const int i) {
      //  int a = 0;
      //  if (j==11 && k ==11) printf("dKhat_d(a=%d,i=%d) = %.9g, stage = %d\n", a, i, dKhat_d(a,i), stage);
      //});
      par_for_inner(member, is, ie, [&](const int i) {
        if (j==11 && k ==11) printf("z4c.Khat(i=%d) = %.9g, stage = %d\n", i, z4c.Khat(m,k,j,i), stage);
      });
      // Vectors
      for(int a = 0; a < NDIM; ++a)
      for(int b = 0; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          dGam_du(b,a,i) = p*Dx(b, 2, idx, z4c.Gam_u, m,a,k,j,i);
        });
      }
      // Tensors
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          dA_ddd(c,a,b,i) = p*Dx(c, 2, idx, z4c.A_dd, m,a,b,k,j,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Compute pseudo-radial vector
      //
      // NOTE: this will need to be changed if the Z4c variables become vertex center
      par_for_inner(member, is, ie, [&](const int i) {
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        r(i) = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        s_u(0,i) = x1v/r(i);
        s_u(1,i) = x2v/r(i);
        s_u(2,i) = x3v/r(i);
      });

      // -----------------------------------------------------------------------------------
      // Boundary RHS for scalars
      //
      par_for_inner(member, is, ie, [&](const int i) {
        rhs.Theta(m,k,j,i) = - z4c.Theta(m,k,j,i)/r(i);
        rhs.Khat(m,k,j,i) = - SQRT2 * z4c.Khat(m,k,j,i)/r(i);
      });
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Theta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
          rhs.Khat(m,k,j,i) -= SQRT2 * s_u(a,i) * dKhat_d(a,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the Gamma's
      //
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Gam_u(m,a,k,j,i) = - z4c.Gam_u(m,a,k,j,i)/r(i);
        });
        for(int b = 0; b < NDIM; ++b) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.Gam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
          });
        }
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the A_ab
      //
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.A_dd(m,a,b,k,j,i) = - z4c.A_dd(m,a,b,k,j,i)/r(i);
        });
        for(int c = 0; c < NDIM; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.A_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
          });
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::outflow) {
      is = is_, ie = ie_;
      js = js_, je = js_;
      ks = ks_, ke = ke_;
      p = +1;
      
      // -----------------------------------------------------------------------------------
      // 1st derivatives
      //
      // Scalars
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          dKhat_d (a,i)  = p*Dx(a, 2, idx, z4c.Khat, m, k, j, i);
          dTheta_d(a,i)  = p*Dx(a, 2, idx, z4c.Theta, m, k, j, i);
        });
      }
      //par_for_inner(member, is, ie, [&](const int i) {
      //  int a = 0;
      //  if (j==11 && k ==11) printf("dKhat_d(a=%d,i=%d) = %.9g, stage = %d\n", a, i, dKhat_d(a,i), stage);
      //});
      par_for_inner(member, is, ie, [&](const int i) {
        if (j==11 && k ==11) printf("z4c.Khat(i=%d) = %.9g, stage = %d\n", i, z4c.Khat(m,k,j,i), stage);
      });
      // Vectors
      for(int a = 0; a < NDIM; ++a)
      for(int b = 0; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          dGam_du(b,a,i) = p*Dx(b, 2, idx, z4c.Gam_u, m,a,k,j,i);
        });
      }
      // Tensors
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          dA_ddd(c,a,b,i) = p*Dx(c, 2, idx, z4c.A_dd, m,a,b,k,j,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Compute pseudo-radial vector
      //
      // NOTE: this will need to be changed if the Z4c variables become vertex center
      par_for_inner(member, is, ie, [&](const int i) {
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        r(i) = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        s_u(0,i) = x1v/r(i);
        s_u(1,i) = x2v/r(i);
        s_u(2,i) = x3v/r(i);
      });

      // -----------------------------------------------------------------------------------
      // Boundary RHS for scalars
      //
      par_for_inner(member, is, ie, [&](const int i) {
        rhs.Theta(m,k,j,i) = - z4c.Theta(m,k,j,i)/r(i);
        rhs.Khat(m,k,j,i) = - SQRT2 * z4c.Khat(m,k,j,i)/r(i);
      });
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Theta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
          rhs.Khat(m,k,j,i) -= SQRT2 * s_u(a,i) * dKhat_d(a,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the Gamma's
      //
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Gam_u(m,a,k,j,i) = - z4c.Gam_u(m,a,k,j,i)/r(i);
        });
        for(int b = 0; b < NDIM; ++b) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.Gam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
          });
        }
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the A_ab
      //
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.A_dd(m,a,b,k,j,i) = - z4c.A_dd(m,a,b,k,j,i)/r(i);
        });
        for(int c = 0; c < NDIM; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.A_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
          });
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::outflow) {
      is = is_, ie = ie_;
      js = je_, je = je_;
      ks = ks_, ke = ke_;
      p = -1;
      
      // -----------------------------------------------------------------------------------
      // 1st derivatives
      //
      // Scalars
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          dKhat_d (a,i)  = p*Dx(a, 2, idx, z4c.Khat, m, k, j, i);
          dTheta_d(a,i)  = p*Dx(a, 2, idx, z4c.Theta, m, k, j, i);
        });
      }
      par_for_inner(member, is, ie, [&](const int i) {
        int a = 0;
        if (j==11 && k ==11) printf("dKhat_d(a=%d,i=%d) = %.9g, stage = %d\n", a, i, dKhat_d(a,i), stage);
      });
      // Vectors
      for(int a = 0; a < NDIM; ++a)
      for(int b = 0; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          dGam_du(b,a,i) = p*Dx(b, 2, idx, z4c.Gam_u, m,a,k,j,i);
        });
      }
      // Tensors
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          dA_ddd(c,a,b,i) = p*Dx(c, 2, idx, z4c.A_dd, m,a,b,k,j,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Compute pseudo-radial vector
      //
      // NOTE: this will need to be changed if the Z4c variables become vertex center
      par_for_inner(member, is, ie, [&](const int i) {
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        r(i) = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        s_u(0,i) = x1v/r(i);
        s_u(1,i) = x2v/r(i);
        s_u(2,i) = x3v/r(i);
      });

      // -----------------------------------------------------------------------------------
      // Boundary RHS for scalars
      //
      par_for_inner(member, is, ie, [&](const int i) {
        rhs.Theta(m,k,j,i) = - z4c.Theta(m,k,j,i)/r(i);
        rhs.Khat(m,k,j,i) = - SQRT2 * z4c.Khat(m,k,j,i)/r(i);
      });
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Theta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
          rhs.Khat(m,k,j,i) -= SQRT2 * s_u(a,i) * dKhat_d(a,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the Gamma's
      //
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Gam_u(m,a,k,j,i) = - z4c.Gam_u(m,a,k,j,i)/r(i);
        });
        for(int b = 0; b < NDIM; ++b) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.Gam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
          });
        }
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the A_ab
      //
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.A_dd(m,a,b,k,j,i) = - z4c.A_dd(m,a,b,k,j,i)/r(i);
        });
        for(int c = 0; c < NDIM; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.A_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
          });
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::outflow) {
      is = is_, ie = ie_;
      js = js_, je = je_;
      ks = ks_, ke = ks_;
      p = +1;
      
      // -----------------------------------------------------------------------------------
      // 1st derivatives
      //
      // Scalars
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          dKhat_d (a,i)  = p*Dx(a, 2, idx, z4c.Khat, m, k, j, i);
          dTheta_d(a,i)  = p*Dx(a, 2, idx, z4c.Theta, m, k, j, i);
        });
      }
      par_for_inner(member, is, ie, [&](const int i) {
        int a = 0;
        if (j==11 && k ==11) printf("dKhat_d(a=%d,i=%d) = %.9g, stage = %d\n", a, i, dKhat_d(a,i), stage);
      });
      // Vectors
      for(int a = 0; a < NDIM; ++a)
      for(int b = 0; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          dGam_du(b,a,i) = p*Dx(b, 2, idx, z4c.Gam_u, m,a,k,j,i);
        });
      }
      // Tensors
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          dA_ddd(c,a,b,i) = p*Dx(c, 2, idx, z4c.A_dd, m,a,b,k,j,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Compute pseudo-radial vector
      //
      // NOTE: this will need to be changed if the Z4c variables become vertex center
      par_for_inner(member, is, ie, [&](const int i) {
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        r(i) = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        s_u(0,i) = x1v/r(i);
        s_u(1,i) = x2v/r(i);
        s_u(2,i) = x3v/r(i);
      });

      // -----------------------------------------------------------------------------------
      // Boundary RHS for scalars
      //
      par_for_inner(member, is, ie, [&](const int i) {
        rhs.Theta(m,k,j,i) = - z4c.Theta(m,k,j,i)/r(i);
        rhs.Khat(m,k,j,i) = - SQRT2 * z4c.Khat(m,k,j,i)/r(i);
      });
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Theta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
          rhs.Khat(m,k,j,i) -= SQRT2 * s_u(a,i) * dKhat_d(a,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the Gamma's
      //
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Gam_u(m,a,k,j,i) = - z4c.Gam_u(m,a,k,j,i)/r(i);
        });
        for(int b = 0; b < NDIM; ++b) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.Gam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
          });
        }
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the A_ab
      //
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.A_dd(m,a,b,k,j,i) = - z4c.A_dd(m,a,b,k,j,i)/r(i);
        });
        for(int c = 0; c < NDIM; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.A_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
          });
        }
      }
    }
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::outflow) {
      is = is_, ie = ie_;
      js = js_, je = je_;
      ks = ke_, ke = ke_;
      p = -1;
      
      // -----------------------------------------------------------------------------------
      // 1st derivatives
      //
      // Scalars
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          dKhat_d (a,i)  = p*Dx(a, 2, idx, z4c.Khat, m, k, j, i);
          dTheta_d(a,i)  = p*Dx(a, 2, idx, z4c.Theta, m, k, j, i);
        });
      }
      par_for_inner(member, is, ie, [&](const int i) {
        int a = 0;
        if (j==11 && k ==11) printf("dKhat_d(a=%d,i=%d) = %.9g, stage = %d\n", a, i, dKhat_d(a,i), stage);
      });
      // Vectors
      for(int a = 0; a < NDIM; ++a)
      for(int b = 0; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          dGam_du(b,a,i) = p*Dx(b, 2, idx, z4c.Gam_u, m,a,k,j,i);
        });
      }
      // Tensors
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b)
      for(int c = 0; c < NDIM; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          dA_ddd(c,a,b,i) = p*Dx(c, 2, idx, z4c.A_dd, m,a,b,k,j,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Compute pseudo-radial vector
      //
      // NOTE: this will need to be changed if the Z4c variables become vertex center
      par_for_inner(member, is, ie, [&](const int i) {
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        r(i) = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
        s_u(0,i) = x1v/r(i);
        s_u(1,i) = x2v/r(i);
        s_u(2,i) = x3v/r(i);
      });

      // -----------------------------------------------------------------------------------
      // Boundary RHS for scalars
      //
      par_for_inner(member, is, ie, [&](const int i) {
        rhs.Theta(m,k,j,i) = - z4c.Theta(m,k,j,i)/r(i);
        rhs.Khat(m,k,j,i) = - SQRT2 * z4c.Khat(m,k,j,i)/r(i);
      });
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Theta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
          rhs.Khat(m,k,j,i) -= SQRT2 * s_u(a,i) * dKhat_d(a,i);
        });
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the Gamma's
      //
      for(int a = 0; a < NDIM; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Gam_u(m,a,k,j,i) = - z4c.Gam_u(m,a,k,j,i)/r(i);
        });
        for(int b = 0; b < NDIM; ++b) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.Gam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
          });
        }
      }

      // -----------------------------------------------------------------------------------
      // Boundary RHS for the A_ab
      //
      for(int a = 0; a < NDIM; ++a)
      for(int b = a; b < NDIM; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.A_dd(m,a,b,k,j,i) = - z4c.A_dd(m,a,b,k,j,i)/r(i);
        });
        for(int c = 0; c < NDIM; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.A_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
          });
        }
      }
    }
*/
  });
#endif
  return TaskStatus::complete;
}
/*
//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cSommerfeld(
//      int const is, int const ie, int const js, int const je, int const ks, int const ke);
// \brief apply Sommerfeld BCs to the given set of points
//
KOKKOS_FUNCTION
void Z4c::Z4cSommerfeld(int const m, 
		        int const is, int const ie,
                        int const ks, int const k,
                        int const js, int const j,
			int const parity, 
                        int const scr_size,
                        int const scr_level,
                        TeamMember_t member)
  {
#if 1
  //printf("In Sommerfeld\n");
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  //
  //int ncells1 = indcs.nx1;
  //int nmb = pmy_pack->nmb_thispack;
 
  //auto &z4c = pmy_pack->pz4c->z4c;
  //auto &rhs = pmy_pack->pz4c->rhs;
  //int &NDIM = pmy_pack->pz4c->NDIM;
  //int scr_level = 1;
  //int der_ord = 1;
  //// 2 1D scratch array and 1 2D scratch array
  //size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)   // 0 tensors
  //                + ScrArray2D<Real>::shmem_size(3,ncells1)*3 // vectors
  //                + ScrArray2D<Real>::shmem_size(9,ncells1)  // 2D tensor with no symm
  //                + ScrArray2D<Real>::shmem_size(18,ncells1); // 3D tensor with symm
  //par_for_outer("Sommerfeld loop",DevExeSpace(),scr_size,scr_level,ks,ke,js,je,
  //KOKKOS_LAMBDA(TeamMember_t member, const int k, const int j) {  
    Real idx[] = {size.d_view(m).idx1, size.d_view(m).idx2, size.d_view(m).idx3};
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
   
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> r;
    
           r.NewAthenaTensor(member, scr_level, nx1);    

    AthenaTensor<Real, TensorSymm::NONE, 1, 1> dKhat_d;
    AthenaTensor<Real, TensorSymm::NONE, 1, 1> dTheta_d;
    AthenaTensor<Real, TensorSymm::NONE, 1, 1> s_u;
    
     dKhat_d.NewAthenaTensor(member, scr_level, nx1);
    dTheta_d.NewAthenaTensor(member, scr_level, nx1);
         s_u.NewAthenaTensor(member, scr_level, nx1);
    
    AthenaTensor<Real, TensorSymm::NONE, 1, 2> dGam_du;
    
    dGam_du.NewAthenaTensor(member, scr_level, nx1);
    
    AthenaTensor<Real, TensorSymm::SYM2, 1, 3> dA_ddd;
    
     dA_ddd.NewAthenaTensor(member, scr_level, nx1);
    //
    //// -----------------------------------------------------------------------------------
    //// 1st derivatives
    ////
    //// Scalars
    //for(int a = 0; a < NDIM; ++a) {
    //  par_for_inner(member, is, ie, [&](const int i) {
    //    dKhat_d (a,i)  = parity*Dx(a, der_ord, idx, z4c.Khat, m, k, j, i);
    //    dTheta_d(a,i)  = parity*Dx(a, der_ord, idx, z4c.Theta, m, k, j, i);
    //  });
    //}
    //// Vectors
    //for(int a = 0; a < NDIM; ++a)
    //for(int b = 0; b < NDIM; ++b) {
    //  par_for_inner(member, is, ie, [&](const int i) {
    //    dGam_du(b,a,i) = parity*Dx(b, der_ord, idx, z4c.Gam_u, m,a,k,j,i);
    //  });
    //}
    //// Tensors
    //for(int a = 0; a < NDIM; ++a)
    //for(int b = a; b < NDIM; ++b)
    //for(int c = 0; c < NDIM; ++c) {
    //  par_for_inner(member, is, ie, [&](const int i) {
    //    dA_ddd(c,a,b,i) = parity*Dx(c, der_ord, idx, z4c.A_dd, m,a,b,k,j,i);
    //  });
    //}

    //// -----------------------------------------------------------------------------------
    //// Compute pseudo-radial vector
    ////
    //// NOTE: this will need to be changed if the Z4c variables become vertex center
    //par_for_inner(member, is, ie, [&](const int i) {
    //  Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    //  r(i) = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    //  s_u(0,i) = x1v/r(i);
    //  s_u(1,i) = x2v/r(i);
    //  s_u(2,i) = x3v/r(i);
    //});

    //// -----------------------------------------------------------------------------------
    //// Boundary RHS for scalars
    ////
    //par_for_inner(member, is, ie, [&](const int i) {
    //  rhs.Theta(m,k,j,i) = - z4c.Theta(m,k,j,i)/r(i);
    //  rhs.Khat(m,k,j,i) = - SQRT2 * z4c.Khat(m,k,j,i)/r(i);
    //});
    //for(int a = 0; a < NDIM; ++a) {
    //  par_for_inner(member, is, ie, [&](const int i) {
    //    rhs.Theta(m,k,j,i) -= s_u(a,i) * dTheta_d(a,i);
    //    rhs.Khat(m,k,j,i) -= SQRT2 * s_u(a,i) * dKhat_d(a,i);
    //  });
    //}

    //// -----------------------------------------------------------------------------------
    //// Boundary RHS for the Gamma's
    ////
    //for(int a = 0; a < NDIM; ++a) {
    //  par_for_inner(member, is, ie, [&](const int i) {
    //    rhs.Gam_u(m,a,k,j,i) = - z4c.Gam_u(m,a,k,j,i)/r(i);
    //  });
    //  for(int b = 0; b < NDIM; ++b) {
    //    par_for_inner(member, is, ie, [&](const int i) {
    //      rhs.Gam_u(m,a,k,j,i) -= s_u(b,i) * dGam_du(b,a,i);
    //    });
    //  }
    //}

    //// -----------------------------------------------------------------------------------
    //// Boundary RHS for the A_ab
    ////
    //for(int a = 0; a < NDIM; ++a)
    //for(int b = a; b < NDIM; ++b) {
    //  par_for_inner(member, is, ie, [&](const int i) {
    //    rhs.A_dd(m,a,b,k,j,i) = - z4c.A_dd(m,a,b,k,j,i)/r(i);
    //  });
    //  for(int c = 0; c < NDIM; ++c) {
    //    par_for_inner(member, is, ie, [&](const int i) {
    //      rhs.A_dd(m,a,b,k,j,i) -= s_u(c,i) * dA_ddd(c,a,b,i);
    //    });
    //  }
    //}
  //});
#endif
  return;
}
*/
}
