//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_relax.cpp
//  \brief thermal relaxation test

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "driver/driver.hpp"
#include "radiation/radiation.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation relaxation test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // return if restart
  if (restart) return;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->prgeo->nangles-1);

  // get problem parameters
  Real erad = pin->GetReal("problem", "erad");
  Real temp = pin->GetReal("problem", "temp");
  Real v1 = pin->GetOrAddReal("problem", "v1", 0.0);
  Real lf = 1.0/sqrt(1.0-(SQR(v1)));

  // set primitive variables
  auto &w0 = pmbp->phydro->w0;
  par_for("pgen_shadow1",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m,IDN,k,j,i) = 1.0;
    w0(m,IVX,k,j,i) = lf*v1;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = temp;  // assumes that gm1=1
  });

  // Convert primitives to conserved
  auto &u0 = pmbp->phydro->u0;
  pmbp->phydro->peos->PrimToCons(w0, u0, 0, (n1-1), 0, (n2-1), 0, (n3-1));

  auto &norm_to_tet_ = pmbp->prad->norm_to_tet;
  auto &nh_c_ = pmbp->prad->nh_c;
  auto &tet_c_ = pmbp->prad->tet_c;
  auto &tetcov_c_ = pmbp->prad->tetcov_c;

  auto &i0 = pmbp->prad->i0;
  par_for("rad_relax",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute fluid velocity in tetrad frame
    Real uu1 = w0(m,IVX,k,j,i);
    Real uu2 = w0(m,IVY,k,j,i);
    Real uu3 = w0(m,IVZ,k,j,i);
    Real uu0 = sqrt(1.0 + SQR(uu1) + SQR(uu2) + SQR(uu3));

    Real u_tet_[4];
    u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                 norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
    u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                 norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
    u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                 norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
    u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                 norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

    // Go through each angle
    for (int n=0; n<=nang1; ++n) {
      // Calculate direction in fluid frame
      Real un_t =  (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
                    u_tet_[3]*nh_c_.d_view(n,3));

      Real n0_f =  u_tet_[0]*nh_c_.d_view(n,0) - un_t;
      Real n1_f = (-u_tet_[1]*nh_c_.d_view(n,0) + u_tet_[1]/(u_tet_[0] + 1.0)*un_t +
                   nh_c_.d_view(n,1));
      Real n2_f = (-u_tet_[2]*nh_c_.d_view(n,0) + u_tet_[2]/(u_tet_[0] + 1.0)*un_t +
                   nh_c_.d_view(n,2));
      Real n3_f = (-u_tet_[3]*nh_c_.d_view(n,0) + u_tet_[3]/(u_tet_[0] + 1.0)*un_t +
                   nh_c_.d_view(n,3));

      // Calculate intensity in fluid frame
      Real ii_f =  erad/(4.0*M_PI);

      // Calculate intensity in tetrad frame
      Real n0 = tet_c_(m,0,0,k,j,i); Real n_0 = 0.0;
      for (int d=0; d<4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }
      i0(m,n,k,j,i) = n0*n_0*ii_f/SQR(SQR(n0_f));
    }
  });

  return;
}
