//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_diffusion.cpp
//  \brief Static and dynamic diffusion test (flat space)

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "srcterms/srcterms.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation diffusion test

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
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nangles_ = pmbp->prad->prgeo->nangles;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;

  Real v1 = pin->GetReal("problem", "v1");
  Real lf = 1.0/sqrt(1.0-(SQR(v1)));
  Real uu1 = lf*v1;

  Real kappa_s_ = pmbp->prad->kappa_s;
  Real dd = 1.0/(3.0*kappa_s_);
  Real nu = pin->GetReal("problem", "nu");
  Real nusq = SQR(nu);
  Real tp0 = 6.0*uu1;

  auto &w0 = pmbp->phydro->w0;
  par_for("pgen_diffusion1",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0(m,IDN,k,j,i) = 1.0;
    w0(m,IVX,k,j,i) = uu1;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = 1.0;
  });

  // Convert primitives to conserved
  auto &u0 = pmbp->phydro->u0;
  pmbp->phydro->peos->PrimToCons(w0, u0, 0, (n1-1), 0, (n2-1), 0, (n3-1));

  auto &nh_c_ = pmbp->prad->nh_c;
  auto &norm_to_tet_ = pmbp->prad->norm_to_tet;
  auto &tet_c_ = pmbp->prad->tet_c;
  auto &tetcov_c_ = pmbp->prad->tetcov_c;

  auto &i0 = pmbp->prad->i0;
  par_for("pgen_diffusion2",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // coordinates
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

    // energy density and flux
    // NOTE(@pdmullen): there is some subtelty here. We need to find the energy density er
    // and flux fr in the comoving frame to pass to @c-white's Minerbo function...but the
    // comoving frame t' is a function of comoving position x' because we enforce that
    // the initial condition be defined at coordinate frame t=0.  We also offset t' by
    // t0' such that at coordinate frame t=0, t0' + t' > 0 at coordinate frame x=x1max.
    Real t = 0.0;
    Real tp = lf*(t-v1*x1v);
    Real xp = lf*(x1v-v1*t);
    Real er = fmax(((1./sqrt(4.*dd*nusq*(tp0+tp)+1.))
                   *exp((-1.*nusq*SQR(xp))/(4.*dd*nusq*(tp0+tp)+1.))),1.0e-20);
    Real tmp_n = (xp - v1*dd - 2.*nusq*dd*(2.*v1*dd*(tp0 - v1*xp) + xp*(v1*xp - 2.*tp0)));
    Real tmp_d = SQR(1. + 4.*nusq*dd*(tp0 - v1*xp));
    Real fr = 2.0*dd*nusq*tmp_n/tmp_d*er;

    // Calculate normalized flux in fluid frame
    Real ee_f  = er;
    Real ff1_f = fr;
    Real ff2_f = 0.0;
    Real ff3_f = 0.0;
    Real ff_f = sqrt(SQR(ff1_f) + SQR(ff2_f) + SQR(ff3_f));
    Real f_f  = ff_f/ee_f;
    Real f1_f = ff1_f/ff_f;
    Real f2_f = ff2_f/ff_f;
    Real f3_f = ff3_f/ff_f;

    Real uu0 = sqrt(1.0 + SQR(uu1));

    // Compute fluid velocity in tetrad frame
    Real u_tet_[4];
    u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1);
    u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1);
    u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1);
    u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1);

    // Go through each angle
    for (int n=0; n<nangles_; ++n) {
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
      Real fn_f = f1_f*n1_f + f2_f*n2_f + f3_f*n3_f;
      Real ii_f = 0.0;
      if (f_f <= 1.0/3.0) {
        ii_f = ee_f/(4.0*M_PI)*(1.0 + 3.0*f_f*fn_f);
      } else {
        ii_f = ee_f/(9.0*M_PI)*(fn_f - 3.0*f_f + 2.0)/SQR(1.0 - f_f);
      }

      // Calculate intensity in tetrad frame
      Real n0 = tet_c_(m,0,0,k,j,i); Real n_0 = 0.0;
      for (int d=0; d<4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }
      i0(m,n,k,j,i) = n0*n_0*ii_f/SQR(SQR(n0_f));
    }
  });

  return;
}
