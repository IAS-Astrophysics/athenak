//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shock_cloud.cpp
//! \brief Problem generator for shock-cloud problem: a planar shock impacting a single
//! spherical cloud. Input parameters are:
//!    - problem/Mach   = Mach number of incident shock
//!    - problem/drat   = density ratio of cloud to ambient
//!    - problem/beta   = ratio of Pgas/Pmag
//! The cloud radius is fixed at 1.0.  The center of the coordinate system defines the
//! center of the cloud, and should be in the middle of the cloud. The shock is initially
//! at x1=-2.0.  A typical grid domain should span x1 in [-3.0,7.0] , y and z in
//! [-2.5,2.5] (see input file).
//========================================================================================

#include <iostream>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::_()
//! \brief Problem Generator for the shock-cloud interaction problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // Read input parameters
  Real xshock = -2.0;
  Real rad    = 1.0;
  Real mach = pin->GetReal("problem","Mach");
  Real drat = pin->GetReal("problem","drat");

  // Set paramters in ambient medium ("R-state" for shock)
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  Real gm = pmbp->phydro->peos->eos_data.gamma;
  Real gm1 = gm - 1.0;
  Real dr = 1.0;
  Real pr = 1.0/gm;
  Real ur = 0.0;

  // Uses Rankine Hugoniot relations for ideal gas to initialize problem
  Real jump1 = (gm + 1.0)/(gm1 + 2.0/(mach*mach));
  Real jump2 = (2.0*gm*mach*mach - gm1)/(gm + 1.0);
  Real jump3 = 2.0*(1.0 - 1.0/(mach*mach))/(gm + 1.0);

  Real dl = dr*jump1;
  Real pl = pr*jump2;
  Real ul = ur + jump3*mach*std::sqrt(gm*pr/dr);

  // set inflow state in BoundaryValues, sync to device
  auto &u_in = pmbp->phydro->pbval_u->u_in;
  u_in.h_view(IDN,BoundaryFace::inner_x1) = dl;
  u_in.h_view(IM1,BoundaryFace::inner_x1) = dl*ul;
  u_in.h_view(IEN,BoundaryFace::inner_x1) = pl/gm1 + 0.5*dl*(ul*ul);
  u_in.template modify<HostMemSpace>();
  u_in.template sync<DevExeSpace>();

  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    par_for("pgen_cloud1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
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

      // postshock flow
      if (x1v < xshock) {
        u0(m,IDN,k,j,i) = dl;
        u0(m,IM1,k,j,i) = ul*dl;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        u0(m,IEN,k,j,i) = pl/gm1 + 0.5*dl*(ul*ul);

        // preshock ambient gas
      } else {
        u0(m,IDN,k,j,i) = dr;
        u0(m,IM1,k,j,i) = ur*dr;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        u0(m,IEN,k,j,i) = pr/gm1 + 0.5*dr*(ur*ur);
      }

      // cloud interior
      Real diag = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
      if (diag < rad) {
        u0(m,IDN,k,j,i) = dr*drat;
        u0(m,IM1,k,j,i) = ur*dr*drat;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        u0(m,IEN,k,j,i) = pr/gm1 + 0.5*dr*drat*(ur*ur);
      }
    });
  }  // End initialization of Hydro variables

  return;
}
