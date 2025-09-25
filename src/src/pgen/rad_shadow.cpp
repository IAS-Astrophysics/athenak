//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_shadow.cpp
//  \brief Shadow test for radiation

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/srcterms.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation shadow test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // set inflow state in BoundaryValues, sync to device
  auto &u_in = pmbp->phydro->pbval_u->u_in;
  auto &i_in = pmbp->prad->pbval_i->i_in;
  int nang1 = (pmbp->prad->prgeo->nangles-1);
  u_in.h_view(IDN,BoundaryFace::inner_x1) = 1.0;
  u_in.h_view(IM1,BoundaryFace::inner_x1) = 0.0;
  u_in.h_view(IM2,BoundaryFace::inner_x1) = 0.0;
  u_in.h_view(IM3,BoundaryFace::inner_x1) = 0.0;
  u_in.h_view(IEN,BoundaryFace::inner_x1) = -1.0;
  for (int n=0; n<=nang1; ++n) {
    if (n==2 || n==5) {
      i_in.h_view(n,BoundaryFace::inner_x1) = -100.0;
    } else {
      i_in.h_view(n,BoundaryFace::inner_x1) = 0.0;
    }
  }
  u_in.template modify<HostMemSpace>();
  u_in.template sync<DevExeSpace>();
  i_in.template modify<HostMemSpace>();
  i_in.template sync<DevExeSpace>();

  // error check input flags
  int geodesic_nlevel = pin->GetInteger("radiation", "nlevel");
  bool geodesic_rotate = pin->GetBoolean("radiation", "rotate_geo");
  if (geodesic_nlevel != 2 || geodesic_rotate) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Shadow test requires an unrotated, nlevel=2 geodesic mesh" << std::endl;
    exit(EXIT_FAILURE);
  }

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
  int nmb1 = (pmbp->nmb_thispack-1);

  // set intensity
  auto &i0 = pmbp->prad->i0;
  par_for("rad_beam",DevExeSpace(),0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    i0(m,n,k,j,i) = 0.0;
  });

  // set hydro field
  auto &w0 = pmbp->phydro->w0;
  par_for("pgen_shadow1",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    w0(m,IDN,k,j,i) = 1.0 + 9.0/(1.0+exp(10.0*(SQR(x1v/0.1)+SQR(x2v/0.06)-1.0)));
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = 1.0;
  });

  // Convert primitives to conserved
  auto &u0 = pmbp->phydro->u0;
  pmbp->phydro->peos->PrimToCons(w0, u0, 0, (n1-1), 0, (n2-1), 0, (n3-1));

  return;
}
