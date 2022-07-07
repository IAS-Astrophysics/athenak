//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain

#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <cmath>  // sin()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"
#include "utils/lagrange_interp.hpp"
//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Interpolator unit test is set up to run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // setup problem parameters

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  int &nghost = indcs.ng;

  par_for("pgen_interp_test1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    u0(m,IDN,k,j,i) = 1.0 + 0.2*std::sin(5.0*M_PI*(x1v))*std::sin(3.0*M_PI*(x2v));
  });
  
  // u0.template modify<HostMemSpace>();
  // u0.template sync<DevExeSpace>();

  int m = 0;
  Real &x1min = size.d_view(m).x1min;
  Real &x1max = size.d_view(m).x1max;
  Real &x2min = size.d_view(m).x2min;
  Real &x2max = size.d_view(m).x2max;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;

  Real x_interp[2];
  x_interp[0] = .43180;
  x_interp[1] = .3892;

  Real delta1 = size.h_view(m).dx1;
  Real delta2 = size.h_view(m).dx2;

  int coordinate_ind[2];
  coordinate_ind[0] = (int) std::floor((x_interp[0]-x1min-delta1/2)/delta1);
  coordinate_ind[1] = (int) std::floor((x_interp[1]-x2min-delta2/2)/delta2);

  int axis[2];
  axis[0] = 1;
  axis[1] = 2;
  LagrangeInterp2D A = LagrangeInterp2D(pmbp, &m, coordinate_ind, x_interp, axis);

  DualArray2D<Real> value;
  Kokkos::realloc(value,2*(nghost+1),2*(nghost+1));
  for (int i=0; i<2*nghost+2; i++) {
    for (int j=0; j<2*nghost+2; j++) {
      value.h_view(i,j) = u0(m,IDN,0,coordinate_ind[1]-nghost+j+js,coordinate_ind[0]-nghost+i+is);
    }
  }

  std::cout << "expected value  " << 1.000 + 0.2000*std::sin(5.000*M_PI*(x_interp[0]))*std::sin(3.0*M_PI*(x_interp[1])) << std::endl;
  
  std::cout << "interpolated value " << A.Evaluate(pmbp,value) << std::endl;

  return;
}
