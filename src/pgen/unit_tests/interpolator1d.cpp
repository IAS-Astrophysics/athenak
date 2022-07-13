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
  Real dl = 3.857143;
  Real pl = 10.33333;
  Real ul = 2.629369;
  Real vl = 0.0;
  Real wl = 0.0;

  // capture variables for kernel
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
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
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    u0(m,IDN,k,j,i) = 1.0 + 0.2*std::sin(5.0*M_PI*(x1v));
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = 1.0/gm1;
  });
  
  // u0.template modify<HostMemSpace>();
  // u0.template sync<DevExeSpace>();

  int m = 0;
  Real &x1min = size.d_view(m).x1min;
  Real &x1max = size.d_view(m).x1max;
  int nx1 = indcs.nx1;

  Real x_interp = .63481905;
  Real delta = size.h_view(m).dx1;
  std::cout << "x1min " <<x1min << std::endl;
  std::cout << "Physical boundary " << CellCenterX(0, nx1, x1min, x1max) << std::endl;
  std::cout << "ghost cell boundary " << CellCenterX(-3, nx1, x1min, x1max) << std::endl;

  int coordinate_ind = (int) std::floor((x_interp-x1min-delta/2)/delta);
  std::cout << x1min << " " << delta << std::endl;
  int axis = 1;
  LagrangeInterp1D A = LagrangeInterp1D(pmbp, &m, &coordinate_ind, &x_interp, &axis);
  std::cout << coordinate_ind << std::endl;
  DualArray1D<Real> value;
  Kokkos::realloc(value,2*nghost);
  for (int i=0; i<2*nghost+2; i++) {
    Real x1v = CellCenterX(coordinate_ind-nghost+i+1, nx1, x1min, x1max);
    value.h_view(i) = u0(m,IDN,0,0,coordinate_ind-nghost+i+is+1);
    //std::cout << x1v << "   " << value.h_view(i) << "   " << A.interp_weight.h_view(i) << std::endl;
  }

  std::cout << "value from interpolator " << A.Evaluate(value) << std::endl;

  std::cout << "expected value  " << 1.000 + 0.2000*std::sin(5.000*M_PI*(x_interp)) << std::endl;





  /*
  std::ofstream spherical_grid_output;
  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_geo_mesh/build/spherical_grid_output.txt", std::ios_base::app);
  for (int i=0;i<A.nangles;++i){
    spherical_grid_output << A.cartcoord.h_view(i,0) << "\t" << A.cartcoord.h_view(i,1) << "\t" << A.cartcoord.h_view(i,2) << "\t" << A.interp_indices.h_view(i,0) 
    << "\t" << A.interp_indices.h_view(i,1) << "\t" << A.interp_indices.h_view(i,2) << "\t" << A.interp_indices.h_view(i,3) 
    << "\t" << A.polarcoord.h_view(i,0) << "\t" << A.polarcoord.h_view(i,1) <<"\n";
  }
  spherical_grid_output.close();

  std::cout<<"Unit Test for Spherical Grid Implementation Initialized."<<std::endl;
  */
  return;
}
