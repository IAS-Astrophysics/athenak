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
#include "geodesic-grid/spherical_grid.hpp"

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
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    int nx2 = indcs.nx2;
    int nx1 = indcs.nx1;

    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // test the InterpToSphere using random 3d sin waves
    u0(m,IDN,k,j,i) = 1.0 + 0.2*std::sin(5.0*M_PI*(x1v))*std::sin(3.0*M_PI*(x2v))*std::sin(3.0*M_PI*(x3v));
  });

  int nlev = 5;
  bool rotate_sphere = true;
  bool fluxes = false;
  Real origin[3] = {0.,0.,0.};
  SphericalGrid S = SphericalGrid(pmbp,nlev,origin, rotate_sphere,fluxes);

  // set to constant radius
  Real radius = 0.2;
  S.SetRadius(radius);
  S.CalculateIndex();

  S.InterpToSphere(u0);
  int test_ind = 127;

  std::cout << "here" << std::endl;
  std::cout << S.cartcoord.h_view(test_ind,0) << "    " << S.cartcoord.h_view(test_ind,1) << "    " << S.cartcoord.h_view(test_ind,2) << std::endl; 
  std::cout << S.interp_indices.h_view(test_ind,0) << "   " << S.interp_indices.h_view(test_ind,1) << "    " << S.interp_indices.h_view(test_ind,2) << "    " << S.interp_indices.h_view(test_ind,3) << std::endl; 

  std::cout << "interpolated value  " << S.intensity.h_view(test_ind) << std::endl;
  std::cout << "analytical value  " << 1.0 + 0.2*std::sin(5.0*M_PI*(S.cartcoord.h_view(test_ind,0)))*
    std::sin(3.0*M_PI*(S.cartcoord.h_view(test_ind,1)))*std::sin(3.0*M_PI*(S.cartcoord.h_view(test_ind,2))) << std::endl;



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
