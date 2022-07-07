//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain
//

#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/spherical_grid.hpp"
//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int nlev = 5;
  bool rotate_sphere = true;
  bool fluxes = false;
  Real origin[3] = {0.,0.,3.};
  SphericalGrid A = SphericalGrid(pmbp,nlev,origin, rotate_sphere,fluxes);
  std::cout << A.polarcoord.h_view(3,0) << std::endl;

  DualArray1D<Real> radius;
  Kokkos::realloc(radius,A.nangles);
  for (int n=0; n<A.nangles; ++n) {
    radius.h_view(n) = 3 - sin(A.polarcoord.h_view(n,0));
  }

  A.SetRadius(radius);
  A.CalculateIndex();
  std::ofstream spherical_grid_output;
  spherical_grid_output.open ("/Users/hawking/Desktop/research/gr/athenak_versions/athenak_geo_mesh/build/spherical_grid_output.txt", std::ios_base::app);
  for (int i=0;i<A.nangles;++i){
    spherical_grid_output << A.cartcoord.h_view(i,0) << "\t" << A.cartcoord.h_view(i,1) << "\t" << A.cartcoord.h_view(i,2) << "\t" << A.interp_indices.h_view(i,0) 
    << "\t" << A.interp_indices.h_view(i,1) << "\t" << A.interp_indices.h_view(i,2) << "\t" << A.interp_indices.h_view(i,3) 
    << "\t" << A.polarcoord.h_view(i,0) << "\t" << A.polarcoord.h_view(i,1) <<"\n";
  }
  spherical_grid_output.close();

  std::cout<<"Unit Test for Spherical Grid Implementation Initialized."<<std::endl;

  return;
}
