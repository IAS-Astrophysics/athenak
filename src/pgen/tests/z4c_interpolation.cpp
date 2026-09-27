//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2026 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_interpolation.cpp
//! \brief Polynomial contracts for Cartesian extraction on CPU and device backends.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "utils/cart_grid.hpp"

void ProblemGenerator::Z4cInterpolation(ParameterInput *pin, const bool restart) {
  Z4cLinearWave(pin, restart);
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  auto ind = pmy_mesh_->mb_indcs;
  auto size = pack->pmb->mb_size;
  DvceArray5D<Real> data("interpolation polynomial",pack->nmb_thispack,1,
                       ind.nx3+2*ind.ng,ind.nx2+2*ind.ng,ind.nx1+2*ind.ng);
  par_for("fill polynomial",DevExeSpace(),0,pack->nmb_thispack-1,
      0,ind.nx3+2*ind.ng-1,0,ind.nx2+2*ind.ng-1,0,ind.nx1+2*ind.ng-1,
      KOKKOS_LAMBDA(int m,int k,int j,int i) {
    auto cell=size.d_view(m);
    Real x=CellCenterX(i-ind.is,ind.nx1,cell.x1min,cell.x1max);
    Real y=CellCenterX(j-ind.js,ind.nx2,cell.x2min,cell.x2max);
    Real z=CellCenterX(k-ind.ks,ind.nx3,cell.x3min,cell.x3max);
    data(m,0,k,j,i)=1+x+2*y+3*z+x*y;
  });
  Real center[3]={0.5,0.5,0.5}, extent[3]={0.1,0.1,0.1};
  int points[3]={5,5,5};
  for (bool cheb : {false,true}) {
    for (int power : {0,2,6}) {
      CartesianGrid grid(pack,center,extent,points,cheb,power);
      for (Real radius : {0.1,0.15}) {
        for (int a=0; a<3; ++a) extent[a]=radius;
        grid.ResetCenterAndExtent(center,extent);
        grid.InterpolateToGrid(0,data);
        for (int i=0; i<5; ++i)
        for (int j=0; j<5; ++j)
        for (int k=0; k<5; ++k) {
          Real x=0.5+radius*(cheb ? cos(i*M_PI/4) : -1+0.5*i);
          Real y=0.5+radius*(cheb ? cos(j*M_PI/4) : -1+0.5*j);
          Real z=0.5+radius*(cheb ? cos(k*M_PI/4) : -1+0.5*k);
          Real expected=1+x+2*y+3*z+x*y;
          Real actual=grid.interp_vals.h_view(i,j,k);
          if (!std::isfinite(actual) || std::abs(actual-expected)>1e-8) {
            std::cerr << "Cartesian interpolation error: " << actual-expected
                      << " cheb=" << cheb << " power=" << power << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      }
    }
  }
  std::cout << "PASS Cartesian polynomial interpolation" << std::endl;
}
