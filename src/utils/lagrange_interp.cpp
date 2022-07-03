//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lagrange_interp.cpp
#include <iostream>
#include <cmath>
#include <list>
#include "utils/lagrange_interp.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"

LagrangeInterp1D::LagrangeInterp1D(MeshBlockPack *pmbp, int *meshblock_ind, int *coordinate_ind, Real *coordinate, int *axis):
  interp_weight("interp_weight",1) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  nghost = indcs.ng;
  mb_ind = *meshblock_ind;
  coord_ind = *coordinate_ind;
  coord = *coordinate;
  ax = *axis;

  // allocate memory for the interpolation weight
  // order of interpolation always match the maximal allowed by the number ghost cell
  // the stencil have size 2*(nghost+1)
  Kokkos::realloc(interp_weight,2*(nghost+1));

  // calculate weight based on the coord_ind
  LagrangeInterp1D::CalculateWeight(pmbp);
}

void LagrangeInterp1D::CalculateWeight(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &nghost_ = nghost;
  int &mb_ind_ = mb_ind;
  auto &interp_weight_ = interp_weight;
  auto &coord_ = coord;
  auto &coord_ind_ = coord_ind;
  int nx;
  Real xmin;
  Real xmax;
  int is; 
  int ie;
  if (ax==1) {
    nx = indcs.nx1;
    xmin = size.h_view(mb_ind_).x1min;
    xmax = size.h_view(mb_ind_).x1max;
    is = indcs.is;
  } else if (ax==2) {
    nx = indcs.nx2;
    xmin = size.h_view(mb_ind_).x2min;
    xmax = size.h_view(mb_ind_).x2max;
    is = indcs.js;
  } else {
    nx = indcs.nx3;
    xmin = size.h_view(mb_ind_).x3min;
    xmax = size.h_view(mb_ind_).x3max;
    is = indcs.ks;
  }
  for (int i=0;i<2*nghost+2;++i) {
    interp_weight_.h_view(i) = 1.;
    // std::cout << CellCenterX(coord_ind-nghost_+i, nx, xmin, xmax) << std::endl;
    for (int j=0;j<2*nghost+2;++j){
      if (j!=i) {
        // std::cout << CellCenterX(coord_ind, nx, xmin, xmax) << std::endl;
        // std::cout << coord_ << std::endl;
        interp_weight_.h_view(i) *= (coord_-CellCenterX(coord_ind-nghost_+j, nx, xmin, xmax))/
        (CellCenterX(coord_ind-nghost_+i, nx, xmin, xmax)-CellCenterX(coord_ind-nghost_+j, nx, xmin, xmax));
      }
    }
  }
}

Real LagrangeInterp1D::Evaluate(DualArray1D<Real> &value) {
  int &nghost_ = nghost;
  Real int_value = 0.;
  auto &interp_weight_ = interp_weight;
  for (int i=0;i<2*nghost+2;++i) {
    int_value +=interp_weight_.h_view(i) * value.h_view(i);
  }
  return int_value;
}

LagrangeInterp2D::LagrangeInterp2D(MeshBlockPack *pmbp, int *meshblock_ind, int coordinate_ind[2], Real coordinate[2], int axis[2]) {
  //Kokkos::realloc(interp_weight,2*(nghost+1),2*(nghost+1));
  auto &indcs = pmbp->pmesh->mb_indcs;
  nghost = indcs.ng;
  mb_ind = *meshblock_ind;
  coord_ind[0] = coordinate_ind[0];
  coord_ind[1] = coordinate_ind[1];
  coord[0] = coordinate[0];
  coord[1] = coordinate[1]; 
  ax[0] = axis[0];
  ax[1] = axis[1];
  // LagrangeInterp2D::CalculateWeight(pmbp);
}

Real LagrangeInterp2D::Evaluate(MeshBlockPack *pmbp, DualArray2D<Real> &value) {
  LagrangeInterp1D X1 = LagrangeInterp1D(pmbp, &mb_ind, &coord_ind[0], &coord[0], &ax[0]);
  LagrangeInterp1D X2 = LagrangeInterp1D(pmbp, &mb_ind, &coord_ind[1], &coord[1], &ax[1]);
  Real int_value = 0; 
  for (int i=0;i<2*nghost+2;++i) {
    // std::cout << X2.interp_weight.h_view(i) << std::endl;
    for (int j=0;j<2*nghost+2;++j) {
      int_value += X1.interp_weight.h_view(i)*X2.interp_weight.h_view(j) * value.h_view(i,j);
      // std::cout << int_value << std::endl;
    }
  }
  return int_value;
}

LagrangeInterp3D::LagrangeInterp3D(MeshBlockPack *pmbp, int *meshblock_ind, int coordinate_ind[3], Real coordinate[3], int axis[3]) {
  //Kokkos::realloc(interp_weight,2*(nghost+1),2*(nghost+1));
  auto &indcs = pmbp->pmesh->mb_indcs;
  nghost = indcs.ng;
  mb_ind = *meshblock_ind;
  coord_ind[0] = coordinate_ind[0];
  coord_ind[1] = coordinate_ind[1];
  coord_ind[2] = coordinate_ind[2];

  coord[0] = coordinate[0];
  coord[1] = coordinate[1]; 
  coord[2] = coordinate[2]; 

  ax[0] = axis[0];
  ax[1] = axis[1];
  ax[2] = axis[2];

  // LagrangeInterp2D::CalculateWeight(pmbp);
}

Real LagrangeInterp3D::Evaluate(MeshBlockPack *pmbp, DualArray3D<Real> &value) {
  LagrangeInterp1D X1 = LagrangeInterp1D(pmbp, &mb_ind, &coord_ind[0], &coord[0], &ax[0]);
  LagrangeInterp1D X2 = LagrangeInterp1D(pmbp, &mb_ind, &coord_ind[1], &coord[1], &ax[1]);
  LagrangeInterp1D X3 = LagrangeInterp1D(pmbp, &mb_ind, &coord_ind[2], &coord[2], &ax[2]);

  Real int_value = 0;
  for (int i=0;i<2*nghost+2;++i) {
    // std::cout << X2.interp_weight.h_view(i) << std::endl;
    for (int j=0;j<2*nghost+2;++j) {
      for (int k=0;k<2*nghost+2;++k) {
        int_value += X1.interp_weight.h_view(i)*X2.interp_weight.h_view(j)*X3.interp_weight.h_view(k) * value.h_view(i,j,k);
      }
      // std::cout << int_value << std::endl;
    }
  }
  return int_value;
}

/*
Real LagrangeInterp2D::Evaluate(DualArray2D<Real> *value) {

}
*/