//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gauss_legendre.cpp
//  \brief Initializes a Gauss-Legendra grid to interpolate data onto

#include <cmath>
#include <iostream>
#include <list>
#include "athena_tensor.hpp"

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "gauss_legendre.hpp"
#include "utils/spherical_harm.hpp"
#include "utils/legendre_roots.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

GaussLegendreGrid::GaussLegendreGrid(MeshBlockPack *pmy_pack, int ntheta, Real rad, int nfilt):
  pmy_pack(pmy_pack),
  radius(rad),
  nfilt(nfilt),
  ntheta(ntheta),
  nangles(0),
  int_weights("int_weights",1),
  polar_pos("polar_pos",1,1),
  cart_pos("cart_pos",1,1),
  pointwise_radius("pointwise_radius",1),
  basis_functions("basis_functions",1,1,1),
  interp_indcs("interp_indcs",1,1),
  interp_wghts("interp_wghts",1,1,1),
  surface_jacobian("surface_jacobian",1,1,1),
  d_surface_jacobian("d_surface_jacobian",1,1,1,1),

  // tangent_vectors("tangent_vectors",1,1,1,1),
  // normal_oneforms("normal_oneforms",1,1,1),
  interp_vals("interp_vals",1,1) {
  
  // reallocate and set interpolation coordinates, indices, and weights
  int &ng = pmy_pack->pmesh->mb_indcs.ng;
  nangles = 2*ntheta*ntheta;

  Kokkos::realloc(int_weights,nangles);

  Kokkos::realloc(polar_pos,nangles,2);
  Kokkos::realloc(cart_pos,nangles,3);
  Kokkos::realloc(pointwise_radius,nangles);
  Kokkos::realloc(surface_jacobian,nangles,3,3);
  Kokkos::realloc(d_surface_jacobian,nangles,3,3,3);
  Kokkos::realloc(basis_functions,3,nfilt,nangles);

  Kokkos::realloc(interp_indcs,nangles,4);
  Kokkos::realloc(interp_wghts,nangles,2*ng,3);

  InitializeAngleAndWeights();
  InitializeRadius();
  EvaluateSphericalHarm();
  SetInterpolationIndices();
  SetInterpolationWeights();
  EvaluateSurfaceJacobian();
  EvaluateSurfaceJacobianDerivative();
  return;
}

//----------------------------------------------------------------------------------------
//! \brief GaussLegendreGrid destructor

GaussLegendreGrid::~GaussLegendreGrid() {
}

void GaussLegendreGrid::InitializeAngleAndWeights() {
  // calculate roots and weights for the legendre polynomial
  auto roots_and_weights = RootsAndWeights(ntheta);

  // order for storing angle, first theta, then phi
  for (int n=0; n<nangles; ++n) {
    // save the weights
    int_weights.h_view(n) =  roots_and_weights[1][n%ntheta]*M_PI/ntheta;
    // calculate theta
    polar_pos.h_view(n,0) = acos(roots_and_weights[0][n%ntheta]);
    // calculate phi
    polar_pos.h_view(n,1) = 2*M_PI/(2*ntheta)*int(n/ntheta);
  }

  // sync to device
  polar_pos.template modify<HostMemSpace>();
  polar_pos.template sync<DevExeSpace>();

  int_weights.template modify<HostMemSpace>();
  int_weights.template sync<DevExeSpace>();
}

void GaussLegendreGrid::InitializeRadius() {
  for (int n=0; n<nangles; ++n) {
    Real &theta = polar_pos.h_view(n,0);
    Real &phi = polar_pos.h_view(n,1);
    pointwise_radius.h_view(n) = radius;
    cart_pos.h_view(n,0) = radius*cos(phi)*sin(theta);
    cart_pos.h_view(n,1) = radius*sin(phi)*sin(theta);
    cart_pos.h_view(n,2) = radius*cos(theta);
  }
  pointwise_radius.template modify<HostMemSpace>();
  pointwise_radius.template sync<DevExeSpace>();
  cart_pos.template modify<HostMemSpace>();
  cart_pos.template sync<DevExeSpace>();
}

void GaussLegendreGrid::SetPointwiseRadius(DualArray1D<Real> rad_tmp, Real ctr[3]) {
  for (int n=0; n<nangles; ++n) {
    Real &theta = polar_pos.h_view(n,0);
    Real &phi = polar_pos.h_view(n,1);
    cart_pos.h_view(n,0) = rad_tmp.h_view(n)*cos(phi)*sin(theta) + ctr[0];
    cart_pos.h_view(n,1) = rad_tmp.h_view(n)*sin(phi)*sin(theta) + ctr[1];
    cart_pos.h_view(n,2) = rad_tmp.h_view(n)*cos(theta) + ctr[2];
    pointwise_radius.h_view(n) = rad_tmp.h_view(n);
  }
  // sync dual arrays
  cart_pos.template modify<HostMemSpace>();
  cart_pos.template sync<DevExeSpace>();
  pointwise_radius.template modify<HostMemSpace>();
  pointwise_radius.template sync<DevExeSpace>();

  // reset interpolation indices and weights
  SetInterpolationIndices();
  SetInterpolationWeights();
  EvaluateSurfaceJacobian();
  EvaluateSurfaceJacobianDerivative();
  return;
}

void GaussLegendreGrid::EvaluateSphericalHarm() {
  for (int a=0; a<nfilt; a++) {
    int l = (int) sqrt(a);
    int m = (int) (a-l*l-l);
    for (int n=0; n<nangles; ++n) {
      Real &theta = polar_pos.h_view(n,0);
      Real &phi = polar_pos.h_view(n,1);
      basis_functions.h_view(0,a,n) = RealSphericalHarm(l, m, theta, phi);
      basis_functions.h_view(1,a,n) = RealSphericalHarm_dtheta(l, m, theta, phi);
      basis_functions.h_view(2,a,n) = RealSphericalHarm_dphi(l, m, theta, phi);
    }
  }
  // sync dual arrays
  basis_functions.template modify<HostMemSpace>();
  basis_functions.template sync<DevExeSpace>();
  return;
}

// gauss-legendre integral assuming unit sphere
Real GaussLegendreGrid::Integrate(DualArray1D<Real> integrand) {
  Real value = 0.;
  for (int n=0; n<nangles; ++n) {
    value += integrand.h_view(n)*int_weights.h_view(n);
  }
  return value;
}

// Integrate tensors of rank 0
Real GaussLegendreGrid::Integrate(AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> integrand) {
  Real value = 0.;
  for (int n=0; n<nangles; ++n) {
    value += integrand(n)*int_weights.h_view(n);
  }
  return value;
}

// calculate spectral representation, maybe change into inline function later
DualArray1D<Real> GaussLegendreGrid::SpatialToSpectral(DualArray1D<Real> scalar_function) {
  DualArray1D<Real> spectral;
  DualArray1D<Real> integrand;
  Kokkos::realloc(spectral,nfilt);
  Kokkos::realloc(integrand,nangles);

  for (int i=0; i<nfilt; ++i) {
    for (int n=0; n<nangles; ++n) {
      integrand.h_view(n) = scalar_function.h_view(n)*basis_functions.h_view(0,i,n);
    }
    spectral.h_view(i) = Integrate(integrand);
  }
  return spectral;
}

DualArray1D<Real> GaussLegendreGrid::SpatialToSpectral(AthenaSurfaceTensor<Real,TensorSymm::NONE,3,0> scalar_function) {
  DualArray1D<Real> spectral;
  DualArray1D<Real> integrand;
  Kokkos::realloc(spectral,nfilt);
  Kokkos::realloc(integrand,nangles);

  for (int i=0; i<nfilt; ++i) {
    for (int n=0; n<nangles; ++n) {
      integrand.h_view(n) = scalar_function(n)*basis_functions.h_view(0,i,n);
    }
    spectral.h_view(i) = Integrate(integrand);
  }
  return spectral;
}

DualArray1D<Real> GaussLegendreGrid::SpectralToSpatial(DualArray1D<Real> scalar_spectrum) {
  DualArray1D<Real> scalar_function;
  Kokkos::realloc(scalar_function,nangles);
  for (int i=0; i<nfilt;++i) {
    for (int n=0; n<nangles; ++n) {
      scalar_function.h_view(n) += scalar_spectrum.h_view(i)*basis_functions.h_view(0,i,n);
    }
  }
  return scalar_function;
}

DualArray1D<Real> GaussLegendreGrid::ThetaDerivative(DualArray1D<Real> scalar_function) {
  // first find spectral representation
  DualArray1D<Real> spectral;
  Kokkos::realloc(spectral,nfilt);
  spectral = SpatialToSpectral(scalar_function);
  // calculate theta derivative
  DualArray1D<Real> scalar_function_dtheta;
  Kokkos::realloc(scalar_function_dtheta,nangles);
  for (int i=0; i<nfilt; ++i) {
    for (int n=0; n<nangles; ++n) {
      scalar_function_dtheta.h_view(n) += spectral.h_view(i)*basis_functions.h_view(1,i,n);
    }
  }
  return scalar_function_dtheta;
}

DualArray1D<Real> GaussLegendreGrid::PhiDerivative(DualArray1D<Real> scalar_function) {
  // first find spectral representation
  DualArray1D<Real> spectral;
  Kokkos::realloc(spectral,nfilt);
  spectral = SpatialToSpectral(scalar_function);
  // calculate theta derivative
  DualArray1D<Real> scalar_function_dphi;
  Kokkos::realloc(scalar_function_dphi,nangles);
  for (int i=0; i<nfilt; ++i) {
    for (int n=0; n<nangles; ++n) {
      scalar_function_dphi.h_view(n) += spectral.h_view(i)*basis_functions.h_view(2,i,n);
    }
  }
  return scalar_function_dphi;
}


//----------------------------------------------------------------------------------------
//! \fn void GaussLegendreGrid::SetInterpolationIndices
//! \brief determine which MeshBlocks and MeshBlock zones therein that will be used in
//         interpolation onto the sphere

void GaussLegendreGrid::SetInterpolationIndices() {
  auto &size = pmy_pack->pmb->mb_size;

  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang1 = nangles - 1;
  auto &rcoord = cart_pos;
  auto &iindcs = interp_indcs;
  for (int n=0; n<=nang1; ++n) {
    // indices default to -1 if angle does not reside in this MeshBlockPack
    iindcs.h_view(n,0) = -1;
    iindcs.h_view(n,1) = -1;
    iindcs.h_view(n,2) = -1;
    iindcs.h_view(n,3) = -1;
    for (int m=0; m<=nmb1; ++m) {
      // extract MeshBlock bounds
      Real &x1min = size.h_view(m).x1min;
      Real &x1max = size.h_view(m).x1max;
      Real &x2min = size.h_view(m).x2min;
      Real &x2max = size.h_view(m).x2max;
      Real &x3min = size.h_view(m).x3min;
      Real &x3max = size.h_view(m).x3max;

      // extract MeshBlock grid cell spacings
      Real &dx1 = size.h_view(m).dx1;
      Real &dx2 = size.h_view(m).dx2;
      Real &dx3 = size.h_view(m).dx3;

      // save MeshBlock and zone indicies for nearest position to spherical patch center
      // if this angle position resides in this MeshBlock
      if ((rcoord.h_view(n,0) >= x1min && rcoord.h_view(n,0) <= x1max) &&
          (rcoord.h_view(n,1) >= x2min && rcoord.h_view(n,1) <= x2max) &&
          (rcoord.h_view(n,2) >= x3min && rcoord.h_view(n,2) <= x3max)) {
        iindcs.h_view(n,0) = m;
        iindcs.h_view(n,1) = static_cast<int>(std::floor((rcoord.h_view(n,0)-
                                                          (x1min+dx1/2.0))/dx1));
        iindcs.h_view(n,2) = static_cast<int>(std::floor((rcoord.h_view(n,1)-
                                                          (x2min+dx2/2.0))/dx2));
        iindcs.h_view(n,3) = static_cast<int>(std::floor((rcoord.h_view(n,2)-
                                                          (x3min+dx3/2.0))/dx3));
      }
    }
  }

  // sync dual arrays
  interp_indcs.template modify<HostMemSpace>();
  interp_indcs.template sync<DevExeSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GaussLegendreGrid::SetInterpolationWeights
//! \brief set weights used by Lagrangian interpolation

void GaussLegendreGrid::SetInterpolationWeights() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &ng = indcs.ng;

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  for (int n=0; n<nangles; ++n) {
    // extract indices
    int &ii0 = iindcs.h_view(n,0);
    int &ii1 = iindcs.h_view(n,1);
    int &ii2 = iindcs.h_view(n,2);
    int &ii3 = iindcs.h_view(n,3);

    if (ii0==-1) {  // angle not on this rank
      for (int i=0; i<2*ng; ++i) {
        iwghts.h_view(n,i,0) = 0.0;
        iwghts.h_view(n,i,1) = 0.0;
        iwghts.h_view(n,i,2) = 0.0;
      }
    } else {
      // extract spherical grid positions
      Real &x0 = cart_pos.h_view(n,0);
      Real &y0 = cart_pos.h_view(n,1);
      Real &z0 = cart_pos.h_view(n,2);

      // extract MeshBlock bounds
      Real &x1min = size.h_view(ii0).x1min;
      Real &x1max = size.h_view(ii0).x1max;
      Real &x2min = size.h_view(ii0).x2min;
      Real &x2max = size.h_view(ii0).x2max;
      Real &x3min = size.h_view(ii0).x3min;
      Real &x3max = size.h_view(ii0).x3max;

      // set interpolation weights
      for (int i=0; i<2*ng; ++i) {
        iwghts.h_view(n,i,0) = 1.;
        iwghts.h_view(n,i,1) = 1.;
        iwghts.h_view(n,i,2) = 1.;
        for (int j=0; j<2*ng; ++j) {
          if (j != i) {
            Real x1vpi1 = CellCenterX(ii1-ng+i+1, indcs.nx1, x1min, x1max);
            Real x1vpj1 = CellCenterX(ii1-ng+j+1, indcs.nx1, x1min, x1max);
            iwghts.h_view(n,i,0) *= (x0-x1vpj1)/(x1vpi1-x1vpj1);
            Real x2vpi1 = CellCenterX(ii2-ng+i+1, indcs.nx2, x2min, x2max);
            Real x2vpj1 = CellCenterX(ii2-ng+j+1, indcs.nx2, x2min, x2max);
            iwghts.h_view(n,i,1) *= (y0-x2vpj1)/(x2vpi1-x2vpj1);
            Real x3vpi1 = CellCenterX(ii3-ng+i+1, indcs.nx3, x3min, x3max);
            Real x3vpj1 = CellCenterX(ii3-ng+j+1, indcs.nx3, x3min, x3max);
            iwghts.h_view(n,i,2) *= (z0-x3vpj1)/(x3vpi1-x3vpj1);
          }
        }
      }
    }
  }

  // sync dual arrays
  interp_wghts.template modify<HostMemSpace>();
  interp_wghts.template sync<DevExeSpace>();

  return;
}
//----------------------------------------------------------------------------------------
//! \fn void GaussLegendreGrid::InterpolateToSphere
//! \brief interpolate Cartesian data to surface of sphere

void GaussLegendreGrid::InterpolateToSphere(int nvars, DvceArray5D<Real> &val) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int &ng = indcs.ng;
  int nang1 = nangles - 1;
  int nvar1 = nvars - 1;

  // reallocate container
  Kokkos::realloc(interp_vals,nangles,nvars);
  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals = interp_vals;
  par_for("int2sph",DevExeSpace(),0,nang1,0,nvar1,
  KOKKOS_LAMBDA(int n, int v) {
    int &ii0 = iindcs.d_view(n,0);
    int &ii1 = iindcs.d_view(n,1);
    int &ii2 = iindcs.d_view(n,2);
    int &ii3 = iindcs.d_view(n,3);

    if (ii0==-1) {  // angle not on this rank
      ivals.d_view(n,v) = 0.0;
    } else {
      Real int_value = 0.0;
      for (int i=0; i<2*ng; i++) {
        for (int j=0; j<2*ng; j++) {
          for (int k=0; k<2*ng; k++) {
            Real iwght = iwghts.d_view(n,i,0)*iwghts.d_view(n,j,1)*iwghts.d_view(n,k,2);
            int_value += iwght*val(ii0,v,ii3-(ng-k-ks)+1,ii2-(ng-j-js)+1,ii1-(ng-i-is)+1);
          }
        }
      }
      ivals.d_view(n,v) = int_value;
    }
  });

  // sync dual arrays
  interp_vals.template modify<DevExeSpace>();
  interp_vals.template sync<HostMemSpace>();

  return;
}

AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,2> GaussLegendreGrid::InterpolateToSphere(AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int &ng = indcs.ng;
  int nang1 = nangles - 1;
  int nvar1 = 5;

  AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,2> tensor_on_sphere;
  tensor_on_sphere.NewAthenaSurfaceTensor(nangles);

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals = tensor_on_sphere;
  par_for("int2sph",DevExeSpace(),0,nang1,0,nvar1,
  KOKKOS_LAMBDA(int n, int v) {
    int &ii0 = iindcs.d_view(n,0);
    int &ii1 = iindcs.d_view(n,1);
    int &ii2 = iindcs.d_view(n,2);
    int &ii3 = iindcs.d_view(n,3);

    // Tensor indices
    int v1;
    int v2;
    if (v<=2) {
      v1 = 0;
      v2 = v;
    } else if (v<=4) {
      v1 = 1;
      v2 = v-2;
    } else {
      v1 = 2;
      v2 = 2;
    }

    if (ii0==-1) {  // angle not on this rank
      ivals(v1,v2,n) = 0.0;
    } else {
      Real int_value = 0.0;
      for (int i=0; i<2*ng; i++) {
        for (int j=0; j<2*ng; j++) {
          for (int k=0; k<2*ng; k++) {
            Real iwght = iwghts.d_view(n,i,0)*iwghts.d_view(n,j,1)*iwghts.d_view(n,k,2);
            int_value += iwght*g_dd(ii0,v1,v2,ii3-(ng-k-ks)+1,ii2-(ng-j-js)+1,ii1-(ng-i-is)+1);
          }
        }
      }
      ivals(v1,v2,n) = int_value;
      // std::cout << n << "   " << v1 << "  " << v2 << std::endl;
    }
  });

  return tensor_on_sphere;
}


//----------------------------------------------------------------------------------------
//! \fn void GaussLegendreGrid::InterpolateToSphere
//! \brief interpolate Rank3 tensors to surface of sphere

AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,3> 
GaussLegendreGrid::InterpolateToSphere(AthenaTensor<Real, TensorSymm::SYM2, 3, 3> &dg_ddd) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  int &ng = indcs.ng;
  int nang1 = nangles - 1;
  int nvar1 = 3;
  int nvar2 = 6;

  // reallocate container

  AthenaSurfaceTensor<Real,TensorSymm::SYM2,3,3> tensor_on_sphere;

  tensor_on_sphere.NewAthenaSurfaceTensor(nangles);

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals = tensor_on_sphere;
  par_for("int2sph",DevExeSpace(),0,nang1,0,nvar1-1,0,nvar2-1,
  KOKKOS_LAMBDA(int n, int u, int v) {
    int &ii0 = iindcs.d_view(n,0);
    int &ii1 = iindcs.d_view(n,1);
    int &ii2 = iindcs.d_view(n,2);
    int &ii3 = iindcs.d_view(n,3);

    // Tensor indices
    int v1;
    int v2;
    if (v<=2) {
      v1 = 0;
      v2 = v;
    } else if (v<=4) {
      v1 = 1;
      v2 = v-2;
    } else {
      v1 = 2;
      v2 = 2;
    }

    if (ii0==-1) {  // angle not on this rank
      ivals(u,v1,v2,n) = 0.0;
    } else {
      Real int_value = 0.0;
      for (int i=0; i<2*ng; i++) {
        for (int j=0; j<2*ng; j++) {
          for (int k=0; k<2*ng; k++) {
            Real iwght = iwghts.d_view(n,i,0)*iwghts.d_view(n,j,1)*iwghts.d_view(n,k,2);
            int_value += iwght*dg_ddd(ii0,u,v1,v2,ii3-(ng-k-ks)+1,ii2-(ng-j-js)+1,ii1-(ng-i-is)+1);
          }
        }
      }
      ivals(u,v1,v2,n) = int_value;
    }
  });

  return tensor_on_sphere;
}


// Jacobian matrix to transform vector to Cartesian basis
// first index r theta phi, second index x,y,z
void GaussLegendreGrid::EvaluateSurfaceJacobian() {
  for (int n=0; n<nangles; ++n) {
    Real x = cart_pos.h_view(n,0);
    Real y = cart_pos.h_view(n,1);
    Real z = cart_pos.h_view(n,2);
    Real r = pointwise_radius.h_view(n);
    Real x2plusy2 = x*x + y*y;
    Real sqrt_x2plusy2 = sqrt(x2plusy2);

    // for x component
    surface_jacobian.h_view(n,0,0) = x/r;
    surface_jacobian.h_view(n,1,0) = x*z/sqrt_x2plusy2/r/r;
    surface_jacobian.h_view(n,2,0) = -y/x2plusy2;
  
    // for y component
    surface_jacobian.h_view(n,0,1) = y/r;
    surface_jacobian.h_view(n,1,1) = y*z/sqrt_x2plusy2/r/r;
    surface_jacobian.h_view(n,2,1) = x/x2plusy2;

    // for z component
    surface_jacobian.h_view(n,0,2) = z/r;
    surface_jacobian.h_view(n,1,2) = -sqrt_x2plusy2/r/r;
    surface_jacobian.h_view(n,2,2) = 0;
  }
  surface_jacobian.template modify<HostMemSpace>();
  surface_jacobian.template sync<DevExeSpace>();
}


// Analytical derivative of the Jacobian matrix
// first index x,y,z, second index r,theta,phi, third index x,y,z
void GaussLegendreGrid::EvaluateSurfaceJacobianDerivative() {
  for (int n=0; n<nangles; ++n) {
    Real x = cart_pos.h_view(n,0);
    Real y = cart_pos.h_view(n,1);
    Real z = cart_pos.h_view(n,2);
    Real r = pointwise_radius.h_view(n);

    Real x2 = x*x;
    Real y2 = y*y;
    Real z2 = z*z;

    Real r2 = r*r;
    Real r3 = r2*r;
    Real r4 = r3*r;
    Real rxy2 = x*x + y*y;
    Real rxy = sqrt(rxy2);
    Real rxy3 = rxy2*rxy;
    Real rxy4 = rxy3*rxy;
    //****************** Partial x ********************
    // for x component
    d_surface_jacobian.h_view(n,0,0,0) = (r2-x2)/r3;
    d_surface_jacobian.h_view(n,0,1,0) = (-2*rxy2*x2 + r2*(rxy2-x2))*z/(r4*rxy3);
    d_surface_jacobian.h_view(n,0,2,0) = 2*x*y/rxy4;
  
    // for y component
    d_surface_jacobian.h_view(n,0,0,1) = -x*y/r3;
    d_surface_jacobian.h_view(n,0,1,1) = -(r2 + 2*rxy2)*x*y*z/(r4*rxy3);
    d_surface_jacobian.h_view(n,0,2,1) = (rxy2 - 2*x2)/rxy4;

    // for z component
    d_surface_jacobian.h_view(n,0,0,2) = -x*z/r3;
    d_surface_jacobian.h_view(n,0,1,2) = -x/(r2*rxy)+2*rxy*x/r4;
    d_surface_jacobian.h_view(n,0,2,2) = 0;

    //****************** Partial y ********************
    // for x component
    d_surface_jacobian.h_view(n,1,0,0) = -x*y/r3;
    d_surface_jacobian.h_view(n,1,1,0) = -(r2 + 2*rxy2)*x*y*z/(r4*rxy3);
    d_surface_jacobian.h_view(n,1,2,0) = -(rxy2 - 2*y2)/rxy4;

    // for y component
    d_surface_jacobian.h_view(n,1,0,1) = (r2-y2)/r3;
    d_surface_jacobian.h_view(n,1,1,1) = (-2*rxy2*y2 + r2*(rxy2-y2))*z/(r4*rxy3);
    d_surface_jacobian.h_view(n,1,2,1) = - 2*x*y/rxy4;

    // for z component
    d_surface_jacobian.h_view(n,1,0,2) = -y*z/r3;
    d_surface_jacobian.h_view(n,1,1,2) = -y/(r2*rxy) +2*rxy*y/r4;
    d_surface_jacobian.h_view(n,1,2,2) = 0;

    //****************** Partial z ********************
    // for x component
    d_surface_jacobian.h_view(n,2,0,0) = -x*z/r3;
    d_surface_jacobian.h_view(n,2,1,0) = x*(r2-2*z2)/(r4*rxy);
    d_surface_jacobian.h_view(n,2,2,0) = 0;

    // for y component
    d_surface_jacobian.h_view(n,2,0,1) = -y*z/r3;
    d_surface_jacobian.h_view(n,2,1,1) = y*(r2-2*z2)/(r4*rxy);
    d_surface_jacobian.h_view(n,2,2,1) = 0;

    // for z component
    d_surface_jacobian.h_view(n,2,0,2) = (r2-z2)/r3;
    d_surface_jacobian.h_view(n,2,1,2) = 2*rxy*z/r4;
    d_surface_jacobian.h_view(n,2,2,2) = 0;
  }
  d_surface_jacobian.template modify<HostMemSpace>();
  d_surface_jacobian.template sync<DevExeSpace>();
}



/*
void GaussLegendreGrid::EvaluateTangentVectors() {
  // tangent vectors are theta and phi derivatives of the x,y,z coordinate
  DualArray1D<Real> spatial_coord;
  Kokkos::realloc(spatial_coord,nangles);

  // i iterates over x, y, and z
  for (int i=0; i<3; ++i) {
    for (int n=0; n<nangles; ++n) {
      spatial_coord.h_view(n) = cart_pos.h_view(n,i);
    }
    DualArray1D<Real> dx;
    Kokkos::realloc(dx,nangles);
    
    // theta component
    dx = ThetaDerivative(spatial_coord);
    for (int n=0; n<nangles; ++n) {
      tangent_vectors.h_view(0,n,i) = dx.h_view(n);
    }
    
    // phi component
    dx = PhiDerivative(spatial_coord);
    for (int n=0; n<nangles; ++n) {
      tangent_vectors.h_view(1,n,i) = dx.h_view(n);
    }
  }

  // sync dual arrays
  tangent_vectors.template modify<HostMemSpace>();
  tangent_vectors.template sync<DevExeSpace>();
}


void GaussLegendreGrid::EvaluateNormalOneForms() {

  // i iterates over x, y, and z components of the one form
  for (int n=0; n<nangles; ++n) {
    normal_oneforms.h_view(n,0) = tangent_vectors.h_view(0,n,1)*tangent_vectors.h_view(1,n,2)
                                - tangent_vectors.h_view(0,n,2)*tangent_vectors.h_view(1,n,1);
    normal_oneforms.h_view(n,1) = tangent_vectors.h_view(0,n,2)*tangent_vectors.h_view(1,n,0)
                                - tangent_vectors.h_view(0,n,0)*tangent_vectors.h_view(1,n,2);
    normal_oneforms.h_view(n,2) = tangent_vectors.h_view(0,n,0)*tangent_vectors.h_view(1,n,1)
                                - tangent_vectors.h_view(0,n,1)*tangent_vectors.h_view(1,n,0);
  }

  // sync dual arrays
  normal_oneforms.template modify<HostMemSpace>();
  normal_oneforms.template sync<DevExeSpace>();
}
*/