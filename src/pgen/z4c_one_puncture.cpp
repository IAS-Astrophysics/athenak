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
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "adm/adm.hpp"
#include "coordinates/cell_locations.hpp"


void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin);

// Prototypes for user-defined history functions
void GWExtract(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "defining spherical grids" << std::endl; 
  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 10.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 15.0));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 5, 20.0));
  user_hist_func = GWExtract;

  ADMOnePuncture(pmbp, pin);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  std::cout<<"OnePuncture initialized."<<std::endl;


  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to single puncture (no spin)

void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  // For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;
  Real ADM_mass = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen one puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

    x1v -= center_x1;
    x2v -= center_x2;
    x3v -= center_x3;

    Real r = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));

    // Minkowski spacetime
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) = (a == b ? 1. : 0.);
    }
    // admK_dd is automatically set to 0 when is initialized as Kokkos View

    // ADMOnePuncture
    adm.psi4(m,k,j,i) = std::pow(1.0 + 0.5*ADM_mass/r,4); // adm.psi4

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) *= adm.psi4(m,k,j,i);
    }
  });
}

//----------------------------------------------------------------------------------------
// Function for computing gravitational wave

void GWExtract(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  std::cout << "Here" << std::endl;
  DvceArray5D<Real> u_weyl_;
  u_weyl_ = pmbp->pz4c->u_weyl;
  int nvars = 2;
  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  int nflux = 2;

  // set number of and names of history variables for z4c
  //  (1) real part of psi4
  //  (2) imaginary part of psi4

  pdata->nhist = nradii*nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  for (int g=0; g<nradii; ++g) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << grids[g]->radius;
    std::string rad_str = stream.str();
    pdata->label[nflux*g+0] = "rpsi4_" + rad_str;
    pdata->label[nflux*g+1] = "ipsi4_" + rad_str;
  }

  // go through angles at each radii:
  for (int g=0; g<nradii; ++g) {
    // zero fluxes at this radius
    pdata->hdata[nflux*g+0] = 0.0;
    pdata->hdata[nflux*g+1] = 0.0;

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    grids[g]->InterpolateToSphere(nvars, u_weyl_);

    for (int n=0; n<grids[g]->nangles; ++n) {
      Real &int_rpsi4 = grids[g]->interp_vals.h_view(n,0);
      Real &int_ipsi4 = grids[g]->interp_vals.h_view(n,1);

      // integrate rpsi4
      pdata->hdata[nflux*g+0] += int_rpsi4;
      // integrate ipsi4
      pdata->hdata[nflux*g+1] += int_ipsi4;
    }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
}