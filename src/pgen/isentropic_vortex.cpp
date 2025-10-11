//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vortex.cpp
//  \brief Problem generator for isentropic vortex test problem
//

// C++ headers
#include <cmath>  // sin()
#include <iostream> // cout

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//  \brief Vortex test problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Isentropic vortex test can only be run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // setup problem parameters
  Real eps = 5.0; // Amplitude, should read in from input file
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real sig2 = x1size*x2size*0.01;
  Real x1c = pmy_mesh_->mesh_size.x1max - 0.5*x1size;
  Real x2c = pmy_mesh_->mesh_size.x2max - 0.5*x2size;
  Real gamma = pmbp->phydro->peos->eos_data.gamma;

  // capture variables for kernel
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  if (gamma <= 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Isentropic vortex test requires gamma > 1 "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;

  par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx1 = indcs.nx1;
    int nx2 = indcs.nx2;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real arg = (SQR(x1c-x1v) + SQR(x2c-x2v))/sig2;
    arg = std::exp(1-arg);

    Real deltaT = -gm1*SQR(eps)*arg*0.125/(gamma*SQR(M_PI));
    Real deltaV = eps*0.5*std::sqrt(arg)/M_PI;
    Real temp = 1.0 + deltaT;
    Real dens = std::pow(temp, 1/gm1);
    Real pres = std::pow(dens, gamma);
    Real delta_u = deltaV*(x2c-x2v);
    Real delta_v = deltaV*(x1v-x1c);

    u0(m,IDN,k,j,i) = dens;
    u0(m,IM1,k,j,i) = dens*(1.0 + delta_u);
    u0(m,IM2,k,j,i) = dens*(1.0 + delta_v);
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = pres/gm1 + 0.5*dens*(SQR(1.0 + delta_v) + SQR(1.0 + delta_u));
  });

  return;
}
