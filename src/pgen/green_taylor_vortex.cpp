//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vortex.cpp
//  \brief Problem generator for compressible Green-Taylor vortex
//  introduce by Lusher & Sandham 2020 (https://doi.org/10.2514/1.J059672)
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
              << "Green-Taylor test can only be run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  // setup problem parameters
  Real M0 = 0.1; // Mach number

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;
  Real lx1 = x1size/M_PI;
  Real lx2 = x2size/M_PI;
  Real lx3 = x3size/M_PI;

  // capture variables for kernel
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  Real P0 = 1.0/(M0*(gm1+1)); // reference pressure

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
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx1 = indcs.nx1;
    int nx2 = indcs.nx2;
    int nx3 = indcs.nx3;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max)/lx1;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max)/lx2;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max)/lx3;

    Real vg =  std::sin(x1v)*std::cos(x2v)*std::cos(x3v);
    Real ug = -std::sin(x1v)*std::cos(x2v)*std::cos(x3v);
    Real p0 = P0 + 0.0625*(std::cos(2*x1v) + std::cos(2*x2v))*(2.0 + std::cos(2*x3v));
    Real d0 = p0/P0;

    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) = vg*d0;
    u0(m,IM2,k,j,i) = ug*d0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*(ug*ug + vg*vg);
    }
  });

  return;
}
