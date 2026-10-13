//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shu_osher.cpp
//  \brief Problem generator for Shu-Osher shocktube test, involving interaction of a
//   Mach 3 shock with a sine wave density distribution.
//
// REFERENCE: C.W. Shu & S. Osher, "Efficient implementation of essentially
//   non-oscillatory shock-capturing schemes, II", JCP, 83, 32 (1998)

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
//  \brief Shu-Osher test problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Shu-Osher test can only be run in Hydro, but no <hydro> block "
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

  par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    if (x1v < -0.8) {
      u0(m,IDN,k,j,i) = dl;
      u0(m,IM1,k,j,i) = ul*dl;
      u0(m,IM2,k,j,i) = vl*dl;
      u0(m,IM3,k,j,i) = wl*dl;
      u0(m,IEN,k,j,i) = pl/gm1 + 0.5*dl*(ul*ul + vl*vl + wl*wl);
    } else {
      u0(m,IDN,k,j,i) = 1.0 + 0.2*std::sin(5.0*M_PI*(x1v));
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = 1.0/gm1;
    }
  });

  return;
}
