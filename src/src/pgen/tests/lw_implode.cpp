//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lw_implode.cpp
//  \brief Problem generator for square implosion problem
//
// REFERENCE: R. Liska & B. Wendroff, SIAM J. Sci. Comput., 25, 995 (2003)
//========================================================================================
#include <iostream> // cout

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::LWImplode_()
//  \brief Problem Generator for LW Implosion test

void ProblemGenerator::LWImplode(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "LW Implosion test can only be run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  Real d_in = pin->GetReal("problem","d_in");
  Real p_in = pin->GetReal("problem","p_in");

  Real d_out = pin->GetReal("problem","d_out");
  Real p_out = pin->GetReal("problem","p_out");

  // capture variables for kernel
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int &nscalars = pmbp->phydro->nscalars;
  int &nhydro = pmbp->phydro->nhydro;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  Real x2min_mesh = pmy_mesh_->mesh_size.x2min;
  Real x2max_mesh = pmy_mesh_->mesh_size.x2max;


  // Set initial conditions
  par_for("pgen_lw_implode", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // to make ICs symmetric, set y0 to be in between cell center and face
    Real y0 = 0.5*(x2max_mesh + x2min_mesh) + 0.25*(size.d_view(m).dx2);

    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    if (x2v > (y0 - x1v)) {
      u0(m,IDN,k,j,i) = d_out;
      u0(m,IEN,k,j,i) = p_out/gm1;
      if (nscalars > 0) u0(m,nhydro,k,j,i) = 0.0;
    } else {
      u0(m,IDN,k,j,i) = d_in;
      u0(m,IEN,k,j,i) = p_in/gm1;
      if (nscalars > 0) u0(m,nhydro,k,j,i) = d_in;
    }
  });

  return;
}
