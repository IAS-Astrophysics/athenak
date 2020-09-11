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

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for advection problems

void ProblemGenerator::LWImplode_(MeshBlock *pmb, ParameterInput *pin)
{
  using namespace hydro;
  Real d_in = pin->GetReal("problem","d_in");
  Real p_in = pin->GetReal("problem","p_in");

  Real d_out = pin->GetReal("problem","d_out");
  Real p_out = pin->GetReal("problem","p_out");

  Real gm1 = pmb->phydro->peos->GetGamma() - 1.0;

  int &is = pmb->mb_cells.is, &ie = pmb->mb_cells.ie;
  int &js = pmb->mb_cells.js, &je = pmb->mb_cells.je;
  int &ks = pmb->mb_cells.ks, &ke = pmb->mb_cells.ke;
  Real &x1min = pmb->mb_size.x1min, &x1max = pmb->mb_size.x1max;
  Real &x2min = pmb->mb_size.x2min, &x2max = pmb->mb_size.x2max;

  // to make ICs symmetric, set y0 to be in between cell center and face
  Real y0 = 0.5*(x2max + x2min) + 0.25*(pmb->mb_cells.dx2);

  // Set initial conditions
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        pmb->phydro->u0(IM1,k,j,i) = 0.0;
        pmb->phydro->u0(IM2,k,j,i) = 0.0;
        pmb->phydro->u0(IM3,k,j,i) = 0.0;
        Real x1v = pmesh_->CellCenterX(i-is, pmb->mb_cells.nx1, x1min, x1max);
        Real x2v = pmesh_->CellCenterX(j-js, pmb->mb_cells.nx2, x2min, x2max);
        if (x2v > (y0 - x1v)) {
          pmb->phydro->u0(IDN,k,j,i) = d_out;
          pmb->phydro->u0(IEN,k,j,i) = p_out/gm1;
        } else {
          pmb->phydro->u0(IDN,k,j,i) = d_in;
          pmb->phydro->u0(IEN,k,j,i) = p_in/gm1;
        }
      }
    }
  }

  return;
}
