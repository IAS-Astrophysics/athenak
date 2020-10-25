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
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "utils/grid_locations.hpp"
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

  // capture variables for kernel
  Real gm1 = pmb->phydro->peos->eos_data.gamma - 1.0;
  int &is = pmb->mb_cells.is, &ie = pmb->mb_cells.ie;
  int &js = pmb->mb_cells.js, &je = pmb->mb_cells.je;
  int &ks = pmb->mb_cells.ks, &ke = pmb->mb_cells.ke;
  Real &x1min = pmb->mb_size.x1min, &x1max = pmb->mb_size.x1max;
  Real &x2min = pmb->mb_size.x2min, &x2max = pmb->mb_size.x2max;
  int &nx1 = pmb->mb_cells.nx1;
  int &nx2 = pmb->mb_cells.nx2;
  int &nscalars = pmb->phydro->nscalars;
  int &nhydro = pmb->phydro->nhydro;
  auto &u0 = pmb->phydro->u0;

  // to make ICs symmetric, set y0 to be in between cell center and face
  Real y0 = 0.5*(x2max + x2min) + 0.25*(pmb->mb_cells.dx2);

  // Set initial conditions
  par_for("pgen_lw_implode", pmb->exe_space, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      u0(IM1,k,j,i) = 0.0;
      u0(IM2,k,j,i) = 0.0;
      u0(IM3,k,j,i) = 0.0;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      if (x2v > (y0 - x1v)) {
        u0(IDN,k,j,i) = d_out;
        u0(IEN,k,j,i) = p_out/gm1;
        if (nscalars > 0) u0(nhydro,k,j,i) = 0.0;
      } else {
        u0(IDN,k,j,i) = d_in;
        u0(IEN,k,j,i) = p_in/gm1;
        if (nscalars > 0) u0(nhydro,k,j,i) = d_in;
      }
    }
  );

  return;
}
