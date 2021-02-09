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
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "utils/grid_locations.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::LWImplode_()
//  \brief Problem Generator for LW Implosion test

void ProblemGenerator::LWImplode_(MeshBlockPack *pmbp, ParameterInput *pin)
{
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
  int &is = pmbp->mb_cells.is, &ie = pmbp->mb_cells.ie;
  int &js = pmbp->mb_cells.js, &je = pmbp->mb_cells.je;
  int &ks = pmbp->mb_cells.ks, &ke = pmbp->mb_cells.ke;
  int &nx1 = pmbp->mb_cells.nx1;
  int &nx2 = pmbp->mb_cells.nx2;
  int &nscalars = pmbp->phydro->nscalars;
  int &nhydro = pmbp->phydro->nhydro;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mbsize;
  Real x2min_mesh = pmbp->pmesh->mesh_size.x2min;
  Real x2max_mesh = pmbp->pmesh->mesh_size.x2max;


  // Set initial conditions
  par_for("pgen_lw_implode", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      // to make ICs symmetric, set y0 to be in between cell center and face
      Real y0 = 0.5*(x2max_mesh + x2min_mesh) + 0.25*(size.dx2.d_view(m));

      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
      Real x2v = CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m));
      if (x2v > (y0 - x1v)) {
        u0(m,IDN,k,j,i) = d_out;
        u0(m,IEN,k,j,i) = p_out/gm1;
        if (nscalars > 0) u0(m,nhydro,k,j,i) = 0.0;
      } else {
        u0(m,IDN,k,j,i) = d_in;
        u0(m,IEN,k,j,i) = p_in/gm1;
        if (nscalars > 0) u0(m,nhydro,k,j,i) = d_in;
      }
    }
  );

  return;
}
