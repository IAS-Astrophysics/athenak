//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advection.c
//  \brief Problem generator for advection problems.  Use with evolve=advect
//
// Input parameters are:
//    - problem/u0   = flow speed

#include <cmath>
#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "utils/grid_locations.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for advection problems

void ProblemGenerator::Advection_(MeshBlockPack *pmbp, ParameterInput *pin)
{
  using namespace hydro;

  // Read input parameters
  int flow_dir = pin->GetInteger("problem","flow_dir");
  int iprob = pin->GetInteger("problem","iproblem");
  Real vel = pin->GetOrAddReal("problem","velocity",1.0);
  Real amp = pin->GetOrAddReal("problem","amplitude",0.1);
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

  // get size of overall domain
  Real length;
  if (flow_dir == 1) {
    length = pmesh_->mesh_size.x1max - pmesh_->mesh_size.x1min;
  } else if (flow_dir == 2) {
    length = pmesh_->mesh_size.x2max - pmesh_->mesh_size.x2min;
  } else if (flow_dir == 3) {
    length = pmesh_->mesh_size.x3max - pmesh_->mesh_size.x3min;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "flow_dir=" << flow_dir << " must be either 1,2, or 3" << std::endl;
    exit(EXIT_FAILURE);
  }

  // check for valid problem flag
  if (iprob <= 0 || iprob > 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "problem/iproblem=" << iprob << " not supported" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  Real &x1mesh = pmesh_->mesh_size.x1min;
  Real &x2mesh = pmesh_->mesh_size.x2min;
  Real &x3mesh = pmesh_->mesh_size.x3min;
  int &nhydro = pmbp->phydro->nhydro;
  int &nscalars = pmbp->phydro->nscalars;
  int &nx1 = pmbp->mb_cells.nx1;
  int &nx2 = pmbp->mb_cells.nx2;
  int &nx3 = pmbp->mb_cells.nx3;
  int &is = pmbp->mb_cells.is, &ie = pmbp->mb_cells.ie;
  int &js = pmbp->mb_cells.js, &je = pmbp->mb_cells.je;
  int &ks = pmbp->mb_cells.ks, &ke = pmbp->mb_cells.ke;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mbsize;

  par_for("pgen_advect", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real r; // coordinate that will span [0->1]
      if (flow_dir == 1) {
        r = (CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m)) - x1mesh)
            /length;
      } else if (flow_dir == 2) {
        r = (CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m)) - x2mesh)
            /length;
      } else {
        r = (CellCenterX(k-ks, nx3, size.x3min.d_view(m), size.x3max.d_view(m)) - x3mesh)
            /length;
      }

      Real f; // value for advected quantity, depending on problem type
      if (iprob == 1) {               // iprob=1: sine wave
        f = 1.0 + amp*sin(2.0*(M_PI)*r);
      } else if (iprob == 2) {        // iprob=2: square wave in second quarter of domain
        f = 1.0;
        if (r >= 0.25 && r <= 0.5) { f += amp; }
      }

      // now compute density  momenta, total energy
      u0(m,IDN,k,j,i) = 1.0;
      if (flow_dir == 1) {
        u0(m,IM1,k,j,i) = vel;
        u0(m,IM2,k,j,i) = f;
        u0(m,IM3,k,j,i) = f;
      } else if (flow_dir == 2) {
        u0(m,IM1,k,j,i) = f;
        u0(m,IM2,k,j,i) = vel;
        u0(m,IM3,k,j,i) = f;
      } else {
        u0(m,IM1,k,j,i) = f;
        u0(m,IM2,k,j,i) = f;
        u0(m,IM3,k,j,i) = vel;
      } 
      u0(m,IEN,k,j,i) = 1.0/gm1 + 0.5*(SQR(u0(m,IM1,k,j,i))
        + SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i)))/
          u0(m,IDN,k,j,i);
      // add passive scalars
      for (int n=nhydro; n<(nhydro+nscalars); ++n) {
        u0(m,n,k,j,i) = f;
      }
    }
  );

  return;
}
