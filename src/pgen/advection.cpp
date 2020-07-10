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
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for advection problems

void ProblemGenerator::Advection_(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin) {
using namespace hydro;

  // Read input parameters
  int flow_dir = pin->GetInteger("problem","flow_dir");
  int iprob = pin->GetInteger("problem","iproblem");
  Real u0 = pin->GetOrAddReal("problem","velocity",1.0);
  Real amp = pin->GetOrAddReal("problem","amplitude",0.1);

  // Initialize the grid
  int &is = pmb->indx.is, &ie = pmb->indx.ie;
  int &js = pmb->indx.js, &je = pmb->indx.je;
  int &ks = pmb->indx.ks, &ke = pmb->indx.ke;

  // get size of overall domain
  Real length;
  if (flow_dir == 1) {
    length = pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min;
  } else if (flow_dir == 2) {
    length = pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min;
  } else if (flow_dir == 3) {
    length = pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "flow_dir=" << flow_dir << " must be either 1,2, or 3" << std::endl;
    exit(EXIT_FAILURE);
  }

  Real &x1min = pmb->mb_size.x1min, &x1max = pmb->mb_size.x1max;
  Real &x2min = pmb->mb_size.x2min, &x2max = pmb->mb_size.x2max;
  Real &x3min = pmb->mb_size.x3min, &x3max = pmb->mb_size.x3max;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real r;
        if (flow_dir == 1) {
          r = pmb->pmy_mesh->CellCenterX(i, pmb->indx.nx1, x1min, x1max)/length;
        } else if (flow_dir == 2) {
          r = pmb->pmy_mesh->CellCenterX(j, pmb->indx.nx2, x2min, x2max)/length;
        } else {
          r = pmb->pmy_mesh->CellCenterX(k, pmb->indx.nx3, x3min, x3max)/length;
        }

        // iprob=1: sine wave
        if (iprob == 1) {
          pmb->phydro->u(IDN,k,j,i) = 1.0 + amp*std::sin(2.0*(M_PI)*r);

        // iprob=2: square wave (needs domain -1 < x < 1)
        } else if (iprob == 2) {
          pmb->phydro->u(IDN,k,j,i) = 1.0;
          if (std::abs(r) <= 0.5) pmb->phydro->u(IDN,k,j,i) += amp;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
             << std::endl << "problem/iproblem=" << iprob 
             << " not supported" << std::endl;
          exit(EXIT_FAILURE);
        }

        // now compute momenta, total energy
        if (flow_dir == 1) {
          pmb->phydro->u(IM1,k,j,i) = u0*pmb->phydro->u(IDN,k,j,i);
          pmb->phydro->u(IM2,k,j,i) = 0.0;
          pmb->phydro->u(IM3,k,j,i) = 0.0;
        } else if (flow_dir == 2) {
          pmb->phydro->u(IM1,k,j,i) = 0.0;
          pmb->phydro->u(IM2,k,j,i) = u0*pmb->phydro->u(IDN,k,j,i);
          pmb->phydro->u(IM3,k,j,i) = 0.0;
        } else {
          pmb->phydro->u(IM1,k,j,i) = 0.0;
          pmb->phydro->u(IM2,k,j,i) = 0.0;
          pmb->phydro->u(IM3,k,j,i) = u0*pmb->phydro->u(IDN,k,j,i);
        } 
        
        pmb->phydro->u(IEN,k,j,i) = 1.0;
      }
    }
  }

  return;
}
