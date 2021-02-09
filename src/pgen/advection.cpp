//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file advection.cpp
//  \brief Problem generator for advection problems.  Use with evolution=kinematic

#include <cmath>
#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "utils/grid_locations.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::Advection_()
//  \brief Problem Generator for advection problems
//  For Hydro: initializes profiles of transverse components of velocity and scalars
//  For MHD: initializes profiles of transverse components of velocity, B, and scalars
//   iprob=1: sine wave
//   iprob=2: square wave


void ProblemGenerator::Advection_(MeshBlockPack *pmbp, ParameterInput *pin)
{
  // Read input parameters
  int flow_dir = pin->GetInteger("problem","flow_dir");
  int iprob = pin->GetInteger("problem","iproblem");
  Real vel = pin->GetOrAddReal("problem","velocity",1.0);
  Real amp = pin->GetOrAddReal("problem","amplitude",0.1);

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
  int &nx1 = pmbp->mb_cells.nx1;
  int &nx2 = pmbp->mb_cells.nx2;
  int &nx3 = pmbp->mb_cells.nx3;
  int &is = pmbp->mb_cells.is, &ie = pmbp->mb_cells.ie;
  int &js = pmbp->mb_cells.js, &je = pmbp->mb_cells.je;
  int &ks = pmbp->mb_cells.ks, &ke = pmbp->mb_cells.ke;
  auto &size = pmbp->pmb->mbsize;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {

    if (pmbp->phydro->peos->eos_data.is_adiabatic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Only isothermal EOS allowed for advection tests" << std::endl;
      exit(EXIT_FAILURE);
    }
    int &nhydro = pmbp->phydro->nhydro;
    int &nscalars = pmbp->phydro->nscalars;
    auto &u0 = pmbp->phydro->u0;

    par_for("hydro_advect", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        Real r; // coordinate that will span [0->1]
        if (flow_dir == 1) {
          r = (CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m))
              - x1mesh)/length;
        } else if (flow_dir == 2) {
          r = (CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m))
              - x2mesh)/length;
        } else {
          r = (CellCenterX(k-ks, nx3, size.x3min.d_view(m), size.x3max.d_view(m))
              - x3mesh)/length;
        }
  
        Real f; // value for advected quantity, depending on problem type

        // iprob=1: sine wave
        if (iprob == 1) {
          f = 1.0 + amp*sin(2.0*(M_PI)*r);

        // iprob=2: square wave in second quarter of domain
        } else if (iprob == 2) {
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
        // add passive scalars
        for (int n=nhydro; n<(nhydro+nscalars); ++n) {
          u0(m,n,k,j,i) = f;
        }
      }
    );
  }  // End initialization of Hydro variables

  // Initialize MHD variables ----------------------------------
  if (pmbp->pmhd != nullptr) {
    if (pmbp->pmhd->peos->eos_data.is_adiabatic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Only isothermal EOS allowed for advection tests" << std::endl;
      exit(EXIT_FAILURE);
    }
    int &nmhd = pmbp->pmhd->nmhd;
    int &nscalars = pmbp->pmhd->nscalars;
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
        
    par_for("mhd_advect", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      { 
        Real r; // coordinate that will span [0->1]
        if (flow_dir == 1) {
          r = (CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m)) 
              - x1mesh)/length;
        } else if (flow_dir == 2) {
          r = (CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m)) 
              - x2mesh)/length;
        } else {
          r = (CellCenterX(k-ks, nx3, size.x3min.d_view(m), size.x3max.d_view(m)) 
              - x3mesh)/length;
        }
        
        Real f; // value for advected quantity, depending on problem type

        // iprob=1: sine wave
        if (iprob == 1) {               
          f = 1.0 + amp*sin(2.0*(M_PI)*r);

        // iprob=2: square wave in second quarter of domain
        } else if (iprob == 2) {        
          f = 1.0;
          if (r >= 0.25 && r <= 0.5) { f += amp; }
        }
        
        // now compute density  momenta, total energy
        u0(m,IDN,k,j,i) = 1.0;

        // Flow in x1-direction
        if (flow_dir == 1) {
          u0(m,IM1,k,j,i) = vel;
          u0(m,IM2,k,j,i) = f;
          u0(m,IM3,k,j,i) = f;

          // initialize By/Bz
          b0.x1f(m,k,j,i) = 0.0;
          b0.x2f(m,k,j,i) = f;
          b0.x3f(m,k,j,i) = f;
          if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
          if (j==je) {b0.x2f(m,k,j+1,i) = f;}
          if (k==ke) {b0.x3f(m,k+1,j,i) = f;}

        // Flow in x2-direction
        } else if (flow_dir == 2) {
          u0(m,IM1,k,j,i) = f;
          u0(m,IM2,k,j,i) = vel;
          u0(m,IM3,k,j,i) = f;

          // initialize Bx/Bz
          b0.x1f(m,k,j,i) = f;
          b0.x2f(m,k,j,i) = 0.0;
          b0.x3f(m,k,j,i) = f;
          if (i==ie) {b0.x1f(m,k,j,i+1) = f;}
          if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
          if (k==ke) {b0.x3f(m,k+1,j,i) = f;}

        // Flow in x3-direction
        } else {
          u0(m,IM1,k,j,i) = f;
          u0(m,IM2,k,j,i) = f;
          u0(m,IM3,k,j,i) = vel;

          // initialize Bx/By
          b0.x1f(m,k,j,i) = f;
          b0.x2f(m,k,j,i) = f;
          b0.x3f(m,k,j,i) = 0.0;
          if (i==ie) {b0.x1f(m,k,j,i+1) = f;}
          if (j==je) {b0.x2f(m,k,j+1,i) = f;}
          if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}
        }

        // add passive scalars
        for (int n=nmhd; n<(nmhd+nscalars); ++n) {
          u0(m,n,k,j,i) = f;
        }
      }
    );
  }  // End initialization of MHD variables

  return;
}
