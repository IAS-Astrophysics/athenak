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
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::Advection_()
//  \brief Problem Generator for advection problems. By default, initializes profiles
//  only in passive scalars (and B for MHD).  Can also set profiles in density by setting
//  input flag advect_dens=true
//   iprob=1: sine wave
//   iprob=2: square wave
//   iprob=2: Gaussian, square, and triangle

void ProblemGenerator::Advection(ParameterInput *pin, const bool restart) {
  // nothing needs to be done on restarts for this pgen
  if (restart) return;

  // Read input parameters
  int flow_dir = pin->GetInteger("problem","flow_dir");
  int iprob = pin->GetInteger("problem","iproblem");
  Real vel = pin->GetOrAddReal("problem","velocity",1.0);
  Real amp = pin->GetOrAddReal("problem","amplitude",0.1);
  bool advect_dens = pin->GetOrAddBoolean("problem","advect_dens",false);

  // get size of overall domain
  Real length;
  if (flow_dir == 1) {
    length = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  } else if (flow_dir == 2) {
    length = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  } else if (flow_dir == 3) {
    length = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "flow_dir=" << flow_dir << " must be either 1,2, or 3" << std::endl;
    exit(EXIT_FAILURE);
  }

  // check for valid problem flag
  if (iprob <= 0 || iprob > 3) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "problem/iproblem=" << iprob << " not supported" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  Real &x1mesh = pmy_mesh_->mesh_size.x1min;
  Real &x2mesh = pmy_mesh_->mesh_size.x2min;
  Real &x3mesh = pmy_mesh_->mesh_size.x3min;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    if (pmbp->phydro->peos->eos_data.is_ideal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Only isothermal EOS allowed for advection tests" << std::endl;
      exit(EXIT_FAILURE);
    }
    int &nhydro = pmbp->phydro->nhydro;
    int &nscalars = pmbp->phydro->nscalars;
    auto &u0 = pmbp->phydro->u0;

    par_for("hydro_advect", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real r; // coordinate that will span [0->1]
      if (flow_dir == 1) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        r = (CellCenterX(i-is, nx1, x1min, x1max) - x1mesh)/length;
      } else if (flow_dir == 2) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        r = (CellCenterX(j-js, nx2, x2min, x2max) - x2mesh)/length;
      } else {
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        r = (CellCenterX(k-ks, nx3, x3min, x3max) - x3mesh)/length;
      }

      Real f; // value for advected quantity, depending on problem type

      // iprob=1: sine wave
      if (iprob == 1) {
        f = 1.0 + amp*sin(2.0*(M_PI)*r);

      // iprob=2: square wave in second quarter of domain
      } else if (iprob == 2) {
        f = 1.0;
        if (r >= 0.25 && r <= 0.5) { f += amp; }
      } else if (iprob == 3) {
        f = 1.0;
        if (r <= 0.45) { f += amp*exp((SQR(r-0.2))/-0.005); }
        if (r >= 0.45 && r <= 0.65) { f += amp; }
        if (r >= 0.75 && r <= 0.85) { f += amp*(10.0*r-7.5); }
        if (r >= 0.85 && r <= 0.95) { f += amp*(9.5-10.0*r); }
        if (r >= 0.95) { f += amp*exp((SQR(r-1.2))/-0.005); }
      }

      // now compute density  momenta, total energy
      if (advect_dens) {
        u0(m,IDN,k,j,i) = f;
      } else {
        u0(m,IDN,k,j,i) = 1.0;
      }
      if (flow_dir == 1) {
        u0(m,IM1,k,j,i) = vel*u0(m,IDN,k,j,i);
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
      } else if (flow_dir == 2) {
        u0(m,IM1,k,j,i) = 0.0;
        u0(m,IM2,k,j,i) = vel*u0(m,IDN,k,j,i);
        u0(m,IM3,k,j,i) = 0.0;
        } else {
        u0(m,IM1,k,j,i) = 0.0;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = vel*u0(m,IDN,k,j,i);
      }
      // add passive scalars
      for (int n=nhydro; n<(nhydro+nscalars); ++n) {
        u0(m,n,k,j,i) = f*u0(m,IDN,k,j,i);
      }
    });
  }  // End initialization of Hydro variables

  // Initialize MHD variables ----------------------------------
  if (pmbp->pmhd != nullptr) {
    if (pmbp->pmhd->peos->eos_data.is_ideal) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Only isothermal EOS allowed for advection tests" << std::endl;
      exit(EXIT_FAILURE);
    }
    int &nmhd = pmbp->pmhd->nmhd;
    int &nscalars = pmbp->pmhd->nscalars;
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;

    par_for("mhd_advect", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real r; // coordinate that will span [0->1]
      if (flow_dir == 1) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        int nx1 = indcs.nx1;
        r = (CellCenterX(i-is, nx1, x1min, x1max) - x1mesh)/length;
      } else if (flow_dir == 2) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        r = (CellCenterX(j-js, nx2, x2min, x2max) - x2mesh)/length;
      } else {
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        int nx3 = indcs.nx3;
        r = (CellCenterX(k-ks, nx3, x3min, x3max) - x3mesh)/length;
      }

      Real f; // value for advected quantity, depending on problem type

      // iprob=1: sine wave
      if (iprob == 1) {
        f = 1.0 + amp*sin(2.0*(M_PI)*r);

      // iprob=2: square wave in second quarter of domain
      } else if (iprob == 2) {
        f = 1.0;
        if (r >= 0.25 && r <= 0.5) { f += amp; }
      } else if (iprob == 3) {
        f = 1.0;
        if (r <= 0.45) { f += amp*exp((SQR(r-0.2))/-0.005); }
        if (r >= 0.45 && r <= 0.65) { f += amp; }
        if (r >= 0.75 && r <= 0.85) { f += amp*(10.0*r-7.5); }
        if (r >= 0.85 && r <= 0.95) { f += amp*(9.5-10.0*r); }
        if (r >= 0.95) { f += amp*exp((SQR(r-1.2))/-0.005); }
      }

      // now compute density  momenta, total energy
      if (advect_dens) {
        u0(m,IDN,k,j,i) = f;
      } else {
        u0(m,IDN,k,j,i) = 1.0;
      }

      // Flow in x1-direction
      if (flow_dir == 1) {
        u0(m,IM1,k,j,i) = vel*u0(m,IDN,k,j,i);
        u0(m,IM2,k,j,i) = 1.0;
        u0(m,IM3,k,j,i) = 1.0;

        // initialize By/Bz
        b0.x1f(m,k,j,i) = 0.0;
        b0.x2f(m,k,j,i) = f;
        b0.x3f(m,k,j,i) = f;
        if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
        if (j==je) {b0.x2f(m,k,j+1,i) = f;}
        if (k==ke) {b0.x3f(m,k+1,j,i) = f;}

      // Flow in x2-direction
      } else if (flow_dir == 2) {
        u0(m,IM1,k,j,i) = 1.0;
        u0(m,IM2,k,j,i) = vel*u0(m,IDN,k,j,i);
        u0(m,IM3,k,j,i) = 1.0;

        // initialize Bx/Bz
        b0.x1f(m,k,j,i) = f;
        b0.x2f(m,k,j,i) = 0.0;
        b0.x3f(m,k,j,i) = f;
        if (i==ie) {b0.x1f(m,k,j,i+1) = f;}
        if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
        if (k==ke) {b0.x3f(m,k+1,j,i) = f;}

      // Flow in x3-direction
      } else {
        u0(m,IM1,k,j,i) = 1.0;
        u0(m,IM2,k,j,i) = 1.0;
        u0(m,IM3,k,j,i) = vel*u0(m,IDN,k,j,i);

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
        u0(m,n,k,j,i) = f*u0(m,IDN,k,j,i);
      }
    });
  }  // End initialization of MHD variables

  return;
}
