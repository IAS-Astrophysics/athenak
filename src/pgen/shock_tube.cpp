//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shock_tube.cpp
//  \brief Problem generator for shock tube problems.
//
// Problem generator for shock tube (1-D Riemann) problems. Initializes plane-parallel
// shock along x1 (in 1D, 2D, 3D), along x2 (in 2D, 3D), and along x3 (in 3D).

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>

#include "athena.hpp"
#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"


//----------------------------------------------------------------------------------------
//! \fn
//  \brief Problem Generator for the shock tube (Riemann problem) tests

void ProblemGenerator::ShockTube_(MeshBlock *pmb, std::unique_ptr<ParameterInput> &pin) {
using namespace hydro;

  // parse shock direction: {1,2,3} -> {x1,x2,x3}
  int shk_dir = pin->GetInteger("problem","shock_dir");

  // parse shock location (must be inside grid)
  Real xshock = pin->GetReal("problem","xshock");
  if (shk_dir == 1 && (xshock < pmb->pmy_mesh->mesh_size.x1min ||
                       xshock > pmb->pmy_mesh->mesh_size.x1max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xshock=" << xshock << " lies outside x1 domain" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (shk_dir == 2 && (xshock < pmb->pmy_mesh->mesh_size.x2min ||
                       xshock > pmb->pmy_mesh->mesh_size.x2max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xshock=" << xshock << " lies outside x2 domain" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (shk_dir == 3 && (xshock < pmb->pmy_mesh->mesh_size.x3min ||
                       xshock > pmb->pmy_mesh->mesh_size.x3max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xshock=" << xshock << " lies outside x3 domain" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Parse left state read from input file: dl,ul,vl,wl,[pl]
  Real wl[5];
  wl[IDN] = pin->GetReal("problem","dl");
  wl[IVX] = pin->GetReal("problem","ul");
  wl[IVY] = pin->GetReal("problem","vl");
  wl[IVZ] = pin->GetReal("problem","wl");
  wl[IPR] = pin->GetReal("problem","pl");

  // Parse right state read from input file: dr,ur,vr,wr,[pr]
  Real wr[5];
  wr[IDN] = pin->GetReal("problem","dr");
  wr[IVX] = pin->GetReal("problem","ur");
  wr[IVY] = pin->GetReal("problem","vr");
  wr[IVZ] = pin->GetReal("problem","wr");
  wr[IPR] = pin->GetReal("problem","pr");

  // Initialize the discontinuity in the Hydro variables ---------------------------------

  int &is = pmb->indx.is, &ie = pmb->indx.ie;
  int &js = pmb->indx.js, &je = pmb->indx.je;
  int &ks = pmb->indx.ks, &ke = pmb->indx.ke;
  switch(shk_dir) {

    //--- shock in 1-direction
    case 1:
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
//            if (pcoord->x1v(i) < xshock) {
            if (i < (ie-is+1)/2) {
              pmb->phydro->u(IDN,k,j,i) = wl[IDN];
              pmb->phydro->u(IM1,k,j,i) = wl[IVX]*wl[IDN];
              pmb->phydro->u(IM2,k,j,i) = wl[IVY]*wl[IDN];
              pmb->phydro->u(IM3,k,j,i) = wl[IVZ]*wl[IDN];
              pmb->phydro->u(IEN,k,j,i) = 1.0;
            } else {
              pmb->phydro->u(IDN,k,j,i) = wr[IDN];
              pmb->phydro->u(IM1,k,j,i) = wr[IVX]*wr[IDN];
              pmb->phydro->u(IM2,k,j,i) = wr[IVY]*wr[IDN];
              pmb->phydro->u(IM3,k,j,i) = wr[IVZ]*wr[IDN];
              pmb->phydro->u(IEN,k,j,i) = 1.0;
            }
          }
        }
      }
      break;

    //--- shock in 2-direction
    case 2:
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
//          if (pcoord->x2v(j) < xshock) {
          if (j < (je-js+1)/2) {
            for (int i=is; i<=ie; ++i) {
              pmb->phydro->u(IDN,k,j,i) = wl[IDN];
              pmb->phydro->u(IM2,k,j,i) = wl[IVX]*wl[IDN];
              pmb->phydro->u(IM3,k,j,i) = wl[IVY]*wl[IDN];
              pmb->phydro->u(IM1,k,j,i) = wl[IVZ]*wl[IDN];
              pmb->phydro->u(IEN,k,j,i) = 1.0;
            }
          } else {
            for (int i=is; i<=ie; ++i) {
              pmb->phydro->u(IDN,k,j,i) = wr[IDN];
              pmb->phydro->u(IM2,k,j,i) = wr[IVX]*wr[IDN];
              pmb->phydro->u(IM3,k,j,i) = wr[IVY]*wr[IDN];
              pmb->phydro->u(IM1,k,j,i) = wr[IVZ]*wr[IDN];
              pmb->phydro->u(IEN,k,j,i) = 1.0;
            }
          }
        }
      }
      break;

    //--- shock in 3-direction
    case 3:
      for (int k=ks; k<=ke; ++k) {
//        if (pcoord->x3v(k) < xshock) {
        if (k < (ke-ks+1)/2) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              pmb->phydro->u(IDN,k,j,i) = wl[IDN];
              pmb->phydro->u(IM3,k,j,i) = wl[IVX]*wl[IDN];
              pmb->phydro->u(IM1,k,j,i) = wl[IVY]*wl[IDN];
              pmb->phydro->u(IM2,k,j,i) = wl[IVZ]*wl[IDN];
              pmb->phydro->u(IEN,k,j,i) = 1.0;
            }
          }
        } else {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              pmb->phydro->u(IDN,k,j,i) = wr[IDN];
              pmb->phydro->u(IM3,k,j,i) = wr[IVX]*wr[IDN];
              pmb->phydro->u(IM1,k,j,i) = wr[IVY]*wr[IDN];
              pmb->phydro->u(IM2,k,j,i) = wr[IVZ]*wr[IDN];
              pmb->phydro->u(IEN,k,j,i) = 1.0;
            }
          }
        }
      }
      break;

    //--- Invaild input value for shk_dir
    default:
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "shock_dir=" << shk_dir << " must be either 1,2, or 3" << std::endl;
      exit(EXIT_FAILURE);
  }

  return;
}
