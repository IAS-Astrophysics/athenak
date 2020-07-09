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

std::cout << "here , dl=" << wl[IDN] << " vzr=" << wr[IVZ] << std::endl; 

/**
  // Initialize the discontinuity in the Hydro variables ---------------------------------

  switch(shk_dir) {
    //--- shock in 1-direction
    case 1:
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            if (pcoord->x1v(i) < xshock) {
              phydro->u(IDN,k,j,i) = wl[IDN];
              phydro->u(IM1,k,j,i) = wl[IVX]*wl[IDN];
              phydro->u(IM2,k,j,i) = wl[IVY]*wl[IDN];
              phydro->u(IM3,k,j,i) = wl[IVZ]*wl[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN,k,j,i) = peos->EgasFromRhoP(wl[IDN], wl[IPR]);
                } else {
                  phydro->u(IEN,k,j,i) = wl[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN,k,j,i) += 0.5*wl[IDN]*(wl[IVX]*wl[IVX] + wl[IVY]*wl[IVY]
                                                     + wl[IVZ]*wl[IVZ]);
              }
            } else {
              phydro->u(IDN,k,j,i) = wr[IDN];
              phydro->u(IM1,k,j,i) = wr[IVX]*wr[IDN];
              phydro->u(IM2,k,j,i) = wr[IVY]*wr[IDN];
              phydro->u(IM3,k,j,i) = wr[IVZ]*wr[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN,k,j,i) = peos->EgasFromRhoP(wr[IDN], wr[IPR]);
                } else {
                  phydro->u(IEN,k,j,i) = wr[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN,k,j,i) += 0.5*wr[IDN]*(wr[IVX]*wr[IVX] + wr[IVY]*wr[IVY]
                                                     + wr[IVZ]*wr[IVZ]);
              }
            }
          }
        }
      }
      break;
      //--- shock in 2-direction
    case 2:
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          if (pcoord->x2v(j) < xshock) {
            for (int i=is; i<=ie; ++i) {
              phydro->u(IDN,k,j,i) = wl[IDN];
              phydro->u(IM2,k,j,i) = wl[IVX]*wl[IDN];
              phydro->u(IM3,k,j,i) = wl[IVY]*wl[IDN];
              phydro->u(IM1,k,j,i) = wl[IVZ]*wl[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN,k,j,i) = peos->EgasFromRhoP(wl[IDN], wl[IPR]);
                } else {
                  phydro->u(IEN,k,j,i) = wl[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN,k,j,i) += 0.5*wl[IDN]*(wl[IVX]*wl[IVX] + wl[IVY]*wl[IVY]
                                                     + wl[IVZ]*wl[IVZ]);
              }
            }
          } else {
            for (int i=is; i<=ie; ++i) {
              phydro->u(IDN,k,j,i) = wr[IDN];
              phydro->u(IM2,k,j,i) = wr[IVX]*wr[IDN];
              phydro->u(IM3,k,j,i) = wr[IVY]*wr[IDN];
              phydro->u(IM1,k,j,i) = wr[IVZ]*wr[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN,k,j,i) = peos->EgasFromRhoP(wr[IDN], wr[IPR]);
                } else {
                  phydro->u(IEN,k,j,i) = wr[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN,k,j,i) += 0.5*wr[IDN]*(wr[IVX]*wr[IVX] + wr[IVY]*wr[IVY]
                                                     + wr[IVZ]*wr[IVZ]);
              }
            }
          }
        }
      }
      break;

      //--- shock in 3-direction
    case 3:
      for (int k=ks; k<=ke; ++k) {
        if (pcoord->x3v(k) < xshock) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              phydro->u(IDN,k,j,i) = wl[IDN];
              phydro->u(IM3,k,j,i) = wl[IVX]*wl[IDN];
              phydro->u(IM1,k,j,i) = wl[IVY]*wl[IDN];
              phydro->u(IM2,k,j,i) = wl[IVZ]*wl[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN,k,j,i) = peos->EgasFromRhoP(wl[IDN], wl[IPR]);
                } else {
                  phydro->u(IEN,k,j,i) = wl[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN,k,j,i) += 0.5*wl[IDN]*(wl[IVX]*wl[IVX] + wl[IVY]*wl[IVY]
                                                     + wl[IVZ]*wl[IVZ]);
              }
            }
          }
        } else {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              phydro->u(IDN,k,j,i) = wr[IDN];
              phydro->u(IM3,k,j,i) = wr[IVX]*wr[IDN];
              phydro->u(IM1,k,j,i) = wr[IVY]*wr[IDN];
              phydro->u(IM2,k,j,i) = wr[IVZ]*wr[IDN];
              if (NON_BAROTROPIC_EOS) {
                if (GENERAL_EOS) {
                  phydro->u(IEN,k,j,i) = peos->EgasFromRhoP(wr[IDN], wr[IPR]);
                } else {
                  phydro->u(IEN,k,j,i) = wr[IPR]/(peos->GetGamma() - 1.0);
                }
                phydro->u(IEN,k,j,i) += 0.5*wr[IDN]*(wr[IVX]*wr[IVX] + wr[IVY]*wr[IVY]
                                                     + wr[IVZ]*wr[IVZ]);
              }
            }
          }
        }
      }
      break;

    default:
      msg << "### FATAL ERROR in Problem Generator" << std::endl
          << "shock_dir=" << shk_dir << " must be either 1,2, or 3" << std::endl;
      ATHENA_ERROR(msg);
  }
***/

  return;
}
