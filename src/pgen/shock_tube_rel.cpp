//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shock_tube.cpp
//  \brief Problem generator for shock tube (1-D Riemann) problems in both relativistic hydro and MHD.
//
// Initializes plane-parallel shock along x1 (in 1D, 2D, 3D), along x2 (in 2D, 3D),
// and along x3 (in 3D).

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
//! \fn
//  \brief Problem Generator for the shock tube (Riemann problem) tests

void ProblemGenerator::ShockTube_Rel_(MeshBlockPack *pmbp, ParameterInput *pin)
{

  // parse shock direction: {1,2,3} -> {x1,x2,x3}
  int shk_dir = pin->GetInteger("problem","shock_dir");
  if (shk_dir < 1 || shk_dir > 3) {
    // Invaild input value for shk_dir
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "shock_dir=" <<shk_dir<< " must be either 1,2, or 3" << std::endl;
    exit(EXIT_FAILURE);
  }
  // set indices of parallel and perpendicular velocities
  int ivx = shk_dir;
  int ivy = IVX + ((ivx - IVX) + 1)%3;
  int ivz = IVX + ((ivx - IVX) + 2)%3;

  // parse shock location (must be inside grid)
  Real xshock = pin->GetReal("problem","xshock");
  if (shk_dir == 1 && (xshock < pmesh_->mesh_size.x1min ||
                       xshock > pmesh_->mesh_size.x1max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xshock=" << xshock << " lies outside x1 domain" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (shk_dir == 2 && (xshock < pmesh_->mesh_size.x2min ||
                       xshock > pmesh_->mesh_size.x2max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xshock=" << xshock << " lies outside x2 domain" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (shk_dir == 3 && (xshock < pmesh_->mesh_size.x3min ||
                       xshock > pmesh_->mesh_size.x3max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "xshock=" << xshock << " lies outside x3 domain" << std::endl;
    exit(EXIT_FAILURE);
  }

 // capture variables for the kernel
  int &is = pmbp->mb_cells.is, &ie = pmbp->mb_cells.ie;
  int &js = pmbp->mb_cells.js, &je = pmbp->mb_cells.je;
  int &ks = pmbp->mb_cells.ks, &ke = pmbp->mb_cells.ke;
  int &nx1 = pmbp->mb_cells.nx1;
  int &nx2 = pmbp->mb_cells.nx2;
  int &nx3 = pmbp->mb_cells.nx3;
  auto size = pmbp->pmb->mbsize;


  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {

    auto& eos = pmbp->phydro->peos->eos_data;

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


    Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
    auto &u0 = pmbp->phydro->u0;

    par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m,int k, int j, int i)
        {
	  Real x;
	  if (shk_dir == 1) {
	    x = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
	  } else if (shk_dir == 2) {
	    x = CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m));
	  } else {
	    x = CellCenterX(k-ks, nx3, size.x3min.d_view(m), size.x3max.d_view(m));
	  }
          if (x < xshock) {

	    Real v_sq = SQR(wl[IVX]) + SQR(wl[IVY]) + SQR(wl[IVZ]);
	    Real gamma_sq = 1./(1.-v_sq);
	    Real gamma = sqrt(gamma_sq);

	    auto const& rho = wl[IDN];
	    auto const& pgas = wl[IPR];
	    auto const& gamma_adi = eos.gamma;
	    Real rho_eps = pgas / gm1;
	    //FIXME ERM: Only ideal fluid for now
	    Real wgas = rho + gamma_adi / gm1 * pgas;

            u0(m,IDN,k,j,i) = rho * gamma;
            u0(m,IM1,k,j,i) = wgas * gamma_sq * wl[IVX];
            u0(m,IM2,k,j,i) = wgas * gamma_sq * wl[IVY];
            u0(m,IM3,k,j,i) = wgas * gamma_sq * wl[IVZ];
            u0(m,IEN,k,j,i) = wgas * gamma_sq  - pgas - rho*gamma; //rho_eps * gamma_sq + (pgas + rho*gamma/(gamma+1.))*(v_sq*gamma_sq);
          } else {
	    Real v_sq = SQR(wr[IVX]) + SQR(wr[IVY]) + SQR(wr[IVZ]);
	    Real gamma_sq = 1./(1.-v_sq);
	    Real gamma = sqrt(gamma_sq);

	    auto const& rho = wr[IDN];
	    auto const& pgas = wr[IPR];
	    auto const& gamma_adi = eos.gamma;
	    Real rho_eps = pgas / gm1;
	    //FIXME ERM: Only ideal fluid for now
	    Real wgas = rho + gamma_adi / gm1 * pgas;

            u0(m,IDN,k,j,i) = rho * gamma;
            u0(m,IM1,k,j,i) = wgas * gamma_sq * wr[IVX];
            u0(m,IM2,k,j,i) = wgas * gamma_sq * wr[IVY];
            u0(m,IM3,k,j,i) = wgas * gamma_sq * wr[IVZ];
            u0(m,IEN,k,j,i) = wgas * gamma_sq  - pgas - rho*gamma; //rho_eps * gamma_sq + (pgas + rho*gamma/(gamma+1.))*(v_sq*gamma_sq);
          }
        }
      );

  } // End initialization of Hydro variables

  // TODO Add rel. MHD.

  return;
}
