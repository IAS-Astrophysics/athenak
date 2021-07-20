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

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_positions.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn
//  \brief Problem Generator for the shock tube (Riemann problem) tests

void ProblemGenerator::KH_Rel_(MeshBlockPack *pmbp, ParameterInput *pin)
{

  // Parse left state read from input file: dl,ul,vl,wl,[pl]
 
  Real A     = pin->GetReal("problem","A");
  Real a     = pin->GetReal("problem","a");
  Real sigma = pin->GetReal("problem","sigma");
  Real Vshear= pin->GetReal("problem","Vshear");
  Real rho0  = pin->GetReal("problem","rho0");
  Real rho1  = pin->GetReal("problem","rho1");

  // Initialize the discontinuity in the Hydro variables ---------------------------------

  // capture variables for kernel
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mbsize;


  par_for("pgen_ot1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real x1 = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
      Real x2 = CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m));

      Real w[5];

	   w[IPR] = 1.;
	   w[IVZ] = 0.;

	  if( x2 <=0.){
	     w[IDN] = rho0 - rho1*std::tanh((x2+0.5)/a);
	     w[IVX] = -Vshear*std::tanh((x2+0.5)/a);
	     w[IVY] = -A*Vshear*std::sin(2.*M_PI*x1)*std::exp(-(x2+0.5)*(x2+0.5)/sigma/sigma);
	   }
	   else{
	     w[IDN] = rho0 + rho1*std::tanh((x2-0.5)/a);
	     w[IVX] =  Vshear*std::tanh((x2-0.5)/a);
	     w[IVY] =  A*Vshear*std::sin(2.*M_PI*x1)*std::exp(-(x2-0.5)*(x2-0.5)/sigma/sigma);
	   }

	    Real v_sq = SQR(w[IVX]) + SQR(w[IVY]) + SQR(w[IVZ]);
	    Real gamma_sq = 1./(1.-v_sq);
	    Real gamma = sqrt(gamma_sq);

	    auto const& rho = w[IDN];
	    auto const& pgas = w[IPR];
	    auto const& gamma_adi = eos.gamma;
	    Real rho_eps = pgas / gm1;
	    //FIXME ERM: Only ideal fluid for now
	    Real wgas = rho + gamma_adi / gm1 * pgas;

            u0(m,IDN,k,j,i) = rho*gamma;
            u0(m,IM1,k,j,i) = wgas * gamma_sq * w[IVX];
            u0(m,IM2,k,j,i) = wgas * gamma_sq * w[IVY];
            u0(m,IM3,k,j,i) = wgas * gamma_sq * w[IVZ];
            u0(m,IEN,k,j,i) = wgas * gamma_sq  - pgas - rho*gamma; 

	}
      );



  return;
}
