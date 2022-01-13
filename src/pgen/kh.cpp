//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kh.cpp
//  \brief Problem generator for KH instability
//  Sets up different initial conditions selected by flag "iprob"
//    - iprob=1 : tanh profile with a single mode perturbation
//    - iprob=2 : double tanh profile with a single mode perturbation

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn
//  \brief Problem Generator for KHI tests

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin) {
  // read problem parameters from input file
  int iprob  = pin->GetReal("problem","iprob");
  Real amp   = pin->GetReal("problem","amp");
  Real sigma = pin->GetReal("problem","sigma");
  Real vshear= pin->GetReal("problem","vshear");
  Real rho0  = pin->GetReal("problem","rho0");
  Real rho1  = pin->GetReal("problem","rho1");

  // capture variables for kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  EOS_Data &eos = pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto &w0 = pmbp->phydro->w0;
  auto &size = pmbp->pmb->mb_size;
  int &nhydro = pmbp->phydro->nhydro;
  int &nscalars = pmbp->phydro->nscalars;
  if (nscalars == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "KH test requires nscalars != 0" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!(eos.use_e)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "KH test requires hydro/use_e=true" << std::endl;
    exit(EXIT_FAILURE);
  }

  // initialize primitive variables
  par_for("pgen_kh1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    w0(m,IEN,k,j,i) = 20.0/gm1;
    w0(m,IVZ,k,j,i) = 0.0;

    // Lorentz factor (needed to initializve 4-velocity in SR)
    Real u00 = 1.0;
    bool is_relativistic = false;
    if (pmbp->phydro->is_special_relativistic ||
        pmbp->phydro->is_general_relativistic) {
      is_relativistic = true;
    }

    Real dens,pres,vx,vy,vz,scal;

    if (iprob == 1) {
      pres = 20.0;
      dens = 1.0;
      vx = -vshear*tanh(x2v/sigma);
      vy = -amp*vshear*sin(2.*M_PI*x1v)*exp( -SQR(x2v/sigma) );
      vz = 0.0;
      scal = 0.0;
      if (x2v > 0.0) scal = 1.0;
    } else if (iprob == 2) {
      pres = 1.0;
      vz = 0.0;
      if(x2v <= 0.0) {
        dens = rho0 - rho1*tanh((x2v-0.5)/sigma);
        vx = -vshear*tanh((x2v-0.5)/sigma);
        vy = -amp*vshear*sin(2.*M_PI*x1v)*exp( -SQR((x2v-0.5)/sigma) );
        if (is_relativistic) {
          u00 = 1.0/sqrt(1.0 - vx*vx - vy*vy);
        }
        scal = 0.0;
        if (x2v < 0.5) scal = 1.0;
      } else {
        dens = rho0 + rho1*tanh((x2v-0.5)/sigma);
        vx = vshear*tanh((x2v-0.5)/sigma);
        vy = amp*vshear*sin(2.*M_PI*x1v)*exp( -SQR((x2v-0.5)/sigma) );
        if (is_relativistic) {
          u00 = 1.0/sqrt(1.0 - vx*vx - vy*vy);
        }
        scal = 0.0;
        if (x2v > 0.5) scal = 1.0;
      }
    }

    // set primitives in both newtonian and SR hydro
    w0(m,IDN,k,j,i) = dens;
    w0(m,IEN,k,j,i) = pres/gm1;
    w0(m,IVX,k,j,i) = u00*vx;
    w0(m,IVY,k,j,i) = u00*vy;
    w0(m,IVZ,k,j,i) = u00*vz;
    // add passive scalars
    for (int n=nhydro; n<(nhydro+nscalars); ++n) {
      w0(m,n,k,j,i) = scal;
    }
  });

  // Convert primitives to conserved
  auto &u0 = pmbp->phydro->u0;
  pmbp->phydro->peos->PrimToCons(w0, u0);

  return;
}
