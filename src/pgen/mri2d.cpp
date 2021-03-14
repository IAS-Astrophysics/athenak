//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mri2d.cpp
//! \brief Problem generator for 2D MRI simulations using the shearing sheet based on
//!  "A powerful local shear instability in weakly magnetized disks. III - Long-term
//!  evolution in a shearing sheet" by Hawley & Balbus.  Based on the hgb.cpp problem
//!  generator in Athena++
//! REFERENCE: Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992).
//!
//! Two different field configurations are possible:
//! - ifield = 1 - Bz=B0 sin(x1) field with zero-net-flux [default]
//! - ifield = 2 - uniform Bz

#include <Kokkos_Random.hpp>

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // cout, endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "srcterms/srcterms.hpp"
#include "utils/grid_locations.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::_()
//  \brief

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  if (pmbp->pmesh->nx3gt1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "mri2d problem generator only works in 2D (nx3=1)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // initialize problem variables
  Real amp   = pin->GetReal("problem","amp");
  Real beta  = pin->GetReal("problem","beta");
  int nwx    = pin->GetOrAddInteger("problem","nwx",1);
  int ifield = pin->GetOrAddInteger("problem","ifield",1);

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  Real d0 = 1.0;
  Real p0 = 10.0/(eos.gamma);
  Real B0 = std::sqrt(2.0*p0/beta);


  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  Real kx = 2.0*(M_PI/x1size)*(static_cast<Real>(nwx));

  // capture variables for kernel
  int &nx1 = pmbp->mb_cells.nx1;
  int &is = pmbp->mb_cells.is, &ie = pmbp->mb_cells.ie;
  int &js = pmbp->mb_cells.js, &je = pmbp->mb_cells.je;
  int &ks = pmbp->mb_cells.ks, &ke = pmbp->mb_cells.ke;
  auto &size = pmbp->pmb->mbsize;

  // Initialize magnetic field first, so entire arrays are initialized before adding 
  // magnetic energy to conserved variables in next loop.  For 2D shearing box
  // B1=Bx, B2=Bz, B3=By
  // ifield = 1 - Bz=B0 sin(kx*xav1) field with zero-net-flux [default]
  // ifield = 2 - uniform Bz
  auto b0 = pmbp->pmhd->b0;
  par_for("mri2d-b", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));

      if (ifield == 1) {
        b0.x1f(m,k,j,i) = 0.0;
        b0.x2f(m,k,j,i) = B0*sin(kx*x1v);
        b0.x3f(m,k,j,i) = 0.0;
        if (i==ie) b0.x1f(m,k,j,ie+1) = 0.0;
        if (j==je) b0.x2f(m,k,je+1,i) = B0*sin(kx*x1v);
        if (k==ke) b0.x3f(m,ke+1,j,i) = 0.0;
      } else if (ifield == 2) {
        b0.x1f(m,k,j,i) = 0.0;
        b0.x2f(m,k,j,i) = B0;
        b0.x3f(m,k,j,i) = 0.0;
        if (i==ie) b0.x1f(m,k,j,ie+1) = 0.0;
        if (j==je) b0.x2f(m,k,je+1,i) = B0;
        if (k==ke) b0.x3f(m,ke+1,j,i) = 0.0;
      }
    }
  );

  // Initialize conserved variables
  Real qshear = pin->GetReal("shearing_box","qshear");
  Real omega0 = pin->GetReal("shearing_box","omega0");
  auto &mbgid = pmbp->pmb->mbgid;
  auto u0 = pmbp->pmhd->u0;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(5374857);
  par_for("mri2d-u", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
      Real rd = d0;
      Real rp = p0;
      auto rand_gen = rand_pool64.get_state();  // get random number state this thread
      Real rval = 1.0 + amp*(rand_gen.frand() - 0.5);
      if (eos.is_adiabatic) {
        rp = rval*p0;
        rd = d0;
      } else {
        rd = rval*d0;
      }
      u0(m,IDN,k,j,i) = rd;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = -rd*qshear*omega0*x1v;

      if (eos.is_adiabatic) {
        u0(m,IEN,k,j,i) = rp/gm1 + 0.5*SQR(u0(m,IM3,k,j,i))/rd
          + 0.5*SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i)));
      }
      rand_pool64.free_state(rand_gen);  // free state for use by other threads
    }
  );

  return;
}
