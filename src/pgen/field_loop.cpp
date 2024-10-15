//========================================================================================
// AthenaK: astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file field_loop.cpp
//! \brief Problem generator for advection tests with and without the shearing box. Can be
//! used with Hydro, but only for advection of density cylinders in shearing box.  Can be
//! used in MHD for advection of field loop both with and without shearing box.
//!
//! Can only be run in 2D or 3D.  Input parameters are:
//!  -  problem/rad   = radius of field loop
//!  -  problem/amp   = amplitude of vector potential (and therefore B)
//!  -  problem/drat  = density ratio in loop, to test density advection and conduction
//!  -  problem/press = amplitude of pressure in mhd
//! Without the shearing box the flow is automatically set to run along the diagonal.
//!
//! Various test cases are possible:
//!  - (iprob=1): field loop in x1-x2 plane (cylinder in 3D)
//!  - (iprob=2): field loop in x2-x3 plane (cylinder in 3D)
//!  - (iprob=3): field loop in x3-x1 plane (cylinder in 3D)
//!  - (iprob=4): rotated cylindrical field loop in 3D.
//! Only iprob=1,4 work with the shearing box.
//!
//! REFERENCE: T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD via
//! constrined transport", JCP, 205, 509 (2005)
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::_()
//! \brief Field loop advection problem generator.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  // Read initial conditions
  Real rad = pin->GetOrAddReal("problem","rad",0.0);
  Real amp = pin->GetOrAddReal("problem","amp",0.0);
  Real drat = pin->GetOrAddReal("problem","drat",1.0);
  Real vx0 = pin->GetOrAddReal("problem","vx0",0.0);
  Real press = pin->GetOrAddReal("problem","press",1.0);
  int iprob = pin->GetInteger("problem","iprob");
  Real cos_a2(0.0), sin_a2(0.0), lambda(0.0);

  // positions of density/field loops
  auto &msize = pmy_mesh_->mesh_size;
  Real lx = msize.x1max - msize.x1min;
  Real ly = msize.x2max - msize.x2min;
  Real lz = msize.x3max - msize.x3min;
  Real x0 = msize.x1min, y0 = msize.x2min;
  Real xpt[5] = {(x0+0.1*lx), (x0+0.3*lx), (x0+0.5*lx), (x0+0.7*lx), (x0+0.9*lx)};
  Real ypt[5] = {(y0+0.3*ly), (y0+0.7*ly), (y0+0.5*ly), (y0+0.2*ly), (y0+0.8*ly)};
  auto three_d = pmy_mesh_->three_d;
  Real diag;
  if (three_d) {
    diag = std::sqrt(lx*lx + ly*ly + lz*lz);
  } else {
    diag = std::sqrt(lx*lx + ly*ly);
  }
  Real vflow = diag; // normalize so crossing time along diagonal is one

  // For (iprob=4) -- rotated cylinder in 3D -- set up rotation angle and wavelength
  if (iprob == 4) {
    // We put 1 wavelength in each direction.  Hence the wavelength
    //     lambda = lx*cos_a;
    //     AND   lambda = lz*sin_a;  are both satisfied.
    if (lx == lz) {
      cos_a2 = sin_a2 = std::sqrt(0.5);
    } else {
      Real ang_2 = std::atan(lx/lz);
      sin_a2 = std::sin(ang_2);
      cos_a2 = std::cos(ang_2);
    }
    // Use the larger angle to determine the wavelength
    if (cos_a2 >= sin_a2) {
      lambda = lx*cos_a2;
    } else {
      lambda = lz*sin_a2;
    }
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  bool is_relativistic = pmbp->pcoord->is_special_relativistic ||
                         pmbp->pcoord->is_general_relativistic ||
                         pmbp->pcoord->is_dynamical_relativistic;
  // Initialize conserved variables in Hydro
  // Hydro only works with shearing box and iprob=1 or 4
  if (pmbp->phydro != nullptr) {
    auto &shearing_box_ = pmbp->phydro->psrc->shearing_box;
    if (!(shearing_box_)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Hydro field loop problem can only be run with shearing box"
         << std::endl;
      exit(EXIT_FAILURE);
    }
    if (iprob != 1 && iprob != 4) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "Hydro field loop problem can only be run with iprob=1 or 4"
         << std::endl;
      exit(EXIT_FAILURE);
    }
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    auto u0 = pmbp->phydro->u0;
    par_for("floop0", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real dens = 1.0;
      if (iprob == 1) {
        for (int n=0; n<5; ++n) {
          if (sqrt(SQR(x1v-xpt[n])+SQR(x2v-ypt[n])) <= (0.1*lx)) {dens *= drat;}
        }
      } else {
        Real x = x1v*cos_a2 + x3v*sin_a2;
        Real y = x2v;
        // shift x back to the domain -0.5*lambda <= x <= 0.5*lambda
        while (x >  0.5*lambda) x -= lambda;
        while (x < -0.5*lambda) x += lambda;
        if ((x*x + y*y) < rad*rad) {dens += amp;}
      }

      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = dens*vx0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) { u0(m,IEN,k,j,i) = 1.0/gm1 + 0.5*dens*vx0*vx0; }
    });
  }

  // Initialize conserved variables and magnetic field in MHD
  if (pmbp->pmhd != nullptr) {
    auto &shearing_box_ = pmbp->pmhd->psrc->shearing_box;
    if (shearing_box_) {
      if (iprob != 1 && iprob != 4) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
           << std::endl << "With shearing box, MHD field loop problem can only be run "
           << "with iprob=1 or 4" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    // start by computing vector potential
    DvceArray4D<Real> ax, ay, az;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = indcs.nx2 + 2*(indcs.ng);
    int ncells3 = indcs.nx3 + 2*(indcs.ng);
    Kokkos::realloc(ax,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);
    Kokkos::realloc(ay,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);
    Kokkos::realloc(az,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);

    auto nmb = pmbp->nmb_thispack;
    par_for("floop1", DevExeSpace(), 0,(nmb-1),ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1f    = LeftEdgeX  (i-is, nx1, x1min, x1max);
      Real x1v    = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3f    = LeftEdgeX  (k-ks, nx3, x3min, x3max);
      Real x3v    = CellCenterX(k-ks, nx3, x3min, x3max);

      ax(m,k,j,i) = 0.0;
      ay(m,k,j,i) = 0.0;
      az(m,k,j,i) = 0.0;
      // (iprob=1): field loop in x1-x2 plane (cylinder in 3D)
      if (iprob==1) {
        if (shearing_box_) {
          for (int n=0; n<5; ++n) {
            if (sqrt(SQR(x1f-xpt[n]) + SQR(x2f-ypt[n])) < (0.1*lx)) {
              az(m,k,j,i) = amp*((0.1*lx) - sqrt(SQR(x1f-xpt[n]) + SQR(x2f-ypt[n])));
            }
          }
        } else {
          if (sqrt(SQR(x1f-xpt[2]) + SQR(x2f-ypt[2])) < rad) {
            az(m,k,j,i) = amp*(rad - sqrt(SQR(x1f-xpt[2]) + SQR(x2f-ypt[2])));
          }
        }
      }

      // (iprob=2): field loop in x2-x3 plane (cylinder in 3D) centered on origin
      if (iprob==2) {
        if ((SQR(x2f) + SQR(x3f)) < rad*rad) {
          ax(m,k,j,i) = amp*(rad - sqrt(SQR(x2f) + SQR(x3f)));
        }
      }

      // (iprob=3): field loop in x3-x1 plane (cylinder in 3D) centered on origin
      if (iprob==3) {
        if ((SQR(x1f) + SQR(x3f)) < rad*rad) {
          ay(m,k,j,i) = amp*(rad - sqrt(SQR(x1f) + SQR(x3f)));
        }
      }

      // (iprob=4): rotated cylindrical field loop in 3D.  Similar to iprob=1 with a
      // rotation about the x2-axis.  Define coordinate systems (x1,x2,x3) and (x,y,z)
      // with the following transformation rules:
      //    x =  x1*std::cos(ang_2) + x3*std::sin(ang_2)
      //    y =  x2
      //    z = -x1*std::sin(ang_2) + x3*std::cos(ang_2)
      // This inverts to:
      //    x1  = x*std::cos(ang_2) - z*std::sin(ang_2)
      //    x2  = y
      //    x3  = x*std::sin(ang_2) + z*std::cos(ang_2)

      if (iprob==4) {
        Real x = x1v*cos_a2 + x3f*sin_a2;
        Real y = x2f;
        // shift x back to the domain -0.5*lambda <= x <= 0.5*lambda
        while (x >  0.5*lambda) x -= lambda;
        while (x < -0.5*lambda) x += lambda;
        if ((x*x + y*y) < rad*rad) {
          ax(m,k,j,i) = amp*(rad - sqrt(x*x + y*y))*(-sin_a2);
        }

        x = x1f*cos_a2 + x3v*sin_a2;
        y = x2f;
        // shift x back to the domain -0.5*lambda <= x <= 0.5*lambda
        while (x >  0.5*lambda) x -= lambda;
        while (x < -0.5*lambda) x += lambda;
        if ((x*x + y*y) < rad*rad) {
          az(m,k,j,i) = amp*(rad - sqrt(x*x + y*y))*(cos_a2);
        }
      }
    });

    // compute face-centered fields
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("floop2", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = (az(m,k,j+1,i) - az(m,k,j,i))/dx2 -
                        (ay(m,k+1,j,i) - ay(m,k,j,i))/dx3;
      b0.x2f(m,k,j,i) = (ax(m,k+1,j,i) - ax(m,k,j,i))/dx3 -
                        (az(m,k,j,i+1) - az(m,k,j,i))/dx1;
      b0.x3f(m,k,j,i) = (ay(m,k,j,i+1) - ay(m,k,j,i))/dx1 -
                        (ax(m,k,j+1,i) - ax(m,k,j,i))/dx2;

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = (az(m,k,j+1,i+1) - az(m,k,j,i+1))/dx2 -
                            (ay(m,k+1,j,i+1) - ay(m,k,j,i+1))/dx3;
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = (ax(m,k+1,j+1,i) - ax(m,k,j+1,i))/dx3 -
                            (az(m,k,j+1,i+1) - az(m,k,j+1,i))/dx1;
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = (ay(m,k+1,j,i+1) - ay(m,k+1,j,i))/dx1 -
                            (ax(m,k+1,j+1,i) - ax(m,k+1,j,i))/dx2;
      }
    });

    // Compute cell-centered fields if relativistic
    if (is_relativistic) {
      par_for("pgen_Bcc", DevExeSpace(), 0, (nmb-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // cell-centered fields are simple linear average of face-centered fields
        Real& w_bx = bcc0(m,IBX,k,j,i);
        Real& w_by = bcc0(m,IBY,k,j,i);
        Real& w_bz = bcc0(m,IBZ,k,j,i);
        w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
        w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
        w_bz = 0.0;
      });
    }

    // Initialize conserved variables in MHD
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    if (pmbp->padm != nullptr) {
      gm1 = 1.0;
    }
    auto u0 = pmbp->pmhd->u0;
    auto w0 = pmbp->pmhd->w0;
    // If relativity is enabled, use the velocity from the parameter file.
    Real vx, vy, vz;
    if (is_relativistic) {
      vx = pin->GetOrAddReal("problem", "vx", 1./1.2);
      vy = pin->GetOrAddReal("problem", "vy", 1./2.4);
      vz = pin->GetOrAddReal("problem", "vz", 0.0);
      Real vsq = vx*vx + vy*vy + vz*vz;
      Real W = 1.0/sqrt(1.0 - vsq);
      vx *= W;
      vy *= W;
      vz *= W;
    }
    par_for("shear1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      if (!is_relativistic) {
        u0(m,IDN,k,j,i) = 1.0;
        if (shearing_box_) {
          u0(m,IM1,k,j,i) = vx0;
          u0(m,IM2,k,j,i) = 0.0;
          u0(m,IM3,k,j,i) = 0.0;
        } else {
          u0(m,IM1,k,j,i) = vflow*lx/diag;
          u0(m,IM2,k,j,i) = vflow*ly/diag;
  //        if (three_d) {
  //          u0(m,IM3,k,j,i) = vflow*lz/diag;
  //        } else {
            u0(m,IM3,k,j,i) = 0.0;
  //        }
        }
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = press/gm1 +
           0.5*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) +
                SQR(u0(m,IM3,k,j,i)))/u0(m,IDN,k,j,i) +
           0.5*(SQR(0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1))) +
                SQR(0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i))) +
                SQR(0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i))));
        }
      } else {
        // If relativity is enabled, initialize the primitive variables instead.
        w0(m,IDN,k,j,i) = 1.0;
        w0(m,IVX,k,j,i) = vx;
        w0(m,IVY,k,j,i) = vy;
        w0(m,IVZ,k,j,i) = vz;
        w0(m,IEN,k,j,i) = press/gm1;
      }
    });

    // If relativity is enabled (but not DynGRMHD), call the C2P because we initialized
    // the primitive variables.
    if (is_relativistic && pmbp->padm == nullptr) {
      pmbp->pmhd->peos->PrimToCons(w0, bcc0, u0, is, ie, js, je, ks, ke);
    }
  }

  // Initialize the ADM variables if necessary
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
    // For DynGRMHD, the conserved variables can't be initialized until the ADM variables
    // have been populated.
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }
  return;
}
