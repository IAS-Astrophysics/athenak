//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shwave.cpp
//! \brief Problem generator for linear HD & MHD in in shearing sheet
//!
//! REFERENCE: Johnson & Gammie 2005, ApJ, 626, 978
//!            Johnson, Guan, & Gammie, ApJS, 177, 373 (2008)
//!
//! Three kinds of problems
//! - ipert = 1 - epicyclic motion
//! - ipert = 2 - Hydro compressive shwave test of JG5
//! - ipert = 3 - MHD compressive shwave test of JGG8


// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // cout, endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "pgen.hpp"

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::_()
//  \brief

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // box size
  auto &msize = pmy_mesh_->mesh_size;
  Real Lx = msize.x1max - msize.x1min;
  Real Ly = msize.x2max - msize.x2min;
  Real Lz = msize.x3max - msize.x3min;

  // read parameters from input file
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real d0 = pin->GetReal("problem", "d0");
  Real p0 = 1.0;
  if (eos.is_ideal) {
    p0 = pin->GetReal("problem", "p0");
  }
  Real amp = pin->GetReal("problem", "amp");
  int ipert = pin->GetInteger("problem", "ipert");
  Real kx, ky, kz;
  Real beta;
  if (ipert == 1) {
    beta = 0.0;
    kx = 0.0;
    ky = 0.0;
    kz = 0.0;
  } else if (ipert == 2) {
    beta = 0.0;
    kx = (2.0*M_PI/Lx)*static_cast<Real>(pin->GetInteger("problem", "nwx"));
    ky = (2.0*M_PI/Ly)*static_cast<Real>(pin->GetInteger("problem", "nwy"));
    kz = 0.0;
  } else if (ipert == 3) {
    beta = pin->GetReal("problem", "beta");
    kx = (2.0*M_PI/Lx)*static_cast<Real>(pin->GetInteger("problem", "nwx"));
    ky = (2.0*M_PI/Ly)*static_cast<Real>(pin->GetInteger("problem", "nwy"));
    kz = (2.0*M_PI/Lz)*static_cast<Real>(pin->GetInteger("problem", "nwz"));
  }
  int error_output_flag = pin->GetInteger("problem", "error_output");

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;

  if (pmbp->phydro != nullptr) {
    if (pmbp->phydro->psrc == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Shearing box source terms are not enabled." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!pmbp->phydro->shearing_box) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "shwave problem generator only works in shearing box"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    if (ipert == 1) {
      Real rvx = 0.1*iso_cs;
      par_for("shwave1_c", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u0(m,IDN,k,j,i) = d0;
        u0(m,IM1,k,j,i) = d0*rvx;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*SQR(rvx);
        }
      });
    } else if (ipert == 2) {
      Real rvx = amp*iso_cs*std::cos(kx*x1 + ky*x2);
      Real rvy = amp*iso_cs*(ky/kx)*std::cos(kx*x1 + ky*x2);
      par_for("shwave2_c", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u0(m,IDN,k,j,i) = d0;
        u0(m,IM1,k,j,i) = -d0*rvx;
        u0(m,IM2,k,j,i) = -d0*rvy;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*(SQR(rvx)+SQR(rvy));
        }
      });
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Hydro test needs to have ipert = 1 or 2." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  if (pmbp->pmhd != nullptr) {
    if (pmbp->pmhd->psrc == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Shearing box source terms are not enabled." << std::endl;
      exit(EXIT_FAILURE);
    }
    if (!pmbp->pmhd->shearing_box) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "jgg problem generator only works in shearing box"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    if (ipert != 3) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "MHD test needs to have ipert = 3." << std::endl;
      exit(EXIT_FAILURE);
    }
    // Initialize conserved variables in MHD
    Real gm1 = eos.gamma - 1.0;
    auto u0 = pmbp->pmhd->u0;
    auto b0 = pmbp->pmhd->b0;
    Real omega0 = pmbp->pmhd->psb->omega0;

    Real B02 = p0/beta;
    Real k2 = SQR(kx)+SQR(ky)+SQR(kz);
    Real rbx = ky*std::sqrt(B02/(SQR(kx)+SQR(ky)));
    Real rby = -kx*std::sqrt(B02/(SQR(kx)+SQR(ky)));
    Real rbz = 0.0;

    Real sch = eos.iso_cs/omega0;
    Real cf1 = std::sqrt(B02*(1.0+beta));
    Real cf2 = amp*std::sqrt(sch*std::sqrt(k2*beta/(1.0+beta)));
    Real vd = cf1/std::sqrt(k2)*cf2;

    par_for("shwave3_c", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

      Real CS = std::cos(kx*x1v+ky*x2v+kz*x3v);
      Real rd = d0*(1.0+cf2*CS);
      u0(m,IDN,k,j,i) = rd;
      u0(m,IM1,k,j,i) = rd*vd*kx*CS;
      u0(m,IM2,k,j,i) = rd*vd*ky*CS;
      u0(m,IM3,k,j,i) = rd*vd*kz*CS;
      if (eos.is_ideal) {
        u0(m,IEN,k,j,i) = p0/gm1 + 0.5*rd*SQR(vd*CS)*k2;
      }
    });

    // compute vector potential
    DvceArray4D<Real> a1, a2, a3;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = indcs.nx2 + 2*(indcs.ng);
    int ncells3 = indcs.nx3 + 2*(indcs.ng);
    Kokkos::realloc(a1,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);
    Kokkos::realloc(a1,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);
    Kokkos::realloc(a1,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);

    par_for("shwave3_a1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke+1,js,je+1,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v    = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2f    = LeftEdgeX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3f    = LeftEdgeX(k-ks, nx3, x3min, x3max);

      Real temp = cf2/k2*std::sin(kx*x1v+ky*x2f+kz*x3f);
      a1(m,k,j,i) = temp*(rby*kz-rbz*ky);
    });

    par_for("shwave3_a2", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke+1,js,je,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1f    = LeftEdgeX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v    = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3f    = LeftEdgeX(k-ks, nx3, x3min, x3max);

      Real temp = cf2/k2*std::sin(kx*x1f+ky*x2v+kz*x3f);
      a2(m,k,j,i) = temp*(rbz*kx-rbx*kz);
    });

    par_for("shwave3_a3", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1f    = LeftEdgeX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2f    = LeftEdgeX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v    = CellCenterX(k-ks, nx3, x3min, x3max);

      Real temp = cf2/k2*std::sin(kx*x1f+ky*x2f+kz*x3v);
      a3(m,k,j,i) = temp*(rbx*ky-rby*kx);
    });

    par_for("shwave3_b1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = rbx
                        +(a3(m,k,j+1,i) - a3(m,k,j,i))/dx2
                        -(a2(m,k+1,j,i) - a2(m,k,j,i))/dx3;
    });
    par_for("shwave3_b2", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je+1,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx3 = size.d_view(m).dx3;

      b0.x2f(m,k,j,i) = rby
                        +(a1(m,k+1,j,i) - a1(m,k,j,i))/dx3
                        -(a3(m,k,j,i+1) - a3(m,k,j,i))/dx1;
    });
    par_for("shwave3_b3", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke+1,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;

      b0.x3f(m,k,j,i) = rbz
                        +(a2(m,k,j,i+1) - a2(m,k,j,i))/dx1
                        -(a1(m,k,j+1,i) - a1(m,k,j,i))/dx2;
    });
    if (eos.is_ideal) {
      par_for("shwave3_e", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &b1m = b0.x1f(m,k,j,i);
        Real &b1p = b0.x1f(m,k,j,i+1);
        Real &b2m = b0.x1f(m,k,j,i);
        Real &b2p = b0.x1f(m,k,j+1,i);
        Real &b3m = b0.x1f(m,k,j,i);
        Real &b3p = b0.x1f(m,k+1,j,i);
        u0(m,IEN,k,j,i) += 0.125*(SQR(b1m+b1p)+SQR(b2m+b2p)+SQR(b3m+b3p));
      });
    }
  }
  return;
}
