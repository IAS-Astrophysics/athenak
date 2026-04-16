//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shwave.cpp
//! \brief Problem generator for linear HD & MHD shearing-wave (shwave) tests
//!
//! REFERENCE: Johnson & Gammie 2005, ApJ, 626, 978
//!            Johnson, Guan, & Gammie, ApJS, 177, 373 (2008)
//!
//! Three kinds of problems
//! - ipert = 1 - epicyclic motion
//! - ipert = 2 - Hydro incompressible (vortical) shwave test of JG5
//! - ipert = 3 - Hydro compressible shwave test of JG5
//! - ipert = 4 - MHD compressible shwave test of JGG8


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
#include "pgen/pgen.hpp"

#include <Kokkos_Random.hpp>

// User-defined history function (only used for compressible Hydro and MHD cases)
void ShwaveHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \struct ShwaveTestVariables
//! \brief container for variables shared with user-history functions

namespace {
struct ShwaveTestVariables {
  Real kx, ky, kz, qshear, omega0;
};

ShwaveTestVariables shw_var;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::_()
//  \brief

void ProblemGenerator::Shwave(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // read parameters from input file
  Real d0 = pin->GetReal("problem", "d0");
  Real p0 = pin->GetOrAddReal("problem", "p0",1.0);
  Real amp = pin->GetReal("problem", "amp");
  int ipert = pin->GetInteger("problem", "ipert");

  // box size and wavenumbers
  auto &msize = pmy_mesh_->mesh_size;
  Real Lx = msize.x1max - msize.x1min;
  Real Ly = msize.x2max - msize.x2min;
  Real Lz = msize.x3max - msize.x3min;
  shw_var.kx = (2.0*M_PI/Lx)*static_cast<Real>(pin->GetInteger("problem", "nwx"));
  shw_var.ky = (2.0*M_PI/Ly)*static_cast<Real>(pin->GetInteger("problem", "nwy"));
  shw_var.kz = (2.0*M_PI/Lz)*static_cast<Real>(pin->GetInteger("problem", "nwz"));

  // capture variables for kernel
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;

  if (pmbp->phydro != nullptr) {
    // Do some error checking in Hydro case
    if (pmbp->phydro->psbox_u == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "shwave problem generator only works in shearing box."
                << std::endl << "Likely missing <shearing_box> block in input file"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    if ((ipert < 1) || (ipert > 3)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "ipert must be 1,2, or 3 for hydro shwaves" << std::endl;
      exit(EXIT_FAILURE);
    }

    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    auto &u0 = pmbp->phydro->u0;
    shw_var.qshear = (pmbp->phydro->psbox_u->qshear);
    shw_var.omega0 = (pmbp->phydro->psbox_u->omega0);
    // epicyclic oscillations
    if (ipert == 1) {
      par_for("shwave1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u0(m,IDN,k,j,i) = d0;
        u0(m,IM1,k,j,i) = amp*d0;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*SQR(amp);
        }
      });
    // incompressible (vortical) hydro shwave of JG05
    } else if (ipert == 2) {
      auto sv = shw_var;
      par_for("shwave2", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real rvx = amp*sin(sv.kx*x1v + sv.ky*x2v);
        Real rvy = -amp*(sv.kx/sv.ky)*sin(sv.kx*x1v + sv.ky*x2v);
        u0(m,IDN,k,j,i) = d0;
        u0(m,IM1,k,j,i) = d0*rvx;
        u0(m,IM2,k,j,i) = d0*rvy;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*(SQR(rvx) + SQR(rvy));
        }
      });
    // compressible hydro shwave of JG05
    } else if (ipert == 3) {
      // enroll user history function for compressible hydro shwaves
      user_hist_func = ShwaveHistory;

      auto sv = shw_var;
      par_for("shwave3", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

        Real rvx = amp*cos(sv.kx*x1v + sv.ky*x2v);
        Real rvy = amp*(sv.ky/sv.kx)*cos(sv.kx*x1v + sv.ky*x2v);
        u0(m,IDN,k,j,i) = d0;
        u0(m,IM1,k,j,i) = -d0*rvx;
        u0(m,IM2,k,j,i) = -d0*rvy;
        u0(m,IM3,k,j,i) = 0.0;
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*(SQR(rvx) + SQR(rvy));
        }
      });
    }
  }

  if (pmbp->pmhd != nullptr) {
    // Do some error checking in MHD case
    if (pmbp->pmhd->psbox_u == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "shwave problem generator only works in shearing box"
                << std::endl << "Likely missing <shearing_box> block in input file"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    if (ipert != 4) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MHD shwave test requires ipert = 4." << std::endl;
      exit(EXIT_FAILURE);
    }

    // enroll user history function for compressible hydro shwaves
    user_hist_func = ShwaveHistory;
    shw_var.qshear = (pmbp->pmhd->psbox_u->qshear);
    shw_var.omega0 = (pmbp->pmhd->psbox_u->omega0);

    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    auto u0 = pmbp->pmhd->u0;
    auto b0 = pmbp->pmhd->b0;
    auto sv = shw_var;

    Real beta = pin->GetReal("problem", "beta");
    Real B02 = p0/beta;
    Real k2 = SQR(sv.kx)+SQR(sv.ky)+SQR(sv.kz);
    Real rbx = sv.ky*std::sqrt(B02/(SQR(sv.kx)+SQR(sv.ky)));
    Real rby = -sv.kx*std::sqrt(B02/(SQR(sv.kx)+SQR(sv.ky)));
    Real rbz = 0.0;

    Real sch = eos.iso_cs/sv.omega0;
    Real cf1 = std::sqrt(B02*(1.0+beta));
    Real cf2 = amp*std::sqrt(sch*std::sqrt(k2*beta/(1.0+beta)));
    Real vd = cf1/std::sqrt(k2)*cf2;

    par_for("shwave4", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
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

      Real csk = cos(sv.kx*x1v+sv.ky*x2v+sv.kz*x3v);
      Real rd = d0*(1.0+cf2*csk);
      u0(m,IDN,k,j,i) = rd;
      u0(m,IM1,k,j,i) = rd*vd*sv.kx*csk;
      u0(m,IM2,k,j,i) = rd*vd*sv.ky*csk;
      u0(m,IM3,k,j,i) = rd*vd*sv.kz*csk;
      if (eos.is_ideal) {
        u0(m,IEN,k,j,i) = p0/gm1 + 0.5*rd*SQR(vd*csk)*k2;
      }
    });

    // compute vector potentials
    DvceArray4D<Real> a1, a2, a3;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = indcs.nx2 + 2*(indcs.ng);
    int ncells3 = indcs.nx3 + 2*(indcs.ng);
    Kokkos::realloc(a1,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);
    Kokkos::realloc(a2,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);
    Kokkos::realloc(a3,(pmbp->nmb_thispack),ncells3,ncells2,ncells1);

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

      Real temp = cf2/k2*std::sin(sv.kx*x1v+sv.ky*x2f+sv.kz*x3f);
      a1(m,k,j,i) = temp*(rby*sv.kz-rbz*sv.ky);
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

      Real temp = cf2/k2*std::sin(sv.kx*x1f+sv.ky*x2v+sv.kz*x3f);
      a2(m,k,j,i) = temp*(rbz*sv.kx-rbx*sv.kz);
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

      Real temp = cf2/k2*std::sin(sv.kx*x1f+sv.ky*x2f+sv.kz*x3v);
      a3(m,k,j,i) = temp*(rbx*sv.ky-rby*sv.kx);
    });

    // Compute face-centered fields from curl(A)
    par_for("shwave3_b1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = rbx
                        +(a3(m,k,j+1,i) - a3(m,k,j,i))/dx2
                        -(a2(m,k+1,j,i) - a2(m,k,j,i))/dx3;
    });
    par_for("shwave3_b2", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je+1,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx3 = size.d_view(m).dx3;

      b0.x2f(m,k,j,i) = rby
                        +(a1(m,k+1,j,i) - a1(m,k,j,i))/dx3
                        -(a3(m,k,j,i+1) - a3(m,k,j,i))/dx1;
    });
    par_for("shwave3_b3", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke+1,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;

      b0.x3f(m,k,j,i) = rbz
                        +(a2(m,k,j,i+1) - a2(m,k,j,i))/dx1
                        -(a1(m,k,j+1,i) - a1(m,k,j,i))/dx2;
    });

    // Compute total energy in ideal-gas EOS
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

//----------------------------------------------------------------------------------------
// Function for computing history variables
// 0 = < dVyc >

void ShwaveHistory(HistoryData *pdata, Mesh *pm) {
  // capture class variabels for kernel
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;
  auto sv = shw_var;
  Real kx = sv.kx + (sv.qshear)*(sv.omega0)*(pm->time)*sv.ky;
  Real omega_t = (sv.omega0)*(pm->time);

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;

  pdata->nhist = 1;
  pdata->label[0] = "dVyc";
  bool is_mhd = false;
  if (pm->pmb_pack->phydro == nullptr) {
    is_mhd = true;
    pdata->label[0] = "dByc";
  }
  auto &w0_ = (is_mhd)? pm->pmb_pack->pmhd->w0 : pm->pmb_pack->phydro->w0;
  DvceArray5D<Real> bcc_temp;
  auto &bcc0_ = (is_mhd)? pm->pmb_pack->pmhd->bcc0 : bcc_temp;

  Kokkos::parallel_reduce("HistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    // summed variables:
    array_sum::GlobalSum hvars;

    // calculate dVyc
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    if (is_mhd) {
      hvars.the_array[0] = vol*2.0*(bcc0_(m,IBY,k,j,i) - (0.2-0.15*omega_t));
      hvars.the_array[0] *= cos(kx*x1v + sv.ky*x2v + sv.kz*x3v);
    } else {
      hvars.the_array[0] = vol*2.0*w0_(m,IVY,k,j,i)*cos(kx*x1v + sv.ky*x2v);
    }

    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
  Kokkos::fence();

  // store data into hdata array
  for (int n=0; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }
  return;
}
