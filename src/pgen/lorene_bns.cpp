//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lorene_bns.cpp
//  \brief Initial data reader for binary neutron star data with LORENE
//
//  LORENE is available at https://lorene.obspm.fr/index.html

#include <stdio.h>
#include <math.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "utils/tov/tov_utils.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"
#include "utils/tov/tov_tabulated.hpp"

// Lorene
#include "bin_ns.h"
#include "unites.h"

// Prototype for user-defined history function
void BNSHistory(HistoryData *pdata, Mesh *pm);
void LoreneBNSRefinementCondition(MeshBlockPack *pmbp);

// Prototypes for magnetic vector potential
KOKKOS_INLINE_FUNCTION
static Real A1(Real x, Real y, Real z, Real I_0, Real r_0);
KOKKOS_INLINE_FUNCTION
static Real A2(Real x, Real y, Real z, Real I_0, Real r_0);

template<class TOVEOS>
void SetupBNS(ParameterInput *pin, Mesh* pmy_mesh_) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  // Conversion constants to translate between Lorene and AthenaK
  const Real c_light  = Lorene::Unites::c_si;      // speed of light [m/s]
  const Real nuc_dens = Lorene::Unites::rhonuc_si; // Nuclear density [kg/m^3]
  const Real G_grav   = Lorene::Unites::g_si;      // Gravitational constant [m^3/kg/s^2]
  const Real M_sun    = Lorene::Unites::msol_si;   // Solar mass [kg]
  const Real mu0      = Lorene::Unites_mag::mu_si; // Mag. vacuum permeability [N/A^2]
  const Real eps0     = 1.0/(mu0 * c_light * c_light);

  const Real athenaM  = M_sun;
  const Real athenaL  = athenaM * G_grav / (c_light * c_light);
  const Real athenaT  = athenaL / c_light;
  // This is just a guess based on what Cactus uses
  const Real athenaB  = (1.0 / athenaL / std::sqrt(eps0 * G_grav / (c_light * c_light)));

  // Other quantities in terms of Athena units
  const Real coord_unit = athenaL / 1.0e3;                 // From km
  const Real rho_unit = athenaM/(athenaL*athenaL*athenaL); // from kg/m^3
  const Real ener_unit = 1.0; // c^2
  const Real vel_unit = athenaL / athenaT / c_light; // c
  const Real B_unit = athenaB / 1.0e9; // 10^9 T

  std::string fname = pin->GetString("problem", "initial_data_file");
  Real rho_cut = pin->GetOrAddReal("problem", "rho_cut", 1e-5);
  Real b_max = pin->GetOrAddReal("problem", "b_max", 1e12) / 8.3519664583273e+19;
  Real r_0 = pin->GetOrAddReal("problem", "r_0_current", 5.0);
  Real I_0 = 4*r_0*b_max/(23.0*M_PI);

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;

  int width = nmb*ncells1*ncells2*ncells3;

  Real *x_coords = new Real[width];
  Real *y_coords = new Real[width];
  Real *z_coords = new Real[width];

  // Set up the 1D EOS
  TOVEOS eos{pin};

  // Enable ye if the EOS supports it.
  constexpr bool use_ye = tov::UsesYe<TOVEOS>;

  if (global_variable::my_rank == 0) {
    std::cout << "Allocated coordinates of size " << width << std::endl;
  }

  // Populate coordinates for LORENE
  int idx = 0;
  for (int m = 0; m < nmb; m++) {
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    int nx1 = indcs.nx1;

    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    int nx2 = indcs.nx2;

    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;
    int nx3 = indcs.nx3;

    for (int k = 0; k < ncells3; k++) {
      Real z = CellCenterX(k - ks, nx3, x3min, x3max);
      for (int j = 0; j < ncells2; j++) {
        Real y = CellCenterX(j - js, nx2, x2min, x2max);
        for (int i = 0; i < ncells1; i++) {
          Real x = CellCenterX(i - is, nx1, x1min, x1max);

          x_coords[idx] = coord_unit*x;
          y_coords[idx] = coord_unit*y;
          z_coords[idx] = coord_unit*z;

          // Increment flat index
          idx++;
        }
      }
    }
  }

  // Interpolate the data
  if (global_variable::my_rank == 0) {
    std::cout << "Coordinates assigned." << std::endl;
  }
  Lorene::Bin_NS *bns = new Lorene::Bin_NS(width, x_coords, y_coords, z_coords,
                                           fname.c_str());

  // Free the coordinates, since we'll no longer need them.
  delete[] x_coords;
  delete[] y_coords;
  delete[] z_coords;

  if (global_variable::my_rank == 0) {
    std::cout << "Coordinates freed." << std::endl;
  }

  // Capture variables for kernel; note that when Z4c is enabled, the gauge variables
  // are part of the Z4c class.
  auto &u_adm = pmbp->padm->u_adm;
  auto &adm = pmbp->padm->adm;
  auto &w0 = pmbp->pmhd->w0;
  // Because Lorene operates only on the CPU, we can't construct the data on the GPU.
  // Instead, we create a mirror guaranteed to be on the CPU, populate the data there,
  // then move it back to the GPU if applicable.
  HostArray5D<Real>::HostMirror host_u_adm = Kokkos::create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0 = Kokkos::create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c;
  adm::ADM::ADMhost_vars host_adm;
  if (pmbp->pz4c != nullptr) {
    host_u_z4c = Kokkos::create_mirror_view(pmbp->pz4c->u0);
    host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
    host_adm.beta_u.InitWithShallowSlice(host_u_z4c,
        z4c::Z4c::I_Z4C_BETAX, z4c::Z4c::I_Z4C_BETAZ);
  } else {
    host_adm.alpha.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_ALPHA);
    host_adm.beta_u.InitWithShallowSlice(host_u_adm,
        adm::ADM::I_ADM_BETAX, adm::ADM::I_ADM_BETAZ);
  }
  host_adm.g_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm,
      adm::ADM::I_ADM_KXX, adm::ADM::I_ADM_KZZ);

  if (global_variable::my_rank == 0) {
    std::cout << "Host mirrors created." << std::endl;
  }

  // Copy the interpolated data into the mirrored views.
  idx = 0;
  for (int m = 0; m < nmb; m++) {
    for (int k = 0; k < ncells3; k++) {
      for (int j = 0; j < ncells2; j++) {
        for (int i = 0; i < ncells1; i++) {
          // Extract metric quantities
          host_adm.alpha(m, k, j, i) = bns->nnn[idx];
          host_adm.beta_u(m, 0, k, j, i) = bns->beta_x[idx];
          host_adm.beta_u(m, 1, k, j, i) = bns->beta_y[idx];
          host_adm.beta_u(m, 2, k, j, i) = bns->beta_z[idx];

          Real g3d[NSPMETRIC];
          host_adm.g_dd(m, 0, 0, k, j, i) = g3d[S11] = bns->g_xx[idx];
          host_adm.g_dd(m, 0, 1, k, j, i) = g3d[S12] = bns->g_xy[idx];
          host_adm.g_dd(m, 0, 2, k, j, i) = g3d[S13] = bns->g_xz[idx];
          host_adm.g_dd(m, 1, 1, k, j, i) = g3d[S22] = bns->g_yy[idx];
          host_adm.g_dd(m, 1, 2, k, j, i) = g3d[S23] = bns->g_yz[idx];
          host_adm.g_dd(m, 2, 2, k, j, i) = g3d[S33] = bns->g_zz[idx];

          host_adm.vK_dd(m, 0, 0, k, j, i) = coord_unit * bns->k_xx[idx];
          host_adm.vK_dd(m, 0, 1, k, j, i) = coord_unit * bns->k_xy[idx];
          host_adm.vK_dd(m, 0, 2, k, j, i) = coord_unit * bns->k_xz[idx];
          host_adm.vK_dd(m, 1, 1, k, j, i) = coord_unit * bns->k_yy[idx];
          host_adm.vK_dd(m, 1, 2, k, j, i) = coord_unit * bns->k_yz[idx];
          host_adm.vK_dd(m, 2, 2, k, j, i) = coord_unit * bns->k_zz[idx];

          // Extract hydro quantities
          // Note that Lorene does not necessarily use the same baryon rest-mass as
          // AthenaK. The most reasonable thing to do, then, is to extract the total
          // energy density, which is invariant, and use that with the 1D EOS.
          Real egas = bns->nbar[idx]*(1.0 + bns->ener_spec[idx] / ener_unit)/rho_unit;
          Real& rho = host_w0(m, IDN, k, j, i);
          rho = eos.template GetRhoFromE<tov::LocationTag::Host>(egas);
          Real vu[3] = {bns->u_euler_x[idx] / vel_unit,
                        bns->u_euler_y[idx] / vel_unit,
                        bns->u_euler_z[idx] / vel_unit};

          // Check for garbage values thrown in by Lorene.
          if (rho <= rho_cut || !Kokkos::isfinite(rho)) {
            rho = 0.0;
            host_w0(m, IPR, k, j, i) = 0.0;
            vu[0] = 0.0;
            vu[1] = 0.0;
            vu[2] = 0.0;
          }

          host_w0(m, IPR, k, j, i) = eos.template
                                     GetPFromRho<tov::LocationTag::Host>(rho);

          // If the electron fraction is available, find it in the 1D EOS.
          if constexpr (use_ye) {
            host_w0(m, IYF, k, j, i) = eos.template
                                       GetYeFromRho<tov::LocationTag::Host>(rho);
          }

          // Before we store the velocity, we need to make sure it's physical and
          // calculate the Lorentz factor. If the velocity is superluminal, we make a
          // last-ditch attempt to salvage the solution by rescaling it to
          // vsq = 1.0 - 1e-15
          Real vsq = Primitive::SquareVector(vu, g3d);
          if (1.0 - vsq <= 0) {
            std::cout << "The velocity is superluminal!" << std::endl
                      << "Attempting to adjust..." << std::endl;
            Real fac = Kokkos::sqrt((1.0 - 1e-15)/vsq);
            vu[0] *= fac;
            vu[1] *= fac;
            vu[2] *= fac;
            vsq = 1.0 - 1.0e-15;
          }
          Real W = Kokkos::sqrt(1.0 / (1.0 - vsq));

          host_w0(m, IVX, k, j, i) = W*vu[0];
          host_w0(m, IVY, k, j, i) = W*vu[1];
          host_w0(m, IVZ, k, j, i) = W*vu[2];

          idx++;
        }
      }
    }
  }

  if (global_variable::my_rank == 0) {
    std::cout << "Host mirrors filled." << std::endl;
  }

  Real sep = bns->dist/coord_unit;
  if (global_variable::my_rank == 0) {
    std::cout << "sep = " << sep << std::endl;
  }

  // Cleanup
  delete bns;

  if (global_variable::my_rank == 0) {
    std::cout << "Lorene freed." << std::endl;
  }

  // Copy the data to the GPU.
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  if (pmbp->pz4c != nullptr) {
    Kokkos::deep_copy(pmbp->pz4c->u0, host_u_z4c);
  }

  if (global_variable::my_rank == 0) {
    std::cout << "Data copied." << std::endl;
  }

  // compute vector potential over all faces
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
  Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

  par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    a1(m,k,j,i) = A1(x1v - 0.5*sep, x2f, x3f, I_0, r_0) +
                  A1(x1v + 0.5*sep, x2f, x3f, I_0, r_0);
    a2(m,k,j,i) = A2(x1f - 0.5*sep, x2v, x3f, I_0, r_0) +
                  A2(x1f + 0.5*sep, x2v, x3f, I_0, r_0);
    a3(m,k,j,i) = 0.0;

    // When neighboring MeshBock is at finer level, compute vector potential as sum of
    // values at fine grid resolution.  This guarantees flux on shared fine/coarse
    // faces is identical.

    // Correct A1 at x2-faces, x3-faces, and x2x3-edges
    if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
        (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
        (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
        (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
        (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
        (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
        (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
        (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
        (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
        (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
        (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
        (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
        (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
        (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
        (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
        (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
      Real xl = x1v + 0.25*dx1;
      Real xr = x1v - 0.25*dx1;
      a1(m,k,j,i) = 0.5*((A1(xl-0.5*sep, x2f, x3f, I_0, r_0) +
                         A1(xl+0.5*sep, x2f, x3f, I_0, r_0)) +
                         (A1(xr-0.5*sep, x2f, x3f, I_0, r_0) +
                         A1(xr+0.5*sep, x2f, x3f, I_0, r_0)));
    }

    // Correct A2 at x1-faces, x3-faces, and x1x3-edges
    if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
        (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
        (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
        (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
        (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
        (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
        (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
        (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
        (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
        (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
        (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
        (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
        (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
        (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
        (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
        (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
        (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
        (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
      Real xl = x2v + 0.25*dx2;
      Real xr = x2v - 0.25*dx2;
      a2(m,k,j,i) = 0.5*((A2(x1f-0.5*sep, xl, x3f, I_0, r_0) +
                         A2(x1f+0.5*sep, xl, x3f, I_0, r_0)) +
                         (A2(x1f-0.5*sep, xr, x3f, I_0, r_0) +
                         A2(x1f+0.5*sep, xr, x3f, I_0, r_0)));
    }
  });

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0, nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute face-centered fields from curl(A).
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m, k, j, i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                          (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m, k, j, i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                          (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m, k, j, i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                          (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);


    // Include extra face-component at edge of block in each direction
    if (i == ie) {
      b0.x1f(m, k, j, i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                              (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
    }
    if (j == je) {
      b0.x2f(m, k, j+1, i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                              (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
    }
    if (k == ke) {
      b0.x3f(m, k+1, j ,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                              (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
    }
  });

  if (global_variable::my_rank == 0) {
    std::cout << "Face-centered fields calculated." << std::endl;
  }

  // Compute cell-centered fields
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_bcc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0(m, IBX, k, j, i) = 0.5*(b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
    bcc0(m, IBY, k, j, i) = 0.5*(b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
    bcc0(m, IBZ, k, j, i) = 0.5*(b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
  });

  if (global_variable::my_rank == 0) {
    std::cout << "Cell-centered fields calculated." << std::endl;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for BNS with LORENE
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Lorene BNS problem must have <adm> block to run"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  user_hist_func = &BNSHistory;
  user_ref_func = &LoreneBNSRefinementCondition;

  if (restart) return;

  // Select the correct EOS template based on the EOS we need.
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupBNS<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupBNS<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
    SetupBNS<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupBNS<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown EOS requested for Lorene BNS problem" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

  pmbp->pdyngr->PrimToConInit(0, (ncells1-1), 0, (ncells2-1), 0, (ncells3-1));
  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
      case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
              break;
      case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
              break;
      case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
              break;
    }
  }

  return;
}

void BNSHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  int &nmhd = pm->pmb_pack->pmhd->nmhd;
  pdata->nhist = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  // Capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  // Loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Kokkos::parallel_reduce("TOVHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
    // coompute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    mb_max = fmax(mb_max, w0_(m,IDN,k,j,i));
    mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
  }, Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we need MPI_MAX
  // and MPI_MIN operations instead. This is a cheap hack to make it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

void LoreneBNSRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

KOKKOS_INLINE_FUNCTION
static Real A1(Real x, Real y, Real z, Real I_0, Real r_0) {
  Real w2 = SQR(x) + SQR(y);
  Real r2 = w2 + SQR(z);
  return -y * M_PI * SQR(r_0)*I_0 / pow(SQR(r_0) + r2, 1.5) *
         (1.0 + 15.0/8.0*SQR(r_0)*(SQR(r_0)+w2)/SQR(SQR(r_0)+r2));
}

KOKKOS_INLINE_FUNCTION
static Real A2(Real x, Real y, Real z, Real I_0, Real r_0) {
  Real w2 = SQR(x) + SQR(y);
  Real r2 = w2 + SQR(z);
  return x * M_PI * SQR(r_0)*I_0 / pow(SQR(r_0) + r2, 1.5) *
         (1.0 + 15.0/8.0*SQR(r_0)*(SQR(r_0)+w2)/SQR(SQR(r_0)+r2));
}
