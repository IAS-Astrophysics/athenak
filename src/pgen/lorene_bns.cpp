//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file elliptica_bns.cpp
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

// Lorene
#include "bin_ns.h"
#include "unites.h"

// Prototype for user-defined history function
void BNSHistory(HistoryData *pdata, Mesh *pm);
void LoreneBNSRefinementCondition(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for BNS with LORENE
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = &BNSHistory;
  user_ref_func = &LoreneBNSRefinementCondition;

  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  /*if (pmbp->pdyngr == nullptr || pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "BNS data requires <mhd> and <z4c> blocks."
              << std::endl;
    exit(EXIT_FAILURE);
  }*/

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

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;

  int width = nmb*ncells1*ncells2*ncells3;

  Real *x_coords = new Real[width];
  Real *y_coords = new Real[width];
  Real *z_coords = new Real[width];

  std::cout << "Allocated coordinates of size " << width << std::endl;

  // Populate coordinates for LORENE
  // TODO(JMF): Replace with a Kokkos loop on Kokkos::DefaultHostExecutionSpace() to
  // improve performance.
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
  std::cout << "Coordinates assigned." << std::endl;
  Lorene::Bin_NS *bns = new Lorene::Bin_NS(width, x_coords, y_coords, z_coords,
                                          fname.c_str());

  // Free the coordinates, since we'll no longer need them.
  delete[] x_coords;
  delete[] y_coords;
  delete[] z_coords;

  std::cout << "Coordinates freed." << std::endl;

  // Capture variables for kernel; note that when Z4c is enabled, the gauge variables
  // are part of the Z4c class.
  auto &u_adm = pmbp->padm->u_adm;
  auto &adm   = pmbp->padm->adm;
  auto &w0    = pmbp->pmhd->w0;
  //auto &u_z4c = pmbp->pz4c->u0;
  // Because Elliptica only operates on the CPU, we can't construct the data on the GPU.
  // Instead, we create a mirror guaranteed to be on the CPU, populate the data there,
  // then move it back to the GPU.
  // TODO(JMF): This needs to be tested on CPUs to ensure that it functions properly;
  // In theory, create_mirror_view shouldn't copy the data unless it's in a different
  // memory space.
  HostArray5D<Real>::HostMirror host_u_adm = create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0 = create_mirror_view(w0);
  //HostArray5D<Real>::HostMirror host_u_z4c = create_mirror_view(u_z4c);
  HostArray5D<Real>::HostMirror host_u_z4c;
  adm::ADM::ADMhost_vars host_adm;
  if (pmbp->pz4c != nullptr) {
    host_u_z4c = create_mirror_view(pmbp->pz4c->u0);
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

  std::cout << "Host mirrors created." << std::endl;

  // TODO(JMF): Replace with a Kokkos loop on Kokkos::DefaultHostExecutionSpace() to
  // improve performance.
  idx = 0;
  for (int m = 0; m < nmb; m++) {
    for (int k = 0; k < ncells3; k++) {
      for (int j = 0; j < ncells2; j++) {
        for (int i = 0; i < ncells1; i++) {
  /*const int nn = nmb;
  const int nk = ncells3;
  const int nj = ncells2;
  const int ni = ncells1;
  const int nnkji = nn * nk * nj * ni;
  const int nkji = nk * nj * ni;
  const int nji = nj * ni;
  Kokkos::parallel_for("pgen_lorene",
  Kokkos::RangePolicy<>(HostExeSpace(), 0, nnkji),
  KOKKOS_LAMBDA(const int &idx) {
          int m = (idx)/nkji;
          int k = (idx - m*nkji)/nji;
          int j = (idx - m*nkji - k*nji)/ni;
          int i = (idx - m*nkji - k*nji - j*ni);*/
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
          host_w0(m, IDN, k, j, i) = bns->nbar[idx] / rho_unit;
          // Lorene only gives the specific internal energy, but PrimitiveSolver needs
          // pressure. Because PrimitiveSolver is templated, it's difficult to call it
          // directly. Thus, the easiest way is to save the internal energy density, IEN,
          // whose index overlaps the pressure, IPR, move the data to the GPU, then
          // make a call to a virtual DynGRMHD EOS function that will call the appropriate
          // template function.
          Real egas = host_w0(m, IDN, k, j, i) * bns->ener_spec[idx] / ener_unit;
          host_w0(m, IEN, k, j, i) = egas;
          Real vu[3] = {bns->u_euler_x[idx] / vel_unit,
                        bns->u_euler_y[idx] / vel_unit,
                        bns->u_euler_z[idx] / vel_unit};

          // Before we store the velocity, we need to make sure it's physical and
          // calculate the Lorentz factor. If the velocity is superluminal, we make a
          // last-ditch attempt to salvage the solution by rescaling it to
          // vsq = 1.0 - 1e-15
          Real vsq = Primitive::SquareVector(vu, g3d);
          if (1.0 - vsq <= 0) {
            std::cout << "The velocity is superluminal!" << std::endl
                      << "Attempting to adjust..." << std::endl;
            Real fac = sqrt((1.0 - 1e-15)/vsq);
            vu[0] *= fac;
            vu[1] *= fac;
            vu[2] *= fac;
            vsq = 1.0 - 1.0e-15;
          }
          Real W = sqrt(1.0 / (1.0 - vsq));

          host_w0(m, IVX, k, j, i) = W*vu[0];
          host_w0(m, IVY, k, j, i) = W*vu[1];
          host_w0(m, IVZ, k, j, i) = W*vu[2];

          idx++;
  //});
        }
      }
    }
  }

  std::cout << "Host mirrors filled." << std::endl;

  // Cleanup
  delete bns;

  std::cout << "Lorene freed." << std::endl;

  // Copy the data to the GPU.
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  if (pmbp->pz4c != nullptr) {
    Kokkos::deep_copy(pmbp->pz4c->u0, host_u_z4c);
  }

  std::cout << "Data copied." << std::endl;

  // Convert internal energy to pressure.
  pmbp->pdyngr->ConvertInternalEnergyToPressure(0, (ncells1-1),
                                                0, (ncells2-1), 0, (ncells3-1));

  // TODO(JMF): Read in scalars if necessary.

  // TODO(JMF): Add magnetic fields
  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0, nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    b0.x1f(m, k, j, i) = 0.0;
    b0.x2f(m, k, j, i) = 0.0;
    b0.x3f(m, k, j, i) = 0.0;

    if (i == ie) {
      b0.x1f(m, k, j, i+1) = 0.0;
    }
    if (j == je) {
      b0.x2f(m, k, j+1, i) = 0.0;
    }
    if (k == ke) {
      b0.x3f(m, k+1, j ,i) = 0.0;
    }
  });

  std::cout << "Face-centered fields zeroed." << std::endl;

  // Compute cell-centered fields
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_bcc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0(m, IBX, k, j, i) = 0.5*(b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
    bcc0(m, IBY, k, j, i) = 0.5*(b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
    bcc0(m, IBZ, k, j, i) = 0.5*(b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
  });

  std::cout << "Cell-centered fields calculated." << std::endl;

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
