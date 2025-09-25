//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file elliptica_bns.cpp
//  \brief Initial data reader for binary neutron star data with Elliptica

#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "elliptica_id_reader_lib.h"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"

void EllipticaBNSHistory(HistoryData *pdata, Mesh *pm);
void EllipticaBNSRefinementCondition(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for BNS with Elliptica
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = &EllipticaBNSHistory;
  user_ref_func = &EllipticaBNSRefinementCondition;

  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs         = pmy_mesh_->mb_indcs;
  auto &size          = pmbp->pmb->mb_size;
  int &is             = indcs.is;
  int &ie             = indcs.ie;
  int &js             = indcs.js;
  int &je             = indcs.je;
  int &ks             = indcs.ks;
  int &ke             = indcs.ke;

  if (pmbp->pdyngr == nullptr || pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "BNS data requires <mhd> and <z4c> blocks." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string fname = pin->GetString("problem", "initial_data_file");
  std::string tab_path =
    pin->GetOrAddString("problem", "initial_data_table", "__na__");

  // Initialize the data reader
  Elliptica_ID_Reader_T *idr =
    elliptica_id_reader_init(fname.c_str(), "generic");

  if (tab_path != "__na__") {
    idr->set_param("NS1_EoS_table_path", tab_path.c_str(), idr);
    idr->set_param("NS2_EoS_table_path", tab_path.c_str(), idr);
  }

  // Fields to interpolate
  idr->ifields =
    "alpha,betax,betay,betaz,"
    "adm_gxx,adm_gxy,adm_gxz,adm_gyy,adm_gyz,adm_gzz,"
    "adm_Kxx,adm_Kxy,adm_Kxz,adm_Kyy,adm_Kyz,adm_Kzz,"
    "grhd_rho,grhd_p,grhd_vx,grhd_vy,grhd_vz";

  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = indcs.nx2 + 2 * (indcs.ng);
  int ncells3 = indcs.nx3 + 2 * (indcs.ng);
  int nmb     = pmbp->nmb_thispack;

  int width = nmb * ncells1 * ncells2 * ncells3;

  Real *x_coords = new Real[width];
  Real *y_coords = new Real[width];
  Real *z_coords = new Real[width];

  std::cout << "Allocated coordinates of size " << width << std::endl;

  // Populate coordinates for Elliptica
  // TODO(JMF): Replace with a Kokkos loop on
  // Kokkos::DefaultHostExecutionSpace() to improve performance.
  int idx = 0;
  for (int m = 0; m < nmb; m++) {
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    int nx1     = indcs.nx1;

    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    int nx2     = indcs.nx2;

    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;
    int nx3     = indcs.nx3;

    for (int k = 0; k < ncells3; k++) {
      Real z = CellCenterX(k - ks, nx3, x3min, x3max);
      for (int j = 0; j < ncells2; j++) {
        Real y = CellCenterX(j - js, nx2, x2min, x2max);
        for (int i = 0; i < ncells1; i++) {
          Real x = CellCenterX(i - is, nx1, x1min, x1max);

          x_coords[idx] = x;
          y_coords[idx] = y;
          z_coords[idx] = z;

          // Increment flat index
          idx++;
        }
      }
    }
  }

  idr->set_param("ADM_B1I_form", "zero", idr);

  // Interpolate the data
  idr->npoints  = width;
  idr->x_coords = x_coords;
  idr->y_coords = y_coords;
  idr->z_coords = z_coords;
  std::cout << "Coordinates assigned." << std::endl;
  elliptica_id_reader_interpolate(idr);

  // Free the coordinates, since we'll no longer need them.
  delete[] x_coords;
  delete[] y_coords;
  delete[] z_coords;

  std::cout << "Coordinates freed." << std::endl;

  // Capture variables for kernel; note that when Z4c is enabled, the gauge
  // variables are part of the Z4c class.
  auto &u_adm = pmbp->padm->u_adm;
  auto &adm   = pmbp->padm->adm;
  auto &w0    = pmbp->pmhd->w0;
  auto &u_z4c = pmbp->pz4c->u0;
  // Because Elliptica only operates on the CPU, we can't construct the data on
  // the GPU. Instead, we create a mirror guaranteed to be on the CPU, populate
  // the data there, then move it back to the GPU.
  // TODO(JMF): This needs to be tested on CPUs to ensure that it functions
  // properly; In theory, create_mirror_view shouldn't copy the data unless it's
  // in a different memory space.
  HostArray5D<Real>::HostMirror host_u_adm = create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0    = create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c = create_mirror_view(u_z4c);
  adm::ADM::ADMhost_vars host_adm;
  host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
  host_adm.beta_u.InitWithShallowSlice(
    host_u_z4c, z4c::Z4c::I_Z4C_BETAX, z4c::Z4c::I_Z4C_BETAZ);
  host_adm.g_dd.InitWithShallowSlice(
    host_u_adm, adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(
    host_u_adm, adm::ADM::I_ADM_KXX, adm::ADM::I_ADM_KZZ);

  std::cout << "Host mirrors created." << std::endl;

  // Save Elliptica field indices for shorthand and a small optimization.
  const int i_alpha = idr->indx("alpha");
  const int i_betax = idr->indx("betax");
  const int i_betay = idr->indx("betay");
  const int i_betaz = idr->indx("betaz");

  const int i_gxx = idr->indx("adm_gxx");
  const int i_gxy = idr->indx("adm_gxy");
  const int i_gxz = idr->indx("adm_gxz");
  const int i_gyy = idr->indx("adm_gyy");
  const int i_gyz = idr->indx("adm_gyz");
  const int i_gzz = idr->indx("adm_gzz");

  const int i_Kxx = idr->indx("adm_Kxx");
  const int i_Kxy = idr->indx("adm_Kxy");
  const int i_Kxz = idr->indx("adm_Kxz");
  const int i_Kyy = idr->indx("adm_Kyy");
  const int i_Kyz = idr->indx("adm_Kyz");
  const int i_Kzz = idr->indx("adm_Kzz");

  const int i_rho = idr->indx("grhd_rho");
  const int i_p   = idr->indx("grhd_p");
  const int i_vx  = idr->indx("grhd_vx");
  const int i_vy  = idr->indx("grhd_vy");
  const int i_vz  = idr->indx("grhd_vz");

  std::cout << "Label indices saved." << std::endl;

  // TODO(JMF): Replace with a Kokkos loop on
  // Kokkos::DefaultHostExecutionSpace() to improve performance.
  idx = 0;
  for (int m = 0; m < nmb; m++) {
    for (int k = 0; k < ncells3; k++) {
      for (int j = 0; j < ncells2; j++) {
        for (int i = 0; i < ncells1; i++) {
          // Extract metric quantities
          host_adm.alpha(m, k, j, i)     = idr->field[i_alpha][idx];
          host_adm.beta_u(m, 0, k, j, i) = idr->field[i_betax][idx];
          host_adm.beta_u(m, 1, k, j, i) = idr->field[i_betay][idx];
          host_adm.beta_u(m, 2, k, j, i) = idr->field[i_betaz][idx];

          Real g3d[NSPMETRIC];
          host_adm.g_dd(m, 0, 0, k, j, i) = g3d[S11] = idr->field[i_gxx][idx];
          host_adm.g_dd(m, 0, 1, k, j, i) = g3d[S12] = idr->field[i_gxy][idx];
          host_adm.g_dd(m, 0, 2, k, j, i) = g3d[S13] = idr->field[i_gxz][idx];
          host_adm.g_dd(m, 1, 1, k, j, i) = g3d[S22] = idr->field[i_gyy][idx];
          host_adm.g_dd(m, 1, 2, k, j, i) = g3d[S23] = idr->field[i_gyz][idx];
          host_adm.g_dd(m, 2, 2, k, j, i) = g3d[S33] = idr->field[i_gzz][idx];

          host_adm.vK_dd(m, 0, 0, k, j, i) = idr->field[i_Kxx][idx];
          host_adm.vK_dd(m, 0, 1, k, j, i) = idr->field[i_Kxy][idx];
          host_adm.vK_dd(m, 0, 2, k, j, i) = idr->field[i_Kxz][idx];
          host_adm.vK_dd(m, 1, 1, k, j, i) = idr->field[i_Kyy][idx];
          host_adm.vK_dd(m, 1, 2, k, j, i) = idr->field[i_Kyz][idx];
          host_adm.vK_dd(m, 2, 2, k, j, i) = idr->field[i_Kzz][idx];

          // Extract hydro quantities
          host_w0(m, IDN, k, j, i) = idr->field[i_rho][idx];
          host_w0(m, IPR, k, j, i) = idr->field[i_p][idx];
          Real vu[3]               = {
            idr->field[i_vx][idx], idr->field[i_vy][idx],
            idr->field[i_vz][idx]};

          // Before we store the velocity, we need to make sure it's physical
          // and calculate the Lorentz factor. If the velocity is superluminal,
          // we make a last-ditch attempt to salvage the solution by rescaling
          // it to vsq = 1.0 - 1e-15
          Real vsq = Primitive::SquareVector(vu, g3d);
          if (1.0 - vsq <= 0) {
            std::cout << "The velocity is superluminal!" << std::endl
                      << "Attempting to adjust..." << std::endl;
            Real fac = sqrt((1.0 - 1e-15) / vsq);
            vu[0] *= fac;
            vu[1] *= fac;
            vu[2] *= fac;
            vsq = 1.0 - 1.0e-15;
          }
          Real W = sqrt(1.0 / (1.0 - vsq));

          host_w0(m, IVX, k, j, i) = W * vu[0];
          host_w0(m, IVY, k, j, i) = W * vu[1];
          host_w0(m, IVZ, k, j, i) = W * vu[2];

          idx++;
        }
      }
    }
  }

  std::cout << "Host mirrors filled." << std::endl;

  // Cleanup
  elliptica_id_reader_free(idr);

  std::cout << "Elliptica freed." << std::endl;

  // Copy the data to the GPU.
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  Kokkos::deep_copy(u_z4c, host_u_z4c);

  std::cout << "Data copied." << std::endl;

  // TODO(JMF): Add magnetic fields
  auto &b0 = pmbp->pmhd->b0;
  par_for(
    "pgen_Bfc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m, k, j, i) = 0.0;
      b0.x2f(m, k, j, i) = 0.0;
      b0.x3f(m, k, j, i) = 0.0;

      if (i == ie) {
        b0.x1f(m, k, j, i + 1) = 0.0;
      }
      if (j == je) {
        b0.x2f(m, k, j + 1, i) = 0.0;
      }
      if (k == ke) {
        b0.x3f(m, k + 1, j, i) = 0.0;
      }
    });

  std::cout << "Face-centered fields zeroed." << std::endl;

  // Compute cell-centered fields
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for(
    "pgen_bcc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bcc0(m, IBX, k, j, i) =
        0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i + 1));
      bcc0(m, IBY, k, j, i) =
        0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j + 1, i));
      bcc0(m, IBZ, k, j, i) =
        0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k + 1, j, i));
    });

  std::cout << "Cell-centered fields calculated." << std::endl;

  pmbp->pdyngr->PrimToConInit(
    0, (ncells1 - 1), 0, (ncells2 - 1), 0, (ncells3 - 1));
  switch (indcs.ng) {
    case 2:
      pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
      break;
    case 3:
      pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
      break;
    case 4:
      pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
      break;
  }

  return;
}

// History function
void EllipticaBNSHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  int &nmhd       = pm->pmb_pack->pmhd->nmhd;
  pdata->nhist    = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs     = pm->pmb_pack->pmesh->mb_indcs;
  int is          = indcs.is;
  int nx1         = indcs.nx1;
  int js          = indcs.js;
  int nx2         = indcs.nx2;
  int ks          = indcs.ks;
  int nx3         = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack) * nx3 * nx2 * nx1;
  const int nkji  = nx3 * nx2 * nx1;
  const int nji   = nx2 * nx1;
  Real rho_max    = std::numeric_limits<Real>::max();
  Real alpha_min  = -rho_max;
  Kokkos::parallel_reduce(
    "TOVHistSums", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
      // coompute n,k,j,i indices of thread
      int m = (idx) / nkji;
      int k = (idx - m * nkji) / nji;
      int j = (idx - m * nkji - k * nji) / nx1;
      int i = (idx - m * nkji - k * nji - j * nx1) + is;
      k += ks;
      j += js;

      mb_max     = fmax(mb_max, w0_(m, IDN, k, j, i));
      mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
    },
    Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we
  // need MPI_MAX and MPI_MIN operations instead. This is a cheap hack to make
  // it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(
      MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(
      MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(
      &rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(
      &alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    rho_max   = 0.;
    alpha_min = 0.;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

void EllipticaBNSRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
