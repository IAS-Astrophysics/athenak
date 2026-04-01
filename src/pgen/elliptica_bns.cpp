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
#include "z4c/compact_object_tracker.hpp"
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
#include "utils/tov/tov_utils.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"
#include "utils/tov/tov_tabulated.hpp"

void EllipticaBinaryHistory(HistoryData *pdata, Mesh *pm);
void EllipticaBinaryRefinementCondition(MeshBlockPack *pmbp);

// Prototypes for magnetic vector potential
KOKKOS_INLINE_FUNCTION
static Real A1(Real x, Real y, Real z, Real I_0, Real r_0);
KOKKOS_INLINE_FUNCTION
static Real A2(Real x, Real y, Real z, Real I_0, Real r_0);

//----------------------------------------------------------------------------------------
//! \fn SetupBinary(ParameterInput *pin, Mesh* pmy_mesh_)
//! \brief Setup of the BHNS/BNS binary with Elliptica
template<class TOVEOS>
void SetupBinary(ParameterInput *pin, Mesh* pmy_mesh_) {
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
    "grhd_rho,grhd_p,grhd_epsl,grhd_vx,grhd_vy,grhd_vz";

  // MHD parameters
  Real rho_cut = pin->GetOrAddReal("problem", "rho_cut", 1e-5);
  Real b_max   = pin->GetOrAddReal("problem", "b_max", 1e12) / 8.3519664583273e+19;
  Real r_0     = pin->GetOrAddReal("problem", "r_0_current", 5.0);
  Real I_0     = 4 * r_0 * b_max / (23.0 * M_PI);

  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = indcs.nx2 + 2 * (indcs.ng);
  int ncells3 = indcs.nx3 + 2 * (indcs.ng);
  int nmb     = pmbp->nmb_thispack;

  int width = nmb * ncells1 * ncells2 * ncells3;

  Real *x_coords = new Real[width];
  Real *y_coords = new Real[width];
  Real *z_coords = new Real[width];

  // Set up the 1D EOS.
  TOVEOS eos{pin};

  // Enable electron fraction if the EOS supports it.
  constexpr bool use_ye = tov::UsesYe<TOVEOS>;

  if (global_variable::my_rank == 0) {
    std::cout << "Allocated coordinates of size " << width << std::endl;
  }

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

  if (global_variable::my_rank == 0) {
    std::cout << "Coordinates assigned." << std::endl;
  }
  elliptica_id_reader_interpolate(idr);

  // Free the coordinates, since we'll no longer need them.
  delete[] x_coords;
  delete[] y_coords;
  delete[] z_coords;

  if (global_variable::my_rank == 0) {
    std::cout << "Coordinates freed." << std::endl;
  }

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

  if (global_variable::my_rank == 0) {
    std::cout << "Host mirrors created." << std::endl;
  }

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
  const int i_eps = idr->indx("grhd_epsl");
  const int i_vx  = idr->indx("grhd_vx");
  const int i_vy  = idr->indx("grhd_vy");
  const int i_vz  = idr->indx("grhd_vz");

  if (global_variable::my_rank == 0) {
    std::cout << "Label indices saved." << std::endl;
  }

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
          // Note that Elliptica does not necessarily use the same baryon rest-mass as
          // AthenaK. The most reasonable thing to do, then, is to extract the total
          // energy density, which is invariant, and use that with the 1D EOS.
          Real egas = idr->field[i_rho][idx] * (1.0 + idr->field[i_eps][idx]);
          Real &rho = host_w0(m, IDN, k, j, i);
          Real vu[3] = {idr->field[i_vx][idx],
                          idr->field[i_vy][idx],
                          idr->field[i_vz][idx]};

          // Check for garbage values thrown in by Elliptica.
          if (idr->field[i_rho][idx] <= rho_cut ||
              !Kokkos::isfinite(idr->field[i_rho][idx])) {
            rho = 0.0;
            host_w0(m, IPR, k, j, i) = 0.0;
            vu[0] = 0.0;
            vu[1] = 0.0;
            vu[2] = 0.0;
          } else {
            rho = eos.template GetRhoFromE<tov::LocationTag::Host>(egas);
            host_w0(m, IPR, k, j, i) = eos.template
                                      GetPFromRho<tov::LocationTag::Host>(rho);
          }

          // If the electron fraction is available, find it in the 1D EOS.
          if constexpr (use_ye) {
            host_w0(m, IYF, k, j, i) = eos.template
                                       GetYeFromRho<tov::LocationTag::Host>(rho);
          }

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

  if (global_variable::my_rank == 0) {
    std::cout << "Host mirrors filled." << std::endl;
  }

  // Binary separation for different systems
  Real sep;
  bool BHNS, BNS;
  if (std::strcmp(idr->system,"BH_NS_binary_initial_data") == 0) {
    sep = idr->get_param_dbl("BHNS_separation",idr);
    BNS = false; BHNS = true;
  } else if (std::strcmp(idr->system,"NS_NS_binary_initial_data") == 0) {
    sep = idr->get_param_dbl("NSNS_separation",idr);
    BNS = true; BHNS = false;
  } else {
    sep = 0.0; BNS = false; BHNS = false;
  }
  if (global_variable::my_rank == 0) {
    std::cout << "Separation = " << sep << std::endl;
  }

  // NS position for the computation of vector potential for BHNS.
  // grid_set_NS = left means that the NS is at -y and
  // grid_set_NS = right means that the NS is at +y.
  // Also, the binary is shifted so that the center of mass
  // is at the center of the grid. We need to adjust this, such that
  // the seed of the magnetic field is at the right position.
  Real NS_x = 0.0, NS_y = 0.0, NS_z = 0.0;
  Real BH_x = 0.0, BH_y = 0.0, BH_z = 0.0;
  Real NS1_x = 0.0, NS1_y = 0.0, NS1_z = 0.0;
  Real NS2_x = 0.0, NS2_y = 0.0, NS2_z = 0.0;
  Real CM_x_corr = 0.0;
  Real CM_y_corr = 0.0;
  Real CM_z_corr = 0.0;
  if (BHNS) {
    NS_x = idr->get_param_dbl("NS_center_x",idr);
    NS_y = idr->get_param_dbl("NS_center_y",idr);
    NS_z = idr->get_param_dbl("NS_center_z",idr);
    BH_x = idr->get_param_dbl("BH_center_x",idr);
    BH_y = idr->get_param_dbl("BH_center_y",idr);
    BH_z = idr->get_param_dbl("BH_center_z",idr);
    CM_x_corr = idr->get_param_dbl("BHNS_x_CM",idr);
    CM_y_corr = idr->get_param_dbl("BHNS_y_CM",idr);
    CM_z_corr = idr->get_param_dbl("BHNS_z_CM",idr);
  }

  if (BNS) {
    NS1_x = idr->get_param_dbl("NS1_center_x",idr);
    NS1_y = idr->get_param_dbl("NS1_center_y",idr);
    NS1_z = idr->get_param_dbl("NS1_center_z",idr);
    NS2_x = idr->get_param_dbl("NS2_center_x",idr);
    NS2_y = idr->get_param_dbl("NS2_center_y",idr);
    NS2_z = idr->get_param_dbl("NS2_center_z",idr);
    CM_x_corr = idr->get_param_dbl("NSNS_x_CM",idr);
    CM_y_corr = idr->get_param_dbl("NSNS_y_CM",idr);
    CM_z_corr = idr->get_param_dbl("NSNS_z_CM",idr);
  }

  // The center of mass shift also needs to be adjusted within
  // the compact object tracker to accurately track the position
  // of the binary.
  if (BHNS) {
    Real COM_corr_NS[3] = {NS_x - CM_x_corr, NS_y - CM_y_corr, NS_z - CM_z_corr};
    Real COM_corr_BH[3] = {BH_x - CM_x_corr, BH_y - CM_y_corr, BH_z - CM_z_corr};
    if (pmbp->pz4c->ptracker[0]->GetType() == 0) { // CompactObjectTracker::BlackHole == 0
      pmbp->pz4c->ptracker[0]->SetPos(COM_corr_BH);
      pmbp->pz4c->ptracker[1]->SetPos(COM_corr_NS);

      if (global_variable::my_rank == 0) {
        std::cout << "Adjusted CompactObjectTracker position by COM." << std::endl;
        std::cout << "BH: cx = " << pmbp->pz4c->ptracker[0]->GetPos(0)
                  << ", cy = " << pmbp->pz4c->ptracker[0]->GetPos(1)
                  << ", cz = " << pmbp->pz4c->ptracker[0]->GetPos(2) << std::endl;
        std::cout << "NS: cx = " << pmbp->pz4c->ptracker[1]->GetPos(0)
                  << ", cy = " << pmbp->pz4c->ptracker[1]->GetPos(1)
                  << ", cz = " << pmbp->pz4c->ptracker[1]->GetPos(2) << std::endl;
      }
    } else {
      pmbp->pz4c->ptracker[1]->SetPos(COM_corr_BH);
      pmbp->pz4c->ptracker[0]->SetPos(COM_corr_NS);

      if (global_variable::my_rank == 0) {
        std::cout << "Adjusted CompactObjectTracker position by COM." << std::endl;
        std::cout << "BH: cx = " << pmbp->pz4c->ptracker[1]->GetPos(0)
                  << ", cy = " << pmbp->pz4c->ptracker[1]->GetPos(1)
                  << ", cz = " << pmbp->pz4c->ptracker[1]->GetPos(2) << std::endl;
        std::cout << "NS: cx = " << pmbp->pz4c->ptracker[0]->GetPos(0)
                  << ", cy = " << pmbp->pz4c->ptracker[0]->GetPos(1)
                  << ", cz = " << pmbp->pz4c->ptracker[0]->GetPos(2) << std::endl;
      }
    }
  }

  if (BNS) {
    Real COM_corr_NS1[3] = {NS1_x - CM_x_corr, NS1_y - CM_y_corr, NS1_z - CM_z_corr};
    Real COM_corr_NS2[3] = {NS2_x - CM_x_corr, NS2_y - CM_y_corr, NS2_z - CM_z_corr};
    pmbp->pz4c->ptracker[0]->SetPos(COM_corr_NS1);
    pmbp->pz4c->ptracker[1]->SetPos(COM_corr_NS2);

    if (global_variable::my_rank == 0) {
      std::cout << "Adjusted CompactObjectTracker position by COM." << std::endl;
      std::cout << "NS1: cx = " << pmbp->pz4c->ptracker[0]->GetPos(0)
                << ", cy = " << pmbp->pz4c->ptracker[0]->GetPos(1)
                << ", cz = " << pmbp->pz4c->ptracker[0]->GetPos(2) << std::endl;
      std::cout << "NS2: cx = " << pmbp->pz4c->ptracker[1]->GetPos(0)
                << ", cy = " << pmbp->pz4c->ptracker[1]->GetPos(1)
                << ", cz = " << pmbp->pz4c->ptracker[1]->GetPos(2) << std::endl;
    }
  }

  // Cleanup
  elliptica_id_reader_free(idr);

  if (global_variable::my_rank == 0) {
    std::cout << "Elliptica freed." << std::endl;
  }

  // Copy the data to the GPU.
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  Kokkos::deep_copy(u_z4c, host_u_z4c);

  if (global_variable::my_rank == 0) {
    std::cout << "Data copied." << std::endl;
  }

  // Compute vector potential over all faces
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
  Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);

  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;

  par_for("pgen_vector_potential", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
    Real x1f   = LeftEdgeX(i - is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j - js, nx2, x2min, x2max);
    Real x2f   = LeftEdgeX(j - js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);
    Real x3f   = LeftEdgeX(k - ks, nx3, x3min, x3max);

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    // Distinguish between BNS and BHNS
    if (BNS) {
      a1(m,k,j,i) = A1(x1v, x2f - 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                    A1(x1v, x2f + 0.5 * sep + CM_y_corr, x3f, I_0, r_0);
      a2(m,k,j,i) = A2(x1f, x2v - 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                    A2(x1f, x2v + 0.5 * sep + CM_y_corr, x3f, I_0, r_0);
      a3(m,k,j,i) = 0.0;
    } else if (BHNS) {
      if (NS_y > 0.0) {
        a1(m,k,j,i) = A1(x1v, x2f - 0.5 * sep + CM_y_corr, x3f, I_0, r_0);
        a2(m,k,j,i) = A2(x1f, x2v - 0.5 * sep + CM_y_corr, x3f, I_0, r_0);
        a3(m,k,j,i) = 0.0;
      } else {
        a1(m,k,j,i) = A1(x1v, x2f + 0.5 * sep + CM_y_corr, x3f, I_0, r_0);
        a2(m,k,j,i) = A2(x1f, x2v + 0.5 * sep + CM_y_corr, x3f, I_0, r_0);
        a3(m,k,j,i) = 0.0;
      }
    } else {
      a1(m,k,j,i) = 0.0;
      a2(m,k,j,i) = 0.0;
      a3(m,k,j,i) = 0.0;
    }

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
      Real xl = x1v + 0.25 * dx1;
      Real xr = x1v - 0.25 * dx1;

      if (BNS) {
        a1(m,k,j,i) = 0.5*((A1(xl, x2f - 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                            A1(xl, x2f + 0.5 * sep + CM_y_corr, x3f, I_0, r_0)) +
                           (A1(xr, x2f - 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                            A1(xr, x2f + 0.5 * sep + CM_y_corr, x3f, I_0, r_0)));
      } else if (BHNS) {
        if (NS_y > 0.0) {
          a1(m,k,j,i) = 0.5*(A1(xl, x2f - 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                             A1(xr, x2f - 0.5 * sep + CM_y_corr, x3f, I_0, r_0));
        } else {
          a1(m,k,j,i) = 0.5*(A1(xl, x2f + 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                             A1(xr, x2f + 0.5 * sep + CM_y_corr, x3f, I_0, r_0));
        }
      } else {
        a1(m,k,j,i) = 0.0;
      }
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
      Real xl = x2v + 0.25 * dx2;
      Real xr = x2v - 0.25 * dx2;

      if (BNS) {
        a2(m,k,j,i) = 0.5*((A2(x1f, xl - 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                            A2(x1f, xl + 0.5 * sep + CM_y_corr, x3f, I_0, r_0)) +
                           (A2(x1f, xr - 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                            A2(x1f, xr + 0.5 * sep + CM_y_corr, x3f, I_0, r_0)));
      } else if (BHNS) {
        if (NS_y > 0.0) {
          a2(m,k,j,i) = 0.5*(A2(x1f, xl - 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                             A2(x1f, xr - 0.5 * sep + CM_y_corr, x3f, I_0, r_0));
        } else {
          a2(m,k,j,i) = 0.5*(A2(x1f, xl + 0.5 * sep + CM_y_corr, x3f, I_0, r_0) +
                             A2(x1f, xr + 0.5 * sep + CM_y_corr, x3f, I_0, r_0));
        }
      } else {
        a2(m,k,j,i) = 0.0;
      }
    }
  });

  auto &b0 = pmbp->pmhd->b0;
  par_for("pgen_Bfc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Compute face-centered fields from curl(A).
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i)) / dx2 -
                          (a2(m,k+1,j,i) - a2(m,k,j,i)) / dx3);
    b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i)) / dx3 -
                          (a3(m,k,j,i+1) - a3(m,k,j,i)) / dx1);
    b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i)) / dx1 -
                          (a1(m,k,j+1,i) - a1(m,k,j,i)) / dx2);

    // Include extra face-component at edge of block in each direction
    if (i == ie) {
      b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1)) / dx2 -
                           (a2(m,k+1,j,i+1) - a2(m,k,j,i+1)) / dx3);
    }
    if (j == je) {
      b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i)) / dx3 -
                           (a3(m,k,j+1,i+1) - a3(m,k,j+1,i)) / dx1);
    }
    if (k == ke) {
      b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i)) / dx1 -
                           (a1(m,k+1,j+1,i) - a1(m,k+1,j,i)) / dx2);
    }
  });

  if (global_variable::my_rank == 0) {
    std::cout << "Face-centered fields calculated." << std::endl;
  }

  // Compute cell-centered fields
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for("pgen_bcc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bcc0(m,IBX,k,j,i) = 0.5 * (b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    bcc0(m,IBY,k,j,i) = 0.5 * (b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    bcc0(m,IBZ,k,j,i) = 0.5 * (b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });

  if (global_variable::my_rank == 0) {
    std::cout << "Cell-centered fields calculated." << std::endl;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Problem Generator for Binary with Elliptica
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pdyngr == nullptr || pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Binary data requires <mhd> and <z4c> blocks." << std::endl;
    exit(EXIT_FAILURE);
  }

  user_hist_func = &EllipticaBinaryHistory;
  user_ref_func = &EllipticaBinaryRefinementCondition;

  if (restart) return;

  // Select the correct EOS template based on the EOS we need.
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_ideal) {
    SetupBinary<tov::PolytropeEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose) {
    SetupBinary<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_hybrid) {
    SetupBinary<tov::TabulatedEOS>(pin, pmy_mesh_);
  } else if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_piecewise_poly) {
    SetupBinary<tov::PiecewisePolytropeEOS>(pin, pmy_mesh_);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown EOS requested for Elliptica problem" << std::endl;
    exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &ng = indcs.ng;
  int ncells1 = indcs.nx1 + 2*ng;
  int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;

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
void EllipticaBinaryHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  int &nmhd       = pm->pmb_pack->pmhd->nmhd;
  pdata->nhist    = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack) * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
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

      mb_max = fmax(mb_max, w0_(m, IDN, k, j, i));
      mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
    },
    Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we
  // need MPI_MAX and MPI_MIN operations instead. This is a cheap hack to make
  // it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

void EllipticaBinaryRefinementCondition(MeshBlockPack *pmbp) {
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
