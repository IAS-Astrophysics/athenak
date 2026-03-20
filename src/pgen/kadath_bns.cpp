//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kadath_bns.cpp
//  \brief Initial data reader for binary neutron star data from Kadath_AEI (FUKa)
//
//  Uses KadathExportBNS() from Kadath_AEI/src/Utilities/Exporters/export_bns.cpp
//  (compiled into libkadath.a) to interpolate spectral BNS initial data onto the
//  AthenaK grid.
//
//  Required input block:
//    <problem>
//      initial_data_file = path/to/bns.info   # Kadath config file (.info)
//
//  The Kadath space file must reside next to the .info file (same stem, .dat extension).

#include <math.h>
#include <stdio.h>

#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "utils/tov/tov_utils.hpp"
#include "utils/tov/tov_polytrope.hpp"
#include "utils/tov/tov_piecewise_poly.hpp"
#include "utils/tov/tov_tabulated.hpp"

// ---------------------------------------------------------------------------
// Output index enumeration for KadathExportBNS.
// Must match export_utils::sim_vac_quants / sim_matter_quants in
// Kadath_AEI/include/for_Kadath/exporter_utilities.hpp
// ---------------------------------------------------------------------------
namespace kadath_out {
enum {
  ALPHA = 0,
  BETAX,
  BETAY,
  BETAZ,
  GXX,
  GXY,
  GXZ,
  GYY,
  GYZ,
  GZZ,
  KXX,
  KXY,
  KXZ,
  KYY,
  KYZ,
  KZZ,  // NUM_VOUT = 16
  RHO,
  EPS,
  PRESS,
  VELX,
  VELY,
  VELZ,
  NUM_OUT  // = 22
};
}  // namespace kadath_out

// Forward declaration of the Kadath BNS exporter function.
// Defined in Kadath_AEI/src/Utilities/Exporters/export_bns.cpp,
// compiled into libkadath.a.
std::array<std::vector<double>, kadath_out::NUM_OUT> KadathExportBNS(
    int npoints, double const *xx, double const *yy, double const *zz,
    char const *fn);

void KadathBNSHistory(HistoryData *pdata, Mesh *pm);
void KadathBNSRefinementCondition(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Problem generator for BNS with Kadath_AEI (FUKa)
template<class TOVEOS>
void SetupBNS(ParameterInput *pin, Mesh* pmy_mesh_) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs         = pmy_mesh_->mb_indcs;
  auto &size          = pmbp->pmb->mb_size;
  int &is             = indcs.is;
  int &ie             = indcs.ie;
  int &js             = indcs.js;
  int &je             = indcs.je;
  int &ks             = indcs.ks;
  int &ke             = indcs.ke;

  std::string fname = pin->GetString("problem", "initial_data_file");

  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = indcs.nx2 + 2 * (indcs.ng);
  int ncells3 = indcs.nx3 + 2 * (indcs.ng);
  int nmb     = pmbp->nmb_thispack;
  int width   = nmb * ncells1 * ncells2 * ncells3;

  // Allocate coordinate arrays as double (Kadath expects double).
  std::vector<double> x_coords(width);
  std::vector<double> y_coords(width);
  std::vector<double> z_coords(width);

  // Set up the 1D EOS
  TOVEOS eos{pin};

  // Enable ye if the EOS supports it.
  constexpr bool use_ye = tov::UsesYe<TOVEOS>;

  if (global_variable::my_rank == 0) {
    std::cout << "Allocated coordinate arrays of size " << width << std::endl;
  }

  // Populate cell-center coordinates for all mesh blocks.
  // TODO(user): Replace with a Kokkos loop on DefaultHostExecutionSpace for
  // improved performance.
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

          x_coords[idx] = static_cast<double>(x);
          y_coords[idx] = static_cast<double>(y);
          z_coords[idx] = static_cast<double>(z);

          idx++;
        }
      }
    }
  }

  if (global_variable::my_rank == 0) {
    std::cout << "Coordinates assigned. Calling KadathExportBNS..." << std::endl;
  }

  // Call the Kadath spectral exporter.  This opens the .info config file and
  // the associated .dat space file, sets up the EOS from the config, and
  // interpolates all metric and hydro fields at the provided coordinates.
  auto out = KadathExportBNS(width, x_coords.data(), y_coords.data(),
                              z_coords.data(), fname.c_str());


  if (global_variable::my_rank == 0) {
    std::cout << "KadathExportBNS complete." << std::endl;
  }

  // Capture device-side views.
  // When Z4c is enabled, gauge variables live in u_z4c; metric/curvature
  // in u_adm; hydro primitives in w0.
  auto &u_adm = pmbp->padm->u_adm;
  auto &w0    = pmbp->pmhd->w0;
  auto &u_z4c = pmbp->pz4c->u0;

  // Create CPU-side mirrors.  Kadath only runs on the CPU, so we populate
  // on the host and deep_copy to the device afterwards.
  HostArray5D<Real>::HostMirror host_u_adm = create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0    = create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c = create_mirror_view(u_z4c);

  adm::ADM::ADMhost_vars host_adm;
  host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
  host_adm.beta_u.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_BETAX,
                                        z4c::Z4c::I_Z4C_BETAZ);
  host_adm.g_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_GXX,
                                      adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_KXX,
                                       adm::ADM::I_ADM_KZZ);

  if (global_variable::my_rank == 0) {
    std::cout << "Host mirrors created." << std::endl;
  }

  // Fill ADM metric and hydrodynamic primitives from Kadath output.
  idx = 0;
  for (int m = 0; m < nmb; m++) {
    for (int k = 0; k < ncells3; k++) {
      for (int j = 0; j < ncells2; j++) {
        for (int i = 0; i < ncells1; i++) {
          using namespace kadath_out;

          // ----- ADM metric -----
          host_adm.alpha(m, k, j, i)     = out[ALPHA][idx];
          host_adm.beta_u(m, 0, k, j, i) = out[BETAX][idx];
          host_adm.beta_u(m, 1, k, j, i) = out[BETAY][idx];
          host_adm.beta_u(m, 2, k, j, i) = out[BETAZ][idx];

          // Spatial metric g_ij.  Kadath outputs the physical (not conformal)
          // metric: g_ij = psi^4 f_ij, where f_ij is the flat metric.
          Real g3d[NSPMETRIC];
          host_adm.g_dd(m, 0, 0, k, j, i) = g3d[S11] = out[GXX][idx];
          host_adm.g_dd(m, 0, 1, k, j, i) = g3d[S12] = out[GXY][idx];
          host_adm.g_dd(m, 0, 2, k, j, i) = g3d[S13] = out[GXZ][idx];
          host_adm.g_dd(m, 1, 1, k, j, i) = g3d[S22] = out[GYY][idx];
          host_adm.g_dd(m, 1, 2, k, j, i) = g3d[S23] = out[GYZ][idx];
          host_adm.g_dd(m, 2, 2, k, j, i) = g3d[S33] = out[GZZ][idx];

          // Extrinsic curvature K_ij.  Kadath outputs K_ij = psi^4 * A_ij
          // (traceless part; trace is zero in maximal slicing XCTS).
          host_adm.vK_dd(m, 0, 0, k, j, i) = out[KXX][idx];
          host_adm.vK_dd(m, 0, 1, k, j, i) = out[KXY][idx];
          host_adm.vK_dd(m, 0, 2, k, j, i) = out[KXZ][idx];
          host_adm.vK_dd(m, 1, 1, k, j, i) = out[KYY][idx];
          host_adm.vK_dd(m, 1, 2, k, j, i) = out[KYZ][idx];
          host_adm.vK_dd(m, 2, 2, k, j, i) = out[KZZ][idx];

          // ----- Hydro primitives -----
          host_w0(m, IDN, k, j, i) = out[RHO][idx];
          host_w0(m, IPR, k, j, i) = out[PRESS][idx];

          if constexpr (use_ye) {
            Real& rho = host_w0(m, IDN, k, j, i);
            host_w0(m, IYF, k, j, i) = eos.template
                                       GetYeFromRho<tov::LocationTag::Host>(rho);
          }

          Real vu[3] = {
            out[VELX][idx],
            out[VELY][idx],
            out[VELZ][idx]
          };

          Real vsq = Primitive::SquareVector(vu, g3d);
          if (1.0 - vsq <= 0.0) {
            std::cout << "Superluminal velocity detected at m=" << m
                      << " k=" << k << " j=" << j << " i=" << i
                      << "; rescaling..." << std::endl;
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

  // Copy data from host mirrors to the device.
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  Kokkos::deep_copy(u_z4c, host_u_z4c);

  if (global_variable::my_rank == 0) {
    std::cout << "Data copied to device." << std::endl;
  }

  // TODO(user): Add magnetic field initialization (e.g., current-loop model).
  // For now, initialize face-centered and cell-centered B fields to zero.
  auto &b0 = pmbp->pmhd->b0;
  par_for(
      "pgen_Bfc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        b0.x1f(m, k, j, i) = 0.0;
        b0.x2f(m, k, j, i) = 0.0;
        b0.x3f(m, k, j, i) = 0.0;

        if (i == ie) b0.x1f(m, k, j, i + 1) = 0.0;
        if (j == je) b0.x2f(m, k, j + 1, i) = 0.0;
        if (k == ke) b0.x3f(m, k + 1, j, i) = 0.0;
      });

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

  if (global_variable::my_rank == 0) {
    std::cout << "Magnetic fields initialized to zero." << std::endl;
  }

  // Convert primitive hydro variables to conserved.
  pmbp->pdyngr->PrimToConInit(0, (ncells1 - 1), 0, (ncells2 - 1),
                               0, (ncells3 - 1));

  // Convert ADM variables to Z4c variables.
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

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for BNS with Kadath FUKa
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Kadath BNS problem must have <adm> block to run"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  user_hist_func = &KadathBNSHistory;
  user_ref_func = &KadathBNSRefinementCondition;

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

//----------------------------------------------------------------------------------------
//! \fn KadathBNSHistory()
//! \brief History function: tracks maximum rest-mass density and minimum lapse
void KadathBNSHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist    = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

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

  Real rho_max   = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;

  Kokkos::parallel_reduce(
      "KadathBNSHistSums",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
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

#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    rho_max   = 0.;
    alpha_min = 0.;
  }
#endif

  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

//----------------------------------------------------------------------------------------
//! \fn KadathBNSRefinementCondition()
//! \brief AMR refinement condition (delegates to Z4c AMR)
void KadathBNSRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
