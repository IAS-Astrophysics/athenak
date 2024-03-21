//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file derived_variables.cpp
//! \brief Calculates various derived variables for outputs, storing them into the
//! "derived_vars" device array located in BaseTypeOutput class.  Variables are only
//! calculated over active zones (ghost zones excluded). Currently implemented are:
//!   - z-component of vorticity Curl(v)_z  [non-relativistic]
//!   - magnitude of vorticity Curl(v)^2  [non-relativistic]
//!   - z-component of current density Jz  [non-relativistic]
//!   - magnitude of current density J^2  [non-relativistic]

#include <iostream>
#include <sstream>
#include <string>   // std::string, to_string()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"
#include "utils/current.hpp"

//----------------------------------------------------------------------------------------
// BaseTypeOutput::ComputeDerivedVariable()

void BaseTypeOutput::ComputeDerivedVariable(std::string name, Mesh *pm) {
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;

  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &multi_d = pm->multi_d;
  auto &three_d = pm->three_d;

  // derived variable index
  int &i_dv = out_params.i_derived;
  int &n_dv = out_params.n_derived;

  // temperature = pressure / density
  if (name.compare("temperature") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &w0_ = (name.compare("hydro_wz") == 0)?
      pm->pmb_pack->phydro->w0 : pm->pmb_pack->pmhd->w0;
    par_for("temperature", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      dv(m,i_dv,k,j,i) = (w0_(m,IEN,k,j,i+1) / w0_(m,IDN,k,j,i-1));
    });
    i_dv += 1; // increment derived variable index
  }

  // z-component of vorticity.
  // Not computed in ghost zones since requires derivative
  if (name.compare("hydro_wz") == 0 ||
      name.compare("mhd_wz") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &w0_ = (name.compare("hydro_wz") == 0)?
      pm->pmb_pack->phydro->w0 : pm->pmb_pack->pmhd->w0;
    par_for("vorz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      dv(m,i_dv,k,j,i) = (w0_(m,IVY,k,j,i+1) - w0_(m,IVY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        dv(m,i_dv,k,j,i) -=(w0_(m,IVX,k,j+1,i) - w0_(m,IVX,k,j-1,i))/size.d_view(m).dx2;
      }
    });
    i_dv += 1; // increment derived variable index
  }

  // magnitude of vorticity.
  // Not computed in ghost zones since requires derivative
  if (name.compare("hydro_w2") == 0 ||
      name.compare("mhd_w2") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &w0_ = (name.compare("hydro_w2") == 0)?
      pm->pmb_pack->phydro->w0 : pm->pmb_pack->pmhd->w0;
    par_for("vor2", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real w1 = 0.0;
      Real w2 = -(w0_(m,IVZ,k,j,i+1) - w0_(m,IVZ,k,j,i-1))/size.d_view(m).dx1;
      Real w3 =  (w0_(m,IVY,k,j,i+1) - w0_(m,IVY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        w1 += (w0_(m,IVZ,k,j+1,i) - w0_(m,IVZ,k,j-1,i))/size.d_view(m).dx2;
        w3 -= (w0_(m,IVX,k,j+1,i) - w0_(m,IVX,k,j-1,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        w1 -= (w0_(m,IVY,k+1,j,i) - w0_(m,IVY,k-1,j,i))/size.d_view(m).dx3;
        w2 += (w0_(m,IVX,k+1,j,i) - w0_(m,IVX,k-1,j,i))/size.d_view(m).dx3;
      }
      dv(m,i_dv,k,j,i) = w1*w1 + w2*w2 + w3*w3;
    });
    i_dv += 1; // increment derived variable index
  }

  // z-component of current density.  Calculated from cell-centered fields.
  // This makes for a large stencil, but approximates volume-averaged value within cell.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_jz") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("jz", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      dv(m,i_dv,k,j,i) = (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        dv(m,i_dv,k,j,i) -=(bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))/size.d_view(m).dx2;
      }
    });
    i_dv += 1; // increment derived variable index
  }

  // magnitude of current density.  Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_j2") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("j2", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real j1 = 0.0;
      Real j2 = -(bcc(m,IBZ,k,j,i+1) - bcc(m,IBZ,k,j,i-1))/size.d_view(m).dx1;
      Real j3 =  (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        j1 += (bcc(m,IBZ,k,j+1,i) - bcc(m,IBZ,k,j-1,i))/size.d_view(m).dx2;
        j3 -= (bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        j1 -= (bcc(m,IBY,k+1,j,i) - bcc(m,IBY,k-1,j,i))/size.d_view(m).dx3;
        j2 += (bcc(m,IBX,k+1,j,i) - bcc(m,IBX,k-1,j,i))/size.d_view(m).dx3;
      }
      dv(m,i_dv,k,j,i) = j1*j1 + j2*j2 + j3*j3;
    });
    i_dv += 1; // increment derived variable index
  }

  // magnitude of curvature = |B^-2 (B.gradB) - B^-4 (BB:gradB)|.
  // or curvature = |(B.gradB).(I - bhat bhat)/B^2|
  // Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_curv") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("curv", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Calculate |B|
      Real &Bx = bcc(m,IBX,k,j,i);
      Real &By = bcc(m,IBY,k,j,i);
      Real &Bz = bcc(m,IBZ,k,j,i);

      Real B_mag_squared = ( Bx*Bx + By*By + Bz*Bz);

      // Calculate gradB tensor
      Real dBx_dx = (bcc(m,IBX,k,j,i+1) - bcc(m,IBX,k,j,i-1))/(2.0*size.d_view(m).dx1);
      Real dBx_dy = (bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))/(2.0*size.d_view(m).dx2);
      Real dBx_dz = (bcc(m,IBX,k+1,j,i) - bcc(m,IBX,k-1,j,i))/(2.0*size.d_view(m).dx3);

      Real dBy_dx = (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))/(2.0*size.d_view(m).dx1);
      Real dBy_dy = (bcc(m,IBY,k,j+1,i) - bcc(m,IBY,k,j-1,i))/(2.0*size.d_view(m).dx2);
      Real dBy_dz = (bcc(m,IBY,k+1,j,i) - bcc(m,IBY,k-1,j,i))/(2.0*size.d_view(m).dx3);

      Real dBz_dx = (bcc(m,IBZ,k,j,i+1) - bcc(m,IBZ,k,j,i-1))/(2.0*size.d_view(m).dx1);
      Real dBz_dy = (bcc(m,IBZ,k,j+1,i) - bcc(m,IBZ,k,j-1,i))/(2.0*size.d_view(m).dx2);
      Real dBz_dz = (bcc(m,IBZ,k+1,j,i) - bcc(m,IBZ,k-1,j,i))/(2.0*size.d_view(m).dx3);

      Real BdotGradB_x = (Bx * dBx_dx + By * dBx_dy + Bz * dBx_dz);
      Real BdotGradB_y = (Bx * dBy_dx + By * dBy_dy + Bz * dBy_dz);
      Real BdotGradB_z = (Bx * dBz_dx + By * dBz_dy + Bz * dBz_dz);

      Real Identity_minus_bhat_bhat_xx = 1.0 - Bx*Bx/B_mag_squared;
      Real Identity_minus_bhat_bhat_xy = 0.0 - Bx*By/B_mag_squared;
      Real Identity_minus_bhat_bhat_xz = 0.0 - Bx*Bz/B_mag_squared;

      Real Identity_minus_bhat_bhat_yx = 0.0 - By*Bx/B_mag_squared;
      Real Identity_minus_bhat_bhat_yy = 1.0 - By*By/B_mag_squared;
      Real Identity_minus_bhat_bhat_yz = 0.0 - By*Bz/B_mag_squared;

      Real Identity_minus_bhat_bhat_zx = 0.0 - Bz*Bx/B_mag_squared;
      Real Identity_minus_bhat_bhat_zy = 0.0 - Bz*By/B_mag_squared;
      Real Identity_minus_bhat_bhat_zz = 1.0 - Bz*Bz/B_mag_squared;

      // Calculate curvature which is |(B.gradB).(I - bhat bhat)/B^2|
      Real curv1 = (
            BdotGradB_x * Identity_minus_bhat_bhat_xx
          + BdotGradB_y * Identity_minus_bhat_bhat_yx
          + BdotGradB_z * Identity_minus_bhat_bhat_zx
        );
      Real curv2 = (
            BdotGradB_x * Identity_minus_bhat_bhat_xy
          + BdotGradB_y * Identity_minus_bhat_bhat_yy
          + BdotGradB_z * Identity_minus_bhat_bhat_zy
        );
      Real curv3 = (
            BdotGradB_x * Identity_minus_bhat_bhat_xz
          + BdotGradB_y * Identity_minus_bhat_bhat_yz
          + BdotGradB_z * Identity_minus_bhat_bhat_zz
        );

      dv(m,i_dv,k,j,i) = sqrt(curv1*curv1 + curv2*curv2 + curv3*curv3)/B_mag_squared;
    });
    i_dv += 1; // increment derived variable index
  }

  // magnitude of curvature = |b_hat dot nabla b_hat|.
  // Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_curv_alt") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("curv_alt", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Calculate |B|
      Real B_mag = sqrt( bcc(m,IBX,k,j,i)*bcc(m,IBX,k,j,i)
                       + bcc(m,IBY,k,j,i)*bcc(m,IBY,k,j,i)
                       + bcc(m,IBZ,k,j,i)*bcc(m,IBZ,k,j,i));
      // Calculate b_hat vector
      Real b1 = bcc(m,IBX,k,j,i)/B_mag;
      Real b2 = bcc(m,IBY,k,j,i)/B_mag;
      Real b3 = bcc(m,IBZ,k,j,i)/B_mag;

      // calculate b_hat vector at i +/- 1
      Real B_mag_ip1 = sqrt( bcc(m,IBX,k,j,i+1)*bcc(m,IBX,k,j,i+1)
                           + bcc(m,IBY,k,j,i+1)*bcc(m,IBY,k,j,i+1)
                           + bcc(m,IBZ,k,j,i+1)*bcc(m,IBZ,k,j,i+1));
      Real b1_ip1 = bcc(m,IBX,k,j,i+1)/B_mag_ip1;
      Real b2_ip1 = bcc(m,IBY,k,j,i+1)/B_mag_ip1;
      Real b3_ip1 = bcc(m,IBZ,k,j,i+1)/B_mag_ip1;

      Real B_mag_im1 = sqrt( bcc(m,IBX,k,j,i-1)*bcc(m,IBX,k,j,i-1)
                           + bcc(m,IBY,k,j,i-1)*bcc(m,IBY,k,j,i-1)
                           + bcc(m,IBZ,k,j,i-1)*bcc(m,IBZ,k,j,i-1));
      Real b1_im1 = bcc(m,IBX,k,j,i-1)/B_mag_im1;
      Real b2_im1 = bcc(m,IBY,k,j,i-1)/B_mag_im1;
      Real b3_im1 = bcc(m,IBZ,k,j,i-1)/B_mag_im1;

      // calculate b_hat vector at j +/- 1
      Real B_mag_jp1 = sqrt( bcc(m,IBX,k,j+1,i)*bcc(m,IBX,k,j+1,i)
                           + bcc(m,IBY,k,j+1,i)*bcc(m,IBY,k,j+1,i)
                           + bcc(m,IBZ,k,j+1,i)*bcc(m,IBZ,k,j+1,i));
      Real b1_jp1 = bcc(m,IBX,k,j+1,i)/B_mag_jp1;
      Real b2_jp1 = bcc(m,IBY,k,j+1,i)/B_mag_jp1;
      Real b3_jp1 = bcc(m,IBZ,k,j+1,i)/B_mag_jp1;

      Real B_mag_jm1 = sqrt( bcc(m,IBX,k,j-1,i)*bcc(m,IBX,k,j-1,i)
                           + bcc(m,IBY,k,j-1,i)*bcc(m,IBY,k,j-1,i)
                           + bcc(m,IBZ,k,j-1,i)*bcc(m,IBZ,k,j-1,i));
      Real b1_jm1 = bcc(m,IBX,k,j-1,i)/B_mag_jm1;
      Real b2_jm1 = bcc(m,IBY,k,j-1,i)/B_mag_jm1;
      Real b3_jm1 = bcc(m,IBZ,k,j-1,i)/B_mag_jm1;

      // calculate b_hat vector at k +/- 1
      Real B_mag_kp1 = sqrt( bcc(m,IBX,k+1,j,i)*bcc(m,IBX,k+1,j,i)
                           + bcc(m,IBY,k+1,j,i)*bcc(m,IBY,k+1,j,i)
                           + bcc(m,IBZ,k+1,j,i)*bcc(m,IBZ,k+1,j,i));
      Real b1_kp1 = bcc(m,IBX,k+1,j,i)/B_mag_kp1;
      Real b2_kp1 = bcc(m,IBY,k+1,j,i)/B_mag_kp1;
      Real b3_kp1 = bcc(m,IBZ,k+1,j,i)/B_mag_kp1;

      Real B_mag_km1 = sqrt( bcc(m,IBX,k-1,j,i)*bcc(m,IBX,k-1,j,i)
                           + bcc(m,IBY,k-1,j,i)*bcc(m,IBY,k-1,j,i)
                           + bcc(m,IBZ,k-1,j,i)*bcc(m,IBZ,k-1,j,i));
      Real b1_km1 = bcc(m,IBX,k-1,j,i)/B_mag_km1;
      Real b2_km1 = bcc(m,IBY,k-1,j,i)/B_mag_km1;
      Real b3_km1 = bcc(m,IBZ,k-1,j,i)/B_mag_km1;

      // Central differencing of b_hat vector
      Real db1_dx1 = (b1_ip1 - b1_im1)/(2.0*size.d_view(m).dx1);
      Real db2_dx1 = (b2_ip1 - b2_im1)/(2.0*size.d_view(m).dx1);
      Real db3_dx1 = (b3_ip1 - b3_im1)/(2.0*size.d_view(m).dx1);

      Real db1_dx2 = (b1_jp1 - b1_jm1)/(2.0*size.d_view(m).dx2);
      Real db2_dx2 = (b2_jp1 - b2_jm1)/(2.0*size.d_view(m).dx2);
      Real db3_dx2 = (b3_jp1 - b3_jm1)/(2.0*size.d_view(m).dx2);

      Real db1_dx3 = (b1_kp1 - b1_km1)/(2.0*size.d_view(m).dx3);
      Real db2_dx3 = (b2_kp1 - b2_km1)/(2.0*size.d_view(m).dx3);
      Real db3_dx3 = (b3_kp1 - b3_km1)/(2.0*size.d_view(m).dx3);

      // Calculate curvature = |b_hat dot nabla b_hat|
      Real curv1 = b1*db1_dx1 + b2*db1_dx2 + b3*db1_dx3;
      Real curv2 = b1*db2_dx1 + b2*db2_dx2 + b3*db2_dx3;
      Real curv3 = b1*db3_dx1 + b2*db3_dx2 + b3*db3_dx3;

      dv(m,i_dv,k,j,i) = sqrt(curv1*curv1 + curv2*curv2 + curv3*curv3);
    });
    i_dv += 1; // increment derived variable index
  }

  // contravariant four-current jcon.  Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_jcon") == 0) {
    Kokkos::realloc(derived_var, nmb, 4, n3, n2, n1);
    auto jcon = derived_var;

    // Coordinates
    auto &coord = pm->pmb_pack->pcoord->coord_data;
    bool &flat = coord.is_minkowski;
    auto &spin = coord.bh_spin;

    auto &dtold = pm->dtold;
    auto w0_ = pm->pmb_pack->pmhd->w0;
    auto bcc_ = pm->pmb_pack->pmhd->bcc0;
    auto wsaved_ = pm->pmb_pack->pmhd->wsaved;
    auto bccsaved_ = pm->pmb_pack->pmhd->bccsaved;

    if (!pm->pmb_pack->pmhd->wbcc_saved) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Cannot compute jcon without saved MHD state" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    par_for("jcon", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      // Extract components of metric
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

      // Get 4-velocity for current step
      const Real &uu1 = w0_(m,IVX,k,j,i);
      const Real &uu2 = w0_(m,IVY,k,j,i);
      const Real &uu3 = w0_(m,IVZ,k,j,i);

      Real uu_sq = glower[1][1]*uu1*uu1 + 2.0*glower[1][2]*uu1*uu2
                 + 2.0*glower[1][3]*uu1*uu3
                 + glower[2][2]*uu2*uu2 + 2.0*glower[2][3]*uu2*uu3
                 + glower[3][3]*uu3*uu3;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real gamma = sqrt(1.0 + uu_sq);

      Real ucon[4];
      ucon[0] = gamma / alpha;
      ucon[1] = uu1 - alpha * gamma * gupper[0][1];
      ucon[2] = uu2 - alpha * gamma * gupper[0][2];
      ucon[3] = uu3 - alpha * gamma * gupper[0][3];

      // Get 4-velocity for last step
      const Real &uu1saved = wsaved_(m,IVX,k,j,i);
      const Real &uu2saved = wsaved_(m,IVY,k,j,i);
      const Real &uu3saved = wsaved_(m,IVZ,k,j,i);

      uu_sq = glower[1][1]*uu1saved*uu1saved + 2.0*glower[1][2]*uu1saved*uu2saved
            + 2.0*glower[1][3]*uu1saved*uu3saved
            + glower[2][2]*uu2saved*uu2saved + 2.0*glower[2][3]*uu2saved*uu3saved
            + glower[3][3]*uu3saved*uu3saved;
      gamma = sqrt(1.0 + uu_sq);

      Real uconsaved[4];
      uconsaved[0] = gamma / alpha;
      uconsaved[1] = uu1saved - alpha * gamma * gupper[0][1];
      uconsaved[2] = uu2saved - alpha * gamma * gupper[0][2];
      uconsaved[3] = uu3saved - alpha * gamma * gupper[0][3];

      // Lower 4-velocities
      Real ucov[4], ucovsaved[4];
      for (int mu=0; mu<4; ++mu) {
        ucov[mu] = 0.0;
        ucovsaved[mu] = 0.0;
        for (int nu=0; nu<4; ++nu) {
          ucov[mu] += glower[mu][nu]*ucon[nu];
          ucovsaved[mu] += glower[mu][nu]*uconsaved[nu];
        }
      }

      // Get bcon and bconsaved
      Real bcon[4];
      Real bconsaved[4];

      bcon[0] = bcc_(m,IBX,k,j,i) * ucov[1] + bcc_(m,IBY,k,j,i) * ucov[2]
              + bcc_(m,IBZ,k,j,i) * ucov[3];
      bcon[1] = (bcc_(m,IBX,k,j,i) + bcon[0] * ucon[1]) / ucon[0];
      bcon[2] = (bcc_(m,IBY,k,j,i) + bcon[0] * ucon[2]) / ucon[0];
      bcon[3] = (bcc_(m,IBZ,k,j,i) + bcon[0] * ucon[3]) / ucon[0];

      bconsaved[0] = bccsaved_(m,IBX,k,j,i) * ucovsaved[1]
                   + bccsaved_(m,IBY,k,j,i) * ucovsaved[2]
                   + bccsaved_(m,IBZ,k,j,i) * ucovsaved[3];
      bconsaved[1] = (bccsaved_(m,IBX,k,j,i) + bconsaved[0]*uconsaved[1]) / uconsaved[0];
      bconsaved[2] = (bccsaved_(m,IBY,k,j,i) + bconsaved[0]*uconsaved[2]) / uconsaved[0];
      bconsaved[3] = (bccsaved_(m,IBZ,k,j,i) + bconsaved[0]*uconsaved[3]) / uconsaved[0];

      // Lower bcon and bconsaved
      Real bcov[4], bcovsaved[4];

      for (int mu=0; mu<4; ++mu) {
        bcov[mu] = 0.0;
        bcovsaved[mu] = 0.0;
        for (int nu=0; nu<4; ++nu) {
          bcov[mu] += glower[mu][nu]*bcon[nu];
          bcovsaved[mu] += glower[mu][nu]*bconsaved[nu];
        }
      }

      // Compute current
      for (int mu=0; mu<4; ++mu) {
        if (dtold > 0) {
          const Real gF0p = get_detg_Fcon(0, mu, ucov, bcov);
          const Real gF0m = get_detg_Fcon(0, mu, ucovsaved, bcovsaved);
          const Real gF1p = get_detg_Fcon(1, mu, ucov, bcov);
          const Real gF1m = get_detg_Fcon(1, mu, ucovsaved, bcovsaved);
          const Real gF2p = (multi_d) ? get_detg_Fcon(2, mu, ucov, bcov) : 0.;
          const Real gF2m = (multi_d) ? get_detg_Fcon(2, mu, ucovsaved, bcovsaved) : 0.;
          const Real gF3p = (three_d) ? get_detg_Fcon(3, mu, ucov, bcov) : 0.;
          const Real gF3m = (three_d) ? get_detg_Fcon(3, mu, ucovsaved, bcovsaved) : 0.;

          const Real dgF0 = (gF0p - gF0m) / dtold;
          const Real dgF1 = (gF1p - gF1m) / (2 * size.d_view(m).dx1);
          const Real dgF2 = (multi_d) ? (gF2p - gF2m) / (2 * size.d_view(m).dx2) : 0.;
          const Real dgF3 = (three_d) ? (gF3p - gF3m) / (2 * size.d_view(m).dx3) : 0.;

          const Real detg = 1.;
          jcon(m,mu,k,j,i) = 1. / (detg * sqrt(4. * M_PI)) * (dgF0 + dgF1 + dgF2 + dgF3);
        } else {
          // zero current if dtold == 0 (e.g., when we don't have a previous step)
          jcon(m,mu,k,j,i) = 0.;
        }
      }
    });
  }

  // get all sgs terms for the MHD equations
  if (name.compare("mhd_sgs") == 0) {
    int n_sgs = 59;
    Kokkos::realloc(derived_var, nmb, n_sgs, n3, n2, n1);
    auto dv = derived_var;
    auto u0_ = pm->pmb_pack->pmhd->u0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("mhd_sgs", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real rho = u0_(m,IDN,k,j,i);
      Real mx = u0_(m,IVX,k,j,i);
      Real my = u0_(m,IVY,k,j,i);
      Real mz = u0_(m,IVZ,k,j,i);
      Real eint = u0_(m,IEN,k,j,i); // eint = P/(gamma-1)
      Real Bx = bcc(m,IBX,k,j,i);
      Real By = bcc(m,IBY,k,j,i);
      Real Bz = bcc(m,IBZ,k,j,i);
      // state variables
      dv(m,0,k,j,i) = rho;
      dv(m,1,k,j,i) = mx;
      dv(m,2,k,j,i) = my;
      dv(m,3,k,j,i) = mz;
      dv(m,4,k,j,i) = eint;
      dv(m,5,k,j,i) = Bx;
      dv(m,6,k,j,i) = By;
      dv(m,7,k,j,i) = Bz;
      // rho * v_i * v_j = m_i * m_j / rho
      dv(m,8,k,j,i) = mx*mx/rho;
      dv(m,9,k,j,i) = mx*my/rho;
      dv(m,10,k,j,i) = mx*mz/rho;
      dv(m,11,k,j,i) = my*my/rho;
      dv(m,12,k,j,i) = my*mz/rho;
      dv(m,13,k,j,i) = mz*mz/rho;
      // B_i * B_j
      dv(m,14,k,j,i) = Bx*Bx;
      dv(m,15,k,j,i) = Bx*By;
      dv(m,16,k,j,i) = Bx*Bz;
      dv(m,17,k,j,i) = By*By;
      dv(m,18,k,j,i) = By*Bz;
      dv(m,19,k,j,i) = Bz*Bz;
      // v_i * B_j
      dv(m,20,k,j,i) = mx*Bx/rho;
      dv(m,21,k,j,i) = mx*By/rho;
      dv(m,22,k,j,i) = mx*Bz/rho;
      dv(m,23,k,j,i) = my*Bx/rho;
      dv(m,24,k,j,i) = my*By/rho;
      dv(m,25,k,j,i) = my*Bz/rho;
      dv(m,26,k,j,i) = mz*Bx/rho;
      dv(m,27,k,j,i) = mz*By/rho;
      dv(m,28,k,j,i) = mz*Bz/rho;
      // rho * v_i * T = m_i * P / rho = m_i * P/rho / (gamma-1)
      dv(m,29,k,j,i) = mx*eint/rho;
      dv(m,30,k,j,i) = my*eint/rho;
      dv(m,31,k,j,i) = mz*eint/rho;
      // rho * v_i * v_j**2 = m_i * m_j**2 / rho**2
      dv(m,32,k,j,i) = mx*mx*mx/rho/rho;
      dv(m,33,k,j,i) = mx*my*my/rho/rho;
      dv(m,34,k,j,i) = mx*mz*mz/rho/rho;
      dv(m,35,k,j,i) = my*mx*mx/rho/rho;
      dv(m,36,k,j,i) = my*my*my/rho/rho;
      dv(m,37,k,j,i) = my*mz*mz/rho/rho;
      dv(m,38,k,j,i) = mz*mx*mx/rho/rho;
      dv(m,39,k,j,i) = mz*my*my/rho/rho;
      dv(m,40,k,j,i) = mz*mz*mz/rho/rho;
      // v_i * B_j**2 = m_i * B_j**2 / rho
      dv(m,41,k,j,i) = mx*Bx*Bx/rho;
      dv(m,42,k,j,i) = mx*By*By/rho;
      dv(m,43,k,j,i) = mx*Bz*Bz/rho;
      dv(m,44,k,j,i) = my*Bx*Bx/rho;
      dv(m,45,k,j,i) = my*By*By/rho;
      dv(m,46,k,j,i) = my*Bz*Bz/rho;
      dv(m,47,k,j,i) = mz*Bx*Bx/rho;
      dv(m,48,k,j,i) = mz*By*By/rho;
      dv(m,49,k,j,i) = mz*Bz*Bz/rho;
      // v_i * B_i * B_j = m_i * B_i * B_j / rho
      dv(m,50,k,j,i) = mx*Bx*Bx/rho;
      dv(m,51,k,j,i) = mx*Bx*By/rho;
      dv(m,52,k,j,i) = mx*Bx*Bz/rho;
      dv(m,53,k,j,i) = my*By*Bx/rho;
      dv(m,54,k,j,i) = my*By*By/rho;
      dv(m,55,k,j,i) = my*By*Bz/rho;
      dv(m,56,k,j,i) = mz*Bz*Bx/rho;
      dv(m,57,k,j,i) = mz*Bz*By/rho;
      dv(m,58,k,j,i) = mz*Bz*Bz/rho;
    });
  }

// get all sgs terms for the MHD equations
  if (name.compare("hydro_sgs") == 0) {
    int n_sgs = 23;
    Kokkos::realloc(derived_var, nmb, n_sgs, n3, n2, n1);
    auto dv = derived_var;
    auto u0_ = pm->pmb_pack->phydro->u0;
    par_for("hydro_sgs", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real rho = u0_(m,IDN,k,j,i);
      Real mx = u0_(m,IVX,k,j,i);
      Real my = u0_(m,IVY,k,j,i);
      Real mz = u0_(m,IVZ,k,j,i);
      Real eint = u0_(m,IEN,k,j,i); // eint = P/(gamma-1)
      // state variables
      dv(m,0,k,j,i) = rho;
      dv(m,1,k,j,i) = mx;
      dv(m,2,k,j,i) = my;
      dv(m,3,k,j,i) = mz;
      dv(m,4,k,j,i) = eint;
      // rho * v_i * v_j = m_i * m_j / rho
      dv(m,5,k,j,i) = mx*mx/rho;
      dv(m,6,k,j,i) = mx*my/rho;
      dv(m,7,k,j,i) = mx*mz/rho;
      dv(m,8,k,j,i) = my*my/rho;
      dv(m,9,k,j,i) = my*mz/rho;
      dv(m,10,k,j,i) = mz*mz/rho;
      // rho * v_i * T = m_i * P / rho = m_i * P/rho / (gamma-1)
      dv(m,11,k,j,i) = mx*eint/rho;
      dv(m,12,k,j,i) = my*eint/rho;
      dv(m,13,k,j,i) = mz*eint/rho;
      // rho * v_i * v_j**2 = m_i * m_j**2 / rho**2
      dv(m,14,k,j,i) = mx*mx*mx/rho/rho;
      dv(m,15,k,j,i) = mx*my*my/rho/rho;
      dv(m,16,k,j,i) = mx*mz*mz/rho/rho;
      dv(m,17,k,j,i) = my*mx*mx/rho/rho;
      dv(m,18,k,j,i) = my*my*my/rho/rho;
      dv(m,19,k,j,i) = my*mz*mz/rho/rho;
      dv(m,20,k,j,i) = mz*mx*mx/rho/rho;
      dv(m,21,k,j,i) = mz*my*my/rho/rho;
      dv(m,22,k,j,i) = mz*mz*mz/rho/rho;
    });
  }

// get all moments terms for |v| and |B|
  if (name.compare("mhd_v_B_moments") == 0) {
    int n_moments = 8;
    Kokkos::realloc(derived_var, n_moments, 1, n3, n2, n1);
    auto dv = derived_var;
    auto &w0_ = pm->pmb_pack->pmhd->w0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("mhd_v_B_moments", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real v = sqrt(w0_(m,IVX,k,j,i)*w0_(m,IVX,k,j,i)
                  + w0_(m,IVY,k,j,i)*w0_(m,IVY,k,j,i)
                  + w0_(m,IVZ,k,j,i)*w0_(m,IVZ,k,j,i));
      Real B = sqrt(bcc(m,IBX,k,j,i)*bcc(m,IBX,k,j,i)
                  + bcc(m,IBY,k,j,i)*bcc(m,IBY,k,j,i)
                  + bcc(m,IBZ,k,j,i)*bcc(m,IBZ,k,j,i));
      // v, v^2, v^3, v^4
      dv(m,0,k,j,i) = v;
      dv(m,1,k,j,i) = v*v;
      dv(m,2,k,j,i) = v*v*v;
      dv(m,3,k,j,i) = v*v*v*v;
      // B, B^2, B^3, B^4
      dv(m,4,k,j,i) = B;
      dv(m,5,k,j,i) = B*B;
      dv(m,6,k,j,i) = B*B*B;
      dv(m,7,k,j,i) = B*B*B*B;
    });
  }

// get all moments terms for v_i and B_i
  if (name.compare("mhd_vi_Bi_moments") == 0) {
    int n_moments = 24;
    Kokkos::realloc(derived_var, n_moments, 1, n3, n2, n1);
    auto dv = derived_var;
    auto &w0_ = pm->pmb_pack->pmhd->w0;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("mhd_vi_Bi_moments", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real vx = w0_(m,IVX,k,j,i);
      Real vy = w0_(m,IVY,k,j,i);
      Real vz = w0_(m,IVZ,k,j,i);
      Real Bx = bcc(m,IBX,k,j,i);
      Real By = bcc(m,IBY,k,j,i);
      Real Bz = bcc(m,IBZ,k,j,i);
      // v_i, v_i^2, v_i^3, v_i^4
      dv(m,0,k,j,i) = vx;
      dv(m,1,k,j,i) = vx*vx;
      dv(m,2,k,j,i) = vx*vx*vx;
      dv(m,3,k,j,i) = vx*vx*vx*vx;
      dv(m,4,k,j,i) = vy;
      dv(m,5,k,j,i) = vy*vy;
      dv(m,6,k,j,i) = vy*vy*vy;
      dv(m,7,k,j,i) = vy*vy*vy*vy;
      dv(m,8,k,j,i) = vz;
      dv(m,9,k,j,i) = vz*vz;
      dv(m,10,k,j,i) = vz*vz*vz;
      dv(m,11,k,j,i) = vz*vz*vz*vz;
      // B_i, B_i^2, B_i^3, B_i^4
      dv(m,12,k,j,i) = Bx;
      dv(m,13,k,j,i) = Bx*Bx;
      dv(m,14,k,j,i) = Bx*Bx*Bx;
      dv(m,15,k,j,i) = Bx*Bx*Bx*Bx;
      dv(m,16,k,j,i) = By;
      dv(m,17,k,j,i) = By*By;
      dv(m,18,k,j,i) = By*By*By;
      dv(m,19,k,j,i) = By*By*By*By;
      dv(m,20,k,j,i) = Bz;
      dv(m,21,k,j,i) = Bz*Bz;
      dv(m,22,k,j,i) = Bz*Bz*Bz;
      dv(m,23,k,j,i) = Bz*Bz*Bz*Bz;
    });
  }

// get all moments terms for |v|
  if (name.compare("hydro_v_moments") == 0) {
    int n_moments = 4;
    Kokkos::realloc(derived_var, n_moments, 1, n3, n2, n1);
    auto dv = derived_var;
    auto &w0_ = pm->pmb_pack->phydro->w0;
    par_for("hydro_v_moments", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real v = sqrt(w0_(m,IVX,k,j,i)*w0_(m,IVX,k,j,i)
                  + w0_(m,IVY,k,j,i)*w0_(m,IVY,k,j,i)
                  + w0_(m,IVZ,k,j,i)*w0_(m,IVZ,k,j,i));
      // v, v^2, v^3, v^4
      dv(m,0,k,j,i) = v;
      dv(m,1,k,j,i) = v*v;
      dv(m,2,k,j,i) = v*v*v;
      dv(m,3,k,j,i) = v*v*v*v;
    });
  }

// get all moments terms for v_i and B_i
  if (name.compare("hydro_vi_moments") == 0) {
    int n_moments = 12;
    Kokkos::realloc(derived_var, n_moments, 1, n3, n2, n1);
    auto dv = derived_var;
    auto &w0_ = pm->pmb_pack->phydro->w0;
    par_for("hydro_moments", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real vx = w0_(m,IVX,k,j,i);
      Real vy = w0_(m,IVY,k,j,i);
      Real vz = w0_(m,IVZ,k,j,i);

      // v_i, v_i^2, v_i^3, v_i^4
      dv(m,0,k,j,i) = vx;
      dv(m,1,k,j,i) = vx*vx;
      dv(m,2,k,j,i) = vx*vx*vx;
      dv(m,3,k,j,i) = vx*vx*vx*vx;
      dv(m,4,k,j,i) = vy;
      dv(m,5,k,j,i) = vy*vy;
      dv(m,6,k,j,i) = vy*vy*vy;
      dv(m,7,k,j,i) = vy*vy*vy*vy;
      dv(m,8,k,j,i) = vz;
      dv(m,9,k,j,i) = vz*vz;
      dv(m,10,k,j,i) = vz*vz*vz;
      dv(m,11,k,j,i) = vz*vz*vz*vz;
    });
  }


  // magnitude of K_JxB = | j x B | / B^2
  // Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_k_jxb") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("mhd_k_jxb", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // calculate j
      Real j1 = 0.0;
      Real j2 = -(bcc(m,IBZ,k,j,i+1) - bcc(m,IBZ,k,j,i-1))/size.d_view(m).dx1;
      Real j3 =  (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        j1 += (bcc(m,IBZ,k,j+1,i) - bcc(m,IBZ,k,j-1,i))/size.d_view(m).dx2;
        j3 -= (bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        j1 -= (bcc(m,IBY,k+1,j,i) - bcc(m,IBY,k-1,j,i))/size.d_view(m).dx3;
        j2 += (bcc(m,IBX,k+1,j,i) - bcc(m,IBX,k-1,j,i))/size.d_view(m).dx3;
      }
      // calculate B
      Real B_mag_sq =    bcc(m,IBX,k,j,i)*bcc(m,IBX,k,j,i)
                       + bcc(m,IBY,k,j,i)*bcc(m,IBY,k,j,i)
                       + bcc(m,IBZ,k,j,i)*bcc(m,IBZ,k,j,i);

      // calculate j x B
      Real jxB1 = j2*bcc(m,IBZ,k,j,i) - j3*bcc(m,IBY,k,j,i);
      Real jxB2 = j3*bcc(m,IBX,k,j,i) - j1*bcc(m,IBZ,k,j,i);
      Real jxB3 = j1*bcc(m,IBY,k,j,i) - j2*bcc(m,IBX,k,j,i);

      // calculate | j x B | / B^2
      dv(m,i_dv,k,j,i) = sqrt(jxB1*jxB1 + jxB2*jxB2 + jxB3*jxB3) / B_mag_sq;
    });
  }

  // magnitude of curv_perp = |(j x B / B^2) - b_hat dot nabla b_hat|
  // Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_curv_perp") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("curv_perp", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // calculate j
      Real j1 = 0.0;
      Real j2 = -(bcc(m,IBZ,k,j,i+1) - bcc(m,IBZ,k,j,i-1))/size.d_view(m).dx1;
      Real j3 =  (bcc(m,IBY,k,j,i+1) - bcc(m,IBY,k,j,i-1))/size.d_view(m).dx1;
      if (multi_d) {
        j1 += (bcc(m,IBZ,k,j+1,i) - bcc(m,IBZ,k,j-1,i))/size.d_view(m).dx2;
        j3 -= (bcc(m,IBX,k,j+1,i) - bcc(m,IBX,k,j-1,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        j1 -= (bcc(m,IBY,k+1,j,i) - bcc(m,IBY,k-1,j,i))/size.d_view(m).dx3;
        j2 += (bcc(m,IBX,k+1,j,i) - bcc(m,IBX,k-1,j,i))/size.d_view(m).dx3;
      }
      // calculate B
      Real B_mag_sq =    bcc(m,IBX,k,j,i)*bcc(m,IBX,k,j,i)
                       + bcc(m,IBY,k,j,i)*bcc(m,IBY,k,j,i)
                       + bcc(m,IBZ,k,j,i)*bcc(m,IBZ,k,j,i);

      // calculate j x B
      Real jxB1_Bsq = (j2*bcc(m,IBZ,k,j,i) - j3*bcc(m,IBY,k,j,i))/(B_mag_sq);
      Real jxB2_Bsq = (j3*bcc(m,IBX,k,j,i) - j1*bcc(m,IBZ,k,j,i))/(B_mag_sq);
      Real jxB3_Bsq = (j1*bcc(m,IBY,k,j,i) - j2*bcc(m,IBX,k,j,i))/(B_mag_sq);


      // now calculate curve_parallel
      // Calculate b_hat vector
      Real b1 = bcc(m,IBX,k,j,i)/sqrt(B_mag_sq);
      Real b2 = bcc(m,IBY,k,j,i)/sqrt(B_mag_sq);
      Real b3 = bcc(m,IBZ,k,j,i)/sqrt(B_mag_sq);

      // calculate b_hat vector at i +/- 1
      Real B_mag_ip1 = sqrt( bcc(m,IBX,k,j,i+1)*bcc(m,IBX,k,j,i+1)
                           + bcc(m,IBY,k,j,i+1)*bcc(m,IBY,k,j,i+1)
                           + bcc(m,IBZ,k,j,i+1)*bcc(m,IBZ,k,j,i+1));
      Real b1_ip1 = bcc(m,IBX,k,j,i+1)/B_mag_ip1;
      Real b2_ip1 = bcc(m,IBY,k,j,i+1)/B_mag_ip1;
      Real b3_ip1 = bcc(m,IBZ,k,j,i+1)/B_mag_ip1;

      Real B_mag_im1 = sqrt( bcc(m,IBX,k,j,i-1)*bcc(m,IBX,k,j,i-1)
                           + bcc(m,IBY,k,j,i-1)*bcc(m,IBY,k,j,i-1)
                           + bcc(m,IBZ,k,j,i-1)*bcc(m,IBZ,k,j,i-1));
      Real b1_im1 = bcc(m,IBX,k,j,i-1)/B_mag_im1;
      Real b2_im1 = bcc(m,IBY,k,j,i-1)/B_mag_im1;
      Real b3_im1 = bcc(m,IBZ,k,j,i-1)/B_mag_im1;

      // calculate b_hat vector at j +/- 1
      Real B_mag_jp1 = sqrt( bcc(m,IBX,k,j+1,i)*bcc(m,IBX,k,j+1,i)
                           + bcc(m,IBY,k,j+1,i)*bcc(m,IBY,k,j+1,i)
                           + bcc(m,IBZ,k,j+1,i)*bcc(m,IBZ,k,j+1,i));
      Real b1_jp1 = bcc(m,IBX,k,j+1,i)/B_mag_jp1;
      Real b2_jp1 = bcc(m,IBY,k,j+1,i)/B_mag_jp1;
      Real b3_jp1 = bcc(m,IBZ,k,j+1,i)/B_mag_jp1;

      Real B_mag_jm1 = sqrt( bcc(m,IBX,k,j-1,i)*bcc(m,IBX,k,j-1,i)
                           + bcc(m,IBY,k,j-1,i)*bcc(m,IBY,k,j-1,i)
                           + bcc(m,IBZ,k,j-1,i)*bcc(m,IBZ,k,j-1,i));
      Real b1_jm1 = bcc(m,IBX,k,j-1,i)/B_mag_jm1;
      Real b2_jm1 = bcc(m,IBY,k,j-1,i)/B_mag_jm1;
      Real b3_jm1 = bcc(m,IBZ,k,j-1,i)/B_mag_jm1;

      // calculate b_hat vector at k +/- 1
      Real B_mag_kp1 = sqrt( bcc(m,IBX,k+1,j,i)*bcc(m,IBX,k+1,j,i)
                           + bcc(m,IBY,k+1,j,i)*bcc(m,IBY,k+1,j,i)
                           + bcc(m,IBZ,k+1,j,i)*bcc(m,IBZ,k+1,j,i));
      Real b1_kp1 = bcc(m,IBX,k+1,j,i)/B_mag_kp1;
      Real b2_kp1 = bcc(m,IBY,k+1,j,i)/B_mag_kp1;
      Real b3_kp1 = bcc(m,IBZ,k+1,j,i)/B_mag_kp1;

      Real B_mag_km1 = sqrt( bcc(m,IBX,k-1,j,i)*bcc(m,IBX,k-1,j,i)
                           + bcc(m,IBY,k-1,j,i)*bcc(m,IBY,k-1,j,i)
                           + bcc(m,IBZ,k-1,j,i)*bcc(m,IBZ,k-1,j,i));
      Real b1_km1 = bcc(m,IBX,k-1,j,i)/B_mag_km1;
      Real b2_km1 = bcc(m,IBY,k-1,j,i)/B_mag_km1;
      Real b3_km1 = bcc(m,IBZ,k-1,j,i)/B_mag_km1;

      // Central differencing of b_hat vector
      Real db1_dx1 = (b1_ip1 - b1_im1)/(2.0*size.d_view(m).dx1);
      Real db2_dx1 = (b2_ip1 - b2_im1)/(2.0*size.d_view(m).dx1);
      Real db3_dx1 = (b3_ip1 - b3_im1)/(2.0*size.d_view(m).dx1);

      Real db1_dx2 = (b1_jp1 - b1_jm1)/(2.0*size.d_view(m).dx2);
      Real db2_dx2 = (b2_jp1 - b2_jm1)/(2.0*size.d_view(m).dx2);
      Real db3_dx2 = (b3_jp1 - b3_jm1)/(2.0*size.d_view(m).dx2);

      Real db1_dx3 = (b1_kp1 - b1_km1)/(2.0*size.d_view(m).dx3);
      Real db2_dx3 = (b2_kp1 - b2_km1)/(2.0*size.d_view(m).dx3);
      Real db3_dx3 = (b3_kp1 - b3_km1)/(2.0*size.d_view(m).dx3);

      // Calculate curvature = |b_hat dot nabla b_hat|
      Real curv1 = b1*db1_dx1 + b2*db1_dx2 + b3*db1_dx3;
      Real curv2 = b1*db2_dx1 + b2*db2_dx2 + b3*db2_dx3;
      Real curv3 = b1*db3_dx1 + b2*db3_dx2 + b3*db3_dx3;

      // calculate |(j x B / B^2) - b_hat dot nabla b_hat|
      dv(m,i_dv,k,j,i) = sqrt((jxB1_Bsq - curv1)*(jxB1_Bsq - curv1)
                         + (jxB2_Bsq - curv2)*(jxB2_Bsq - curv2)
                         + (jxB3_Bsq - curv3)*(jxB3_Bsq - curv3));
    });
    i_dv += 1; // increment derived variable index
  }

  // magnitude of bmag = |bcc|
  // Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_bmag") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);
    auto dv = derived_var;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    par_for("bmag", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      dv(m,i_dv,k,j,i) = sqrt( bcc(m,IBX,k,j,i)*bcc(m,IBX,k,j,i)
                          + bcc(m,IBY,k,j,i)*bcc(m,IBY,k,j,i)
                          + bcc(m,IBZ,k,j,i)*bcc(m,IBZ,k,j,i));
    });
    i_dv += 1; // increment derived variable index
  }

  // Calculated from cell-centered fields.
  // Not computed in ghost zones since requires derivative
  if (name.compare("mhd_dynamo_ks") == 0) {
    // 0 = < B^2 >
    // 1 = < B^4 >
    // 2 = < (d_j B_i)(d_j B_i) >
    // 3 = < (B_j d_j B_i)(B_k d_k B_i) >
    // 4 = < |BxJ|^2 >
    // 5 = < |B.J|^2 >
    // 6 = < U^2 >
    // 7 = < (d_j U_i)(d_j U_i) >
    Kokkos::realloc(derived_var, nmb, 8, n3, n2, n1);
    auto dv = derived_var;
    auto &bcc = pm->pmb_pack->pmhd->bcc0;
    auto &b = pm->pmb_pack->pmhd->b0;
    auto &w0_ = pm->pmb_pack->pmhd->w0;
    par_for("bmag", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx_squared = size.d_view(m).dx1 * size.d_view(m).dx1;
      // 0 = < B^4 >
      Real B_mag_sq = bcc(m,IBX,k,j,i)*bcc(m,IBX,k,j,i)
                    + bcc(m,IBY,k,j,i)*bcc(m,IBY,k,j,i)
                    + bcc(m,IBZ,k,j,i)*bcc(m,IBZ,k,j,i);
      dv(m,0,k,j,i) = B_mag_sq;
      Real B_fourth = B_mag_sq*B_mag_sq;
      dv(m,1,k,j,i) = B_fourth;
      // 1 = < (d_j B_i)(d_j B_i) >
      dv(m,2,k,j,i) =(  (b.x1f(m,k,j,i+1)-b.x1f(m,k,j,i))
                       *(b.x1f(m,k,j,i+1)-b.x1f(m,k,j,i))
                      + (b.x2f(m,k,j+1,i)-b.x2f(m,k,j,i))
                       *(b.x2f(m,k,j+1,i)-b.x2f(m,k,j,i))
                      + (b.x3f(m,k+1,j,i)-b.x3f(m,k,j,i))
                       *(b.x3f(m,k+1,j,i)-b.x3f(m,k,j,i))
                      + 0.25*(bcc(m,IBX,k,j+1,i)-bcc(m,IBX,k,j-1,i))
                            *(bcc(m,IBX,k,j+1,i)-bcc(m,IBX,k,j-1,i))
                      + 0.25*(bcc(m,IBX,k+1,j,i)-bcc(m,IBX,k-1,j,i))
                            *(bcc(m,IBX,k+1,j,i)-bcc(m,IBX,k-1,j,i))
                      + 0.25*(bcc(m,IBY,k,j,i+1)-bcc(m,IBY,k,j,i-1))
                            *(bcc(m,IBY,k,j,i+1)-bcc(m,IBY,k,j,i-1))
                      + 0.25*(bcc(m,IBY,k+1,j,i)-bcc(m,IBY,k-1,j,i))
                            *(bcc(m,IBY,k+1,j,i)-bcc(m,IBY,k-1,j,i))
                      + 0.25*(bcc(m,IBZ,k,j,i+1)-bcc(m,IBZ,k,j,i-1))
                            *(bcc(m,IBZ,k,j,i+1)-bcc(m,IBZ,k,j,i-1))
                      + 0.25*(bcc(m,IBZ,k,j+1,i)-bcc(m,IBZ,i,j-1,i))
                            *(bcc(m,IBZ,k,j+1,i)-bcc(m,IBZ,i,j-1,i)))
                      / dx_squared;
      // 2 = < (B_j d_j B_i)(B_k d_k B_i) >
      Real bdb1 = bcc(m,IBX,k,j,i)*(b.x1f(m,k,j,i+1)-b.x1f(m,k,j,i))
                  +0.5*bcc(m,IBY,k,j,i)*(bcc(m,IBX,k,j+1,i)-bcc(m,IBX,k,j-1,i))
                  +0.5*bcc(m,IBZ,k,j,i)*(bcc(m,IBX,k+1,j,i)-bcc(m,IBX,k-1,j,i));
      Real bdb2 = bcc(m,IBY,k,j,i)*(b.x2f(m,k,j+1,i)-b.x2f(m,k,j,i))
                  +0.5*bcc(m,IBZ,k,j,i)*(bcc(m,IBY,k+1,j,i)-bcc(m,IBY,k-1,j,i))
                  +0.5*bcc(m,IBX,k,j,i)*(bcc(m,IBY,k,j,i+1)-bcc(m,IBY,k,j,i-1));
      Real bdb3 = bcc(m,IBZ,k,j,i)*(b.x3f(m,k+1,j,i)-b.x3f(m,k,j,i))
                  +0.5*bcc(m,IBX,k,j,i)*(bcc(m,IBZ,k,j,i+1)-bcc(m,IBZ,k,j,i-1))
                  +0.5*bcc(m,IBY,k,j,i)*(bcc(m,IBZ,k,j+1,i)-bcc(m,IBZ,k,j-1,i));
      dv(m,3,k,j,i) = (bdb1*bdb1 + bdb2*bdb2 + bdb3*bdb3) / dx_squared;
      // 3 = < |BxJ|^2 >
      Real Jx = 0.5*(bcc(m,IBZ,k,j+1,i)-bcc(m,IBZ,k,j-1,i))
              - 0.5*(bcc(m,IBY,k+1,j,i)-bcc(m,IBY,k-1,j,i));
      Real Jy = 0.5*(bcc(m,IBX,k+1,j,i)-bcc(m,IBX,k-1,j,i))
              - 0.5*(bcc(m,IBZ,k,j,i+1)-bcc(m,IBZ,k,j,i-1));
      Real Jz = 0.5*(bcc(m,IBY,k,j,i+1)-bcc(m,IBY,k,j,i-1))
              - 0.5*(bcc(m,IBX,k,j+1,i)-bcc(m,IBX,k,j-1,i));
      dv(m,4,k,j,i) =( (bcc(m,IBY,k,j,i)*Jz - bcc(m,IBZ,k,j,i)*Jy)
                      *(bcc(m,IBY,k,j,i)*Jz - bcc(m,IBZ,k,j,i)*Jy)
                     + (bcc(m,IBZ,k,j,i)*Jx - bcc(m,IBX,k,j,i)*Jz)
                      *(bcc(m,IBZ,k,j,i)*Jx - bcc(m,IBX,k,j,i)*Jz)
                     + (bcc(m,IBX,k,j,i)*Jy - bcc(m,IBY,k,j,i)*Jx)
                      *(bcc(m,IBX,k,j,i)*Jy - bcc(m,IBY,k,j,i)*Jx))
                     / dx_squared;
      // 4 = < |B.J|^2 >
      dv(m,5,k,j,i) = ((bcc(m,IBX,k,j,i)*Jx + bcc(m,IBY,k,j,i)*Jy + bcc(m,IBZ,k,j,i)*Jz)
                      *(bcc(m,IBX,k,j,i)*Jx + bcc(m,IBY,k,j,i)*Jy + bcc(m,IBZ,k,j,i)*Jz))
                      /dx_squared;
      // 5 = < U^2 >
      dv(m,6,k,j,i) += (w0_(m,IVX,k,j,i)*w0_(m,IVX,k,j,i))
                     + (w0_(m,IVY,k,j,i)*w0_(m,IVY,k,j,i))
                     + (w0_(m,IVZ,k,j,i)*w0_(m,IVZ,k,j,i));
      // 6 = < (d_j U_i)(d_j U_i) >
      dv(m,7,k,j,i) +=((0.25*(w0_(m,IVX,k,j,i+1)-w0_(m,IVX,k,j,i-1))
                            *(w0_(m,IVX,k,j,i+1)-w0_(m,IVX,k,j,i-1))
                      + 0.25*(w0_(m,IVY,k,j+1,i)-w0_(m,IVY,k,j-1,i))
                            *(w0_(m,IVY,k,j+1,i)-w0_(m,IVY,k,j-1,i))
                      + 0.25*(w0_(m,IVZ,k+1,j,i)-w0_(m,IVZ,k-1,j,i))
                            *(w0_(m,IVZ,k+1,j,i)-w0_(m,IVZ,k-1,j,i))
                      + 0.25*(w0_(m,IVX,k,j+1,i)-w0_(m,IVX,k,j-1,i))
                            *(w0_(m,IVX,k,j+1,i)-w0_(m,IVX,k,j-1,i))
                      + 0.25*(w0_(m,IVX,k+1,j,i)-w0_(m,IVX,k-1,j,i))
                            *(w0_(m,IVX,k+1,j,i)-w0_(m,IVX,k-1,j,i))
                      + 0.25*(w0_(m,IVY,k,j,i+1)-w0_(m,IVY,k,j,i-1))
                            *(w0_(m,IVY,k,j,i+1)-w0_(m,IVY,k,j,i-1))
                      + 0.25*(w0_(m,IVY,k+1,j,i)-w0_(m,IVY,k-1,j,i))
                            *(w0_(m,IVY,k+1,j,i)-w0_(m,IVY,k-1,j,i))
                      + 0.25*(w0_(m,IVZ,k,j,i+1)-w0_(m,IVZ,k,j,i-1))
                            *(w0_(m,IVZ,k,j,i+1)-w0_(m,IVZ,k,j,i-1))
                      + 0.25*(w0_(m,IVZ,k,j+1,i)-w0_(m,IVZ,k,j-1,i))
                            *(w0_(m,IVZ,k,j+1,i)-w0_(m,IVZ,k,j-1,i))))
                      / dx_squared;
    });
  }

  // divergence of B, including ghost zones
  if (name.compare("mhd_divb") == 0) {
    if (derived_var.extent(4) <= 1)
      Kokkos::realloc(derived_var, nmb, n_dv, n3, n2, n1);

    // set the loop limits for 1D/2D/3D problems
    int jl = js, ju = je, kl = ks, ku = ke;
    if (multi_d) {
      jl = js-ng, ju = je+ng;
    } else if (three_d) {
      jl = js-ng, ju = je+ng, kl = ks-ng, ku = ke+ng;
    }

    auto dv = derived_var;
    auto b0 = pm->pmb_pack->pmhd->b0;
    par_for("divb", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, (is-ng), (ie+ng),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real divb = (b0.x1f(m,k,j,i+1) - b0.x1f(m,k,j,i))/size.d_view(m).dx1;
      if (multi_d) {
        divb += (b0.x2f(m,k,j+1,i) - b0.x2f(m,k,j,i))/size.d_view(m).dx2;
      }
      if (three_d) {
        divb += (b0.x3f(m,k+1,j,i) - b0.x3f(m,k,j,i))/size.d_view(m).dx3;
      }
      dv(m,i_dv,k,j,i) = divb;
    });
    i_dv += 1; // increment derived variable index
  }

  // radiation moments
  if (name.compare(0, 3, "rad") == 0) {
    // Determine if coordinate and/or fluid frame moments required
    bool needs_coord_only = (name.compare("rad_coord") == 0);
    bool needs_fluid_only = (name.compare("rad_fluid") == 0);
    bool needs_both = !(needs_coord_only || needs_fluid_only);
    int mom_var_size = (needs_both) ? 20 : 10;
    int moments_offset = (needs_both) ? 10 : 0;
    Kokkos::realloc(derived_var, nmb, mom_var_size, n3, n2, n1);
    auto dv = derived_var;

    // Coordinates
    auto &coord = pm->pmb_pack->pcoord->coord_data;
    bool &flat = coord.is_minkowski;
    Real &spin = coord.bh_spin;

    // Radiation
    int nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
    auto nh_c_ = pm->pmb_pack->prad->nh_c;
    auto tet_c_ = pm->pmb_pack->prad->tet_c;
    auto tetcov_c_ = pm->pmb_pack->prad->tetcov_c;
    auto solid_angles_ = pm->pmb_pack->prad->prgeo->solid_angles;
    auto i0_ = pm->pmb_pack->prad->i0;
    auto norm_to_tet_ = pm->pmb_pack->prad->norm_to_tet;

    // Select either Hydro or MHD (if fluid enabled)
    DvceArray5D<Real> w0_;
    if (pm->pmb_pack->phydro != nullptr) {
      w0_ = pm->pmb_pack->phydro->w0;
    } else if (pm->pmb_pack->pmhd != nullptr) {
      w0_ = pm->pmb_pack->pmhd->w0;
    }

    par_for("moments",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      // Extract components of metric
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);

      // coordinate component n^0
      Real n0 = tet_c_(m,0,0,k,j,i);

      // set coordinate frame components
      for (int n1=0, n12=0; n1<4; ++n1) {
        for (int n2=n1; n2<4; ++n2, ++n12) {
          dv(m,n12,k,j,i) = 0.0;
          for (int n=0; n<=nang1; ++n) {
            Real nmun1 = 0.0; Real nmun2 = 0.0; Real n_0 = 0.0;
            for (int d=0; d<4; ++d) {
              nmun1 += tet_c_   (m,d,n1,k,j,i)*nh_c_.d_view(n,d);
              nmun2 += tet_c_   (m,d,n2,k,j,i)*nh_c_.d_view(n,d);
              n_0   += tetcov_c_(m,d,0, k,j,i)*nh_c_.d_view(n,d);
            }
            dv(m,n12,k,j,i) += (nmun1*nmun2*(i0_(m,n,k,j,i)/(n0*n_0))*
                                solid_angles_.d_view(n));
          }
        }
      }

      if (needs_fluid_only || needs_both) {
        // stash coordinate frame moments
        Real moments_coord[4][4];
        moments_coord[0][0] = dv(m,0,k,j,i);
        moments_coord[0][1] = dv(m,1,k,j,i);
        moments_coord[0][2] = dv(m,2,k,j,i);
        moments_coord[0][3] = dv(m,3,k,j,i);
        moments_coord[1][1] = dv(m,4,k,j,i);
        moments_coord[1][2] = dv(m,5,k,j,i);
        moments_coord[1][3] = dv(m,6,k,j,i);
        moments_coord[2][2] = dv(m,7,k,j,i);
        moments_coord[2][3] = dv(m,8,k,j,i);
        moments_coord[3][3] = dv(m,9,k,j,i);
        moments_coord[1][0] = moments_coord[0][1];
        moments_coord[2][0] = moments_coord[0][2];
        moments_coord[3][0] = moments_coord[0][3];
        moments_coord[2][1] = moments_coord[1][2];
        moments_coord[3][1] = moments_coord[1][3];
        moments_coord[3][2] = moments_coord[2][3];

        // fluid velocity in tetrad frame
        Real uu1 = w0_(m,IVX,k,j,i);
        Real uu2 = w0_(m,IVY,k,j,i);
        Real uu3 = w0_(m,IVZ,k,j,i);
        Real q = glower[1][1]*uu1*uu1+2.0*glower[1][2]*uu1*uu2+2.0*glower[1][3]*uu1*uu3
               + glower[2][2]*uu2*uu2+2.0*glower[2][3]*uu2*uu3
               + glower[3][3]*uu3*uu3;
        Real uu0 = sqrt(1.0 + q);
        Real u_tet_[4];
        u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                     norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
        u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                     norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
        u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                     norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
        u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                     norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

        // Construct Lorentz boost from tetrad frame to orthonormal fluid frame
        Real tet_to_fluid[4][4];
        tet_to_fluid[0][0] =  u_tet_[0];
        tet_to_fluid[0][1] = -u_tet_[1];
        tet_to_fluid[0][2] = -u_tet_[2];
        tet_to_fluid[0][3] = -u_tet_[3];
        tet_to_fluid[1][1] = u_tet_[1]*u_tet_[1]/(1.0 + u_tet_[0]) + 1.0;
        tet_to_fluid[1][2] = u_tet_[1]*u_tet_[2]/(1.0 + u_tet_[0]);
        tet_to_fluid[1][3] = u_tet_[1]*u_tet_[3]/(1.0 + u_tet_[0]);
        tet_to_fluid[2][2] = u_tet_[2]*u_tet_[2]/(1.0 + u_tet_[0]) + 1.0;
        tet_to_fluid[2][3] = u_tet_[2]*u_tet_[3]/(1.0 + u_tet_[0]);
        tet_to_fluid[3][3] = u_tet_[3]*u_tet_[3]/(1.0 + u_tet_[0]) + 1.0;
        tet_to_fluid[1][0] = tet_to_fluid[0][1];
        tet_to_fluid[2][0] = tet_to_fluid[0][2];
        tet_to_fluid[3][0] = tet_to_fluid[0][3];
        tet_to_fluid[2][1] = tet_to_fluid[1][2];
        tet_to_fluid[3][1] = tet_to_fluid[1][3];
        tet_to_fluid[3][2] = tet_to_fluid[2][3];

        // set tetrad frame moments
        for (int n1=0, n12=0; n1<4; ++n1) {
          for (int n2=n1; n2<4; ++n2, ++n12) {
            dv(m,moments_offset+n12,k,j,i) = 0.0;
            for (int m1=0; m1<4; ++m1) {
              for (int m2=0; m2<4; ++m2) {
                dv(m,moments_offset+n12,k,j,i) += (tetcov_c_(m,n1,m1,k,j,i)*
                                                   tetcov_c_(m,n2,m2,k,j,i)*
                                                   moments_coord[m1][m2]);
              }
            }
          }
        }
        dv(m,moments_offset+1,k,j,i) *= -1.0;
        dv(m,moments_offset+2,k,j,i) *= -1.0;
        dv(m,moments_offset+3,k,j,i) *= -1.0;

        // stash tetrad frame moments
        Real moments_tetrad[4][4];
        moments_tetrad[0][0] = dv(m,moments_offset+0,k,j,i);
        moments_tetrad[0][1] = dv(m,moments_offset+1,k,j,i);
        moments_tetrad[0][2] = dv(m,moments_offset+2,k,j,i);
        moments_tetrad[0][3] = dv(m,moments_offset+3,k,j,i);
        moments_tetrad[1][1] = dv(m,moments_offset+4,k,j,i);
        moments_tetrad[1][2] = dv(m,moments_offset+5,k,j,i);
        moments_tetrad[1][3] = dv(m,moments_offset+6,k,j,i);
        moments_tetrad[2][2] = dv(m,moments_offset+7,k,j,i);
        moments_tetrad[2][3] = dv(m,moments_offset+8,k,j,i);
        moments_tetrad[3][3] = dv(m,moments_offset+9,k,j,i);
        moments_tetrad[1][0] = moments_tetrad[0][1];
        moments_tetrad[2][0] = moments_tetrad[0][2];
        moments_tetrad[3][0] = moments_tetrad[0][3];
        moments_tetrad[2][1] = moments_tetrad[1][2];
        moments_tetrad[3][1] = moments_tetrad[1][3];
        moments_tetrad[3][2] = moments_tetrad[2][3];

        // set R^{\mu \nu} (fluid frame)
        for (int n1=0, n12=0; n1<4; ++n1) {
          for (int n2=n1; n2<4; ++n2, ++n12) {
            dv(m,moments_offset+n12,k,j,i) = 0.0;
            for (int m1=0; m1<4; ++m1) {
              for (int m2=0; m2<4; ++m2) {
                dv(m,moments_offset+n12,k,j,i) += (tet_to_fluid[n1][m1]*
                                                   tet_to_fluid[n2][m2]*
                                                   moments_tetrad[m1][m2]);
              }
            }
          }
        }
      }
    });
  }

  // Particle density binned to mesh.
  if (name.compare("prtcl_d") == 0) {
    Kokkos::realloc(derived_var, nmb, 1, n3, n2, n1);
    auto pdens = derived_var;
    auto pr = pm->pmb_pack->ppart->prtcl_rdata;
    auto pi = pm->pmb_pack->ppart->prtcl_idata;
    int &npart = pm->nprtcl_thisrank;
    int gids = pm->pmb_pack->gids;

    par_for("pdens0", DevExeSpace(), 0, (nmb-1), ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      pdens(m,0,k,j,i) = 0.0;
    });

    par_for("pdens", DevExeSpace(), 0, (npart-1),
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID,p) - gids;
      int ip = (pr(IPX,p) - size.d_view(m).x1min)/size.d_view(m).dx1 + is;
      int jp = (pr(IPY,p) - size.d_view(m).x2min)/size.d_view(m).dx2 + js;
      int kp = ks;
      if (three_d) {
        kp = (pr(IPZ,p) - size.d_view(m).x3min)/size.d_view(m).dx3 + ks;
      }
      pdens(m,0,kp,jp,ip) += 1.0;
    });
  }
  i_dv = i_dv % n_dv; // reset derived variable index
}
