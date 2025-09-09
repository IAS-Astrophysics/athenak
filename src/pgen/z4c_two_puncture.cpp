//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_two_puncture.cpp
//  \brief Problem generator for two punctures

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "TwoPunctures.h"
#include "coordinates/adm.hpp"
#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"

static ini_data *data;

void ADMTwoPunctures(MeshBlockPack *pmbp, ini_data *data);
void RefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for two punctures
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func  = RefinementCondition;

  if (restart)
    return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs         = pmy_mesh_->mb_indcs;

  TwoPunctures_params_set_default();
  std::string set_name = "problem";
  TwoPunctures_params_set_default();
  TwoPunctures_params_set_Boolean(
    const_cast<char *>("verbose"),
    pin->GetOrAddBoolean(set_name, "verbose", 0));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_b"),
    pin->GetOrAddReal(set_name, "par_b", 1.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_m_plus"),
    pin->GetOrAddReal(set_name, "par_m_plus", 1.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_m_minus"),
    pin->GetOrAddReal(set_name, "par_m_minus", 1.));

  TwoPunctures_params_set_Real(
    const_cast<char *>("target_M_plus"),
    pin->GetOrAddReal(set_name, "target_M_plus", 1.));

  TwoPunctures_params_set_Real(
    const_cast<char *>("target_M_minus"),
    pin->GetOrAddReal(set_name, "target_M_minus", 1.));

  TwoPunctures_params_set_Real(
    const_cast<char *>("par_P_plus1"),
    pin->GetOrAddReal(set_name, "par_P_plus1", 0.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_P_plus2"),
    pin->GetOrAddReal(set_name, "par_P_plus2", 0.5));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_P_plus3"),
    pin->GetOrAddReal(set_name, "par_P_plus3", 0.));

  TwoPunctures_params_set_Real(
    const_cast<char *>("par_P_minus1"),
    pin->GetOrAddReal(set_name, "par_P_minus1", 0.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_P_minus2"),
    pin->GetOrAddReal(set_name, "par_P_minus2", 0.5));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_P_minus3"),
    pin->GetOrAddReal(set_name, "par_P_minus3", 0.));

  TwoPunctures_params_set_Real(
    const_cast<char *>("par_S_plus1"),
    pin->GetOrAddReal(set_name, "par_S_plus1", 0.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_S_plus2"),
    pin->GetOrAddReal(set_name, "par_S_plus2", 0.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_S_plus3"),
    pin->GetOrAddReal(set_name, "par_S_plus3", 0.));

  TwoPunctures_params_set_Real(
    const_cast<char *>("par_S_minus1"),
    pin->GetOrAddReal(set_name, "par_S_minus1", 0.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_S_minus2"),
    pin->GetOrAddReal(set_name, "par_S_minus2", 0.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("par_S_minus3"),
    pin->GetOrAddReal(set_name, "par_S_minus3", 0.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("center_offset1"),
    pin->GetOrAddReal(set_name, "center_offset1", 0.));

  TwoPunctures_params_set_Real(
    const_cast<char *>("center_offset2"),
    pin->GetOrAddReal(set_name, "center_offset2", 0.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("center_offset3"),
    pin->GetOrAddReal(set_name, "center_offset3", 0.));

  TwoPunctures_params_set_Boolean(
    const_cast<char *>("give_bare_mass"),
    pin->GetOrAddBoolean(set_name, "give_bare_mass", 1));

  TwoPunctures_params_set_Int(
      const_cast<char *>("grid_setup_method"),
      pin->GetOrAddInteger(set_name, "grid_setup_method", 0));
  TwoPunctures_params_set_Int(
    const_cast<char *>("npoints_A"),
    pin->GetOrAddInteger(set_name, "npoints_A", 30));
  TwoPunctures_params_set_Int(
    const_cast<char *>("npoints_B"),
    pin->GetOrAddInteger(set_name, "npoints_B", 30));
  TwoPunctures_params_set_Int(
    const_cast<char *>("npoints_phi"),
    pin->GetOrAddInteger(set_name, "npoints_phi", 16));

  TwoPunctures_params_set_Real(
    const_cast<char *>("Newton_tol"),
    pin->GetOrAddReal(set_name, "Newton_tol", 1.e-10));

  TwoPunctures_params_set_Int(
    const_cast<char *>("Newton_maxit"),
    pin->GetOrAddInteger(set_name, "Newton_maxit", 5));

  TwoPunctures_params_set_Real(
    const_cast<char *>("TP_epsilon"),
    pin->GetOrAddReal(set_name, "TP_epsilon", 0.));

  TwoPunctures_params_set_Real(
    const_cast<char *>("TP_Tiny"),
    pin->GetOrAddReal(set_name, "TP_Tiny", 0.));
  TwoPunctures_params_set_Real(
    const_cast<char *>("TP_Extend_Radius"),
    pin->GetOrAddReal(set_name, "TP_Extend_Radius", 0.));

  TwoPunctures_params_set_Real(
    const_cast<char *>("adm_tol"),
    pin->GetOrAddReal(set_name, "adm_tol", 1.e-10));

  TwoPunctures_params_set_Boolean(
    const_cast<char *>("do_residuum_debug_output"),
    pin->GetOrAddBoolean(set_name, "do_residuum_debug_output", 0));

  TwoPunctures_params_set_Boolean(
    const_cast<char *>("solve_momentum_constraint"),
    pin->GetOrAddBoolean(set_name, "solve_momentum_constraint", 0));

  TwoPunctures_params_set_Real(
    const_cast<char *>("initial_lapse_psi_exponent"),
    pin->GetOrAddReal(set_name, "initial_lapse_psi_exponent", -2.0));

  TwoPunctures_params_set_Boolean(
    const_cast<char *>("swap_xz"),
    pin->GetOrAddBoolean(set_name, "swap_xz", 0));
  data = TwoPunctures_make_initial_data();
  ADMTwoPunctures(pmbp, data);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
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
  TwoPunctures_finalise(data);

  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  std::cout << "TwoPuncture initialized." << std::endl;
  return;
}

//! \fn void ADMTwoPunctures(MeshBlockPack *pmbp, ini_data *data)
//! \brief Interpolate two puncture initial data in cartesian grid
//
// p  = detgbar^(-1/3)
// p0 = psi^(-4)
//
// gtilde_ij = p gbar_ij
// Ktilde_ij = p p0 K_ij
//
// phi = - log(p) / 4
// K   = gtildeinv^ij Ktilde_ij
// Atilde_ij = Ktilde_ij - gtilde_ij K / 3
//
// G^i = - del_j gtildeinv^ji
//
void ADMTwoPunctures(MeshBlockPack *pmbp, ini_data *data) {
  // capture variables for the kernel
  auto &u_adm = pmbp->padm->u_adm;

  HostArray5D<Real>::HostMirror host_u_adm = create_mirror(u_adm);
  z4c::Z4c::ADMhost_vars host_adm;
  host_adm.psi4.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_PSI4);
  host_adm.g_dd.InitWithShallowSlice(
    host_u_adm, adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(
    host_u_adm, adm::ADM::I_ADM_KXX, adm::ADM::I_ADM_KZZ);
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size  = pmbp->pmb->mb_size;
  int &is     = indcs.is;
  int &ie     = indcs.ie;
  int &js     = indcs.js;
  int &je     = indcs.je;
  int &ks     = indcs.ks;
  int &ke     = indcs.ke;
  // For GLOOPS
  int isg = is - indcs.ng;
  int ieg = ie + indcs.ng;
  int jsg = js - indcs.ng;
  int jeg = je + indcs.ng;
  int ksg = ks - indcs.ng;
  int keg = ke + indcs.ng;

  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = indcs.nx2 + 2 * (indcs.ng);
  int ncells3 = indcs.nx3 + 2 * (indcs.ng);
  int nmb     = pmbp->nmb_thispack;
  for (int m = 0; m < nmb; ++m) {
    int imin[3] = {0, 0, 0};

    int n[3] = {ncells1, ncells2, ncells3};

    int sz = n[0] * n[1] * n[2];
    // this could be done instead by accessing and casting the Athena vars but
    // then it is coupled to implementation details etc.
    Real *gxx = new Real[sz], *gyy = new Real[sz], *gzz = new Real[sz];
    Real *gxy = new Real[sz], *gxz = new Real[sz], *gyz = new Real[sz];

    Real *Kxx = new Real[sz], *Kyy = new Real[sz], *Kzz = new Real[sz];
    Real *Kxy = new Real[sz], *Kxz = new Real[sz], *Kyz = new Real[sz];

    Real *psi = new Real[sz];
    Real *alp = new Real[sz];

    Real *x = new Real[n[0]];
    Real *y = new Real[n[1]];
    Real *z = new Real[n[2]];

    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    int nx1     = indcs.nx1;

    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    int nx2     = indcs.nx2;

    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;
    int nx3     = indcs.nx3;
    // need to populate coordinates
    for (int ix_I = isg; ix_I < ieg + 1; ix_I++) {
      x[ix_I] = CellCenterX(ix_I - is, nx1, x1min, x1max);
    }

    for (int ix_J = jsg; ix_J < jeg + 1; ix_J++) {
      y[ix_J] = CellCenterX(ix_J - js, nx2, x2min, x2max);
    }

    for (int ix_K = ksg; ix_K < keg + 1; ix_K++) {
      z[ix_K] = CellCenterX(ix_K - ks, nx3, x3min, x3max);
    }
    TwoPunctures_Cartesian_interpolation(
      data, // struct containing the previously calculated solution
      imin, // min, max idxs of Cartesian Grid in the three directions
      n,    // <-imax, but this collapses
      n,    // total number of indices in each direction
      x,    // x,         // Cartesian coordinates
      y,    // y,
      z,    // z,
      alp,  // alp,       // lapse
      psi,  // psi,       // conformal factor and derivatives
      NULL, // psix,
      NULL, // psiy,
      NULL, // psiz,
      NULL, // psixx,
      NULL, // psixy,
      NULL, // psixz,
      NULL, // psiyy,
      NULL, // psiyz,
      NULL, // psizz,
      gxx,  // gxx,       // metric components
      gxy,  // gxy,
      gxz,  // gxz,
      gyy,  // gyy,
      gyz,  // gyz,
      gzz,  // gzz,
      Kxx,  // kxx,       // extrinsic curvature components
      Kxy,  // kxy,
      Kxz,  // kxz,
      Kyy,  // kyy,
      Kyz,  // kyz,
      Kzz   // kzz
    );

    for (int k = ksg; k <= keg; k++)
      for (int j = jsg; j <= jeg; j++)
        for (int i = isg; i <= ieg; i++) {
          int flat_ix               = i + n[0] * (j + n[1] * k);
          host_adm.psi4(m, k, j, i) = std::pow(psi[flat_ix], 4);

          host_adm.g_dd(m, 0, 0, k, j, i) =
            host_adm.psi4(m, k, j, i) * gxx[flat_ix];
          host_adm.g_dd(m, 1, 1, k, j, i) =
            host_adm.psi4(m, k, j, i) * gyy[flat_ix];
          host_adm.g_dd(m, 2, 2, k, j, i) =
            host_adm.psi4(m, k, j, i) * gzz[flat_ix];
          host_adm.g_dd(m, 0, 1, k, j, i) =
            host_adm.psi4(m, k, j, i) * gxy[flat_ix];
          host_adm.g_dd(m, 0, 2, k, j, i) =
            host_adm.psi4(m, k, j, i) * gxz[flat_ix];
          host_adm.g_dd(m, 1, 2, k, j, i) =
            host_adm.psi4(m, k, j, i) * gyz[flat_ix];

          host_adm.vK_dd(m, 0, 0, k, j, i) = Kxx[flat_ix];
          host_adm.vK_dd(m, 1, 1, k, j, i) = Kyy[flat_ix];
          host_adm.vK_dd(m, 2, 2, k, j, i) = Kzz[flat_ix];
          host_adm.vK_dd(m, 0, 1, k, j, i) = Kxy[flat_ix];
          host_adm.vK_dd(m, 0, 2, k, j, i) = Kxz[flat_ix];
          host_adm.vK_dd(m, 1, 2, k, j, i) = Kyz[flat_ix];
        }

    delete[] gxx;
    delete[] gyy;
    delete[] gzz;
    delete[] gxy;
    delete[] gxz;
    delete[] gyz;

    delete[] Kxx;
    delete[] Kyy;
    delete[] Kzz;
    delete[] Kxy;
    delete[] Kxz;
    delete[] Kyz;

    delete[] psi;
    delete[] alp;
    delete[] x;
    delete[] y;
    delete[] z;
  }
  Kokkos::deep_copy(u_adm, host_u_adm);
  return;
}

// how decide the refinement
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
