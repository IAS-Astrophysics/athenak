//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file blast.cpp
//! \brief Problem generator for spherical blast wave problem.
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

#include <algorithm>
#include <cmath>
#include <sstream>
#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"

#include "TwoPunctures.h"

static ini_data *data;

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  TwoPunctures_params_set_default();
  std::string set_name = "problem";
  TwoPunctures_params_set_default();
  TwoPunctures_params_set_Boolean(const_cast<char *> ("verbose"),
                              pin->GetOrAddBoolean(set_name, "verbose", 0));
  TwoPunctures_params_set_Real(const_cast<char *>("par_b"),
                               pin->GetOrAddReal(set_name, "par_b", 1.));
  TwoPunctures_params_set_Real(const_cast<char *>("par_m_plus"),
                               pin->GetOrAddReal(set_name, "par_m_plus", 1.));
  TwoPunctures_params_set_Real(const_cast<char *>("par_m_minus"),
                               pin->GetOrAddReal(set_name, "par_m_minus", 1.));

  TwoPunctures_params_set_Real(const_cast<char *>("target_M_plus"),
                               pin->GetOrAddReal(set_name, "target_M_plus", 1.));

  TwoPunctures_params_set_Real(const_cast<char *>("target_M_minus"),
                               pin->GetOrAddReal(set_name, "target_M_minus", 1.));

  TwoPunctures_params_set_Real(const_cast<char *>("par_P_plus1"),
                               pin->GetOrAddReal(set_name, "par_P_plus1", 0.));
  TwoPunctures_params_set_Real(const_cast<char *>("par_P_plus2"),
                               pin->GetOrAddReal(set_name, "par_P_plus2", 0.5));
  TwoPunctures_params_set_Real(const_cast<char *>("par_P_plus3"),
                               pin->GetOrAddReal(set_name, "par_P_plus3", 0.));


  TwoPunctures_params_set_Real(const_cast<char *>("par_P_minus1"),
                               pin->GetOrAddReal(set_name, "par_P_minus1", 0.));
  TwoPunctures_params_set_Real(const_cast<char *>("par_P_minus2"),
                               pin->GetOrAddReal(set_name, "par_P_minus2", 0.5));
  TwoPunctures_params_set_Real(const_cast<char *>("par_P_minus3"),
                               pin->GetOrAddReal(set_name, "par_P_minus3", 0.));


  TwoPunctures_params_set_Real(const_cast<char *>("par_S_plus1"),
                               pin->GetOrAddReal(set_name, "par_S_plus1", 0.));
  TwoPunctures_params_set_Real(const_cast<char *>("par_S_plus2"),
                               pin->GetOrAddReal(set_name, "par_S_plus2", 0.));
  TwoPunctures_params_set_Real(const_cast<char *>("par_S_plus3"),
                               pin->GetOrAddReal(set_name, "par_S_plus3", 0.));

  TwoPunctures_params_set_Real(const_cast<char *>("par_S_minus1"),
                               pin->GetOrAddReal(set_name, "par_S_minus1", 0.));
  TwoPunctures_params_set_Real(const_cast<char *>("par_S_minus2"),
                               pin->GetOrAddReal(set_name, "par_S_minus2", 0.));
  TwoPunctures_params_set_Real(const_cast<char *>("par_S_minus3"),
                               pin->GetOrAddReal(set_name, "par_S_minus3", 0.));
  TwoPunctures_params_set_Real(const_cast<char *>("center_offset1"),
                               pin->GetOrAddReal(set_name, "center_offset1", 0.));

  TwoPunctures_params_set_Real(const_cast<char *>("center_offset2"),
                               pin->GetOrAddReal(set_name, "center_offset2", 0.));
  TwoPunctures_params_set_Real(const_cast<char *>("center_offset3"),
                               pin->GetOrAddReal(set_name, "center_offset3", 0.));

  TwoPunctures_params_set_Boolean(const_cast<char *>("give_bare_mass"),
                               pin->GetOrAddBoolean(set_name, "give_bare_mass", 1));

  TwoPunctures_params_set_Int(const_cast<char *>("npoints_A"),
                               pin->GetOrAddInteger(set_name, "npoints_A", 30));
  TwoPunctures_params_set_Int(const_cast<char *>("npoints_B"),
                               pin->GetOrAddInteger(set_name, "npoints_B", 30));
  TwoPunctures_params_set_Int(const_cast<char *>("npoints_phi"),
                               pin->GetOrAddInteger(set_name, "npoints_phi", 16));


  TwoPunctures_params_set_Real(const_cast<char *>("Newton_tol"),
                               pin->GetOrAddReal(set_name, "Newton_tol", 1.e-10));

  TwoPunctures_params_set_Int(const_cast<char *>("Newton_maxit"),
                               pin->GetOrAddInteger(set_name, "Newton_maxit", 5));


  TwoPunctures_params_set_Real(const_cast<char *>("TP_epsilon"),
                               pin->GetOrAddReal(set_name, "TP_epsilon", 0.));

  TwoPunctures_params_set_Real(const_cast<char *>("TP_Tiny"),
                               pin->GetOrAddReal(set_name, "TP_Tiny", 0.));
  TwoPunctures_params_set_Real(const_cast<char *>("TP_Extend_Radius"),
                               pin->GetOrAddReal(set_name, "TP_Extend_Radius", 0.));


  TwoPunctures_params_set_Real(const_cast<char *>("adm_tol"),
                               pin->GetOrAddReal(set_name, "adm_tol", 1.e-10));


  TwoPunctures_params_set_Boolean(const_cast<char *>("do_residuum_debug_output"),
                      pin->GetOrAddBoolean(set_name, "do_residuum_debug_output", 0));

  TwoPunctures_params_set_Boolean(const_cast<char *>("solve_momentum_constraint"),
                      pin->GetOrAddBoolean(set_name, "solve_momentum_constraint", 0));

  TwoPunctures_params_set_Real(const_cast<char *>("initial_lapse_psi_exponent"),
                      pin->GetOrAddReal(set_name, "initial_lapse_psi_exponent", -2.0));

  TwoPunctures_params_set_Boolean(const_cast<char *>("swap_xz"),
                               pin->GetOrAddBoolean(set_name, "swap_xz", 0));
  data = TwoPunctures_make_initial_data();
  pmbp->pz4c->ADMTwoPunctures(pmbp, data);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  TwoPunctures_finalise(data);
  std::cout<<"TwoPuncture initialized."<<std::endl;
  return;
}
