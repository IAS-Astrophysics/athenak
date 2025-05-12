#ifndef RADIATION_M1_PARAMS_HPP
#define RADIATION_M1_PARAMS_HPP

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "bns_nurates/include/bns_nurates.hpp"
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_params.hpp
//  \brief enums/structs for various params of Grey M1

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \enum RadiationM1Closure
//  \brief choice of M1 closure
enum RadiationM1Closure {
  Minerbo,
  Eddington,
  Thin,
  Kershaw,
};

//----------------------------------------------------------------------------------------
//! \enum RadiationM1OpacityType
//  \brief choice of neutrino opacity library
enum RadiationM1OpacityType {
  None,
  Toy,
  BnsNurates,
};

//----------------------------------------------------------------------------------------
//! \enum RadiationM1SrcUpdate
//  \brief method to treat radiation sources
enum RadiationM1SrcUpdate {
  Explicit,
  Implicit,
};

//----------------------------------------------------------------------------------------
//! \struct RadiationM1Params
//  \brief parameters for the Grey M1 class
struct RadiationM1Params {
  RadiationM1Closure closure_type;      // choice of closure
  RadiationM1OpacityType opacity_type;  // choice of opacity library
  RadiationM1SrcUpdate src_update;      // choice of source update

  bool gr_sources;      // include GR sources
  bool matter_sources;  // include matter sources
  bool theta_limiter;   // activate theta limiter
  bool beam_sources;    // include beam sources

  int nspecies;              // number of neutrino species
  Real closure_epsilon;      // precision with which to find closure
  int closure_maxiter;       // maximum number of iterations in closure root finder
  Real inv_closure_epsilon;  // precision with which to find inverse closure
  int inv_closure_maxiter;  // maximum number of iterations in inverse closure root finder

  Real minmod_theta;    // value of theta for minmod limiter
  Real rad_E_floor;     // radiation energy density floor
  Real rad_N_floor;     // radiation number density floor
  Real rad_eps;         // Impose F_a F^a < (1 - rad_E_eps) E2
  Real source_Ye_max;   // maximum allowed Ye for matter
  Real source_Ye_min;   // minimum allowed Ye for matter
  Real source_limiter;  // limiter for matter source (0: sources disabled, 1: sources
                        // limited to avoid negative energies)
  Real source_epsabs;   // target absolute precision for non-linear solver
  Real source_epsrel;   // target relative precision for non-linear solver
  int source_maxiter;   // maximum number of iterations for non-linear solver

  Real source_thick_limit;  // Use the optically thick limit if the equilibration time is
                            // less than the timestep over this factor
  Real source_therm_limit;  // Assume neutrinos to be thermalized above this optical depth
  Real source_scat_limit;   // Use the scattering limit if the isotropization time is less
                            // than the timestep over this factor
};

enum SrcSignal {
  SrcThin,
  SrcEquil,
  SrcScat,
  SrcOk,
  SrcEddington,
  SrcFail,
};

struct RadiationM1Beam {
  DvceArray1D<Real> beam_source_vals;
  Real beam_ymin;
  Real beam_ymax;
};

struct SrcParams {
  SrcParams(const Real _cdt, const Real _alp,
            const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &_g_dd,
            const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &_g_uu,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_n_d,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_n_u,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &_gamma_ud,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_u_d,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_u_u,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_v_d,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &_v_u,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &_proj_ud,
            const Real _W, const Real Estar,
            const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d,
            const Real _chi, const Real _eta, const Real _kabs, const Real _kscat)
      : cdt(_cdt),
        alp(_alp),
        g_dd(_g_dd),
        g_uu(_g_uu),
        n_d(_n_d),
        n_u(_n_u),
        gamma_ud(_gamma_ud),
        u_d(_u_d),
        u_u(_u_u),
        v_d(_v_d),
        v_u(_v_u),
        proj_ud(_proj_ud),
        W(_W),
        Estar(Estar),
        Fstar_d(Fstar_d),
        chi(_chi),
        eta(_eta),
        kabs(_kabs),
        kscat(_kscat) {}
  const Real cdt;
  const Real alp;
  const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd;
  const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud;
  const Real W;
  const Real Estar;
  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d;
  Real chi;
  const Real eta;
  const Real kabs;
  const Real kscat;

  Real E{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_u{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
  Real J{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> S_d{};
  Real Edot{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> tS_d{};
};

struct NuratesParams {
  int nurates_quad_nx;     // no. of quadrature points for 1d integration (bns_nurates)
  int nurates_quad_ny;     // no. of quadrature points for 2d integration (bns_nurates)
  Real opacity_tau_trap;   // incl. effects of neutrino trapping above this optical depth
  Real opacity_tau_delta;  // range of optical depths over which trapping is introduced
  Real opacity_corr_fac_max;  // maximum correction factor for optically thin regime
  Real rho_min_cgs;
  Real temp_min_mev;

  bool use_abs_em;
  bool use_pair;
  bool use_brem;
  bool use_iso;
  bool use_inelastic_scatt;
  bool use_WM_ab;
  bool use_WM_sc;
  bool use_dU;
  bool use_dm_eff;
  bool use_equilibrium_distribution;
  bool use_kirchhoff_law;

  MyQuadrature my_quadrature_1d;
  MyQuadrature my_quadrature_2d;
};

}  // namespace radiationm1
#endif  // RADIATION_M1_PARAMS_HPP
