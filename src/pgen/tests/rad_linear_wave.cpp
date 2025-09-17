//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_linear_wave.cpp
//  \brief GR radiation linear wave test

// C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "srcterms/srcterms.hpp"
#include "pgen/pgen.hpp"

// function to compute errors in solution at end of run
void RadiationLinearWaveErrors(ParameterInput *pin, Mesh *pm);

namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;

//----------------------------------------------------------------------------------------
//! \struct RadLinWaveVariables
//  \brief container for variables shared with vector potential and error functions

struct RadLinWaveVariables {
  Real k_par;
  Real cos_a2, cos_a3, sin_a2, sin_a3;
};

struct RadEigensystem {
  Real omega_real, omega_imag;
  Real rho, pgas, ux, uy, uz, erad, fxrad, fyrad, fzrad;
  Real delta;
  Real drho_real, drho_imag, dpgas_real, dpgas_imag;
  Real dux_real, dux_imag, duy_real, duy_imag, duz_real, duz_imag;
  Real derad_real, derad_imag;
  Real dfxrad_real, dfxrad_imag, dfyrad_real, dfyrad_imag, dfzrad_real, dfzrad_imag;
};

  RadLinWaveVariables rlw;
  RadEigensystem eig;

} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationLinearWave(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation linear wave test

void ProblemGenerator::RadiationLinearWave(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  // set linear wave errors function
  pgen_final_func = RadiationLinearWaveErrors;
  if (restart) return;

  // read global parameters
  bool along_x1 = pin->GetOrAddBoolean("problem", "along_x1", false);
  bool along_x2 = pin->GetOrAddBoolean("problem", "along_x2", false);
  bool along_x3 = pin->GetOrAddBoolean("problem", "along_x3", false);
  // error check input flags
  if ((along_x1 && (along_x2 || along_x3)) || (along_x2 && along_x3)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Can only specify one of along_x1/2/3 to be true" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((along_x2 || along_x3) && pmy_mesh_->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x2 or x3 axis in 1D" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (along_x3 && pmy_mesh_->two_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x3 axis in 2D" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // start with wavevector along x1 axis
  rlw.cos_a3 = 1.0;
  rlw.sin_a3 = 0.0;
  rlw.cos_a2 = 1.0;
  rlw.sin_a2 = 0.0;
  if (pmy_mesh_->multi_d && !(along_x1)) {
    Real ang_3 = std::atan(x1size/x2size);
    rlw.sin_a3 = std::sin(ang_3);
    rlw.cos_a3 = std::cos(ang_3);
  }
  if (pmy_mesh_->three_d && !(along_x1)) {
    Real ang_2 = std::atan(0.5*(x1size*rlw.cos_a3 + x2size*rlw.sin_a3)/x3size);
    rlw.sin_a2 = std::sin(ang_2);
    rlw.cos_a2 = std::cos(ang_2);
  }

  // hardcode wavevector along x2 axis, override ang_2, ang_3
  if (along_x2) {
    rlw.cos_a3 = 0.0;
    rlw.sin_a3 = 1.0;
    rlw.cos_a2 = 1.0;
    rlw.sin_a2 = 0.0;
  }

  // hardcode wavevector along x3 axis, override ang_2, ang_3
  if (along_x3) {
    rlw.cos_a3 = 0.0;
    rlw.sin_a3 = 1.0;
    rlw.cos_a2 = 0.0;
    rlw.sin_a2 = 1.0;
  }

  // choose the smallest projection of the wavelength in each direction that is > 0
  Real lambda = std::numeric_limits<float>::max();
  if (rlw.cos_a2*rlw.cos_a3 > 0.0) {
    lambda = std::min(lambda, x1size*rlw.cos_a2*rlw.cos_a3);
  }
  if (rlw.cos_a2*rlw.sin_a3 > 0.0) {
    lambda = std::min(lambda, x2size*rlw.cos_a2*rlw.sin_a3);
  }
  if (rlw.sin_a2 > 0.0) lambda = std::min(lambda, x3size*rlw.sin_a2);

  // Initialize k_parallel
  rlw.k_par = 2.0*(M_PI)/lambda;

  // Set eigensystem
  eig.omega_real  = pin->GetReal     ("problem", "omega_real");
  eig.omega_imag  = pin->GetReal     ("problem", "omega_imag");
  eig.rho         = pin->GetReal     ("problem", "rho");
  eig.pgas        = pin->GetReal     ("problem", "pgas");
  eig.ux          = pin->GetOrAddReal("problem", "ux", 0.0);
  eig.uy          = pin->GetOrAddReal("problem", "uy", 0.0);
  eig.uz          = pin->GetOrAddReal("problem", "uz", 0.0);
  eig.erad        = pin->GetReal     ("problem", "erad");
  eig.fxrad       = pin->GetOrAddReal("problem", "fxrad", 0.0);
  eig.fyrad       = pin->GetOrAddReal("problem", "fyrad", 0.0);
  eig.fzrad       = pin->GetOrAddReal("problem", "fzrad", 0.0);
  eig.delta       = pin->GetReal     ("problem", "delta");
  eig.drho_real   = pin->GetReal     ("problem", "drho_real");
  eig.drho_imag   = pin->GetReal     ("problem", "drho_imag");
  eig.dpgas_real  = pin->GetReal     ("problem", "dpgas_real");
  eig.dpgas_imag  = pin->GetReal     ("problem", "dpgas_imag");
  eig.dux_real    = pin->GetReal     ("problem", "dux_real");
  eig.dux_imag    = pin->GetReal     ("problem", "dux_imag");
  eig.duy_real    = pin->GetOrAddReal("problem", "duy_real", 0.0);
  eig.duy_imag    = pin->GetOrAddReal("problem", "duy_imag", 0.0);
  eig.duz_real    = pin->GetOrAddReal("problem", "duz_real", 0.0);
  eig.duz_imag    = pin->GetOrAddReal("problem", "duz_imag", 0.0);
  eig.derad_real  = pin->GetReal     ("problem", "derad_real");
  eig.derad_imag  = pin->GetReal     ("problem", "derad_imag");
  eig.dfxrad_real = pin->GetReal     ("problem", "dfxrad_real");
  eig.dfxrad_imag = pin->GetReal     ("problem", "dfxrad_imag");
  eig.dfyrad_real = pin->GetOrAddReal("problem", "dfyrad_real", 0.0);
  eig.dfyrad_imag = pin->GetOrAddReal("problem", "dfyrad_imag", 0.0);
  eig.dfzrad_real = pin->GetOrAddReal("problem", "dfzrad_real", 0.0);
  eig.dfzrad_imag = pin->GetOrAddReal("problem", "dfzrad_imag", 0.0);

  // set time limit (input tlim interpreted as damping times)
  if (set_initial_conditions) {
    Real tlim = pin->GetReal("time", "tlim");
    pin->SetReal("time", "tlim", tlim*log(2.0)/fabs(eig.omega_imag));
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;
  int nangles_ = pmbp->prad->prgeo->nangles;
  auto eig_ = eig;
  auto wv_ = rlw;

  // time for reference solution
  Real tf = pmbp->pmesh->time;

  // set EOS data
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;

  // set primitive variables.  Note solution explicitly depends on time since wave
  // is damped.  Initial conditions will ise amplitudes at t=0.  Final time will
  // compute damped solution at appropriate spatial position.
  auto &w0 = pmbp->phydro->w0;
  par_for("rad_wave",DevExeSpace(),0,(pmbp->nmb_thispack-1),0,(n3-1),0,(n2-1),0,(n1-1),
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

    // compute analytic solution
    Real x = wv_.cos_a2*(x1v*wv_.cos_a3 + x2v*wv_.sin_a3) + x3v*wv_.sin_a2;
    Real sn = sin(eig_.omega_real*tf - wv_.k_par*x);
    Real cn = cos(eig_.omega_real*tf - wv_.k_par*x);
    Real en = exp(eig_.omega_imag*tf);
    Real rhon  = eig_.rho  + eig_.delta*en*(eig_.drho_real *cn + eig_.drho_imag *sn);
    Real pgasn = eig_.pgas + eig_.delta*en*(eig_.dpgas_real*cn + eig_.dpgas_imag*sn);
    Real uxn   = eig_.ux   + eig_.delta*en*(eig_.dux_real  *cn + eig_.dux_imag  *sn);
    Real uyn   = eig_.uy   + eig_.delta*en*(eig_.duy_real  *cn + eig_.duy_imag  *sn);
    Real uzn   = eig_.uz   + eig_.delta*en*(eig_.duz_real  *cn + eig_.duz_imag  *sn);

    w0(m,IDN,k,j,i) = rhon;
    w0(m,IVX,k,j,i) = uxn*wv_.cos_a2*wv_.cos_a3-uyn*wv_.sin_a3-uzn*wv_.sin_a2*wv_.cos_a3;
    w0(m,IVY,k,j,i) = uxn*wv_.cos_a2*wv_.sin_a3+uyn*wv_.cos_a3-uzn*wv_.sin_a2*wv_.sin_a3;
    w0(m,IVZ,k,j,i) = uxn*wv_.sin_a2                          +uzn*wv_.cos_a2;
    w0(m,IEN,k,j,i) = pgasn/gm1;
  });

  // Convert primitives to conserved
  if (set_initial_conditions) {
    pmbp->phydro->peos->PrimToCons(w0, pmbp->phydro->u0, 0,(n1-1), 0,(n2-1), 0,(n3-1));
  } else {
    pmbp->phydro->peos->PrimToCons(w0, pmbp->phydro->u1, 0,(n1-1), 0,(n2-1), 0,(n3-1));
  }

  // initialize specific intensity over angles in initial conditions
  if (set_initial_conditions) {
    auto &nh_c_ = pmbp->prad->nh_c;
    auto &norm_to_tet_ = pmbp->prad->norm_to_tet;
    auto &tet_c_ = pmbp->prad->tet_c;
    auto &tetcov_c_ = pmbp->prad->tetcov_c;

    auto &i0 = pmbp->prad->i0;
    par_for("rad_wave2",DevExeSpace(),0,(pmbp->nmb_thispack-1),0,(n3-1),0,(n2-1),0,(n1-1),
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

      Real x = wv_.cos_a2*(x1v*wv_.cos_a3 + x2v*wv_.sin_a3) + x3v*wv_.sin_a2;
      Real sn = sin(wv_.k_par*x);
      Real cn = cos(wv_.k_par*x);

      // Calculate wave-aligned coordinate-frame fluid velocity
      Real u_wave[4];
      u_wave[1] = eig_.ux + eig_.delta*(eig_.dux_real*cn - eig_.dux_imag*sn);
      u_wave[2] = eig_.uy + eig_.delta*(eig_.duy_real*cn - eig_.duy_imag*sn);
      u_wave[3] = eig_.uz + eig_.delta*(eig_.duz_real*cn - eig_.duz_imag*sn);
      u_wave[0] = hypot(1.0, hypot(u_wave[1], hypot(u_wave[2], u_wave[3])));

      // Calculate coordinate-frame fluid velocity
      Real u[4];
      u[0] = u_wave[0];
      u[1] = (u_wave[1]*wv_.cos_a2*wv_.cos_a3 -u_wave[2]*wv_.sin_a3 -
              u_wave[3]*wv_.sin_a2*wv_.cos_a3);
      u[2] = (u_wave[1]*wv_.cos_a2*wv_.sin_a3 +u_wave[2]*wv_.cos_a3 -
              u_wave[3]*wv_.sin_a2*wv_.sin_a3);
      u[3] = u_wave[1]*wv_.sin_a2 + u_wave[3]*wv_.cos_a2;

      // Calculate wave-aligned fluid-frame radiation moments
      Real rf_wave[4][4];
      rf_wave[0][0] =  eig_.erad + eig_.delta*(eig_.derad_real *cn - eig_.derad_imag *sn);
      rf_wave[0][1] = eig_.fxrad + eig_.delta*(eig_.dfxrad_real*cn - eig_.dfxrad_imag*sn);
      rf_wave[0][2] = eig_.fyrad + eig_.delta*(eig_.dfyrad_real*cn - eig_.dfyrad_imag*sn);
      rf_wave[0][3] = eig_.fzrad + eig_.delta*(eig_.dfzrad_real*cn - eig_.dfzrad_imag*sn);
      rf_wave[1][1] = 1.0/3.0*rf_wave[0][0];
      rf_wave[2][2] = 1.0/3.0*rf_wave[0][0];
      rf_wave[3][3] = 1.0/3.0*rf_wave[0][0];
      rf_wave[1][2] = 0.0;
      rf_wave[1][3] = 0.0;
      rf_wave[2][3] = 0.0;
      rf_wave[1][0] = rf_wave[0][1];
      rf_wave[2][0] = rf_wave[0][2];
      rf_wave[3][0] = rf_wave[0][3];
      rf_wave[2][1] = rf_wave[1][2];
      rf_wave[3][1] = rf_wave[1][3];
      rf_wave[3][2] = rf_wave[2][3];

      // Calculate wave-aligned coordinate-frame radiation moments
      Real lambda_c_f_wave[4][4];
      lambda_c_f_wave[0][0] = u_wave[0];
      lambda_c_f_wave[0][1] = u_wave[1];
      lambda_c_f_wave[0][2] = u_wave[2];
      lambda_c_f_wave[0][3] = u_wave[3];
      lambda_c_f_wave[1][1] = 1.0 + 1.0/(1.0 + u_wave[0])*SQR(u_wave[1]);
      lambda_c_f_wave[2][2] = 1.0 + 1.0/(1.0 + u_wave[0])*SQR(u_wave[2]);
      lambda_c_f_wave[3][3] = 1.0 + 1.0/(1.0 + u_wave[0])*SQR(u_wave[3]);
      lambda_c_f_wave[1][2] = 1./(1.+u_wave[0])*u_wave[1]*u_wave[2];
      lambda_c_f_wave[1][3] = 1./(1.+u_wave[0])*u_wave[1]*u_wave[3];
      lambda_c_f_wave[2][3] = 1./(1.+u_wave[0])*u_wave[2]*u_wave[3];
      lambda_c_f_wave[1][0] = lambda_c_f_wave[0][1];
      lambda_c_f_wave[2][0] = lambda_c_f_wave[0][2];
      lambda_c_f_wave[3][0] = lambda_c_f_wave[0][3];
      lambda_c_f_wave[2][1] = lambda_c_f_wave[1][2];
      lambda_c_f_wave[3][1] = lambda_c_f_wave[1][3];
      lambda_c_f_wave[3][2] = lambda_c_f_wave[2][3];

      Real r_wave[4][4];
      for (int alpha=0; alpha<4; ++alpha) {
        for (int beta=0; beta<4; ++beta) {
          r_wave[alpha][beta] = 0.0;
          for (int mu=0; mu<4; ++mu) {
            for (int nu=0; nu<4; ++nu) {
              r_wave[alpha][beta] += lambda_c_f_wave[alpha][mu]
                 *lambda_c_f_wave[beta][nu]*rf_wave[mu][nu];
            }
          }
        }
      }

      // Calculate coordinate-frame radiation moments
      Real r[4][4];
      r[0][0] = r_wave[0][0];
      r[0][1] = (wv_.cos_a2*wv_.cos_a3*r_wave[0][1] - wv_.sin_a3*r_wave[0][2] -
                 wv_.cos_a3*wv_.sin_a2*r_wave[0][3]);
      r[0][2] = (wv_.cos_a2*wv_.sin_a3*r_wave[0][1] + wv_.cos_a3*r_wave[0][2] -
                 wv_.sin_a2*wv_.sin_a3*r_wave[0][3]);
      r[0][3] = wv_.cos_a2*r_wave[0][3] + wv_.sin_a2*r_wave[0][1];
      r[1][1] = (r_wave[1][1]*SQR(wv_.cos_a2)*SQR(wv_.cos_a3) +
                 r_wave[2][2]*SQR(wv_.sin_a3) +
                 r_wave[3][3]*SQR(wv_.cos_a3)*SQR(wv_.sin_a2) -
                 2.0*r_wave[1][3]*wv_.cos_a2*SQR(wv_.cos_a3)*wv_.sin_a2 -
                 2.0*r_wave[1][2]*wv_.cos_a2*wv_.cos_a3*wv_.sin_a3 +
                 2.0*r_wave[2][3]*wv_.cos_a3*wv_.sin_a2*wv_.sin_a3);
      r[2][2] = (r_wave[1][1]*SQR(wv_.cos_a2)*SQR(wv_.sin_a3) +
                 r_wave[2][2]*SQR(wv_.cos_a3) +
                 r_wave[3][3]*SQR(wv_.sin_a2)*SQR(wv_.sin_a3) +
                 2.0*r_wave[1][2]*wv_.cos_a2*wv_.cos_a3*wv_.sin_a3 -
                 2.0*r_wave[2][3]*wv_.cos_a3*wv_.sin_a2*wv_.sin_a3 -
                 2.0*r_wave[1][3]*wv_.cos_a2*wv_.sin_a2*SQR(wv_.sin_a3));
      r[3][3] = (r_wave[3][3]*SQR(wv_.cos_a2) + 2.0*r_wave[1][3]*wv_.cos_a2*wv_.sin_a2 +
                 r_wave[1][1]*SQR(wv_.sin_a2));
      r[1][2] = (r_wave[1][2]*wv_.cos_a2*SQR(wv_.cos_a3) -
                 r_wave[2][3]*SQR(wv_.cos_a3)*wv_.sin_a2 -
                 r_wave[2][2]*wv_.cos_a3*wv_.sin_a3 +
                 r_wave[1][1]*SQR(wv_.cos_a2)*wv_.cos_a3*wv_.sin_a3  +
                 r_wave[3][3]*wv_.cos_a3*SQR(wv_.sin_a2)*wv_.sin_a3 -
                 r_wave[1][2]*wv_.cos_a2*SQR(wv_.sin_a3) +
                 r_wave[2][3]*wv_.sin_a2*SQR(wv_.sin_a3) -
                 2.0*r_wave[1][3]*wv_.cos_a2*wv_.cos_a3*wv_.sin_a2*wv_.sin_a3);
      r[1][3] = (r_wave[1][3]*SQR(wv_.cos_a2)*wv_.cos_a3 +
                 r_wave[1][1]*wv_.cos_a2*wv_.cos_a3*wv_.sin_a2 -
                 r_wave[3][3]*wv_.cos_a2*wv_.cos_a3*wv_.sin_a2 -
                 r_wave[1][3]*wv_.cos_a3*SQR(wv_.sin_a2) -
                 r_wave[2][3]*wv_.cos_a2*wv_.sin_a3 -
                 r_wave[1][2]*wv_.sin_a2*wv_.sin_a3);
      r[2][3] = (r_wave[2][3]*wv_.cos_a2*wv_.cos_a3 +
                 r_wave[1][2]*wv_.cos_a3*wv_.sin_a2 +
                 r_wave[1][3]*SQR(wv_.cos_a2)*wv_.sin_a3 +
                 r_wave[1][1]*wv_.cos_a2*wv_.sin_a2*wv_.sin_a3 -
                 r_wave[3][3]*wv_.cos_a2*wv_.sin_a2*wv_.sin_a3 -
                 r_wave[1][3]*SQR(wv_.sin_a2)*wv_.sin_a3);
      r[1][0] = r[0][1];
      r[2][0] = r[0][2];
      r[3][0] = r[0][3];
      r[2][1] = r[1][2];
      r[3][1] = r[1][3];
      r[3][2] = r[2][3];

      // Calculate fluid-frame radiation moments
      Real lambda_f_c[4][4];
      lambda_f_c[0][0] =  u[0];
      lambda_f_c[0][1] = -u[1];
      lambda_f_c[0][2] = -u[2];
      lambda_f_c[0][3] = -u[3];
      lambda_f_c[1][1] = 1.0 + 1.0/(1.0 + u[0])*u[1]*u[1];
      lambda_f_c[2][2] = 1.0 + 1.0/(1.0 + u[0])*u[2]*u[2];
      lambda_f_c[3][3] = 1.0 + 1.0/(1.0 + u[0])*u[3]*u[3];
      lambda_f_c[1][2] = 1.0/(1.0 + u[0])*u[1]*u[2];
      lambda_f_c[1][3] = 1.0/(1.0 + u[0])*u[1]*u[3];
      lambda_f_c[2][3] = 1.0/(1.0 + u[0])*u[2]*u[3];
      lambda_f_c[1][0] = lambda_f_c[0][1];
      lambda_f_c[2][0] = lambda_f_c[0][2];
      lambda_f_c[3][0] = lambda_f_c[0][3];
      lambda_f_c[2][1] = lambda_f_c[1][2];
      lambda_f_c[3][1] = lambda_f_c[1][3];
      lambda_f_c[3][2] = lambda_f_c[2][3];

      Real rf[4][4];
      for (int alpha=0; alpha<4; ++alpha) {
        for (int beta=0; beta<4; ++beta) {
          rf[alpha][beta] = 0.0;
          for (int mu=0; mu<4; ++mu) {
            for (int nu=0; nu<4; ++nu) {
              rf[alpha][beta] += lambda_f_c[alpha][mu]*lambda_f_c[beta][nu]*r[mu][nu];
            }
          }
        }
      }

      // Calculate normalized flux in fluid frame
      Real ee_f  = rf[0][0];
      Real ff1_f = rf[0][1];
      Real ff2_f = rf[0][2];
      Real ff3_f = rf[0][3];
      Real ff_f = sqrt(SQR(ff1_f) + SQR(ff2_f) + SQR(ff3_f));
      Real f_f  = ff_f/ee_f;
      Real f1_f = ff1_f/ff_f;
      Real f2_f = ff2_f/ff_f;
      Real f3_f = ff3_f/ff_f;

      // Compute fluid velocity in tetrad frame
      Real uu1 = u[1];
      Real uu2 = u[2];
      Real uu3 = u[3];
      Real uu0 = sqrt(1.0 + SQR(uu1) + SQR(uu2) + SQR(uu3));

      Real u_tet_[4];
      u_tet_[0] = (norm_to_tet_(m,0,0,k,j,i)*uu0 + norm_to_tet_(m,0,1,k,j,i)*uu1 +
                   norm_to_tet_(m,0,2,k,j,i)*uu2 + norm_to_tet_(m,0,3,k,j,i)*uu3);
      u_tet_[1] = (norm_to_tet_(m,1,0,k,j,i)*uu0 + norm_to_tet_(m,1,1,k,j,i)*uu1 +
                   norm_to_tet_(m,1,2,k,j,i)*uu2 + norm_to_tet_(m,1,3,k,j,i)*uu3);
      u_tet_[2] = (norm_to_tet_(m,2,0,k,j,i)*uu0 + norm_to_tet_(m,2,1,k,j,i)*uu1 +
                   norm_to_tet_(m,2,2,k,j,i)*uu2 + norm_to_tet_(m,2,3,k,j,i)*uu3);
      u_tet_[3] = (norm_to_tet_(m,3,0,k,j,i)*uu0 + norm_to_tet_(m,3,1,k,j,i)*uu1 +
                   norm_to_tet_(m,3,2,k,j,i)*uu2 + norm_to_tet_(m,3,3,k,j,i)*uu3);

      // Go through each angle
      for (int n=0; n<nangles_; ++n) {
        // Calculate direction in fluid frame
        Real un_t =  (u_tet_[1]*nh_c_.d_view(n,1) + u_tet_[2]*nh_c_.d_view(n,2) +
                      u_tet_[3]*nh_c_.d_view(n,3));

        Real n0_f =  u_tet_[0]*nh_c_.d_view(n,0) - un_t;
        Real n1_f = (-u_tet_[1]*nh_c_.d_view(n,0) + u_tet_[1]/(u_tet_[0] + 1.0)*un_t +
                     nh_c_.d_view(n,1));
        Real n2_f = (-u_tet_[2]*nh_c_.d_view(n,0) + u_tet_[2]/(u_tet_[0] + 1.0)*un_t +
                     nh_c_.d_view(n,2));
        Real n3_f = (-u_tet_[3]*nh_c_.d_view(n,0) + u_tet_[3]/(u_tet_[0] + 1.0)*un_t +
                     nh_c_.d_view(n,3));

        // Calculate intensity in fluid frame
        Real fn_f = f1_f*n1_f + f2_f*n2_f + f3_f*n3_f;
        Real ii_f = 0.0;
        if (f_f <= 1.0/3.0) {
          ii_f = ee_f/(4.0*M_PI)*(1.0 + 3.0*f_f*fn_f);
        } else {
          ii_f = ee_f/(9.0*M_PI)*(fn_f - 3.0*f_f + 2.0)/SQR(1.0 - f_f);
        }

        // Calculate intensity in tetrad frame
        Real n0 = tet_c_(m,0,0,k,j,i); Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {  n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);  }
        i0(m,n,k,j,i) = n0*n_0*ii_f/SQR(SQR(n0_f));
      }
    });
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void RadiationLinearWaveErrors()
//! \brief Computes errors in linear wave solution by calling initialization function
//! again to compute initial condictions, and subtracting current solution from ICs, and
//! outputs errors to file. Problem must be run for an integer number of wave periods.

void RadiationLinearWaveErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1 when flag is false.
  set_initial_conditions = false;
  pm->pgen->RadiationLinearWave(pin, false);
  pm->pgen->OutputErrors(pin, pm);
  return;
}
