//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sc_linwave.cpp
//! \brief Radiatively damped acoustic waves (Davis, Stone & Jiang 2012 Sec. 5.4 /
//! Eq. 38; Stein & Spiegel 1967). 1D port of Athena-C linear_wave_rad1d.c for Figs. 9-10,
//! with the 2D non-grid-aligned extension of linear_wave_rad2d.c.
//!
//! Sec. 5.4 protocol: rho0 = 1, v0 = 0, a = 1 (adiabatic sound speed), A = 1e-6,
//! periodic, t_final = lambda / a. The wave is characterised by the optical depth per
//! wavelength tau and the Boltzmann number Bo. The deck supplies the module's own
//! parameters and the pgen derives the two from them (rho0 = 1, T0 = 1/gamma):
//!   tau = kappa_a * lambda / (2 pi)          (chi0 = kappa_a rho0 = tau k)
//!   Bo  = 4 gamma^4 / ((gamma - 1) crat prat) (from B0 = gamma Etherm0 / (Bo pi),
//!                                              crat prat = 4 pi B0 / T0^4)
//! Initial intensity I = T0^4 (thermal equilibrium).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {
using Complex = std::complex<Real>;

struct WaveVars {
  Real amp, vflow, Bo, tau;
  Real sin_a, cos_a, lambda;
  Real vph, rdamp;           // analytic: vph = Re(ω)/(k a) with a=1; rdamp = 2π Im(ω)
  Real omega_r, omega_i;     // analytic ω/(k a)
  Real V0R, V0I, E0R, E0I;
  Real dens, pgas;
  Real gamma;
  Real chi0, T0;
  bool is_1d;
};
WaveVars wv;

//! Newton iteration for a root of the cubic c3 x^3 + c2 x^2 + c1 x + c0 from a starting
//! guess. Returns false if it fails to converge.
bool CubicRootNewton(const Complex c[4], Complex *x) {
  for (int it = 0; it < 100; ++it) {
    Complex p  = ((c[3]*(*x) + c[2])*(*x) + c[1])*(*x) + c[0];
    Complex dp = (3.0*c[3]*(*x) + 2.0*c[2])*(*x) + c[1];
    if (std::abs(dp) == 0.0) return false;
    Complex dx = p / dp;
    *x -= dx;
    if (std::abs(dx) <= 1.0e-15 * std::max(std::abs(*x), 1.0)) return true;
  }
  return false;
}

//! Davis 2012 Eq. 38 cubic for radiatively modified acoustic modes
void AcousticWaveRad(Real Bo, Real tau, Real cs, Real d0,
                     Real *vph, Real *rdamp,
                     Real *V0R, Real *V0I, Real *E0R, Real *E0I,
                     Real *omega_r, Real *omega_i) {
  Real mu = 1.0 - tau * std::atan(1.0/tau);
  Real theta = 16.0 * tau * mu / Bo;
  // omega^3 - i theta gamma omega^2 - omega + i theta = 0; the acoustic branch continues
  // from the undamped sound wave omega/(k a) = 1, so Newton starts there.
  Complex coeff[4];
  coeff[0] = Complex(0.0, theta);
  coeff[1] = -1.0;
  coeff[2] = Complex(0.0, -theta * wv.gamma);
  coeff[3] = 1.0;
  Complex omega(1.0, 0.0);
  if (!CubicRootNewton(coeff, &omega) || omega.real() <= 0.5) {
    std::cout << "### FATAL ERROR in sc_linwave: dispersion relation did not converge "
              << "to the acoustic branch (Bo=" << Bo << ", tau=" << tau << ")"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Complex V0 = cs * omega / d0;
  Complex E0 = cs*cs * omega*omega / (wv.gamma - 1.0);
  *V0R = V0.real(); *V0I = V0.imag();
  *E0R = E0.real(); *E0I = E0.imag();
  *omega_r = omega.real();
  *omega_i = omega.imag();
  *vph = omega.real();
  *rdamp = 2.0 * M_PI * omega.imag();
}

//----------------------------------------------------------------------------------------
//! Fourier fit of the fundamental density mode (Davis §5.4 Fig. 9)

void FitFourierOmega(Mesh *pm, Real *omega_r_fit, Real *omega_i_fit, Real *amp_meas) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = pmbp->phydro->u0;
  auto u_h = Kokkos::create_mirror_view(u0);
  Kokkos::deep_copy(u_h, u0);
  auto size_h = size.h_view;

  // Volume-weighted Fourier projection: on an SMR/AMR mesh cells differ in size, so a
  // per-cell count would over-weight refined regions and bias amplitude and phase.
  Real sum_s = 0.0, sum_c = 0.0, vol = 0.0;
  int nmb = pmbp->nmb_thispack;
  for (int m=0; m<nmb; ++m) {
    Real x1min = size_h(m).x1min, x1max = size_h(m).x1max;
    Real x2min = size_h(m).x2min, x2max = size_h(m).x2max;
    Real x3min = size_h(m).x3min, x3max = size_h(m).x3max;
    Real dv = ((x1max-x1min)/indcs.nx1) * ((x2max-x2min)/indcs.nx2)
              * ((x3max-x3min)/indcs.nx3);
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
        for (int i=is; i<=ie; ++i) {
          Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
          Real r = (x1v*wv.cos_a + x2v*wv.sin_a) / wv.lambda;
          Real dpert = u_h(m,IDN,k,j,i) - wv.dens;
          sum_s += dv * dpert * std::sin(2.0*M_PI*r);
          sum_c += dv * dpert * std::cos(2.0*M_PI*r);
          vol += dv;
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  Real buf[3] = {sum_s, sum_c, vol};
  MPI_Allreduce(MPI_IN_PLACE, buf, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  sum_s = buf[0]; sum_c = buf[1]; vol = buf[2];
#endif
  Real As = 2.0 * sum_s / vol;
  Real Ac = 2.0 * sum_c / vol;
  *amp_meas = std::sqrt(As*As + Ac*Ac);
  Real phase = std::atan2(-Ac, As);
  Real t = pm->time;
  if (t <= 0.0 || *amp_meas <= 0.0) {
    *omega_r_fit = 0.0;
    *omega_i_fit = 0.0;
    return;
  }
  Real phase_an = 2.0*M_PI * wv.vph * t / wv.lambda;
  while (phase < phase_an - M_PI) phase += 2.0*M_PI;
  while (phase > phase_an + M_PI) phase -= 2.0*M_PI;
  *omega_r_fit = phase * wv.lambda / (2.0*M_PI * t);
  *omega_i_fit = -std::log((*amp_meas)/wv.amp) * wv.lambda / (2.0*M_PI * t);
}

void SCLinwaveErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) return;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  Real time = pm->time;
  Real amp_t = wv.amp * std::exp(-time * wv.rdamp);
  int nmb1 = pmbp->nmb_thispack - 1;

  // 1. Fill u1 with the analytic damped, phase-shifted eigenmode at the final time,
  //    then hand the state comparison to the shared ProblemGenerator::OutputErrors
  //    (volume-weighted L1 and L-infty over u0 - u1, written to <basename>-errs.dat in
  //    the standard column layout used by every other linear-wave test). Same pattern as
  //    rad_linear_wave.cpp.
  auto u1 = pmbp->phydro->u1;
  WaveVars wv_ = wv;
  par_for("sc_linwave_u1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real r = (x1v*wv_.cos_a + x2v*wv_.sin_a)/wv_.lambda - wv_.vph*time;
    Real sinr = sin(2.0*M_PI*r);
    Real cosr = cos(2.0*M_PI*r);
    Real d0 = wv_.dens;
    Real p0 = wv_.pgas;
    Real uflow = wv_.vflow;
    u1(m,IDN,k,j,i) = d0 + amp_t*sinr;
    u1(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*uflow*uflow + amp_t*(sinr*wv_.E0R - cosr*wv_.E0I);
    u1(m,IM1,k,j,i) = wv_.cos_a*(d0*uflow + amp_t*(sinr*wv_.V0R - cosr*wv_.V0I));
    u1(m,IM2,k,j,i) = wv_.sin_a*(d0*uflow + amp_t*(sinr*wv_.V0R - cosr*wv_.V0I));
    u1(m,IM3,k,j,i) = 0.0;
  });
  pm->pgen->OutputErrors(pin, pm);

  // 2. Davis sec. 5.4 dispersion check: Fourier-fit the density mode (volume-weighted,
  // so it is
  //    valid on SMR/AMR meshes) and record fitted vs analytic omega_R, omega_I.
  Real omega_r_fit=0, omega_i_fit=0, amp_meas=0;
  FitFourierOmega(pm, &omega_r_fit, &omega_i_fit, &amp_meas);

  if (global_variable::my_rank == 0) {
    int nx1_tot = pm->mesh_indcs.nx1;
    std::cout << "sc_linwave §5.4: Bo=" << wv.Bo << " tau=" << wv.tau
              << " N=" << nx1_tot << " t=" << time
              << " 1D=" << (wv.is_1d?1:0) << std::endl;
    std::cout << "  analytic: ωR/(ka)=" << wv.omega_r
              << " ωI/(ka)=" << wv.omega_i
              << " vph=" << wv.vph << " rdamp=" << wv.rdamp << std::endl;
    std::cout << "  fitted:   ωR/(ka)=" << omega_r_fit
              << " ωI/(ka)=" << omega_i_fit
              << " A_meas=" << amp_meas << std::endl;

    FILE *fp = std::fopen("SCLinWave-davis54.dat", "a");
    if (fp) {
      std::fprintf(fp, "%.6g %.6g %d %d %.10e %.10e %.10e %.10e\n",
        wv.Bo, wv.tau, nx1_tot, pm->mesh_indcs.nx2,
        wv.omega_r, wv.omega_i, omega_r_fit, omega_i_fit);
      std::fclose(fp);
    }
  }
}
}  // namespace

void ProblemGenerator::SCLinwave(ParameterInput *pin, const bool restart) {
  pgen_final_func = SCLinwaveErrors;
  wv.amp   = pin->GetOrAddReal("problem", "amp", 1.0e-6);
  wv.vflow = pin->GetOrAddReal("problem", "vflow", 0.0);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr || pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: sc_linwave needs <hydro> and <nr_radiation>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // everything the final error check needs is derived here, on the restart path too
  wv.gamma = pmbp->phydro->peos->eos_data.gamma;
  Real gm1 = wv.gamma - 1.0;

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  wv.is_1d = pmy_mesh_->one_d || (pmy_mesh_->mesh_indcs.nx2 <= 1);

  if (wv.is_1d) {
    wv.cos_a = 1.0;
    wv.sin_a = 0.0;
    wv.lambda = x1size;
  } else {
    Real angle = std::atan(x1size/x2size);
    wv.sin_a = std::sin(angle);
    wv.cos_a = std::cos(angle);
    wv.lambda = (wv.cos_a >= wv.sin_a) ? (x1size*wv.cos_a) : (x2size*wv.sin_a);
  }

  wv.dens = 1.0;
  wv.pgas = wv.dens / wv.gamma;  // a = sqrt(gamma*p/rho) = 1
  wv.T0 = 1.0 / wv.gamma;        // (gamma-1) Etherm0 / rho0 with Etherm0 = p0/(gamma-1)

  // tau and Bo from the module's parameters (see the file header)
  nr_radiation::SC *psc = pmbp->pnrrad;
  wv.chi0 = psc->kappa_a * wv.dens;
  wv.tau  = wv.chi0 * wv.lambda / (2.0 * M_PI);
  Real g4 = wv.gamma * wv.gamma * wv.gamma * wv.gamma;
  wv.Bo   = 4.0 * g4 / (gm1 * psc->crat * psc->prat);

  AcousticWaveRad(wv.Bo, wv.tau, 1.0, wv.dens,
                  &wv.vph, &wv.rdamp, &wv.V0R, &wv.V0I, &wv.E0R, &wv.E0I,
                  &wv.omega_r, &wv.omega_i);

  if (global_variable::my_rank == 0) {
    std::cout << "sc_linwave: kappa_a=" << psc->kappa_a << " crat*prat="
              << psc->crat*psc->prat
              << " -> tau=" << wv.tau << " Bo=" << wv.Bo << " lambda=" << wv.lambda
              << " 1D=" << (wv.is_1d?1:0) << std::endl;
    std::cout << "  omega/(ka)=(" << wv.omega_r << "," << wv.omega_i
              << ") vph=" << wv.vph << " rdamp=" << wv.rdamp << " T0=" << wv.T0
              << std::endl;
  }
  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = pmbp->phydro->u0;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto wv_ = wv;

  par_for("sc_linwave_ic", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real r = (x1v*wv_.cos_a + x2v*wv_.sin_a)/wv_.lambda;
    Real sinr = sin(2.0*M_PI*r);
    Real cosr = cos(2.0*M_PI*r);
    Real d0 = wv_.dens;
    Real p0 = wv_.pgas;
    Real uflow = wv_.vflow;
    u0(m,IDN,k,j,i) = d0 + wv_.amp*sinr;
    u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*uflow*uflow
                    + wv_.amp*(sinr*wv_.E0R - cosr*wv_.E0I);
    u0(m,IM1,k,j,i) = wv_.cos_a*d0*uflow
                    + wv_.cos_a*wv_.amp*(sinr*wv_.V0R - cosr*wv_.V0I);
    u0(m,IM2,k,j,i) = wv_.sin_a*d0*uflow
                    + wv_.sin_a*wv_.amp*(sinr*wv_.V0R - cosr*wv_.V0I);
    u0(m,IM3,k,j,i) = 0.0;
  });

  // Initial intensity: thermal equilibrium I = S = T0^4
  Real T04 = wv.T0 * wv.T0 * wv.T0 * wv.T0;
  Kokkos::deep_copy(pmbp->pnrrad->ir, T04);
}
