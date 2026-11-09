//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nuclear_burn.cpp
//! \brief Nuclear::BurnStep: operator-split burn of the species mass fractions over one
//! network substep, on DynGRMHD with the Helmholtz EOS.
//!
//! UNITS. The hydro state lives in code units: G = c = 1 with the mass unit set by
//! <mhd>/units (geometric_solar: [M] = M_sun, lengths in G*M_sun/c^2, T in MeV). The
//! reaction network is written in cgs. BurnStep(h) receives h in code time and converts
//! everything to cgs before the kernel, so that INSIDE THE KERNEL ALL QUANTITIES ARE CGS:
//!   rho [g/cm^3], p [erg/cm^3], eps [erg/g], T [K], dt and t_heat [s],
//!   NuclearParams tolerances on rho*X [g/cm^3].
//! The conversion factors come from the code unit system of the DynGRMHD EOS itself
//! (not units::Units, whose M_sun differs from the EOS's), so the cgs state seen by the
//! network is exactly the one the hydro EOS describes. Only dimensionless X[n] (through
//! D*X[n]) and dt_nuc [code time] are written back.

#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "nuclear/integrate/backward_euler.hpp"
#include "nuclear/microphysics.hpp"
#include "nuclear/nuclear.hpp"

namespace nuclear {

//----------------------------------------------------------------------------------------
//! \fn void Nuclear::InitBurn()
//! \brief Called once by the constructor (Nuclear is built after DynGRMHD in
//! MeshBlockPack::AddPhysics). Checks that the burn is supported: only DynGRMHD with the
//! Helmholtz EOS (and the ResetFloor error policy) for now; stops the run at startup
//! otherwise. Then sets the cgs -> code conversion factors from the code unit system of
//! the DynGRMHD EOS (<mhd>/units), which is fixed for the whole run.

void Nuclear::InitBurn() {
  using DynGRHelm = dyngr::DynGRMHDPS<Primitive::EOSHelmholtz, Primitive::ResetFloor>;
  auto *pdyngr = dynamic_cast<DynGRHelm*>(pmy_pack->pdyngr);
  if (pdyngr == nullptr || pmy_pack->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "the nuclear burn (<nuclear> block) needs DynGRMHD with "
              << "mhd/dyn_eos = helmholtz and mhd/dyn_error = reset_floor" << std::endl;
    exit(EXIT_FAILURE);
  }

  const Primitive::UnitSystem &code = pdyngr->eos.ps.GetEOS().GetCodeUnitSystem();
  const Primitive::UnitSystem cgs = Primitive::MakeCGS();
  time_c2c = cgs.TimeConversion(code);
  rho_c2c  = cgs.MassDensityConversion(code);
  pres_c2c = cgs.PressureConversion(code);
}

//----------------------------------------------------------------------------------------
//! \fn void Nuclear::BurnStep(const Real h)
//! \brief Burn over one network substep h [code time]. Selects the kernel for the
//! DynGRMHD EOS policy (checked by InitBurn at startup); only the Helmholtz EOS for now.
//! Returns the number of failed cells on this rank.

int Nuclear::BurnStep(const Real h) {
  return BurnStepImpl<Primitive::EOSHelmholtz>(h);
}

//----------------------------------------------------------------------------------------
//! \fn void Nuclear::BurnStepImpl<EOSPolicy>(const Real h)
//! \brief Operator-split burn of the mass fractions over h [code time]. Only the partial
//! densities D*X[n] change: with the Helmholtz EOS, eps includes the nuclear binding
//! energy, so the release keeps tau (and eps, at these low velocities, where
//! tau ~ rho*eps) fixed and shows up as a higher T and p after ConToPrim.
//! The network runs in cgs on a shallow copy of the EOS (it shares the table).
//! Returns the number of cells of this rank where the Newton solver failed (singular
//! matrix, no convergence within max_iter, or a non-finite result); the run is stopped
//! by Nuclear::Integrate, on all ranks, at the end of the hydro step.

template <class EOSPolicy>
int Nuclear::BurnStepImpl(const Real h) {
  using EOSType = Primitive::EOS<EOSPolicy, Primitive::ResetFloor>;
  auto *pdyngr = static_cast<dyngr::DynGRMHDPS<EOSPolicy, Primitive::ResetFloor>*>(
                 pmy_pack->pdyngr);

  // cgs copy of the DynGRMHD EOS, captured by value into the per-cell kernel below.
  // A shallow copy: the table is shared, only its handles are copied. Rebuilt on every
  // call rather than cached at construction, so it carries the EOS floors set later
  // (e.g. by the pgen).
  EOSType eos = pdyngr->eos.ps.GetEOS();
  eos.SetCodeUnitSystem(Primitive::MakeCGS());

  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  int is=indcs.is, ie=indcs.ie, js=indcs.js, je=indcs.je, ks=indcs.ks, ke=indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &u0 = pmy_pack->pmhd->u0;
  auto &w0 = pmy_pack->pmhd->w0;
  int nmhd = pmy_pack->pmhd->nmhd;

  // Local copies of the members, captured by value into the kernel (no "this" on device)
  const NuclearParams prm = params;  // tolerances in cgs
  const Real dt = h/time_c2c;        // substep [s]
  const Real rho_c2c_ = rho_c2c, pres_c2c_ = pres_c2c;

  // Burn every cell over dt, and at the same time find min(cv*T/|edot|) [s] at the
  // burned state, which sets the network substep size (dt_nuc) of the next hydro step.
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (nmb1 + 1)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real min_theat = std::numeric_limits<float>::max();
  int nfail_total = 0;  // number of cells where the Newton solver failed

  Kokkos::parallel_reduce("nuclear_burn",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_th, int &nfail) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Thermo state;

    // Primitives in cgs: rho, X[n], and T from the pressure (DynGRMHD stores p in
    // IPR); eps at that T is held fixed during the burn.
    const Real rho = w0(m,IDN,k,j,i)/rho_c2c_;  // [g/cm^3], fixed during the burn
    Real X[NSPECIES];
    state.rho = rho;
    state.p = w0(m,IPR,k,j,i)/pres_c2c_;        // [erg/cm^3]
    for (int n=0; n<NSPECIES; n++) {
      X[n] = w0(m,nmhd+n,k,j,i);
      state.xn[n] = X[n];
    }

    burn(eos_input_rp, state, eos);
    const Real eps = state.e;  // [erg/g], held fixed during the burn (tau is unchanged)

    // Backward Euler burn over dt (cgs): X[n] is returned burned and renormalized,
    // t_heat = cv*T/|edot| [s] at the burned state. A failure (singular matrix, no
    // convergence within max_iter, or a non-finite result) is counted and returned.
    Real t_heat;
    if (!BackwardEulerBurn(rho, eps, dt, X, eos, prm, state, t_heat)) nfail++;

    // Conserved D*X[n]; D and tau are unchanged by the burn
    for (int n=0; n<NSPECIES; n++) {
      u0(m,nmhd+n,k,j,i) = u0(m,IDN,k,j,i)*X[n];
    }

    min_th = Kokkos::fmin(t_heat, min_th);
  }, Kokkos::Min<Real>(min_theat), Kokkos::Sum<int>(nfail_total));

  // Network substep size for the next hydro step, back in code time: Nuclear::Integrate
  // reads it once at the start of the next hydro step, before subcycling. Without any
  // burning min_theat stays huge, and the hydro step is covered by a single substep.
  dt_nuc = prm.dt_nuc_factor*min_theat*time_c2c;

  // New primitives of the active cells (T and p rise at fixed eps); the ghost zones
  // are refreshed by the Nuclear tasks afterwards
  pdyngr->ConToPrimBC(is, ie, js, je, ks, ke);
  return nfail_total;
}

}  // namespace nuclear
