#ifndef DYN_GRMHD_RSOLVERS_HLLD_DYN_GRMHD_HPP_
#define DYN_GRMHD_RSOLVERS_HLLD_DYN_GRMHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlld_dyngrmhd.hpp
//! \brief HLLE Riemann solver for GRMHD that first transforms to local flat space

#include <math.h>

#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "flux_dyn_grmhd.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void HLLD_DYN_GRMHD
//! \brief inline function for calculating GRMHD fluxes via HLLE with frame transform
//----------------------------------------------------------------------------------------
template<int ivx, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void HLLD_DYNGR(TeamMember_t const &member,
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
     const RegionIndcs &indcs, const DualArray1D<RegionSize> &size,
     const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     const int& nhyd, const int& nscal,
     const adm::ADM::ADM_vars& adm,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  par_for_inner(member, il, iu, [&](const int i) {
    constexpr int ivy = IVX + ((ivx - IVX) + 1)%3;
    constexpr int ivz = IVX + ((ivx - IVX) + 2)%3;

    constexpr int ibx = ivx - IVX;
    constexpr int iby = ((ivx - IVX) + 1)%3;
    constexpr int ibz = ((ivx - IVX) + 2)%3;

    constexpr int pvx = PVX + (ivx - IVX);
    constexpr int csx = CSX + (ivx - IVX);
    constexpr int csy = CSX + ((ivx - IVX) + 1) % 3;
    constexpr int csz = CSX + ((ivx - IVX) + 2) % 3;

    Real g3d[NSPMETRIC];
    Real beta_u[3];
    Real alpha;
    if constexpr (ivx == IVX) {
      adm::Face1Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    } else if (ivx == IVY) {
      adm::Face2Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    } else if (ivx == IVZ) {
      adm::Face3Metric(m, k, j, i, adm.g_dd, adm.beta_u, adm.alpha, g3d, beta_u, alpha);
    }

    Real sdetg = sqrt(Primitive::GetDeterminant(g3d));
    Real isdetg = 1.0/sdetg;

    // Extract left and right primitives
    Real prim_l[NPRIM], prim_r[NPRIM];
    Real Bu_l[NMAG], Bu_r[NMAG];
    ExtractPrimitives(eos, wl, bl, bx, isdetg, prim_l, Bu_l,
                      nhyd, nscal, m, k, j, i, ibx, iby, ibz);
    ExtractPrimitives(eos, wr, br, bx, isdetg, prim_r, Bu_r,
                      nhyd, nscal, m, k, j, i, ibx, iby, ibz);

    // Compute tetrad transformation
    Real e_cov[4][4], e_cont[4][4];
    ComputeOrthonormalTetrad<ivx>(g3d, beta_u, alpha, isdetg*isdetg, e_cov, e_cont);
    Real ialpha = e_cov[0][0]; // Save a division this way
    TransformPrimitivesToTetrad(prim_l, Bu_l, e_cont);
    TransformPrimitivesToTetrad(prim_r, Bu_r, e_cont);

    // LEFT STATE
    Real cons_l[NCONS], flux_l[NCONS], bflux_l[NMAG], bsq_l;
    SingleStateTetradFlux<ivx>(eos, prim_l, Bu_l, cons_l, flux_l, bflux_l, bsq_l);


    // RIGHT STATE
    Real cons_r[NCONS], flux_r[NCONS], bflux_r[NMAG], bsq_r;
    SingleStateTetradFlux<ivx>(eos, prim_r, Bu_r, cons_r, flux_r, bflux_r, bsq_r);


    // Calculate the magnetosonic speeds for both states
    Real lambda_pl, lambda_pr, lambda_ml, lambda_mr;
    eos.GetSRFastMagnetosonicSpeeds(lambda_pl, lambda_ml, prim_l, bsq_l, pvx);
    eos.GetSRFastMagnetosonicSpeeds(lambda_pr, lambda_mr, prim_r, bsq_r, pvx);
    
    // Get the extremal wavespeeds
    Real lambda_l = Kokkos::fmin(lambda_ml, lambda_mr);
    Real lambda_r = Kokkos::fmax(lambda_pl, lambda_pr);

    Real vint = beta_u[ibx]*ialpha*e_cont[ivx][ivx];
    Real *f_interface, *bf_interface, *cons_interface, *b_interface;
    Real cons_int[NCONS], b_int[NMAG], fint[NCONS], bfint[NMAG];
    if (lambda_l >= vint) {
      // Supersonic wave going left to right
      f_interface = &flux_l[0];
      bf_interface = &bflux_l[0];
      cons_interface = &cons_l[0];
      b_interface = &Bu_l[0];
    } else if (lambda_r <= vint) {
      // Supersonic wave going right to left
      f_interface = &flux_r[0];
      bf_interface = &bflux_r[0];
      cons_interface = &cons_r[0];
      b_interface = &Bu_r[0];
    } else {
      // Intermediate state, need to check contact wave and Alfven waves

      //----------------------------------------------------------------------------------
      // STEP 1: Compute jump conditions
      //----------------------------------------------------------------------------------
      Real Rl[NCONS];
      Real RBl[NMAG];
      Rl[CDN] = lambda_l*cons_l[CDN] - flux_l[CDN];
      Rl[CSX] = lambda_l*cons_l[CSX] - flux_l[CSX];
      Rl[CSY] = lambda_l*cons_l[CSY] - flux_l[CSY];
      Rl[CSZ] = lambda_l*cons_l[CSZ] - flux_l[CSZ];
      Rl[CTA] = lambda_l*cons_l[CTA] - flux_l[CTA];
      RBl[ibx] = lambda_l*Bu_l[ibx];
      RBl[iby] = lambda_l*Bu_l[iby] - bflux_l[iby];
      RBl[ibz] = lambda_l*Bu_l[ibz] - bflux_l[ibz];

      Real Rr[NCONS];
      Real RBr[NMAG];
      Rr[CDN] = lambda_r*cons_r[CDN] - flux_r[CDN];
      Rr[CSX] = lambda_r*cons_r[CSX] - flux_r[CSX];
      Rr[CSY] = lambda_r*cons_r[CSY] - flux_r[CSY];
      Rr[CSZ] = lambda_r*cons_r[CSZ] - flux_r[CSZ];
      Rr[CTA] = lambda_r*cons_r[CTA] - flux_r[CTA];
      RBr[ibx] = lambda_r*Bu_r[ibx];
      RBr[iby] = lambda_r*Bu_r[iby] - bflux_r[iby];
      RBr[ibz] = lambda_r*Bu_r[ibz] - bflux_r[ibz];

      Real qb = 1.0/(lambda_r - lambda_l);

      //----------------------------------------------------------------------------------
      // STEP 2: Store HLL solution, will replace if possible.
      //----------------------------------------------------------------------------------
      cons_int[CDN] = (Rr[CDN] - Rl[CDN])*qb;
      cons_int[CSX] = (Rr[CSX] - Rl[CSX])*qb;
      cons_int[CSY] = (Rr[CSY] - Rl[CSY])*qb;
      cons_int[CSZ] = (Rr[CSZ] - Rl[CSZ])*qb;
      cons_int[CTA] = (Rr[CTA] - Rl[CTA])*qb;
      b_int[ibx] = Bu_l[ibx];
      b_int[iby] = (RBr[iby] - RBl[iby])*qb;
      b_int[ibz] = (RBr[ibz] - RBl[ibz])*qb;

      fint[CDN] = (lambda_l*Rr[CDN] - lambda_r*Rl[CDN])*qb;
      fint[CSX] = (lambda_l*Rr[CSX] - lambda_r*Rl[CSX])*qb;
      fint[CSY] = (lambda_l*Rr[CSY] - lambda_r*Rl[CSY])*qb;
      fint[CSZ] = (lambda_l*Rr[CSZ] - lambda_r*Rl[CSZ])*qb;
      fint[CTA] = (lambda_l*Rr[CTA] - lambda_r*Rl[CTA])*qb;
      bfint[ibx] = 0.0;
      bfint[iby] = (lambda_l*RBr[iby] - lambda_r*RBl[iby])*qb;
      bfint[ibz] = (lambda_l*RBr[ibz] - lambda_r*RBl[ibz])*qb;

      // Passive scalar advection
      if (fint[CDN] > 0.0) {
        for (int s = 0; s < nscal; s++) {
          cons_int[CYD + s] = prim_l[PYF + s];
          fint[CYD + s] = prim_l[PYF + s]*cons_int[CDN];
        }
      } else {
        for (int s = 0; s < nscal; s++) {
          cons_int[CYD + s] = prim_r[PYF + s];
          fint[CYD + s] = prim_r[PYF + s]*cons_int[CDN];
        }
      }

      //----------------------------------------------------------------------------------
      // STEP 3: Compute pressure across contact discontinuity
      //----------------------------------------------------------------------------------
      // Initial guess for pressure
      // FIXME(JMF): Doing a C2P here is absolutely horrible. We need something better.
      Real flat[NSPMETRIC] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
      Real prim_hll[NPRIM];
      eos.ps.ConToPrim(prim_hll, cons_int, b_int, flat, flat);
      Real Wsq_hll = prim_hll[IVX]*prim_hll[IVX] +
                     prim_hll[IVY]*prim_hll[IVY] +
                     prim_hll[IVZ]*prim_hll[IVZ] + 1.0;
      Real Bu_hll = b_int[0]*prim_hll[IVX] + b_int[1]*prim_hll[IVY] +
                    b_int[2]*prim_hll[IVZ];
      Real bsq_hll = ((b_int[0]*b_int[0] + b_int[1]*b_int[1] + b_int[2]*b_int[2]) +
                      Bu_hll*Bu_hll)/Wsq_hll;
      Real ptot_hll = prim_hll[PPR] + 0.5*bsq_hll;
      Real ptot;
      if (b_int[ibx]*b_int[ibx]/ptot_hll < 0.1) {
        // If the flow is not strongly magnetized, we can initialize it assuming Bx = 0,
        // which is a degenerate case with an exact solution.
        Real& sx_hll = cons_int[csx];
        Real& fsx_hll = fint[csx];
        Real e_hll = cons_int[CTA] + cons_int[CDN];
        Real fe_hll = fint[CTA] + fint[CDN];

        ptot = 0.5*(fsx_hll - e_hll +
                   Kokkos::sqrt((e_hll + fsx_hll)*(e_hll + fsx_hll) - 4.0*sx_hll*fe_hll));
      } else {
        ptot = ptot_hll;
      }

      // Function for computing Alfven state
      auto alfven = [](Real va[NMAG], Real Ba[NMAG], Real& Hb, Real& eta, Real Ua[NCONS],
                       Real K[NCONS], const Real lambda, const Real P,
                       const Real R[NCONS], const Real Rb[NMAG], const Real Bx,
                       const int sign) {
        // Utility variables
        Real Re = R[CTA] + R[CDN];
        Real A = R[csx] - lambda*Re + P*(1. - lambda*lambda);
        Real G = Rb[iby]*Rb[iby] + Rb[ibz]*Rb[ibz];
        Real C = R[csy]*Rb[iby] + R[csz]*Rb[ibz];
        Real Q = -A - G + Bx*Bx*(1. - lambda*lambda);
        Real iX = 1.0/(Bx*(A*lambda*Bx + C) - (A + G)*(lambda*P + Re));

        // Velocity in Alfven state
        va[ibx] = (Bx*(A*Bx + lambda*C) - (A + G)*(P + R[csx]))*iX;
        va[iby] = (Q*R[csy] + Rb[iby]*(C + Bx*(lambda*R[csx] - Re)))*iX;
        va[ibz] = (Q*R[csz] + Rb[ibz]*(C + Bx*(lambda*R[csx] - Re)))*iX;

        // Magnetic field in Alfven state
        Real qc = 1.0/(lambda - va[ibx]);
        Ba[ibx] = Bx;
        Ba[iby] = (Rb[iby] - Bx*va[iby])*qc;
        Ba[ibz] = (Rb[ibz] - Bx*va[ibz])*qc;

        Hb = P + (Re - (va[ibx]*R[csx] + va[iby]*R[csy] + va[ibz]*R[csz]))*qc;

        // Conserved variables in Alfven state
        Real Bva = (va[ibx]*Ba[ibx] + va[iby]*Ba[iby] + va[ibz]*Ba[ibz]);
        Ua[CDN] = R[CDN]*qc;
        Ua[CTA] = (R[CTA] + P*va[ibx] - Bva*Bx)*qc;
        Ua[csx] = (Ua[CTA] + Ua[CDN] + P)*va[ibx] - Bva*Bx;
        Ua[csy] = (Ua[CTA] + Ua[CDN] + P)*va[iby] - Bva*Ba[iby];
        Ua[csz] = (Ua[CTA] + Ua[CDN] + P)*va[ibz] - Bva*Ba[ibz];

        // K-vector
        eta = sign*( (Bx > 0) ? 1 : -1)*Kokkos::sqrt(Hb);
        Real qd = 1.0/(lambda*P + Re + Bx*eta);
        K[ibx] = (R[csx] + P + Rb[ibx]*eta)*qd;
        K[iby] = (R[csy] + Rb[iby]*eta)*qd;
        K[ibz] = (R[csz] + Rb[ibz]*eta)*qd;
      };

      // Master function for computing pressure
      Real Ual[NCONS], Uar[NCONS], Bal[NMAG], Bar[NMAG], Bc[NMAG], vc, lal, lar;
      Real Ucl[NCONS], Ucr[NCONS], valx, varx, Hbl, Hbr;
      auto froot = [&](Real P) {
        Real val[NMAG], Kl[NMAG], etal;
        Real var[NMAG], Kr[NMAG], etar;

        // Compute Alfven states
        alfven(val, Bal, Hbl, etal, Ual, Kl, lambda_l, P, Rl, RBl, Bu_l[ibx], -1);
        alfven(var, Bar, Hbr, etar, Uar, Kr, lambda_r, P, Rr, RBr, Bu_l[ibx], 1);
        // The Alfven wave speeds are the x-components of the K vector.
        lal = Kl[ibx];
        lar = Kr[ibx];

        // Compute field in contact state
        Real qe = 1.0/(lar - lal);
        Bc[ibx] = Bu_l[ibx];
        Bc[iby] = ((Bar[iby]*(lar - var[ibx]) + Bu_l[ibx]*var[iby]) -
                    (Bal[iby]*(lal - val[ibx]) + Bu_l[ibx]*val[iby]))*qe;
        Bc[ibz] = ((Bar[ibz]*(lar - var[ibx]) + Bu_l[ibx]*var[ibz]) -
                    (Bal[ibz]*(lal - val[ibx]) + Bu_l[ibx]*val[ibz]))*qe;

        valx = val[ibx];
        varx = var[ibx];

        // Velocity in contact state
        Real vcl[NMAG], vcr[NMAG];
        Real qfl = 1.0/(etal - (Kl[ibx]*Bc[ibx] + Kl[iby]*Bc[iby] + Kl[ibz]*Bc[ibz]));
        Real Ksql = Kl[ibx]*Kl[ibx] + Kl[iby]*Kl[iby] + Kl[ibz]*Kl[ibz];
        vcl[ibx] = Kl[ibx] - Bc[ibx]*(1.0 - Ksql)*qfl;
        vcl[iby] = Kl[iby] - Bc[iby]*(1.0 - Ksql)*qfl;
        vcl[ibz] = Kl[ibz] - Bc[ibz]*(1.0 - Ksql)*qfl;

        Real qfr = 1.0/(etar - (Kr[ibx]*Bc[ibx] + Kr[iby]*Bc[iby] + Kr[ibz]*Bc[ibz]));
        Real Ksqr = Kr[ibx]*Kr[ibx] + Kr[iby]*Kr[iby] + Kr[ibz]*Kr[ibz];
        vcr[ibx] = Kr[ibx] - Bc[ibx]*(1.0 - Ksqr)*qfr;
        vcr[iby] = Kr[iby] - Bc[iby]*(1.0 - Ksqr)*qfr;
        vcr[ibz] = Kr[ibz] - Bc[ibz]*(1.0 - Ksqr)*qfr;

        // Compute the conserved variables in the contact regions
        Real qgl = 1.0/(lal - vcl[ibx]);
        Real Bvcl = Bc[ibx]*vcl[ibx] + Bc[iby]*vcl[iby] + Bc[ibz]*vcl[ibz];
        Ucl[CDN] = Ual[CDN]*(lal - val[ibx])*qgl;
        Ucl[CTA] = ((Ual[CTA] + Ual[CDN])*lal - Ual[csx] + P*vcl[ibx] - Bvcl*Bc[ibx])*qgl;
        Ucl[csx] = (Ucl[CTA] + P)*vcl[ibx] - Bvcl*Bc[ibx];
        Ucl[csy] = (Ucl[CTA] + P)*vcl[iby] - Bvcl*Bc[iby];
        Ucl[csz] = (Ucl[CTA] + P)*vcl[ibz] - Bvcl*Bc[ibz];
        Ucl[CTA] -= Ucl[CDN];

        Real qgr = 1.0/(lar - vcr[ibx]);
        Real Bvcr = Bc[ibx]*vcr[ibx] + Bc[iby]*vcr[iby] + Bc[ibz]*vcr[ibz];
        Ucr[CDN] = Uar[CDN]*(lar - var[ibx])*qgr;
        Ucr[CTA] = ((Uar[CTA] + Uar[CDN])*lar - Uar[csx] + P*vcr[ibx] - Bvcr*Bc[ibx])*qgr;
        Ucr[csx] = (Ucr[CTA] + P)*vcr[ibx] - Bvcr*Bc[ibx];
        Ucr[csy] = (Ucr[CTA] + P)*vcr[iby] - Bvcr*Bc[iby];
        Ucr[csz] = (Ucr[CTA] + P)*vcr[ibz] - Bvcr*Bc[ibz];
        Ucr[CTA] -= Ucr[CDN];

        
        // In principle the left and right normal velocities are identical across the
        // contact interface, but this will only be true up to some tolerance in our
        // root solver. Therefore, we average the left and right states together and call
        // that the contact wave speed.
        vc = 0.5*(vcl[ibx] + vcr[ibx]);

        // Try to enforce vcl[ibx] = vcr[ibx]
        Real DK = Kr[ibx] - Kl[ibx];
        Real Yl = (1.0 - Ksql)*qfl;
        Real Yr = (1.0 - Ksqr)*qfr;

        return DK - Bc[ibx]*(Yr - Yl);
      };

      // Secant method to find intermediate pressure.
      // We need two guesses to initialize the secant method. We choose the second guess
      // as follows: if Ptot is less than both Pl and Pr, we take the minimum of the two.
      // If Ptot is greater than both Pl and Pr, we choose the maximum of the two. If it
      // sits in between, we take the average.
      Real pmin = Kokkos::fmin(prim_l[PPR], prim_r[PPR]);
      Real pmax = Kokkos::fmax(prim_l[PPR], prim_r[PPR]);
      Real ptot_old;
      if (ptot < pmin) {
        ptot_old = pmin;
      } else if (ptot > pmax) {
        ptot_old = pmax;
      } else {
        ptot_old = 0.5*(pmin + pmax);
      }
      Real fold = froot(ptot_old);
      Real ptot_old2;
      const Real tol = 1e-6;
      const int max_iters = 15;
      int count = 0;
      do {
        Real f = froot(ptot);
        count++;

        ptot_old2 = ptot_old;
        ptot_old = ptot;
        ptot = (ptot_old2*f - ptot*fold)/(f - fold);
        fold = f;
      } while (Kokkos::fabs(ptot - ptot_old) > tol*Kokkos::fabs(ptot_old - ptot_old2) &&
               count < max_iters);
      
      // STEP 4: Check for correctness and compute intermediate state if possible.
      bool fail = false;
      if (count == max_iters || !Kokkos::isfinite(ptot)) {
        // Root solver did not converge. Do nothing.
        fail = true;
      }
      else if (Hbl <= ptot || Hbr <= ptot || valx <= lambda_l || varx >= lambda_r) {
        // Either the interface pressure is too large or the eigenvalues are not
        // well-ordered, so return failure.
        fail = false;
      } else if (vc <= lal || vc >= lar) {
        // Check an edge case where lar and lal are effectively the same due to very small
        // Bx. Only fail if the edge case doesn't apply because numerical errors can cause
        // sign issues.
        if (Kokkos::fabs(lar - lal) > tol*Kokkos::fabs(vc)) {
          fail = true;
        }
      }
      if (!fail) {
        // If we reach this point, we have a valid solution. Otherwise, we will revert to
        // the HLL solution.
        if (vint < vc) {
          // Compute the left Alfven state
          for (int v = 0; v < nhyd; v++) {
            cons_int[v] = Ual[v];
            fint[v] = flux_l[v] + lambda_l*(Ual[v] - cons_l[v]);
          }

          b_int[iby] = Bal[iby];
          b_int[ibz] = Bal[ibz];
          bfint[iby] = bflux_l[iby] + lambda_l*(Bal[iby] - Bu_l[iby]);
          bfint[ibz] = bflux_l[ibz] + lambda_l*(Bal[ibz] - Bu_l[ibz]);

          if (vint >= lal) {
            // Compute the left contact state using the RH condition on the Alfven state.
            for (int v = 0; v < nhyd; v++) {
              cons_int[v] = Ucl[v];
              fint[v] = fint[v] + lal*(Ucl[v] - Ual[v]);
            }

            b_int[iby] = Bc[iby];
            b_int[ibz] = Bc[ibz];
            bfint[iby] = bfint[iby] + lal*(Bal[iby] - Bu_l[iby]);
            bfint[ibz] = bfint[ibz] + lal*(Bal[ibz] - Bu_l[ibz]);
          }
        } else {
          // Compute the right Alfven state
          for (int v = 0; v < nhyd; v++) {
            cons_int[v] = Uar[v];
            fint[v] = flux_r[v] + lambda_r*(Uar[v] - cons_r[v]);
          }

          b_int[iby] = Bar[iby];
          b_int[ibz] = Bar[ibz];
          bfint[iby] = bflux_r[iby] + lambda_r*(Bar[iby] - Bu_r[iby]);
          bfint[ibz] = bflux_r[ibz] + lambda_r*(Bar[ibz] - Bu_r[ibz]);
         
          if (vint >= lar) {
            // Compute the right contact state using the RH conditions on the Alfven
            // state.
            for (int v = 0; v < nhyd; v++) {
              cons_int[v] = Ucr[v];
              fint[v] = fint[v] + lar*(Ucr[v] - Uar[v]);
            }

            b_int[iby] = Bc[iby];
            b_int[ibz] = Bc[ibz];
            bfint[iby] = bfint[iby] + lar*(Bar[iby] - Bu_r[iby]);
            bfint[ibz] = bfint[ibz] + lar*(Bar[ibz] - Bu_r[ibz]);
          }
        }
      }

      cons_interface = &cons_int[0];
      f_interface = &fint[0];
      b_interface = &b_int[0];
      bf_interface = &bfint[0];
    }

    Real vol = alpha*sdetg;

    // Transform the fluxes and store them in the global flux arrays.
    TransformFluxesToGlobal(cons_interface, f_interface, b_interface, bf_interface,
                    e_cont, e_cov, flx, ey, ez, vol, m, k, j, i, ivx, ivy, ivz,
                    ibx, iby, ibz, csx, csy, csz);
  });
}

} // namespace dyngr


#endif // DYN_GRMHD_RSOLVERS_HLLD_DYN_GRMHD_HPP_
