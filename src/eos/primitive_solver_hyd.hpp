#ifndef EOS_PRIMITIVE_SOLVER_HYD_HPP_
#define EOS_PRIMITIVE_SOLVER_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file primitive_solver_hyd.hpp
//  \brief Contains the template class for PrimitiveSolverHydro, which is independent
//  of the EquationOfState class used elsewhere in AthenaK.

// C++ headers
#include <string>
#include <float.h>
#include <math.h>
#include <type_traits>

// PrimitiveSolver headers
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/primitive_solver.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

// AthenaK headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "adm/adm.hpp"
#include "hydro/hydro.hpp"

template<class EOSPolicy, class ErrorPolicy>
class PrimitiveSolverHydro {
  protected:
    void SetPolicyParams(std::string block, ParameterInput *pin) {
      if constexpr(std::is_same_v<Primitive::IdealGas, EOSPolicy>) {
        eos.SetGamma(pin->GetOrAddReal(block, "gamma", 5.0/3.0));
      }
    }
  public:
    Primitive::EOS<EOSPolicy, ErrorPolicy> eos;
    Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> ps;
    MeshBlockPack* pmy_pack;

    PrimitiveSolverHydro(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
        pmy_pack(pp), ps{&eos} {
      eos.SetDensityFloor(pin->GetOrAddReal(block, "dfloor", (FLT_MIN)));
      eos.SetPressureFloor(pin->GetOrAddReal(block, "pfloor", (FLT_MIN)));
      eos.SetThreshold(pin->GetOrAddReal(block, "dthreshold", 1.0));
      SetPolicyParams(block, pin);
    }

    // The prim to con function used on the reconstructed states inside the Riemann solver.
    // It also extracts the primitives into a form usable by PrimitiveSolver.
    KOKKOS_INLINE_FUNCTION
    void PrimToConsPt(const ScrArray2D<Real> &w, Real prim_pt[NPRIM], Real cons_pt[NCONS],
                      Real g3d[NSPMETRIC], Real sdetg,
                      const int i, const int &nhyd, const int &nscal) const {
      Real mb = eos.GetBaryonMass();
      Real b[NMAG] = {0.0};
      prim_pt[PRH] = w(IDN, i)/mb;
      prim_pt[PVX] = w(IVX, i);
      prim_pt[PVY] = w(IVY, i);
      prim_pt[PVZ] = w(IVZ, i);
      for (int n = 0; n < nscal; n++) {
        prim_pt[PYF + n] = w(nhyd + n, i);
      }
      // FIXME: Debug only! Use specific energy to validate other
      // hydro functions before breaking things
      Real e = w(IDN, i) + w(IEN, i);
      prim_pt[PTM] = eos.GetTemperatureFromE(prim_pt[PRH], e, &prim_pt[PYF]);
      prim_pt[PPR] = eos.GetPressure(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);

      // Apply the floor to make sure these values are physical.
      // FIXME: Is this needed if the first-order flux correction is enabled?
      bool floor = eos.ApplyPrimitiveFloor(prim_pt[PRH], &prim_pt[PVX],
                                           prim_pt[PPR], prim_pt[PTM], &prim_pt[PYF]);
      
      ps.PrimToCon(prim_pt, cons_pt, b, g3d);

      // Densitize the variables
      for (int i = 0; i < nhyd + nscal; i++) {
        cons_pt[i] *= sdetg;
      }

      // Copy floored primitives back into the original array.
      // TODO: Check if this is necessary
      if (floor) {
        w(IDN, i) = prim_pt[PRH]*mb;
        w(IVX, i) = prim_pt[PVX];
        w(IVY, i) = prim_pt[PVY];
        w(IVZ, i) = prim_pt[PVZ];
        // FIXME: Debug only! Switch to temperature or pressure after validating.
        w(IEN, i) = eos.GetEnergy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]) - w(IDN, i);
        for (int n = 0; n < nscal; n++) {
          w(nhyd + n, i) = prim_pt[PYF + n];
        }
      }
    }

    void PrimToCons(DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                    const int il, const int iu, const int jl, const int ju,
                    const int kl, const int ku) {
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
      auto &size = pmy_pack->pmb->mb_size;
      auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;

      auto &adm = pmy_pack->padm->adm;

      int &nhyd = pmy_pack->phydro->nhydro;
      int &nscal = pmy_pack->phydro->nscalars;
      int &nmb = pmy_pack->nmb_thispack;

      Real mb = eos.GetBaryonMass();


      par_for("pshyd_prim2cons", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Extract metric at a single point
        Real g3d[NSPMETRIC];
        g3d[S11] = adm.g_dd(m, 0, 0, k, j, i);
        g3d[S12] = adm.g_dd(m, 0, 1, k, j, i);
        g3d[S13] = adm.g_dd(m, 0, 2, k, j, i);
        g3d[S22] = adm.g_dd(m, 1, 1, k, j, i);
        g3d[S23] = adm.g_dd(m, 1, 2, k, j, i);
        g3d[S33] = adm.g_dd(m, 2, 2, k, j, i);
        Real sdetg = sqrt(Primitive::GetDeterminant(g3d));

        Real b[NMAG] = {0.0};

        // Extract primitive variables at a single point
        Real prim_pt[NPRIM], cons_pt[NCONS];
        prim_pt[PRH] = prim(m, IDN, k, j, i)/mb;
        prim_pt[PVX] = prim(m, IVX, k, j, i);
        prim_pt[PVY] = prim(m, IVY, k, j, i);
        prim_pt[PVZ] = prim(m, IVZ, k, j, i);
        for (int n = 0; n < nscal; n++) {
          prim_pt[PYF + n] = prim(m, nhyd + n, k, j, i);
        }
        // FIXME: Debug only! Use specific energy to validate other
        // hydro functions before breaking things.
        Real e = prim(m, IDN, k, j, i) + prim(m, IEN, k, j, i);
        prim_pt[PTM] = eos.GetTemperatureFromE(prim_pt[PRH], e, &prim_pt[PYF]);
        prim_pt[PPR] = eos.GetPressure(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);

        // Apply the floor to make sure these values are physical.
        bool floor = eos.ApplyPrimitiveFloor(prim_pt[PRH], &prim_pt[PVX],
                                             prim_pt[PPR], prim_pt[PTM], &prim_pt[PYF]);
        
        ps.PrimToCon(prim_pt, cons_pt, b, g3d);

        // Save the densitized conserved variables.
        cons(m, IDN, k, j, i) = cons_pt[CDN]*sdetg;
        cons(m, IM1, k, j, i) = cons_pt[CSX]*sdetg;
        cons(m, IM2, k, j, i) = cons_pt[CSY]*sdetg;
        cons(m, IM3, k, j, i) = cons_pt[CSZ]*sdetg;
        cons(m, IEN, k, j, i) = cons_pt[CTA]*sdetg;
        for (int n = 0; n < nscal; n++) {
          cons(m, nhyd + n, k, j, i) = cons_pt[CYD + n]*sdetg;
        }

        // If we floored the primitive variables, we need to adjust those, too.
        if (floor) {
          prim(m, IDN, k, j, i) = prim_pt[PRH]*mb;
          prim(m, IVX, k, j, i) = prim_pt[PVX];
          prim(m, IVY, k, j, i) = prim_pt[PVY];
          prim(m, IVZ, k, j, i) = prim_pt[PVZ];
          prim(m, IEN, k, j, i) = eos.GetEnergy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]) 
                                  - prim(m, IDN, k, j, i);
          for (int n = 0; n < nscal; n++) {
            prim(m, nhyd + n, k, j, i) = prim_pt[PYF + n];
          }
        }
      });

      return;
    }

    void ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                    const int il, const int iu, const int jl, const int ju,
                    const int kl, const int ku) {
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
      auto &size = pmy_pack->pmb->mb_size;

      int &nhyd = pmy_pack->phydro->nhydro;
      int &nscal = pmy_pack->phydro->nscalars;
      int &nmb = pmy_pack->nmb_thispack;

      // Some problem-specific parameters
      auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
      auto &cc_mask_ = pmy_pack->pcoord->cc_mask;
      auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
      auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

      auto &adm = pmy_pack->padm->adm;

      const int ni = (iu - il + 1);
      const int nji = (ju - jl + 1)*ni;
      const int nkji = (ku - kl + 1)*nji;
      const int nmkji = nmb*nkji;

      Real mb = eos.GetBaryonMass();

      Kokkos::parallel_for("pshyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx) {
        int m = (idx)/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/ni;
        int i = (idx - m*nkji - k*nji - j*ni) + il;
        j += jl;
        k += kl;

        // Extract the metric
        Real g3d[NSPMETRIC], g3u[NSPMETRIC], detg, sdetg;
        g3d[S11] = adm.g_dd(m, 0, 0, k, j, i);
        g3d[S12] = adm.g_dd(m, 0, 1, k, j, i);
        g3d[S13] = adm.g_dd(m, 0, 2, k, j, i);
        g3d[S22] = adm.g_dd(m, 1, 1, k, j, i);
        g3d[S23] = adm.g_dd(m, 1, 2, k, j, i);
        g3d[S33] = adm.g_dd(m, 2, 2, k, j, i);
        detg = Primitive::GetDeterminant(g3d);
        sdetg = sqrt(detg);
        SpatialInv(1.0/detg, g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                   &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);

        // Extract the conserved variables
        Real cons_pt[NCONS], cons_pt_old[NCONS], prim_pt[NPRIM];
        cons_pt[CDN] = cons_pt_old[CDN] = cons(m, IDN, k, j, i)/sdetg;
        cons_pt[CSX] = cons_pt_old[CSX] = cons(m, IM1, k, j, i)/sdetg;
        cons_pt[CSY] = cons_pt_old[CSY] = cons(m, IM2, k, j, i)/sdetg;
        cons_pt[CSZ] = cons_pt_old[CSZ] = cons(m, IM3, k, j, i)/sdetg;
        cons_pt[CTA] = cons_pt_old[CTA] = cons(m, IEN, k, j, i)/sdetg;
        for (int n = 0; n < nscal; n++) {
          cons_pt[CYD + n] = cons(m, nhyd + n, k, j, i)/sdetg;
        }
        Real b3u[NMAG] = {0.0};

        // If we're in an excised region, set the primitives to some default value.
        Primitive::SolverResult result;
        if (excise) {
          if (cc_mask_(m,k,j,i)) {
            prim_pt[PRH] = dexcise_/mb;
            prim_pt[PVX] = 0.0;
            prim_pt[PVY] = 0.0;
            prim_pt[PVZ] = 0.0;
            prim_pt[PPR] = pexcise_;
            for (int n = 0; n < nscal; n++) {
              // FIXME: Particle abundances should probably be set to a
              // default inside an excised region.
              prim_pt[PYF + n] = cons_pt[CYD]/cons_pt[CDN];
            }
            prim_pt[PTM] = eos.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
            result.error = Primitive::Error::SUCCESS;
            result.iterations = 0;
            result.cons_floor = false;
            result.prim_floor = false;
            result.cons_adjusted = true;
            ps.PrimToCon(prim_pt, cons_pt, b3u, g3d);
          }
          else {
            result = ps.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);
          }
        }
        else {
          result = ps.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);
        }

        if (result.error != Primitive::Error::SUCCESS) {
          // FIXME: Proper error response needed!
        }

        // Regardless of failure, we need to copy the primitives.
        prim(m, IDN, k, j, i) = prim_pt[PRH]*mb;
        prim(m, IVX, k, j, i) = prim_pt[PVX];
        prim(m, IVY, k, j, i) = prim_pt[PVY];
        prim(m, IVZ, k, j, i) = prim_pt[PVZ];
        prim(m, IEN, k, j, i) = eos.GetEnergy(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]) -
                                prim(m, IDN, k, j, i);
        for (int n = 0; n < nscal; n++) {
          prim(m, nhyd + n, k, j, i);
        }

        // If the conservative variables were floored or adjusted for consistency,
        // we need to copy the conserved variables, too.
        if (result.cons_floor || result.cons_adjusted) {
          cons(m, IDN, k, j, i) = cons_pt[CDN]*sdetg;
          cons(m, IM1, k, j, i) = cons_pt[CSX]*sdetg;
          cons(m, IM2, k, j, i) = cons_pt[CSY]*sdetg;
          cons(m, IM3, k, j, i) = cons_pt[CSZ]*sdetg;
          cons(m, IEN, k, j, i) = cons_pt[CTA]*sdetg;
          for (int n = 0; n < nscal; n++) {
            cons(m, nhyd + n, k, j, i) = cons_pt[CYD + n]*sdetg;
          }
        }
      });
    }

    // Get the transformed sound speeds at a point in a given direction.
    KOKKOS_INLINE_FUNCTION
    void GetGRSoundSpeeds(Real& lambda_p, Real& lambda_m, Real prim[NPRIM], Real g3d[NSPMETRIC],
                          Real beta_u[3], Real alpha, Real gii, int pvx) const {
      Real uu[3] = {prim[PVX], prim[PVY], prim[PVZ]};
      Real usq = Primitive::SquareVector(uu, g3d);
      int index = pvx - PVX;

      // Get the Lorentz factor and the 3-velocity.
      Real Wsq = 1.0 + usq;
      Real W = sqrt(Wsq);
      Real vsq = usq/Wsq;
      Real vu[3] = {uu[0]/W, uu[1]/W, uu[2]/W};

      Real cs = eos.GetSoundSpeed(prim[PRH], prim[PTM], &prim[PYF]);
      Real csq = cs*cs;

      Real iWsq_ad = 1.0 - vsq*csq;
      Real dis = (csq/Wsq)*(gii*iWsq_ad - vu[index]*vu[index]*(1.0 - csq));
      Real sdis = sqrt(dis);

      lambda_p = alpha*(vu[index]*(1.0 - csq) + sdis)/iWsq_ad - beta_u[index];
      lambda_m = alpha*(vu[index]*(1.0 - csq) - sdis)/iWsq_ad - beta_u[index];
    }
};
#endif
