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

// C headers
#include <float.h>
#include <math.h>

// C++ headers
#include <string>
#include <type_traits>
#include <iostream>
#include <sstream>

// PrimitiveSolver headers
#include "eos/primitive-solver/eos.hpp"
#include "eos/primitive-solver/primitive_solver.hpp"
#include "eos/primitive-solver/idealgas.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/reset_floor.hpp"

// AthenaK headers
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "coordinates/adm.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"

template<class EOSPolicy, class ErrorPolicy>
class PrimitiveSolverHydro {
 protected:
  void SetPolicyParams(std::string block, ParameterInput *pin) {
    // Parameters for an ideal gas
    if constexpr(std::is_same_v<Primitive::IdealGas, EOSPolicy>) {
      ps.GetEOSMutable().SetGamma(pin->GetOrAddReal(block, "gamma", 5.0/3.0));
      ps.GetEOSMutable().SetNSpecies(pin->GetOrAddInteger(block, "nscalars", 0));
    }
    // Parameters for a piecewise polytrope
    if constexpr(std::is_same_v<Primitive::PiecewisePolytrope, EOSPolicy>) {
      bool result = ps.GetEOSMutable().ReadParametersFromInput(block, pin);
      if (!result) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "There was an error while constructing the EOS."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    // Parameters for CompOSE EoS
    if constexpr(std::is_same_v<Primitive::EOSCompOSE, EOSPolicy>) {
      // Get and set number of scalars in table. This will currently fail if not 1.
      ps.GetEOSMutable().SetNSpecies(pin->GetOrAddInteger(block, "nscalars", 1));
      std::string units = pin->GetOrAddString(block, "units", "geometric_solar");
      if (!units.compare("geometric_solar")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricSolar());
      } else if (!units.compare("geometric_kilometer")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeGeometricKilometer());
      } else if (!units.compare("nuclear")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeNuclear());
      } else if (!units.compare("cgs")) {
        ps.GetEOSMutable().SetCodeUnitSystem(Primitive::MakeCGS());
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Unknown unit system " << units << " requested."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }

      // Get table filename, then read the table,
      std::string fname = pin->GetString(block, "table");
      ps.GetEOSMutable().ReadTableFromFile(fname);

      // Ensure table was read properly
      assert(ps.GetEOSMutable().IsInitialized());
    }
  }

 public:
  Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> ps;
  MeshBlockPack* pmy_pack;
  unsigned int nerrs;
  unsigned int errcap;

  PrimitiveSolverHydro(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
//        pmy_pack(pp), ps{&eos} {
        pmy_pack(pp), nerrs(0) {
    SetPolicyParams(block, pin);
    Real mb = ps.GetEOS().GetBaryonMass();
    ps.GetEOSMutable().SetDensityFloor(pin->GetOrAddReal(block, "dfloor", (FLT_MIN))/mb);
    ps.GetEOSMutable().SetTemperatureFloor(pin->GetOrAddReal(block, "tfloor", (FLT_MIN)));
    ps.GetEOSMutable().SetThreshold(pin->GetOrAddReal(block, "dthreshold", 1.0));
    ps.tol = pin->GetOrAddReal(block, "c2p_tol", 1e-15);
    ps.GetRootSolverMutable().iterations = pin->GetOrAddInteger(block, "c2p_iter", 50);
    errcap = pin->GetOrAddInteger(block, "c2perrs", 1000);

    // Calculate maximum allowed velocity
    Real Wmax = pin->GetOrAddReal(block, "gamma_max", 50.0);
    Real vmax = sqrt(1.0 - 1.0/(Wmax*Wmax));
    ps.GetEOSMutable().SetMaxVelocity(vmax);

    // Set maximum B^2/D
    ps.GetEOSMutable().SetMaximumMagnetization(pin->GetOrAddReal(block, "max_bsq", 1e6));

    for (int n = 0; n < ps.GetEOS().GetNSpecies(); n++) {
      std::stringstream spec_name;
      spec_name << "s" << (n + 1) << "_atmosphere";
      ps.GetEOSMutable().SetSpeciesAtmosphere(
          pin->GetOrAddReal(block, spec_name.str(), 0.0), n);
    }
  }

  // The prim to con function used on the reconstructed states inside the Riemann solver.
  // It also extracts the primitives into a form usable by PrimitiveSolver.
  KOKKOS_INLINE_FUNCTION
  void PrimToConsPt(const ScrArray2D<Real> &w, const ScrArray2D<Real> &brc,
                    const DvceArray4D<Real> &bx,
                    Real prim_pt[NPRIM], Real cons_pt[NCONS], Real b[NMAG],
                    Real g3d[NSPMETRIC], Real sdetg,
                    const int m, const int k, const int j, const int i,
                    const int &nhyd, const int &nscal,
                    const int ibx, const int iby, const int ibz) const {
    auto &eos = ps.GetEOS();
    Real mb = eos.GetBaryonMass();
    // The magnetic field is densitized, but the PrimToCon call
    // needs undensitized variables.
    Real isdetg = 1.0/sdetg;
    Real bin[NMAG];
    bin[ibx] = bx(m, k, j, i)*isdetg;
    bin[iby] = brc(iby, i)*isdetg;
    bin[ibz] = brc(ibz, i)*isdetg;
    Real prim_pt_old[NPRIM];
    prim_pt[PRH] = prim_pt_old[PRH] = w(IDN, i)/mb;
    prim_pt[PVX] = prim_pt_old[PVX] = w(IVX, i);
    prim_pt[PVY] = prim_pt_old[PVY] = w(IVY, i);
    prim_pt[PVZ] = prim_pt_old[PVZ] = w(IVZ, i);
    for (int n = 0; n < nscal; n++) {
      prim_pt[PYF + n] = prim_pt_old[PYF + n] = w(nhyd + n, i);
    }
    prim_pt[PPR] = prim_pt_old[PPR] = w(IPR, i);

    // Apply the floor to make sure these values are physical.
    // FIXME(JF): Is this needed if the first-order flux correction is enabled?
    prim_pt[PTM] = prim_pt_old[PTM] = eos.GetTemperatureFromP(prim_pt[PRH],
                                        prim_pt[PPR], &prim_pt[PYF]);
    bool floored = ps.GetEOS().ApplyPrimitiveFloor(prim_pt[PRH], &prim_pt[PVX],
                                         prim_pt[PPR], prim_pt[PTM], &prim_pt[PYF]);

    ps.PrimToCon(prim_pt, cons_pt, bin, g3d);

    // Check for NaNs
    /*if (CheckForConservedNaNs(cons_pt)) {
      printf("Location: PrimToConsPt\n");
      DumpPrimitiveVars(prim_pt);
    }*/

    // Densitize the variables
    for (int n = 0; n < nhyd + nscal; n++) {
      cons_pt[n] *= sdetg;
    }
    b[ibx] = bx(m, k, j, i);
    b[iby] = brc(iby, i);
    b[ibz] = brc(ibz, i);

    // Previously we checked if the floor was applied and copied these variables back
    // into the original array. However, this is pointless because only the extracted
    // variables in the C-style array are used from this point forward.
  }

  void PrimToCons(DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                  DvceArray5D<Real> &cons,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku) {
    //int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
    //auto &size = pmy_pack->pmb->mb_size;
    //auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
    auto &eos_ = ps.GetEOS();
    auto &ps_  = ps;

    auto &adm = pmy_pack->padm->adm;

    int &nhyd = pmy_pack->pmhd->nmhd;
    int &nscal = pmy_pack->pmhd->nscalars;
    int &nmb = pmy_pack->nmb_thispack;

    Real mb = eos_.GetBaryonMass();


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

      // The magnetic field is densitized, but the PrimToCon calculation is
      // done with undensitized variables.
      Real b[NMAG] = {bcc(m, IBX, k, j, i)/sdetg,
                      bcc(m, IBY, k, j, i)/sdetg,
                      bcc(m, IBZ, k, j, i)/sdetg};

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
      //Real e = prim(m, IDN, k, j, i) + prim(m, IEN, k, j, i);
      //prim_pt[PTM] = eos_.GetTemperatureFromE(prim_pt[PRH], e, &prim_pt[PYF]);
      //prim_pt[PPR] = eos_.GetPressure(prim_pt[PRH], prim_pt[PTM], &prim_pt[PYF]);
      prim_pt[PPR] = prim(m, IPR, k, j, i);

      // Apply the floor to make sure these values are physical.
      prim_pt[PTM] = eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
      bool floor = eos_.ApplyPrimitiveFloor(prim_pt[PRH], &prim_pt[PVX],
                                           prim_pt[PPR], prim_pt[PTM], &prim_pt[PYF]);

      ps_.PrimToCon(prim_pt, cons_pt, b, g3d);

      // Check for NaNs
      if (CheckForConservedNaNs(cons_pt)) {
        Kokkos::printf("Error occurred in PrimToCons at (%d, %d, %d, %d)\n", m, k, j, i);
        DumpPrimitiveVars(prim_pt);
      }

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
        prim(m, IPR, k, j, i) = prim_pt[PPR];
        for (int n = 0; n < nscal; n++) {
          prim(m, nhyd + n, k, j, i) = prim_pt[PYF + n];
        }
      }
    });

    return;
  }

  void ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &bfc,
                  DvceArray5D<Real> &bcc0, DvceArray5D<Real> &prim,
                  const int il, const int iu, const int jl, const int ju,
                  const int kl, const int ku, bool floors_only=false) {
    int &nhyd = pmy_pack->pmhd->nmhd;
    int &nscal = pmy_pack->pmhd->nscalars;
    int &nmb = pmy_pack->nmb_thispack;
    auto &fofc_ = pmy_pack->pmhd->fofc;

    // Some problem-specific parameters
    auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
    auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
    auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
    auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
    auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

    auto &adm  = pmy_pack->padm->adm;
    auto &eos_ = ps.GetEOS();
    auto &ps_  = ps;

    const int ni = (iu - il + 1);
    const int nji = (ju - jl + 1)*ni;
    const int nkji = (ku - kl + 1)*nji;
    const int nmkji = nmb*nkji;

    const int rank = global_variable::my_rank;
    const int nerrs_ = nerrs;
    const int errcap_ = errcap;

    Real mb = eos_.GetBaryonMass();

    // FIXME: This only works for a flooring policy that has these functions!
    bool prim_failure, cons_failure;
    if (floors_only) {
      prim_failure = ps.GetEOSMutable().IsPrimitiveFlooringFailure();
      cons_failure = ps.GetEOSMutable().IsConservedFlooringFailure();
      ps.GetEOSMutable().SetPrimitiveFloorFailure(true);
      ps.GetEOSMutable().SetConservedFloorFailure(true);
    }

    // FIXME(JMF): We can short-circuit the primitive solve if FOFC is already enabled
    // due to a maximum principle violation.
    int count_errs=0;
    Kokkos::parallel_reduce("pshyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, int &sumerrs) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/ni;
      int i = (idx - m*nkji - k*nji - j*ni) + il;
      j += jl;
      k += kl;

      // Add in a short circuit where FOFC is guaranteed.
      if (floors_only && fofc_(m, k, j, i)) {
        return;
      }
      if (floors_only && excise) {
        if (excision_flux_(m,k,j,i)) {
          return;
        }
      }

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
      Real isdetg = 1.0/sdetg;
      adm::SpatialInv(1.0/detg,
                  g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                 &g3u[S11], &g3u[S12], &g3u[S13], &g3u[S22], &g3u[S23], &g3u[S33]);

      // Extract the conserved variables
      Real cons_pt[NCONS], cons_pt_old[NCONS], prim_pt[NPRIM];
      cons_pt[CDN] = cons_pt_old[CDN] = cons(m, IDN, k, j, i)*isdetg;
      cons_pt[CSX] = cons_pt_old[CSX] = cons(m, IM1, k, j, i)*isdetg;
      cons_pt[CSY] = cons_pt_old[CSY] = cons(m, IM2, k, j, i)*isdetg;
      cons_pt[CSZ] = cons_pt_old[CSZ] = cons(m, IM3, k, j, i)*isdetg;
      cons_pt[CTA] = cons_pt_old[CTA] = cons(m, IEN, k, j, i)*isdetg;
      for (int n = 0; n < nscal; n++) {
        cons_pt[CYD + n] = cons_pt_old[CYD + n] = cons(m, nhyd + n, k, j, i)*isdetg;
      }
      // If we're only testing the floors, we can use the CC fields.
      Real b3u[NMAG];
      if (floors_only) {
        b3u[IBX] = bcc0(m, IBX, k, j, i)*isdetg;
        b3u[IBY] = bcc0(m, IBY, k, j, i)*isdetg;
        b3u[IBZ] = bcc0(m, IBZ, k, j, i)*isdetg;
      } else {
        // Otherwise we don't have the correct CC fields yet, so use
        // the FC fields.
        bcc0(m, IBX, k, j, i) = 0.5*(bfc.x1f(m,k,j,i) + bfc.x1f(m,k,j,i+1));
        bcc0(m, IBY, k, j, i) = 0.5*(bfc.x2f(m,k,j,i) + bfc.x2f(m,k,j+1,i));
        bcc0(m, IBZ, k, j, i) = 0.5*(bfc.x3f(m,k,j,i) + bfc.x3f(m,k+1,j,i));
        b3u[IBX] = bcc0(m, IBX, k, j, i)*isdetg;
        b3u[IBY] = bcc0(m, IBY, k, j, i)*isdetg;
        b3u[IBZ] = bcc0(m, IBZ, k, j, i)*isdetg;
      }

      // If we're in an excised region, set the primitives to some default value.
      Primitive::SolverResult result;
      if (excise) {
        if (excision_floor_(m,k,j,i)) {
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
          prim_pt[PTM] =
            eos_.GetTemperatureFromP(prim_pt[PRH], prim_pt[PPR], &prim_pt[PYF]);
          result.error = Primitive::Error::SUCCESS;
          result.iterations = 0;
          result.cons_floor = false;
          result.prim_floor = false;
          result.cons_adjusted = true;
          ps_.PrimToCon(prim_pt, cons_pt, b3u, g3d);
        } else {
          result = ps_.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);
        }
      } else {
        result = ps_.ConToPrim(prim_pt, cons_pt, b3u, g3d, g3u);
      }

      if (result.error != Primitive::Error::SUCCESS && floors_only) {
        fofc_(m,k,j,i) = true;
      } else if (!floors_only) {
        if (result.error != Primitive::Error::SUCCESS && (nerrs_ + sumerrs < errcap_)) {
          // TODO(JF): put in a proper error response here.
          sumerrs++;
          Kokkos::printf("An error occurred during the primitive solve: %s\n"
                 "  Location: (%d, %d, %d, %d)\n"
                 "  Conserved vars: \n"
                 "    D   = %.17g\n"
                 "    Sx  = %.17g\n"
                 "    Sy  = %.17g\n"
                 "    Sz  = %.17g\n"
                 "    tau = %.17g\n"
                 "    Dye = %.17g\n"
                 "    Bx  = %.17g\n"
                 "    By  = %.17g\n"
                 "    Bz  = %.17g\n"
                 "  Metric vars: \n"
                 "    detg = %.17g\n"
                 "    g_dd = {%.17g, %.17g, %.17g, %.17g, %.17g, %.17g}\n"
                 "    alp  = %.17g\n"
                 "    beta = {%.17g, %.17g, %.17g}\n"
                 "    psi4 = %.17g\n"
                 "    K_dd = {%.17g, %.17g, %.17g, %.17g, %.17g, %.17g}\n",
                 ErrorToString(result.error),
                 m, k, j, i,
                 cons_pt_old[CDN], cons_pt_old[CSX], cons_pt_old[CSY], cons_pt_old[CSZ],
                 cons_pt_old[CTA], cons_pt_old[CYD], b3u[IBX], b3u[IBY], b3u[IBZ], detg,
                 g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33],
                 adm.alpha(m, k, j, i),
                 adm.beta_u(m, 0, k, j, i),
                 adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                 adm.psi4(m, k, j, i),
                 adm.vK_dd(m, 0, 0, k, j, i), adm.vK_dd(m, 0, 1, k, j, i),
                 adm.vK_dd(m, 0, 2, k, j, i),
                 adm.vK_dd(m, 1, 1, k, j, i), adm.vK_dd(m, 1, 2, k, j, i),
                 adm.vK_dd(m, 2, 2, k, j, i));
          if (nerrs_ + sumerrs == errcap_) {
            Kokkos::printf("%d C2P errors have been detected on rank %d."
                   "All future C2P errors\n"
                   "on this rank will be suppressed. Fix your code!\n",
                   nerrs_ + sumerrs,rank);
          }
        }
        // Regardless of failure, we need to copy the primitives.
        prim(m, IDN, k, j, i) = prim_pt[PRH]*mb;
        prim(m, IVX, k, j, i) = prim_pt[PVX];
        prim(m, IVY, k, j, i) = prim_pt[PVY];
        prim(m, IVZ, k, j, i) = prim_pt[PVZ];
        prim(m, IPR, k, j, i) = prim_pt[PPR];
        for (int n = 0; n < nscal; n++) {
          prim(m, nhyd + n, k, j, i) = prim_pt[PYF + n];
        }

        // If the conservative variables were floored or adjusted for consistency,
        // we need to copy the conserved variables, too.
        if (result.cons_floor || result.cons_adjusted) {
          /*if (fabs((cons_pt[CDN] - cons_pt_old[CDN])/cons_pt_old[CDN]) > 1e-12) {
            Real &x1min = size.d_view(m).x1min;
            Real &x1max = size.d_view(m).x1max;
            Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

            Real &x2min = size.d_view(m).x2min;
            Real &x2max = size.d_view(m).x2max;
            Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

            Real &x3min = size.d_view(m).x3min;
            Real &x3max = size.d_view(m).x3max;
            Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
            bool is_ghost = (i < is) || (i > ie) ||
                            (j < js) || (j > je) ||
                            (k < ks) || (k > ke);

            printf("Density was nontrivially adjusted on MeshBlock %d!\n"
                   "  Grid index: (i=%d, j=%d, k=%d)\n"
                   "  Physical position: (%g, %g, %g)\n"
                   "  D (old): %.17g\n"
                   "  D (new): %.17g\n"
                   "  Ghost zone? %s\n",
                   m, i, j, k,
                   x1v, x2v, x3v, cons_pt_old[CDN], cons_pt[CDN],
                   is_ghost ? "true" : "false");
          }*/
          cons(m, IDN, k, j, i) = cons_pt[CDN]*sdetg;
          cons(m, IM1, k, j, i) = cons_pt[CSX]*sdetg;
          cons(m, IM2, k, j, i) = cons_pt[CSY]*sdetg;
          cons(m, IM3, k, j, i) = cons_pt[CSZ]*sdetg;
          cons(m, IEN, k, j, i) = cons_pt[CTA]*sdetg;
          for (int n = 0; n < nscal; n++) {
            cons(m, nhyd + n, k, j, i) = cons_pt[CYD + n]*sdetg;
          }
        }
      }
    }, Kokkos::Sum<int>(count_errs));

    if (floors_only) {
      ps.GetEOSMutable().SetPrimitiveFloorFailure(prim_failure);
      ps.GetEOSMutable().SetConservedFloorFailure(cons_failure);
    } else {
      nerrs += count_errs;
    }
  }

  // Get the transformed magnetosonic speeds at a point in a given direction.
  KOKKOS_INLINE_FUNCTION
  void GetGRFastMagnetosonicSpeeds(Real& lambda_p, Real& lambda_m,
                                   Real prim[NPRIM], Real bsq, Real g3d[NSPMETRIC],
                                   Real beta_u[3], Real alpha, Real gii,
                                   int pvx) const {
    Real uu[3] = {prim[PVX], prim[PVY], prim[PVZ]};
    Real usq = Primitive::SquareVector(uu, g3d);
    int index = pvx - PVX;

    // Get spacetime quantities
    Real Wsq = 1.0 + usq;
    Real ialpha = 1.0/alpha;
    Real W = sqrt(Wsq);
    Real u0 = W*ialpha;
    Real u1 = uu[index] - u0*beta_u[index];
    Real g00 = -ialpha*ialpha;
    Real g01 = -g00*beta_u[index];
    Real g11 = gii - g01*beta_u[index];

    // Calculate the sound speed and the Alfven speed
    Real cs = ps.GetEOS().GetSoundSpeed(prim[PRH], prim[PTM], &prim[PYF]);
    Real csq = cs*cs;
    Real H = ps.GetEOS().GetBaryonMass()*prim[PRH]*
             ps.GetEOS().GetEnthalpy(prim[PRH], prim[PTM], &prim[PYF]);
    Real vasq = bsq/(bsq + H);
    Real cmsq = csq + vasq - csq*vasq;

    // Set fast magnetosonic speed in appropriate coordinates
    Real a = u0*u0 - (g00 + u0*u0)*cmsq;
    Real b = -2.0 * (u0 * u1 - (g01 + u0 * u1) *cmsq);
    Real c = u1*u1 - (g11 + u1*u1)*cmsq;
    Real a1 = b / a;
    Real a0 = c / a;
    Real s = fmax(a1*a1 - 4.0 * a0, 0.0);
    s = sqrt(s);
    lambda_p = (a1 >= 0.0) ? -2.0 * a0 / (a1 + s) : (-a1 + s) / 2.0;
    lambda_m = (a1 >= 0.0) ? (-a1 - s) / 2.0 : -2.0 * a0 / (a1 - s);
  }

  // A function for converting PrimitiveSolver errors to strings
  KOKKOS_INLINE_FUNCTION
  static const char * ErrorToString(Primitive::Error e) {
    switch(e) {
      case Primitive::Error::SUCCESS:
        return "SUCCESS";
        break;
      case Primitive::Error::RHO_TOO_BIG:
        return "RHO_TOO_BIG";
        break;
      case Primitive::Error::RHO_TOO_SMALL:
        return "RHO_TOO_SMALL";
        break;
      case Primitive::Error::NANS_IN_CONS:
        return "NANS_IN_CONS";
        break;
      case Primitive::Error::MAG_TOO_BIG:
        return "MAG_TOO_BIG";
        break;
      case Primitive::Error::BRACKETING_FAILED:
        return "BRACKETING_FAILED";
        break;
      case Primitive::Error::NO_SOLUTION:
        return "NO_SOLUTION";
        break;
      default:
        return "OTHER";
        break;
    }
  }

  // A function for checking for NaNs in the conserved variables.
  KOKKOS_INLINE_FUNCTION
  static int CheckForConservedNaNs(const Real cons_pt[NCONS]) {
    int nans = 0;
    if (!isfinite(cons_pt[CDN])) {
      Kokkos::printf("D is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CSX])) {
      Kokkos::printf("Sx is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CSY])) {
      Kokkos::printf("Sy is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CSZ])) {
      Kokkos::printf("Sz is NaN!\n"); // NOLINT
      nans = 1;
    }
    if (!isfinite(cons_pt[CTA])) {
      Kokkos::printf("Tau is NaN!\n"); // NOLINT
      nans = 1;
    }

    return nans;
  }

  KOKKOS_INLINE_FUNCTION
  static void DumpPrimitiveVars(const Real prim_pt[NPRIM]) {
    Kokkos::printf("Primitive vars: \n" // NOLINT
           "  rho = %.17g\n"
           "  ux  = %.17g\n"
           "  uy  = %.17g\n"
           "  uz  = %.17g\n"
           "  P   = %.17g\n"
           "  T   = %.17g\n",
           prim_pt[PRH], prim_pt[PVX], prim_pt[PVY],
           prim_pt[PVZ], prim_pt[PPR], prim_pt[PTM]);
  }
};
#endif  // EOS_PRIMITIVE_SOLVER_HYD_HPP_
