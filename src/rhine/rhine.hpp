#ifndef RHINE_RHINE_HPP_
#define RHINE_RHINE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rhine.hpp
//! \brief Definitions for the RHINE class (neural-network r-process heating source term),
//!        coupled to the transition EOS with a 7-species mass-fraction composition.

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "rhine_net.hpp"

// Forward declarations.
class MeshBlockPack;
class Driver;

namespace rhine {

//----------------------------------------------------------------------------------------
//! Passive-scalar layout: matches the transition EOS composition Y[SCYE..SCEB],
//! offset by IYF into the MHD conserved/primitive arrays (r_0..r_6).
enum RhineScalar {
  I_YE = IYF,       // electron fraction              (SCYE)
  I_XN = IYF + 1,   // free-neutron mass fraction      (SCXN)
  I_XP = IYF + 2,   // free-proton mass fraction       (SCXP)
  I_XA = IYF + 3,   // alpha-particle mass fraction    (SCXA)
  I_XH = IYF + 4,   // heavy-nucleus mass fraction      (SCXH)
  I_AH = IYF + 5,   // average heavy mass number        (SCAH)
  I_EB = IYF + 6    // binding-energy factor E_B        (SCEB)
};

//----------------------------------------------------------------------------------------
//! Auxiliary (diagnostic) output channels; mirrors GR-Athena++ IX_HEAT..IX_DMA.
enum RhineAux {
  A_HEAT = 0,   // heating rate per unit volume [erg/cm^3/s]
  A_TRANS,      // EOS transition weight
  A_XERR,       // sum of mass fractions - 1
  A_FNU,        // neutrino-loss fraction of the nuclear release
  A_QDOT,       // densitized heating rate [code energy / code time / vol]
  A_LNU,        // densitized neutrino-loss rate [code units]
  A_DYE,        // network rate dYe/dt  [1/s]
  A_DYN,        // network rate dXn/dt  [1/s]
  A_DYP,        // network rate dXp/dt  [1/s]
  A_DYA,        // network rate dYalpha/dt [1/s]
  A_DYH,        // network rate dYh/dt  [1/s]
  A_DAH,        // network rate dAh/dt  [1/s]
  A_DMA,        // network rate dma/dt  [MeV/baryon/s]
  A_QPHYS,      // densitized heating from the reaction increments only
                // (A_QDOT - A_QPHYS - A_QEXIT is the mixing release)
  A_NSEST,      // NSE crossing state: 0 unknown (init/restart/regrid), 1 out of NSE and
                // released, 2 resynced and not yet released, 3 first release this step
  A_QEXIT,      // densitized RHINE heating of cells making their first release since the
                // last resync
  A_QREENT,     // densitized heating by the NSE resync of cells that re-entered NSE after
                // a release, -D W dE_B/dt
  A_QNSE,       // same for EVERY resynced cell (binding-energy shifts inside NSE)
  N_RHINE_AUX
};

constexpr int N_RHINE_SCALARS = 7;

//----------------------------------------------------------------------------------------
//! Which network pass to run. GR-Athena++ evaluates the network twice per step:
//! `source` = TransitionNetworkApply (per substep, back-reaction, '0' reference frozen at
//! full-step start); `poststep` = Mesh::CalculateTransitionNetwork (once per step after the
//! final C2P: NSE resync, then diagnostics-only with the current state as reference);
//! `diag` = the diagnostics-only network evaluation alone (no resync).
enum class RhinePass {source, poststep, diag};

//----------------------------------------------------------------------------------------
//! \class RHINE
//! \brief Operator-split r-process heating + composition source term for the transition EOS.
class RHINE {
 public:
  RHINE(MeshBlockPack *ppack, ParameterInput *pin);
  ~RHINE();

  //! Per-substep pass: add RHINE source terms to the MHD conserved variables.
  TaskStatus AddSources(Driver *pdrive, int stage);

  //! Once-per-step pass (final stage only): NSE resync over the full block, then
  //! the diagnostics-only network evaluation.
  TaskStatus PostStep(Driver *pdrive, int stage);

  //! Diagnostics-only network evaluation on the current state (no resync). Used by
  //! Driver::Finalize when the last PostStep skipped it, so final outputs see fresh aux.
  void Diagnostics(Driver *pdrive);

  RhineNets nets;
  int  pmode;
  bool apply;        // back-react on the conserved variables (else diagnostics only)
  bool clamp_cold;   // evaluate at T9 = 0.1 below RHINE's cutoff while free n remain
  bool fnu_physical; // apply the neutrino fraction to the reaction release only
  bool use_nqt;      // EOS LogPolicy selector (mirrors <mhd> use_NQT)
  bool post_after_integrator = false;
  bool diag_due = true;
  bool diag_fresh = false;
  int  amr_seq = 0;  // mesh-update sequence the NSE-crossing state belongs to

  // Diagnostic array (nmb, N_RHINE_AUX, nc3, nc2, nc1).
  DvceArray5D<Real> aux;
  // Composition frozen at full-step start; the network's '0' reference in the source
  // pass (GR-Athena++ pscalars->r0). (nmb, N_RHINE_SCALARS, nc3, nc2, nc1).
  DvceArray5D<Real> r0;

  //! `dt_apply_code <= 0` makes the evaluation diagnostics-only.
  template<class EOSPolicy, class ErrorPolicy>
  TaskStatus NetworkStepEOS(Real dt_apply_code);

  template<class EOSPolicy, class ErrorPolicy>
  TaskStatus NSEResyncEOS();

  void SnapshotReference();
  void ResetCrossingStateOnRegrid();

 private:
  //! Resolve the EOS/error policy template arguments and run `pass`.
  TaskStatus Dispatch(Driver *pdrive, int stage, RhinePass pass);

  template<class EOSPolicy, class ErrorPolicy>
  TaskStatus RunPassEOS(RhinePass pass, Real dt_apply_code);

  MeshBlockPack *pmy_pack;
};

}  // namespace rhine

#endif  // RHINE_RHINE_HPP_
