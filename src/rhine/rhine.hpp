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
  N_RHINE_AUX
};

//----------------------------------------------------------------------------------------
//! \class RHINE
//! \brief Operator-split r-process heating + composition source term for the transition EOS.
class RHINE {
 public:
  RHINE(MeshBlockPack *ppack, ParameterInput *pin);
  ~RHINE();

  //! Apply RHINE per cell and add source terms to the MHD conserved variables.
  TaskStatus AddSources(Driver *pdrive, int stage);

  RhineNets nets;
  int  pmode;
  bool apply;        // back-react on the conserved variables (else diagnostics only)
  bool use_nqt;      // EOS LogPolicy selector (mirrors <mhd> use_NQT)

  // Diagnostic array (nmb, N_RHINE_AUX, nc3, nc2, nc1).
  DvceArray5D<Real> aux;

 private:
  //! EOS-typed implementation dispatched from AddSources.
  template<class EOSPolicy, class ErrorPolicy>
  TaskStatus AddSourcesEOS(Driver *pdrive, int stage);

  MeshBlockPack *pmy_pack;
};

}  // namespace rhine

#endif  // RHINE_RHINE_HPP_
