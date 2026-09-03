#ifndef EOS_PRIMITIVE_SOLVER_PS_TYPES_HPP_
#define EOS_PRIMITIVE_SOLVER_PS_TYPES_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ps_types.hpp
//  \brief contains some basic type definitions consistent with Athena++.
//
//  Ideally this file shouldn't be required when the code is dropped into Athena.
//  Therefore, all type definitions should be consistent with Athena.
//

#include "athena.hpp"

#define MAX_SPECIES 3
#define NHYDRO ((5) + (MAX_SPECIES))

enum ConsIndex {CDN=0, CSX=1, CSY=2, CSZ=3, CTA=4, CYD=5, NCONS=(NHYDRO)};
// FIXME: Make sure that the position of IYF makes sense.
// It should be okay, since if we're not using any species,
// IBY gets aliased to 6, and IYF should never get called.
// Note that NPRIM does not include IBY and IBZ because NHYDRO doesn't.
enum PrimIndex {PRH=0,PVX=1, PVY=2, PVZ=3, PPR=4, PTM=5, PYF=6, NPRIM=((NHYDRO)+1)};
enum SpatialMetricIndex{S11=0, S12=1, S13=2, S22=3, S23=4, S33=5, NSPMETRIC=6};

class SupportsEntropy{};
class SupportsChemicalPotentials{};

//! Indices into the weight array of the partially-equilibrated weak-equilibrium
//! residuals: one weight per neutrino channel, per species wherever the two members of
//! a pair have different equilibrium densities. Dimensionless and in [0, 1]; all five
//! equal to 1 is the fully trapped equilibrium and all five 0 leaves the matter state
//! untouched.
//
//  The heavy-lepton pairs share one weight because nothing in the opacities
//  distinguishes nu_x from its antiparticle: any difference between the two is
//  numerical, not physical. That will stop being true as more physics goes in, and at
//  that point they need splitting too -- eta = 0 makes their *equilibrium* densities
//  coincide, which is not on its own enough, since the weight also multiplies the
//  actual densities and those differ once radiation is advected or floored.
//
//  The residuals consume the pair mean and half-difference,
//
//      wbar_c = (w_nue + w_anue)/2,   dw_c = (w_nue - w_anue)/2
//
//  which is what makes the closed-form polynomials usable: wbar multiplies them
//  unchanged and dw multiplies the two combinations that are not polynomial. See
//  EOSCompOSE::BetaEquilibriumPartial.
enum PeqWeightIndex {
  PEQ_W1_NUE   = 0,  //! nu_e     energy density
  PEQ_W1_ANUE  = 1,  //! nubar_e  energy density
  PEQ_W1_X     = 2,  //! heavy-lepton pair energy densities
  PEQ_W0_NUE   = 3,  //! nu_e     number density
  PEQ_W0_ANUE  = 4,  //! nubar_e  number density
  PEQ_NWEIGHTS = 5
};

#endif  // EOS_PRIMITIVE_SOLVER_PS_TYPES_HPP_
