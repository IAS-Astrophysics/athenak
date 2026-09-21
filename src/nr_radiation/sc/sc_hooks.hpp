#ifndef NR_RADIATION_SC_SC_HOOKS_HPP_
#define NR_RADIATION_SC_SC_HOOKS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_hooks.hpp
//! \brief User-enrollable opacity and emission hooks for the SC radiation module.
//!
//! Both hooks are HOST function pointers, enrolled from the problem generator via
//! SC::EnrollOpacityFunction / SC::EnrollEmissionFunction (pmbp->pnrrad->Enroll...).
//! The enrolled function is called on the host from SC::UpdateOpacity /
//! SC::UpdateEmission (at the top of every SolveTransfer, before the iteration loop;
//! UpdateOpacity also runs at initialization, after a remesh, and in SC::NewTimeStep) and
//! must launch its own par_for over the mesh. Parameters reach the kernel via the usual
//! pgen idiom: a file-scope POD struct copied to a local and captured by value.
//!
//! sigma_a and brad are recomputed at the start of every solve: by the enrolled hook,
//! else from the fluid (sigma_a = kappa_a * rho, brad = T^4). Without a fluid both hooks
//! are required (a startup error otherwise). A direct write to either array from a pgen
//! is overwritten before the first sweep; a static pattern is expressed as a hook that
//! rewrites it each solve (two cell-array fills, negligible next to the sweep).
//!
//! Contract:
//!  - Fill ALL cells INCLUDING ghosts (loop bounds 0..ncells-1 in every active
//!    dimension).
//!    sigma_a and brad have no ghost-zone communication; the sweep reads both at ghost
//!    footpoints, so stale ghosts corrupt the formal solution.
//!  - Opacity hook: fill sigma_a(m,0,k,j,i), the absorption coefficient per unit volume.
//!    The module then sets chi = sigma_a. Do NOT write chi directly. When an opacity hook
//!    is enrolled, <nr_radiation>/kappa_a is not read.
//!  - Emission hook: fill brad(m,k,j,i), the LTE source function per cell: the
//!    emissivity divided by the absorption coefficient (NOT the emissivity itself), in
//!    intensity units. Default without a hook is the Planck function of the gas, T^4 in
//!    code units. A hook may write anything (a temperature floor, a modified thermal
//!    law, a non-thermal source). In LTE the source iterate is S = brad exactly.
//!    Note: Athena++'s EnrollEmissionFunction is a different contract (per-group
//!    fractions of a fixed T^4 total); this hook fills the source per cell.
//!  - Enroll BEFORE any `if (restart) return;` in the pgen so restarted runs re-enroll
//!    (the pgen runs on the restart path with restart=true).

class MeshBlockPack;

namespace nr_radiation {

// user-enrollable opacity function: fills sigma_a over all cells incl ghosts
using SCOpacityFnPtr = void (*)(MeshBlockPack *pmbp);
// user-enrollable emission function: fills brad (the LTE source) over all cells
// including ghosts
using SCEmissionFnPtr = void (*)(MeshBlockPack *pmbp);

}  // namespace nr_radiation

#endif  // NR_RADIATION_SC_SC_HOOKS_HPP_
