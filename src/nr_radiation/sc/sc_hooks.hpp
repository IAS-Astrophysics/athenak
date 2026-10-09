#ifndef NR_RADIATION_SC_SC_HOOKS_HPP_
#define NR_RADIATION_SC_SC_HOOKS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sc_hooks.hpp
//! \brief User-enrollable opacity and emission hooks for the SC radiation module: host
//! function pointers enrolled from the problem generator, called at the start of every
//! solve, each filling its array over all cells including ghosts.

class MeshBlockPack;

namespace nr_radiation {

// user-enrollable opacity function: fills sigma_a over all cells incl ghosts
using SCOpacityFnPtr = void (*)(MeshBlockPack *pmbp);
// user-enrollable emission function: fills brad (the LTE source) over all cells
// including ghosts
using SCEmissionFnPtr = void (*)(MeshBlockPack *pmbp);

}  // namespace nr_radiation

#endif  // NR_RADIATION_SC_SC_HOOKS_HPP_
