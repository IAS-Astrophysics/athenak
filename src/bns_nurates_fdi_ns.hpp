#ifndef BNS_NURATES_FDI_NS_HPP_
#define BNS_NURATES_FDI_NS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bns_nurates_fdi_ns.hpp
//  \brief bns_nurates' Fermi-Dirac integrals and constants, in namespace bns_nurates.
//
//  This and bns_nurates_ns.hpp, which pulls the rest of the API on top of it, are the
//  ONLY two files that may include a bns_nurates header. Two translation units that
//  disagree about whether the API is namespaced would silently get two sets of inline
//  functions, so
//
//      grep -rn 'bns_nurates/include' src/ | grep -v bns_nurates_
//
//  must come back empty.
//
//  Include this one, not bns_nurates_ns.hpp, wherever only the FDI_* functions are
//  wanted. It stops short of m1_opacities.hpp, whose kernels reach GP19Table.hpp -- 13 MB
//  of constexpr tables, which every translation unit including the header would
//  otherwise have to parse. eos_compose.hpp is reached by most of dyn_grmhd, so that
//  matters there.
//
//  The system headers below are pre-included at global scope so that the copies nested
//  inside the namespace hit their own include guards and expand to nothing; without
//  them size_t, printf and std land in the wrong scope. Re-derive the list on every
//  bns_nurates bump with
//
//      grep -rhoE '#include *<[^>]+>' bns_nurates/include/*.hpp | sort -u

#include "config.hpp"

#if ENABLE_NURATES

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>

namespace bns_nurates {
#include "bns_nurates/include/bns_nurates.hpp"      // NOLINT(build/include)
#include "bns_nurates/include/constants.hpp"        // NOLINT(build/include)
#include "bns_nurates/include/functions.hpp"        // NOLINT(build/include)
}  // namespace bns_nurates

#endif  // ENABLE_NURATES
#endif  // BNS_NURATES_FDI_NS_HPP_
