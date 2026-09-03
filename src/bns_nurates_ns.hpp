#ifndef BNS_NURATES_NS_HPP_
#define BNS_NURATES_NS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bns_nurates_ns.hpp
//  \brief The whole bns_nurates API, in namespace bns_nurates.
//
//  bns_nurates declares its API at global scope, including thirteen unprefixed constants
//  (one, two, zero, five, ten, one_third, ...) that athenak also uses as local variable
//  names in more than fifty files. Enclosing the includes here keeps them apart.
//
//  This and bns_nurates_fdi_ns.hpp are the ONLY two files that may include a
//  bns_nurates header; that header carries the invariant and the system-header list, and
//  is included first here so both spellings agree on them. Where only the Fermi-Dirac
//  integrals are wanted, include the fdi header instead -- it is much cheaper to parse.

#include "bns_nurates_fdi_ns.hpp"

#if ENABLE_NURATES

namespace bns_nurates {
#include "bns_nurates/include/distribution.hpp"     // NOLINT(build/include)
#include "bns_nurates/include/integration.hpp"      // NOLINT(build/include)
#include "bns_nurates/include/m1_opacities.hpp"     // NOLINT(build/include)
}  // namespace bns_nurates

#endif  // ENABLE_NURATES
#endif  // BNS_NURATES_NS_HPP_
