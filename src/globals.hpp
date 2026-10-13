#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file globals.hpp
//  \brief namespace containing external global variables

namespace global_variable {
extern int my_rank, nranks;
extern int mpi_tag_lid_bits;
// Six bits encode the receiving boundary buffer. Reserve only as many local-ID
// bits as the MPI implementation actually supports (some Aurora MPICH builds
// provide 18 tag bits, below the old hard-coded 20-bit assumption).
inline int MPILocalIDBits(int tag_upper_bound, int maximum_bits) {
  int bits=maximum_bits;
  while (bits>0 && ((1ULL << (bits+6))-1) >
                   static_cast<unsigned long long>(tag_upper_bound)) --bits;
  return bits;
}
}

#endif // GLOBALS_HPP_
