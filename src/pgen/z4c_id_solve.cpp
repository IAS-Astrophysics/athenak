//========================================================================================
// AthenaK: astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License
//========================================================================================
//! \file z4c_id_solve.cpp
//! \brief Compatibility entry point for the id_solve initial-data reader.

// The current branch carries the maintained reader as pgen/id_solve.cpp.  The
// remote id_solve branch introduced the public PROBLEM name z4c_id_solve, so
// keep that entry point while avoiding a second copy of the reader.
#include "id_solve.cpp"
