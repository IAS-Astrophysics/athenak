//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file create_mpitag.hpp

//----------------------------------------------------------------------------------------
//! \fn int CreateMPITag(int lid, int bufid, int phys)
//  \brief calculate an MPI tag for boundary buffer communications
//  MPI tag = lid (remaining bits) + bufid (6 bits) + physics(4 bits)
//  Note the convention in Athena++ is lid and bufid are both for the *receiving* process

// WARNING: Generating unsigned integer bitfields from signed integer types and converting
// output to signed integer tags (required by MPI) may lead to unsafe conversions (and
// overflows from built-in types and MPI_TAG_UB).  Note, the MPI standard requires signed
// int tag, with MPI_TAG_UB>= 2^15-1 = 32,767 (inclusive)

static int CreateMPITag(int lid, int bufid, int phys)
{
  return (lid<<10) | (bufid<<4) | phys;
}
