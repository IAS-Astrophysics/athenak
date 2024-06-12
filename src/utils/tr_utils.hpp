#ifndef UTILS_TR_UTILS_HPP_
#define UTILS_TR_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tr_utils.hpp
//! \brief Various utility functions needed by the table reader which aren't necessarily
//!        specific to the table reader itself.
#include <climits>

namespace TableReader {

// This function swaps the endianness of an input type. Shamelessly stolen from this
// Stack Overflow answer: https://stackoverflow.com/a/4956493/9715636
// On some compilers there are built-in functions to do this which are certainly faster,
// but this is more portable.
template<typename T>
T SwapEndianness(T var) {
  static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");

  // FIXME: This may be undefined behavior in C++!
  union {
    T var;
    unsigned char u8[sizeof(T)];
  } source, dest;

  source.var = var;

  for (size_t k = 0; k < sizeof(T); k++) {
    dest.u8[k] = source.u8[sizeof(T) - k - 1];
  }

  return dest.var;
  // Alternative method?
  //unsigned char su8[sizeof(T)], du8[sizeof(T)];
  //std::memcpy(su8, &var);
  //for (int k = 0; k < sizeof(T); k++) {
  //  du8[k] = su8[sizeof(T)-k-1];
  //}
  //T dest;
  //std::memcpy(&dest, du8);
  //return dest
}

inline bool IsLittleEndian() {
  const int value = 0x01;
  const void * address = static_cast<const void*>(&value);
  const unsigned char *lsa = static_cast<const unsigned char*>(address);
  return *lsa == 0x01;
}

} // namespace TableReader

#endif // UTILS_TR_UTILS_HPP_
