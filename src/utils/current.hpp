#ifndef UTILS_CURRENT_HPP_
#define UTILS_CURRENT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file current.hpp
//  \brief helper functions to compute current

//----------------------------------------------------------------------------------------
// Parity calculation due to Norm Hardy. Templated over number of dimensions n.
template<int n>
KOKKOS_INLINE_FUNCTION int pp(int P[n]) {
  int x;
  int p = 0;
  int v[n] = {0};

  for (int j = 0; j < n; ++j) {
    if (v[j]) {
      p++;
    } else {
      x = j;
      do {
        x = P[x];
        v[x] = 1;
      } while (x != j);
    }
  }

  if (p % 2 == 0) {
    return 1;
  } else {
    return -1;
  }
}

//----------------------------------------------------------------------------------------
// Antisymmetric symbol in 4 dimensions
KOKKOS_INLINE_FUNCTION int antisym(int a, int b, int c, int d) {
  // Entries must be different
  if (a == b) return 0;
  if (a == c) return 0;
  if (a == d) return 0;
  if (b == c) return 0;
  if (b == d) return 0;
  if (c == d) return 0;

  // Determine parity of permutation
  int p[4] = {a, b, c, d};

  return pp<4>(p);
}

//----------------------------------------------------------------------------------------
// Compute contravariant (mu, nu) component of Faraday tensor multiplied by detg
KOKKOS_INLINE_FUNCTION double get_detg_Fcon(int mu, int nu, const Real ucov[4],
                                            const Real bcov[4]) {
    if (mu == nu) return 0.;

    double Fcon = 0.;

    for (int kap = 0; kap < 4; ++kap) {
        for (int lam = 0; lam < 4; ++lam) {
            Fcon -= antisym(mu, nu, kap, lam) * ucov[kap] * bcov[lam];
        }
    }

    return Fcon;
}

#endif // UTILS_CURRENT_HPP_
