#ifndef NR_RADIATION_ANGULAR_GRID_HPP_
#define NR_RADIATION_ANGULAR_GRID_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file angular_grid.hpp
//! \brief Angular quadrature grid for the LTE short-characteristics SC solver.
//!
//! Implements the Bruls et al. (1999, A&A 348, 233) "type-A" discrete-ordinate grid
//! (Carlson 1963 symmetric S_N method), as described in Davis, Stone & Jiang (2012)
//! Sec. 3.2. The construction (permutation-family weight solve, n_mu <= 6) follows
//! Athena-C's radiation/angles.c::carlson().

#include "athena.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \class SCAngularGrid
//! \brief discrete-ordinate angular grid: direction cosines and quadrature weights

class SCAngularGrid {
 public:
  SCAngularGrid(int ndim, int nmu);

  int ndim;   // 1, 2, or 3
  int nmu;    // input <nr_radiation>/nmu parameter (number of polar levels)
  int nang;   // number of unique rays per octant
  int noct;   // number of octants: 2 (1D), 4 (2D), 8 (3D)

  // mu(oct,ang,0..2) = direction cosines (n-hat . x1hat, x2hat, x3hat) of ray 'ang' in
  // octant 'oct'; wmu(ang) = quadrature weight (same for every octant, already includes
  // the 1/noct octant-degeneracy factor, so moments are plain sums over (oct,ang) with
  // no extra prefactor -- Eq. 17-19 of Davis 2012).
  DualArray3D<Real> mu;
  DualArray1D<Real> wmu;

 private:
  void BuildCarlson1D();
  void BuildCarlsonND();
};

}  // namespace nr_radiation

#endif  // NR_RADIATION_ANGULAR_GRID_HPP_
