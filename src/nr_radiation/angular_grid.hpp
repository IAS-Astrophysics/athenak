#ifndef NR_RADIATION_ANGULAR_GRID_HPP_
#define NR_RADIATION_ANGULAR_GRID_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file angular_grid.hpp
//! \brief Angular quadrature grid for the SC solver: Bruls et al. (1999) type-A set,
//! n_mu <= 6.

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
  // octant 'oct'; wmu(ang) = quadrature weight, the same in every octant and summing to
  // one over all (oct, ang) because the 1/noct octant factor is folded in. The moments
  // are therefore plain sums, J = Sum wmu I, with no normalisation in front.
  DualArray3D<Real> mu;
  DualArray1D<Real> wmu;

 private:
  void BuildCarlson1D();
  void BuildCarlsonND();
};

}  // namespace nr_radiation

#endif  // NR_RADIATION_ANGULAR_GRID_HPP_
