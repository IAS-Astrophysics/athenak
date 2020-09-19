#ifndef UTILS_GRID_LOCATIONS_HPP_
#define UTILS_GRID_LOCATIONS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grid_locations.cpp
//  \brief functions to compute locations on a uniform Cartesian grid
// They provide functionality of the Coordinates class in the C++ version of the code.

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn void LeftEdgeX()
// returns x-posn of left edge of i^th cell of N in range xmin->xmax
// Averages of linear interpolation from each side used to symmetrize r.o. error

static inline Real LeftEdgeX(int ith, int n, Real xmin, Real xmax)
{
  Real x = (static_cast<Real>(ith)) / (static_cast<Real>(n));
  return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenterX()
// returns cell center position of i^th cell of N in range xmin->xmax
// Averages of linear interpolation from each side used to symmetrize r.o. error

static inline Real CellCenterX(int ith, int n, Real xmin, Real xmax)
{
  Real x = (static_cast<Real>(ith) + 0.5) / (static_cast<Real>(n));
  return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenterIndex()
// returns i-index of cell containing x position

// TODO: set trap if out-of-range
static inline int CellCenterIndex(Real x, int n, Real xmin, Real xmax)
{
  return static_cast<int>(((x-xmin)/(xmax-xmin))*static_cast<Real>(n));
}

#endif // UTILS_GRID_LOCATIONS_HPP_
