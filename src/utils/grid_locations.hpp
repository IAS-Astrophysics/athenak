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
// returns x-posn of left edge of i^th cell where index range [0,N] maps to [xmin,xmax]
// Works for ghost cells if i outside range [0,N] (e.g. i=-1 is x-posn of first ghost cell)
// Averages of linear interpolation from each side used to symmetrize r.o. error

KOKKOS_INLINE_FUNCTION
static Real LeftEdgeX(int ith, int n, Real xmin, Real xmax)
{
  Real x = (static_cast<Real>(ith)) / (static_cast<Real>(n));
  return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenterX()
// returns cell-center posn of i^th cell where index range [0,N] maps to [xmin,xmax]
// Works for ghost cells if i outside range [0,N] (e.g. i=-1 is cc-posn of first ghost cell)
// Averages of linear interpolation from each side used to symmetrize r.o. error

KOKKOS_INLINE_FUNCTION
static Real CellCenterX(int ith, int n, Real xmin, Real xmax)
{
  Real x = (static_cast<Real>(ith) + 0.5) / (static_cast<Real>(n));
  return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
}

//----------------------------------------------------------------------------------------
//! \fn void CellCenterIndex()
// returns i-index of cell containing x position

// TODO: set trap if out-of-range

KOKKOS_INLINE_FUNCTION
static int CellCenterIndex(Real x, int n, Real xmin, Real xmax)
{
  return static_cast<int>(((x-xmin)/(xmax-xmin))*static_cast<Real>(n));
}

#endif // UTILS_GRID_LOCATIONS_HPP_
