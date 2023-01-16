//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lagrange_interpolator.hpp

#include <iostream>
#include <cmath>
#include <list>
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "athena.hpp"
#include "athena_tensor.hpp"

class LagrangeInterpolator {
 public:
  LagrangeInterpolator(MeshBlockPack *pmy_pack, Real rcoords[3]);
  ~LagrangeInterpolator();

  void SetInterpolationIndices();
  void CalculateWeight();
  Real Interpolate(DvceArray5D<Real> &val, int nvars);
  Real Interpolate(AthenaTensor<Real, TensorSymm::NONE, 3, 1> &val,int nvars);

  Real ResetPointAndInterpolate(DvceArray5D<Real> &val, int nvars, Real rcoords2[3]);
 private:
  DvceArray1D<Real> rcoord; // xyz coordinate for interpolated value

  int nvars; // index of the variable for interpolation
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  int ng; // number of ghost cell, which sets interpolation level automatically
  DvceArray1D<int> interp_indcs;   // indices of MeshBlock and zones therein for interp
  DvceArray2D<Real> interp_wghts;  // weights for interpolation
};