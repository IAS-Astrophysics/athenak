#ifndef UTILS_LAGRANGE_INTERPOLATOR_HPP_
#define UTILS_LAGRANGE_INTERPOLATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lagrange_interpolator.hpp

#include <cmath>
#include <iostream>
#include <list>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"

class LagrangeInterpolator {
 public:
  LagrangeInterpolator(MeshBlockPack *pmy_pack, Real rcoords[3]);
  ~LagrangeInterpolator() = default;

  void SetInterpolationIndices();
  void CalculateWeight();
  Real Interpolate(DvceArray5D<Real> &val, int nvars);
  Real InterpolateTensor(
    AthenaTensor<Real, TensorSymm::NONE, 3, 1> &val, int nvars);
  Real ResetPointAndInterpolate(
    DvceArray5D<Real> &val, int nvars, Real rcoords2[3]);
  bool point_exist; // point exist on this rank (meshblock pack)
 private:
  HostArray1D<Real> rcoord; // xyz coordinate for interpolated value

  int nvars;               // index of the variable for interpolation
  MeshBlockPack *pmy_pack; // ptr to MeshBlockPack containing this Hydro
  int
    ng; // number of ghost cell, which sets interpolation level automatically
  HostArray1D<int>
    interp_indcs; // indices of MeshBlock and zones therein for interp
  HostArray2D<Real> interp_wghts; // weights for interpolation
};

#endif // UTILS_LAGRANGE_INTERPOLATOR_HPP_
