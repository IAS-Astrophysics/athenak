#ifndef GEODESIC_GRID_GEODESIC_GRID_HPP_
#define GEODESIC_GRID_GEODESIC_GRID_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geodesic_grid.hpp
//  \brief definitions for GeodesicGrid class

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \class GeodesicGrid

class GeodesicGrid {
 public:
  GeodesicGrid(int nlev, bool rotate, bool fluxes);
  ~GeodesicGrid();

  int nangles;  // number of angles (derived from nlevel, 5*(2*nlevel^2) + 2)
  DualArray1D<int>  num_neighbors;        // number of neighbors
  DualArray2D<int>  ind_neighbors;        // indices of neighbors
  DualArray2D<int>  ind_neighbors_edges;  // indices of neighbor edge
  DualArray1D<Real> solid_angles;         // solid angles
  DualArray2D<Real> arc_lengths;          // arc lengths
  DualArray2D<Real> cart_pos;             // coord position (cartesian) at face center
  DualArray3D<Real> cart_pos_mid;         // coord position (cartesian) at face edges
  DualArray2D<Real> polar_pos;            // polar coordinates at face center
  DualArray3D<Real> polar_pos_mid;        // polar coordinates at face edges
  DualArray3D<Real> unit_flux;            // angular unit vectors computed at face edges

  // functions
  void GridCartPosition(int n, Real& x, Real& y, Real& z);
  void GridCartPositionMid(int n, int nb, Real& x, Real& y, Real& z);
  void Neighbors(int n, int& num_nghbr, int neighbors[6]);
  void CircumcenterNormalized(Real x1, Real x2, Real x3, Real y1, Real y2, Real y3,
                              Real z1, Real z2, Real z3, Real& x, Real& y, Real& z);
  void SolidAngleAndArcLengths(int n, Real& weight, Real length[6]);
  Real ArcLength(int n1, int n2);
  void OptimalAngles(Real ang[2]);
  void RotateGrid(Real znew, Real pnew);
  void UnitFluxDir(Real zv, Real pv, Real zf, Real pf, Real& dz, Real& dp);
  void GreatCircleParam(Real z1, Real z2, Real p1, Real p2, Real& apar, Real& psi0);

 private:
  int nlevel;       // level of the geodesic mesh (==0 is 1 angle per octant for testing)
  bool rotate_geo;  // flag to enable the rotation of geodesic mesh
  bool geo_fluxes;  // flag to compute angular unit vectors (used for fluxes on geo grid)
  HostArray4D<Real> amesh_normals;   // normal components (regular faces)
  HostArray2D<Real> ameshp_normals;  // normal components (at poles)
  HostArray3D<Real> amesh_indices;   // indexing (regular faces)
  HostArray1D<Real> ameshp_indices;  // indexing (at poles)
};

#endif // GEODESIC_GRID_GEODESIC_GRID_HPP_
