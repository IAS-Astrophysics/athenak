//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geodesic_grid.hpp
//  \brief defines GeodesicGrid, a geodesic grid on the unit sphere

#include "athena.hpp"


//----------------------------------------------------------------------------------------
//! \class GeodesicGrid

class GeodesicGrid {
 public:
  GeodesicGrid(MeshBlockPack *ppack, int *nlev, bool *rotate_g);

  // Angular mesh parameters and functions
  int nlevel;                         // geodesic nlevel
  int nangles;                        // number of angles
  bool rotate_geo;                    // rotate geodesic mesh
  DualArray1D<Real> solid_angle;      // solid angles
  DualArray2D<Real> polarcoord;       // polar coordinate for grid points
  DualArray3D<Real> amesh_indices;    // indexing (regular faces)
  DualArray1D<Real> ameshp_indices;   // indexing (at poles)
  DualArray4D<Real> amesh_normals;    // normal components (regular faces)
  DualArray2D<Real> ameshp_normals;   // normal components (at poles)
  //int nvertices;                      // number of vertices
  //DualArray1D<int>  num_neighbors;    // number of neighbors
  //DualArray2D<int>  ind_neighbors;    // indices of neighbors
  //DualArray2D<Real> arc_lengths;      // arc lengths
  //DualArray2D<Real> nh_c;             // normal vector computed at face center
  //DualArray3D<Real> nh_f;             // normal vector computed at face edges
  void InitAngularMesh();

  // intensity arrays
  //DvceArray5D<Real> i0;         // intensities

  // Boundary communication buffers and functions for i
  BoundaryValuesCC *pbval_i;

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

