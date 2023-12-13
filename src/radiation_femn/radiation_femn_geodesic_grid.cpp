//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_geodesicgrid.cpp
//  \brief implementation of functions for generating the geodesic grid
//         Functions in this file are:
//         (a) CartesianToSpherical:     convert from cartesian to spherical coordinates
//         (b) FindEdgesIndex:           from an array of edge index pairs, find the location of the edge given two vertex indices
//         (c) GeodesicGridBaseGenerate: generate the icosahedral grid with 12 angles
//         (d) GeodesicGridRefine:       refine the geodesic grid by one level

#include "athena.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid_matrices.hpp"
#include "radiation_femn.hpp"

namespace radiationfemn {

// ------------------------------------------
// Convert cartesian to spherical coordinates
inline
void CartesianToSpherical(Real xvar, Real yvar, Real zvar, Real &rvar, Real &thetavar, Real &phivar) {
  rvar = sqrt(xvar * xvar + yvar * yvar + zvar * zvar);
  thetavar = acos(zvar / rvar);
  phivar = atan2(yvar, xvar);
}

// -------------------------------------------------------------------------------
// Given two points of an edge, find the index of the edge array of their location
// Two vertex indices e1 and e2 goes in along with the edge information array --> return -42 if they don't share an edge, otherwise return index in edge array
inline
int FindEdgesIndex(int e1, int e2, HostArray2D<int> &edges) {
  int index{-42};
  for (int i = 0; i < edges.size() / 2; i++) {
    if ((edges(i, 0) == e1 && edges(i, 1) == e2) || (edges(i, 0) == e2 && edges(i, 1) == e1)) {
      index = i;
      break;
    }
  }
  return index;
}

// -------------------------------
// Generate the base geodesic grid
// Base grid --> regular icosahedron with vertices lying on unit sphere
// Populate metadata, vertex information in cartesian & spherical, edge information and triangle information
void
GeodesicGridBaseGenerate(int &geogrid_level,
                         int &geogrid_num_points,
                         int &geogrid_num_edges,
                         int &geogrid_num_triangles,
                         HostArray1D<Real> &x,
                         HostArray1D<Real> &y,
                         HostArray1D<Real> &z,
                         HostArray1D<Real> &r,
                         HostArray1D<Real> &theta,
                         HostArray1D<Real> &phi,
                         HostArray2D<int> &edges,
                         HostArray2D<int> &triangles) {

  // Geodesic grid metadata
  geogrid_level = 0;
  geogrid_num_points = 12;
  geogrid_num_edges = 30;
  geogrid_num_triangles = 20;

  // cartesian coordinates of the grid vertices
  Kokkos::realloc(x, geogrid_num_points);
  Kokkos::realloc(y, geogrid_num_points);
  Kokkos::realloc(z, geogrid_num_points);

  // corresponding polar coordinates
  Kokkos::realloc(r, geogrid_num_points);
  Kokkos::realloc(theta, geogrid_num_points);
  Kokkos::realloc(phi, geogrid_num_points);

  // edge and triangle information
  Kokkos::realloc(edges, geogrid_num_edges, 2);
  Kokkos::realloc(triangles, geogrid_num_triangles, 3);

  Real golden_ratio{(1.0 + sqrt(5.0)) / 2.0};
  Real normalization_factor{1.0 / sqrt(1. + golden_ratio * golden_ratio)};

  x(0) = normalization_factor * 0.;
  x(1) = normalization_factor * 0.;
  x(2) = normalization_factor * 0.;
  x(3) = normalization_factor * 0.;
  x(4) = normalization_factor * 1.;
  x(5) = normalization_factor * 1.;
  x(6) = normalization_factor * -1.;
  x(7) = normalization_factor * -1.;
  x(8) = normalization_factor * golden_ratio;
  x(9) = normalization_factor * golden_ratio;
  x(10) = normalization_factor * -golden_ratio;
  x(11) = normalization_factor * -golden_ratio;

  y(0) = normalization_factor * 1.;
  y(1) = normalization_factor * 1.;
  y(2) = normalization_factor * -1.;
  y(3) = normalization_factor * -1.;
  y(4) = normalization_factor * golden_ratio;
  y(5) = normalization_factor * -golden_ratio;
  y(6) = normalization_factor * golden_ratio;
  y(7) = normalization_factor * -golden_ratio;
  y(8) = normalization_factor * 0.;
  y(9) = normalization_factor * 0.;
  y(10) = normalization_factor * 0.;
  y(11) = normalization_factor * 0.;

  z(0) = normalization_factor * golden_ratio;
  z(1) = normalization_factor * -golden_ratio;
  z(2) = normalization_factor * golden_ratio;
  z(3) = normalization_factor * -golden_ratio;
  z(4) = normalization_factor * 0.;
  z(5) = normalization_factor * 0.;
  z(6) = normalization_factor * 0.;
  z(7) = normalization_factor * 0.;
  z(8) = normalization_factor * 1.;
  z(9) = normalization_factor * -1.;
  z(10) = normalization_factor * 1.;
  z(11) = normalization_factor * -1.;

  for (size_t i = 0; i < geogrid_num_points; i++) {
    CartesianToSpherical(x(i), y(i), z(i), r(i), theta(i), phi(i));
  }

  edges(0, 0) = 2;
  edges(0, 1) = 8;
  edges(1, 0) = 1;
  edges(1, 1) = 4;
  edges(2, 0) = 4;
  edges(2, 1) = 6;
  edges(3, 0) = 3;
  edges(3, 1) = 9;
  edges(4, 0) = 4;
  edges(4, 1) = 9;
  edges(5, 0) = 10;
  edges(5, 1) = 11;
  edges(6, 0) = 5;
  edges(6, 1) = 9;
  edges(7, 0) = 6;
  edges(7, 1) = 11;
  edges(8, 0) = 0;
  edges(8, 1) = 6;
  edges(9, 0) = 7;
  edges(9, 1) = 10;
  edges(10, 0) = 0;
  edges(10, 1) = 2;
  edges(11, 0) = 0;
  edges(11, 1) = 4;
  edges(12, 0) = 3;
  edges(12, 1) = 5;
  edges(13, 0) = 1;
  edges(13, 1) = 6;
  edges(14, 0) = 5;
  edges(14, 1) = 8;
  edges(15, 0) = 3;
  edges(15, 1) = 11;
  edges(16, 0) = 1;
  edges(16, 1) = 3;
  edges(17, 0) = 3;
  edges(17, 1) = 7;
  edges(18, 0) = 0;
  edges(18, 1) = 10;
  edges(19, 0) = 7;
  edges(19, 1) = 11;
  edges(20, 0) = 2;
  edges(20, 1) = 7;
  edges(21, 0) = 0;
  edges(21, 1) = 8;
  edges(22, 0) = 5;
  edges(22, 1) = 7;
  edges(23, 0) = 1;
  edges(23, 1) = 9;
  edges(24, 0) = 2;
  edges(24, 1) = 10;
  edges(25, 0) = 1;
  edges(25, 1) = 11;
  edges(26, 0) = 8;
  edges(26, 1) = 9;
  edges(27, 0) = 6;
  edges(27, 1) = 10;
  edges(28, 0) = 2;
  edges(28, 1) = 5;
  edges(29, 0) = 4;
  edges(29, 1) = 8;

  triangles(0, 0) = 0;
  triangles(0, 1) = 6;
  triangles(0, 2) = 10;
  triangles(1, 0) = 7;
  triangles(1, 1) = 10;
  triangles(1, 2) = 11;
  triangles(2, 0) = 3;
  triangles(2, 1) = 5;
  triangles(2, 2) = 9;
  triangles(3, 0) = 1;
  triangles(3, 1) = 4;
  triangles(3, 2) = 6;
  triangles(4, 0) = 0;
  triangles(4, 1) = 4;
  triangles(4, 2) = 6;
  triangles(5, 0) = 2;
  triangles(5, 1) = 7;
  triangles(5, 2) = 10;
  triangles(6, 0) = 3;
  triangles(6, 1) = 7;
  triangles(6, 2) = 11;
  triangles(7, 0) = 6;
  triangles(7, 1) = 10;
  triangles(7, 2) = 11;
  triangles(8, 0) = 1;
  triangles(8, 1) = 6;
  triangles(8, 2) = 11;
  triangles(9, 0) = 2;
  triangles(9, 1) = 5;
  triangles(9, 2) = 8;
  triangles(10, 0) = 0;
  triangles(10, 1) = 2;
  triangles(10, 2) = 10;
  triangles(11, 0) = 4;
  triangles(11, 1) = 8;
  triangles(11, 2) = 9;
  triangles(12, 0) = 1;
  triangles(12, 1) = 3;
  triangles(12, 2) = 9;
  triangles(13, 0) = 1;
  triangles(13, 1) = 3;
  triangles(13, 2) = 11;
  triangles(14, 0) = 0;
  triangles(14, 1) = 4;
  triangles(14, 2) = 8;
  triangles(15, 0) = 0;
  triangles(15, 1) = 2;
  triangles(15, 2) = 8;
  triangles(16, 0) = 2;
  triangles(16, 1) = 5;
  triangles(16, 2) = 7;
  triangles(17, 0) = 1;
  triangles(17, 1) = 4;
  triangles(17, 2) = 9;
  triangles(18, 0) = 5;
  triangles(18, 1) = 8;
  triangles(18, 2) = 9;
  triangles(19, 0) = 3;
  triangles(19, 1) = 5;
  triangles(19, 2) = 7;

}

// -------------------------------
// Refine a geodesic grid
// Provide metadata, vertex infrormation in cartesian & spherical, edge and triangle information --> these get updated with new values (refine one level up)
void GeodesicGridRefine(int &geogrid_level,
                        int &geogrid_num_points,
                        int &geogrid_num_edges,
                        int &geogrid_num_triangles,
                        HostArray1D<Real> &x,
                        HostArray1D<Real> &y,
                        HostArray1D<Real> &z,
                        HostArray1D<Real> &r,
                        HostArray1D<Real> &theta,
                        HostArray1D<Real> &phi,
                        HostArray2D<int> &edges,
                        HostArray2D<int> &triangles) {

  int new_num_ref = geogrid_level + 1;
  int new_num_points = int(12 * pow(4, new_num_ref));
  if (new_num_ref != 0) {
    for (int i = 0; i < new_num_ref; i++) {
      new_num_points -= 6 * int(pow(4, i));
    }
  }
  int new_num_edges = 3 * (new_num_points - 2);
  int new_num_triangles = 2 * (new_num_points - 2);

  HostArray1D<Real> xnew("xnew", new_num_points);
  HostArray1D<Real> ynew("ynew", new_num_points);
  HostArray1D<Real> znew("znew", new_num_points);
  HostArray1D<Real> rnew("rnew", new_num_points);
  HostArray1D<Real> thetanew("thetanew", new_num_points);
  HostArray1D<Real> phinew("phinew", new_num_points);

  HostArray2D<int> edgesnewtemp("edgesnewtemp", 9 * geogrid_num_triangles, 2);
  HostArray2D<int> edgesnew("edgesnew", new_num_edges, 2);
  HostArray2D<int> trianglesnew("trianglesnew", new_num_triangles, 3);

  for (size_t i = 0; i < geogrid_num_points; i++) {
    xnew(i) = x(i);
    ynew(i) = y(i);
    znew(i) = z(i);
  }

  for (size_t i = 0; i < geogrid_num_edges; i++) {
    int e1 = edges(i, 0);
    int e2 = edges(i, 1);

    xnew(geogrid_num_points + i) = (x(e1) + x(e2)) / 2.0;
    ynew(geogrid_num_points + i) = (y(e1) + y(e2)) / 2.0;
    znew(geogrid_num_points + i) = (z(e1) + z(e2)) / 2.0;

    Real mod_point = sqrt(xnew(geogrid_num_points + i) * xnew(geogrid_num_points + i) +
        ynew(geogrid_num_points + i) * ynew(geogrid_num_points + i) +
        znew(geogrid_num_points + i) * znew(geogrid_num_points + i));
    Real scaling_factor = sqrt(x(0) * x(0) + y(0) * y(0) + z(0) * z(0)) / mod_point;

    xnew(geogrid_num_points + i) = scaling_factor * xnew(geogrid_num_points + i);
    ynew(geogrid_num_points + i) = scaling_factor * ynew(geogrid_num_points + i);
    znew(geogrid_num_points + i) = scaling_factor * znew(geogrid_num_points + i);

  }

  for (size_t i = 0; i < new_num_points; i++) {
    CartesianToSpherical(xnew(i), ynew(i), znew(i), rnew(i), thetanew(i), phinew(i));
  }

  for (size_t i = 0; i < geogrid_num_triangles; i++) {
    int t1 = triangles(i, 0);
    int t2 = triangles(i, 1);
    int t3 = triangles(i, 2);

    int midpoint_index_0 = geogrid_num_points + FindEdgesIndex(t1, t2, edges);
    int midpoint_index_1 = geogrid_num_points + FindEdgesIndex(t2, t3, edges);
    int midpoint_index_2 = geogrid_num_points + FindEdgesIndex(t1, t3, edges);

    trianglesnew(4 * i, 0) = t1;
    trianglesnew(4 * i, 1) = midpoint_index_0;
    trianglesnew(4 * i, 2) = midpoint_index_2;
    trianglesnew(4 * i + 1, 0) = midpoint_index_0;
    trianglesnew(4 * i + 1, 1) = midpoint_index_1;
    trianglesnew(4 * i + 1, 2) = t2;
    trianglesnew(4 * i + 2, 0) = midpoint_index_1;
    trianglesnew(4 * i + 2, 1) = midpoint_index_2;
    trianglesnew(4 * i + 2, 2) = t3;
    trianglesnew(4 * i + 3, 0) = midpoint_index_0;
    trianglesnew(4 * i + 3, 1) = midpoint_index_1;
    trianglesnew(4 * i + 3, 2) = midpoint_index_2;

    edgesnewtemp(9 * i, 0) = t1;
    edgesnewtemp(9 * i, 1) = midpoint_index_0;
    edgesnewtemp(9 * i + 1, 0) = t1;
    edgesnewtemp(9 * i + 1, 1) = midpoint_index_2;
    edgesnewtemp(9 * i + 2, 0) = t2;
    edgesnewtemp(9 * i + 2, 1) = midpoint_index_0;
    edgesnewtemp(9 * i + 3, 0) = t2;
    edgesnewtemp(9 * i + 3, 1) = midpoint_index_1;
    edgesnewtemp(9 * i + 4, 0) = t3;
    edgesnewtemp(9 * i + 4, 1) = midpoint_index_1;
    edgesnewtemp(9 * i + 5, 0) = t3;
    edgesnewtemp(9 * i + 5, 1) = midpoint_index_2;
    edgesnewtemp(9 * i + 6, 0) = midpoint_index_0;
    edgesnewtemp(9 * i + 6, 1) = midpoint_index_1;
    edgesnewtemp(9 * i + 7, 0) = midpoint_index_0;
    edgesnewtemp(9 * i + 7, 1) = midpoint_index_2;
    edgesnewtemp(9 * i + 8, 0) = midpoint_index_1;
    edgesnewtemp(9 * i + 8, 1) = midpoint_index_2;
  }

  size_t index{0};
  for (size_t i = 0; i < 9 * geogrid_num_triangles; i++) {
    bool is_present{false};
    for (size_t j = 0; j < new_num_edges; j++) {
      if ((edgesnew(j, 0) == edgesnewtemp(i, 0) && edgesnew(j, 1) == edgesnewtemp(i, 1)) ||
          (edgesnew(j, 0) == edgesnewtemp(i, 1) && edgesnew(j, 1) == edgesnewtemp(i, 0))) {
        is_present = true;
      }

    }

    if (!is_present) {
      edgesnew(index, 0) = (edgesnewtemp(i, 0) < edgesnewtemp(i, 1)) * edgesnewtemp(i, 0) +
          (edgesnewtemp(i, 0) > edgesnewtemp(i, 1)) * edgesnewtemp(i, 1);
      edgesnew(index, 1) = (edgesnewtemp(i, 0) < edgesnewtemp(i, 1)) * edgesnewtemp(i, 1) +
          (edgesnewtemp(i, 0) > edgesnewtemp(i, 1)) * edgesnewtemp(i, 0);
      index++;
    }
  }

  Kokkos::realloc(x, new_num_points);
  Kokkos::realloc(y, new_num_points);
  Kokkos::realloc(z, new_num_points);
  Kokkos::realloc(r, new_num_points);
  Kokkos::realloc(theta, new_num_points);
  Kokkos::realloc(phi, new_num_points);
  Kokkos::realloc(edges, new_num_edges, 2);
  Kokkos::realloc(triangles, new_num_triangles, 3);

  Kokkos::deep_copy(x, xnew);
  Kokkos::deep_copy(y, ynew);
  Kokkos::deep_copy(z, znew);
  Kokkos::deep_copy(r, rnew);
  Kokkos::deep_copy(theta, thetanew);
  Kokkos::deep_copy(phi, phinew);
  Kokkos::deep_copy(edges, edgesnew);
  Kokkos::deep_copy(triangles, trianglesnew);

  geogrid_level = new_num_ref;
  geogrid_num_points = new_num_points;
  geogrid_num_edges = new_num_edges;
  geogrid_num_triangles = new_num_triangles;

}
} // namespace radiationfemn