#ifndef MESH_NGHBR_INDEX_HPP_
#define MESH_NGHBR_INDEX_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nghbr_index.hpp
// \brief Finds ID of neighbor given input offsets.  The latter are measured relative
// to the center of the MeshBlock (0,0,0).  Thus (-1,0,0) is the inner_x1 face, and
// (0,0,1) the outer_x3 face.  Edges and corners are also specified in this way, e.g.
// (0,1,-1) is the outer_x2-inner_x3 x2x3 edge, and (1,1,1) is the outer_x1/x2/x3 corner
//
// Faces (edges) are further subdivided into 4 (2) blocks given the last two integer
// arguments.  So the 4 subblocks [0,1,2,3] of the inner_x1 face are specified by the
// pairs (0,0),(1,0),(0,1),(1,1) respectively.  For edges only the FIRST argument n1 is
// used to specify the two subblocks.
//
// The neighbor (boundary buffer) indexing scheme is as follows:
//   x1faces:    [0-3],  [4-7]
//   x2faces:    [8-11], [12-15]
//   x1x2edges:  [16-23]
//   x3faces:    [24-27], [28-31]
//   x3x1edges:  [32-39]
//   x2x3edges:  [40-47]
//   corners:    [48-55]

KOKKOS_INLINE_FUNCTION
static int NeighborIndex(int ix, int iy, int iz, int n1, int n2) {
  // do some error checking on input parameters
  if ((std::abs(ix) + std::abs(iy) + std::abs(iz)) == 0) {return -1;}
  if (std::abs(ix*iy*iz) > 1) {return -1;}

  if (iz == 0) {
    // x1faces or x2faces
    if (ix*iy == 0) {
      int subface = n1 + 2*n2;
      return std::abs(ix)*2*(ix + 1) + std::abs(iy)*2*(iy + 5) + subface;
    // x1x2 edges
    } else {
      return 16 + (ix + 1) + 2*(iy + 1) + n1;
    }
  } else {
    // x3faces, x3x1 edges, and x2x3 edges
    if (ix*iy == 0) {
      int subface = n1 + 2*n2;
      return 24 + std::abs(ix)*(ix + 9) + std::abs(iy)*(iy + 17) + 2*(iz + 1) + subface;
    // corners
    } else {
      return 48 + (ix + 1)/2 + (iy + 1) + 2*(iz + 1);
    }
  }
  return -1;
}
#endif // MESH_NGHBR_INDEX_HPP_
