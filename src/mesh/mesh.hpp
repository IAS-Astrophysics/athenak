#ifndef MESH_MESH_HPP_
#define MESH_MESH_HPP_
//==================================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==================================================================================================
//! \file mesh.hpp
//  \brief defines Mesh class.
//  The Mesh is the overall grid structure, and MeshBlocks are local patches of data
//  (potentially on different levels) that tile the entire domain.

#include <cstdint>     // int64_t
#include <vector>

//Forward declarations
class MeshBlock;

//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size and number of cells in a Mesh or a MeshBlock

struct RegionSize {  // aggregate and POD type; do NOT reorder member declarations:
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real x1rat, x2rat, x3rat; // ratio of dxf(i)/dxf(i-1)
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
  int nghost;               // number of ghost cells
};

//--------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock

struct LogicalLocation { // aggregate and POD type
  // These values can exceed the range of std::int32_t even if the root grid has only a
  // single MeshBlock if >30 levels of AMR are used, since the corresponding max index =
  // 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer type impelementations
  std::int64_t lx1, lx2, lx3;
  int level;
  // comparison functions for sorting
  static bool Lesser(const LogicalLocation &left, const LogicalLocation &right) {
    return left.level < right.level;
  }
  static bool Greater(const LogicalLocation & left, const LogicalLocation &right) {
    return left.level > right.level;
  }
};

//--------------------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh {
 public:
  // 2x function overloads of ctor: normal and restarted simulation
  // Note ParameterInput is smart pointer passed by reference: 
  explicit Mesh(std::unique_ptr<ParameterInput> &pin);
  Mesh(std::unique_ptr<ParameterInput> &pin, IOWrapper &resfile);
  ~Mesh();

  // accessors
  int GetNumMeshThreads() const {return num_mesh_threads_;}

  // data
  RegionSize root_size;
  bool adaptive, multilevel;
  int nrmbx1, nrmbx2, nrmbx3; // number of MeshBlocks in root grid in each direction
  int nmbtotal;   // total number of MeshBlocks across all levels
  int root_level; // logical level of root (physical) grid (e.g. Fig. 3 of method paper)
  int max_level;  // maximum number if logical levels in Mesh

  // 1D vector of MeshBlocks belonging to this MPI rank
  std::vector<MeshBlock> my_blocks;

  // functions
  void OutputMeshStructure();

 private:
  // data
  int num_mesh_threads_;

  LogicalLocation *loclist;

  // functions

};

#endif  // MESH_MESH_HPP_
