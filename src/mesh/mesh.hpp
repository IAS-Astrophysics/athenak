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

// Forward declarations
class MeshBlock;
class MeshBlockTree;

//--------------------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size and number of cells in a Mesh or a MeshBlock

struct RegionSize {  // aggregate and POD type; do NOT reorder member declarations:
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real x1rat, x2rat, x3rat; // ratio of dxf(i)/dxf(i-1)
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
  int nghost;               // number of ghost cells
};

//--------------------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock

struct LogicalLocation {
  // WARNING: values of lx? can exceed the range of std::int32_t with >30 levels
  // of AMR, even if the root grid consists of a single MeshBlock, since the corresponding
  // max index = 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer type impelementations
  // lx1/2/3 = logical location in x1/2/3 = integer index in array of nodes at current level
  std::int32_t lx1, lx2, lx3, level;  
  // comparison functions for sorting
  static bool Lesser(const LogicalLocation &left, const LogicalLocation &right) {
    return left.level < right.level;
  }
  static bool Greater(const LogicalLocation & left, const LogicalLocation &right) {
    return left.level > right.level;
  }
  // overload comparison operator
  bool operator==(LogicalLocation const &rhs) const {
    return ((this->lx1 == rhs.lx1) && (this->lx2 == rhs.lx2) &&
            (this->lx3 == rhs.lx3) && (this->level == rhs.level));
  }
};

//--------------------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh {
 friend class MeshBlock;
 friend class MeshBlockTree;
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
  BoundaryFlag mesh_bcs[6];   // physical boundary conditions at 6 faces of root grid
  bool adaptive, multilevel;
  int nrmbx1, nrmbx2, nrmbx3; // number of MeshBlocks in root grid in each dir
  int nmbtotal;               // total number of MeshBlocks across all levels

  std::vector<MeshBlock> my_blocks; // MeshBlocks belonging to this MPI rank

  // functions
  void OutputMeshStructure();
  // compute l-edge posn of i^{th} MeshBlock (counting from 0) in total of n spanning (xmin->xmax)
  inline Real LeftEdgePosition(std::int32_t ith, std::int32_t n, Real xmin, Real xmax) {
    Real x = (static_cast<Real>(ith)) / (static_cast<Real>(n));
    return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax); //symmetrize round-off
  }

 private:
  // data
  int num_mesh_threads_;
  int root_level; // logical level of root (physical) grid (e.g. Fig. 3 of method paper)
  int max_level;  // logical level of maximum refinement grid in Mesh
  bool nx2gt1_, nx3gt1_; // flags to indictate 2D/3D calculations

  LogicalLocation *loclist;
  MeshBlockTree *ptree;    // binary/quad/oct-tree

  // functions

};

#endif  // MESH_MESH_HPP_
