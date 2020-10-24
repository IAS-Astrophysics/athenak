#ifndef MESH_MESH_HPP_
#define MESH_MESH_HPP_
//========================================================================================
// Athena++K astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh.hpp
//  \brief defines Mesh class.
//  The Mesh is the overall grid structure, and MeshBlocks are local patches of data
//  (potentially on different levels) that tile the entire domain.

#include <cstdint>  // int32_t
#include <vector>

// Define following structure before other "include" files to resolve declarations
//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size in a Mesh or a MeshBlock

struct RegionSize
{
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
};

//----------------------------------------------------------------------------------------
//! \struct RegionCells
//  \brief Number of cells and cell indexing in a Mesh or a MeshBlock

struct RegionCells
{  
  int ng;                   // number of ghost cells
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
  int is,ie,js,je,ks,ke;    // indices of ACTIVE cells
  Real dx1, dx2, dx3;       // (uniform) grid spacing
};

//----------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock
// WARNING: values of lx? can exceed the range of std::int32_t with >30 levels
// of AMR, even if the root grid consists of a single MeshBlock, since the corresponding
// max index = 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer types

struct LogicalLocation
{
  // lx1/2/3 = logical location in x1/2/3 = index in array of nodes at current level
  std::int32_t lx1, lx2, lx3, level;  
  // comparison functions for sorting, and overloaded operator==
  static bool Lesser(const LogicalLocation &left, const LogicalLocation &right)
  {
    return left.level < right.level;
  }
  static bool Greater(const LogicalLocation & left, const LogicalLocation &right)
  {
    return left.level > right.level;
  }
  bool operator==(LogicalLocation const &rhs) const
  {
    return ((this->lx1 == rhs.lx1) && (this->lx2 == rhs.lx2) &&
            (this->lx3 == rhs.lx3) && (this->level == rhs.level));
  }
};

// Forward declarations
class Mesh;

#include "bvals/bvals.hpp"
#include "meshblock_tree.hpp"
#include "meshblock.hpp"

//----------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh
{
 // the three mesh classes (Mesh, MeshBlock, MeshBlockTree) like to play together
 friend class MeshBlock;
 friend class MeshBlockTree;
 public:
  // 2x function overloads of ctor: normal and restarted simulation
  explicit Mesh(ParameterInput *pin);
  Mesh(ParameterInput *pin, IOWrapper &resfile);
  ~Mesh();

  // accessors
  MeshBlock* FindMeshBlock(int tgid)
  {
//    assert (tgid >= gids && tgid <= gide);
    return &(mblocks[tgid - gids]);
  }

  // data
  RegionSize  mesh_size;      // physical size of mesh (physical root level)
  RegionCells mesh_cells;     // number of cells in mesh (physical root level)
  BoundaryFlag mesh_bcs[6];   // physical boundary conditions at 6 faces of mesh
  bool nx2gt1, nx3gt1;        // flags to indictate 2D/3D calculations
  bool adaptive, multilevel;

  int nmbroot_x1, nmbroot_x2, nmbroot_x3; // # of MeshBlocks at root level in each dir
  int nmbtotal;                  // total number of MeshBlocks across all levels
  int nmbthisrank;               // number of MeshBlocks on this MPI rank (local)
  int nmb_created;               // number of MeshBlcoks created via AMR during run
  int nmb_deleted;               // number of MeshBlcoks deleted via AMR during run

  int root_level; // logical level of root (physical) grid (e.g. Fig. 3 of method paper)
  int max_level;  // logical level of maximum refinement grid in Mesh
  int gids, gide; // start/end of global IDs on this MPI rank

  // following 2x arrays allocated with length [nmbtotal]
  int *ranklist;             // rank of each MeshBlock
  double *costlist;          // cost of each MeshBlock
  LogicalLocation *loclist;  // LogicalLocations for each MeshBlocks

  // following 2x arrays allocated with length [nranks]
  int *gidslist;        // starting global ID of MeshBlocks in each rank
  int *nmblist;        // number of MeshBlocks on each rank

  // following 8x arrays allocated with length [nranks] only with AMR
  int *nref, *nderef;
  int *rdisp, *ddisp;
  int *bnref, *bnderef;
  int *brdisp, *bddisp;

  Real time, dt, cfl_no;           
  int ncycle;

  std::vector<MeshBlock> mblocks; // MeshBlocks belonging to this MPI rank

  // functions
  void BuildTree(ParameterInput *pin);
  void NewTimeStep(const Real tlim);
  void OutputMeshStructure(int flag);
  BoundaryFlag GetBoundaryFlag(const std::string& input_string);
  std::string GetBoundaryString(BoundaryFlag input_flag);

 private:
  // variables for load balancing control
  bool lb_flag_;
  bool lb_automatic_, lb_manual_;
  double lb_tolerance_;
  int lb_cyc_interval_;
  int cyc_since_lb_;

  std::unique_ptr<MeshBlockTree> ptree;  // binary/quad/oct-tree

  // functions
  void SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &size,
                                 RegionCells &cells, BoundaryFlag *bcs);
  void LoadBalance(double *clist, int *rlist, int *slist, int *nlist, int nb);
  void ResetLoadBalanceCounters();
};

#endif  // MESH_MESH_HPP_
