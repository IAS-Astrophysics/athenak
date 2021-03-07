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
#include "athena.hpp"
#include <memory>

// Define following structure before other "include" files to resolve declarations
//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size in a Mesh or a MeshBlock

struct MeshSize
{
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real dx1, dx2, dx3;       // (uniform) grid spacing
};

// size data stored as Views to enable loops over all MeshBlocks in a single kernel
struct MeshBlockSize
{
  DualArray1D<Real> x1min, x2min, x3min;
  DualArray1D<Real> x1max, x2max, x3max;
  DualArray1D<Real> dx1, dx2, dx3;       // (uniform) grid spacing
  // constructor
  MeshBlockSize(int nmb) :
    x1min("x1min",nmb), x1max("x1max",nmb),
    x2min("x2min",nmb), x2max("x2max",nmb),
    x3min("x3min",nmb), x3max("x3max",nmb),
    dx1("dx1",nmb), dx2("dx2",nmb), dx3("dx3",nmb) {}
};

//----------------------------------------------------------------------------------------
//! \struct RegionCells
//  \brief Number of cells and cell indexing in a Mesh or a MeshBlock

struct RegionCells
{
  int ng;                   // number of ghost cells
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
  int is,ie,js,je,ks,ke;    // indices of ACTIVE cells
};

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief Information about neighboring MeshBlocks, stored in DualArrays of length
//  (# of neighboring blocks).  Latter is 26 for a uniform grid in 3D.

struct NeighborBlock
{
  DualArray1D<int> gid;      // global ID
  DualArray1D<int> lev;      // logical level
  DualArray1D<int> rank;    // MPI rank     
  DualArray1D<int> destn;   // index of recv buffer in target vector of NeighborBlocks
  // default constructor
  NeighborBlock(){};
};

//----------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock
//  lx1/2/3 = logical location in x1/2/3 = index in array of nodes at current level
// WARNING: values of lx? can exceed the range of std::int32_t with >30 levels
// of AMR, even if the root grid consists of a single MeshBlock, since the corresponding
// max index = 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer types

struct LogicalLocation
{
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

#include "parameter_input.hpp"
#include "meshblock_tree.hpp"
#include "meshblock.hpp"
#include "meshblock_pack.hpp"

//----------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh
{
 // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
 friend class MeshBlock;
 friend class MeshBlockPack;
 friend class MeshBlockTree;

 public:
  explicit Mesh(ParameterInput *pin);
  Mesh(ParameterInput *pin, IOWrapper &resfile);  // ctor for restarts
  ~Mesh();

  // accessors
  int FindMeshBlockIndex(int tgid)
  {
    for (int m=0; m<pmb_pack->pmb->nmb; ++m) {
      if (pmb_pack->pmb->mbgid.h_view(m) == tgid) return m;
    }
    return -1;
  }

  // data
  MeshSize  mesh_size;        // physical size of mesh (physical root level)
  RegionCells mesh_cells;     // number of cells in mesh (physical root level)
  BoundaryFlag mesh_bcs[6];   // physical boundary conditions at 6 faces of mesh
  bool nx2gt1, nx3gt1;        // flags to indictate 2D/3D calculations
  bool shearing_periodic;     // flag to indicate periodic x1/x2 boundaries are sheared
  bool adaptive, multilevel;

  int nmb_rootx1, nmb_rootx2, nmb_rootx3; // # of MeshBlocks at root level in each dir
  int nmb_total;                 // total number of MeshBlocks across all levels
  int nmb_thisrank;              // number of MeshBlocks on this MPI rank (local)
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

  MeshBlockPack* pmb_pack;  // container for MeshBlocks on this rank

  // functions
  void BuildTree(ParameterInput *pin);
  void NewTimeStep(const Real tlim);
  void PrintMeshDiagnostics();
  void WriteMeshStructure();
  BoundaryFlag GetBoundaryFlag(const std::string& input_string);
  std::string GetBoundaryString(BoundaryFlag input_flag);

 private:
  // variables for load balancing control
  bool lb_flag_;
  bool lb_automatic_, lb_manual_;
  double lb_tolerance_;
  int lb_cyc_interval_;
  int cyc_since_lb_;

  std::unique_ptr<MeshBlockTree> ptree;  // pointer to root node in binary/quad/oct-tree

  // functions
  void LoadBalance(double *clist, int *rlist, int *slist, int *nlist, int nb);
  void ResetLoadBalanceCounters();
};

#endif  // MESH_MESH_HPP_
