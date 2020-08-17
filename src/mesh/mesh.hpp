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

// Define following two structure before other "include" files to resolve declarations
//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size in a Mesh or a MeshBlock

struct RegionSize {
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
};

//----------------------------------------------------------------------------------------
//! \struct RegionCells
//  \brief Number of cells and cell indexing in a Mesh or a MeshBlock

struct RegionCells {  
  int nghost;               // number of ghost cells
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
  int is,ie,js,je,ks,ke;    // indices of ACTIVE cells
  Real dx1, dx2, dx3;       // (uniform) grid spacing
};


//----------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock

struct LogicalLocation {
  // WARNING: values of lx? can exceed the range of std::int32_t with >30 levels
  // of AMR, even if the root grid consists of a single MeshBlock, since the corresponding
  // max index = 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer types
  // lx1/2/3 = logical location in x1/2/3 = index in array of nodes at current level
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

#include <vector>
// Forward declarations
class Mesh;

#include "bvals/bvals.hpp"
#include "meshblock.hpp"
#include "meshblock_tree.hpp"

//----------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh {
 // the three mesh classes (Mesh, MeshBlock, MeshBlockTree) like to play together
 friend class MeshBlock;
 friend class MeshBlockTree;
 public:
  // 2x function overloads of ctor: normal and restarted simulation
  // Note ParameterInput is smart pointer passed by reference: 
  explicit Mesh(std::unique_ptr<ParameterInput> &pin);
  Mesh(std::unique_ptr<ParameterInput> &pin, IOWrapper &resfile);
  ~Mesh();

  // accessors
  int GetNumMeshThreads() const {return num_mesh_threads;}

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

  Real time, dt, cfl_no;           
  int ncycle;

  std::vector<MeshBlock> mblocks; // MeshBlocks belonging to this MPI rank

  // functions
  void SelectPhysics(std::unique_ptr<ParameterInput> &pin);
  void NewTimeStep(const Real tlim);
  void OutputMeshStructure(int flag);
  void SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &size,
                                 RegionCells &cells, BoundaryFlag *bcs);
  void LoadBalance(double *clist, int *rlist, int *slist, int *nlist, int nb);
  void ResetLoadBalance();

  // Following functions compute positions on a regular Cartesian grid.
  // They provide functionality of the Coordinates class in the C++ version of the code.

  // returns x-posn of left edge of i^th cell of N in range xmin->xmax
  // Averages of linear interpolation from each side used to symmetrize r.o. error
  inline Real LeftEdgeX(std::int32_t ith, std::int32_t n, Real xmin, Real xmax) {
    Real x = (static_cast<Real>(ith)) / (static_cast<Real>(n));
    return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
  }
  // returns cell center position of i^th cell of N in range xmin->xmax
  inline Real CellCenterX(int ith, int n, Real xmin, Real xmax) {
    Real x = (static_cast<Real>(ith) + 0.5) / (static_cast<Real>(n));
    return (x*xmax - x*xmin) - (0.5*xmax - 0.5*xmin) + (0.5*xmin + 0.5*xmax);
  }
  // returns i-index of cell containing x position
  inline int CellCenterIndex(Real x, int n, Real xmin, Real xmax) {
    return static_cast<int>(((x-xmin)/(xmax-xmin))*static_cast<Real>(n));
  }

 private:
  // data
  int num_mesh_threads;
  int root_level; // logical level of root (physical) grid (e.g. Fig. 3 of method paper)
  int max_level;  // logical level of maximum refinement grid in Mesh
  // following 2x arrays allocated with length [nmbtotal]
  int *ranklist;
  double *costlist;
  // following 2x arrays allocated with length [nranks]
  int *nslist;
  int *nblist;
  // following 8x arrays allocated with length [nranks] only with AMR
  int *nref, *nderef;
  int *rdisp, *ddisp;
  int *bnref, *bnderef;
  int *brdisp, *bddisp;
  int gids, gide; // start/end of grid IDs on this MPI rank

  // variables for load balancing control
  bool lb_flag;
  bool lb_automatic, lb_manual;
  double lb_tolerance;
  int lb_cyc_interval;
  int cyc_since_lb;

  MeshBlockTree tree;     // binary/quad/oct-tree
  LogicalLocation *loclist; // array of LogicalLocations for ALL MeshBlocks

  // functions

};

#endif  // MESH_MESH_HPP_
