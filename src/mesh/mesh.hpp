#ifndef MESH_MESH_HPP_
#define MESH_MESH_HPP_
//========================================================================================
// Athena++K astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh.hpp
//  \brief defines Mesh class.
//  The Mesh is the overall grid structure, which is divided into local patches called
//  MeshBlocks (potentially on different levels) that tile the entire domain.  MeshBlocks
//  are grouped together into MeshBlockPacks for better performance on GPUs.

#include <cstdint>  // int32_t
#include <memory>

#include "athena.hpp"

// Define following structure before other "include" files to resolve declarations
//----------------------------------------------------------------------------------------
//! \struct RegionSize
//! \brief physical size in a Mesh or a MeshBlock

struct RegionSize
{
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real dx1, dx2, dx3;       // (uniform) grid spacing
};

//----------------------------------------------------------------------------------------
//! \struct RegionIndcs
//! \brief Cell indices and number of active and ghost cells in a Mesh or a MeshBlock

struct RegionIndcs
{
  int ng;                   // number of ghost cells
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
  int is,ie,js,je,ks,ke;    // indices of ACTIVE cells
  int cnx1, cnx2, cnx3;         // number of active coarse cells (not including gzs)
  int cis,cie,cjs,cje,cks,cke;  // indices of ACTIVE coarse cells
};

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief Information about neighboring MeshBlocks stored as 2D DualArray in MeshBlock

struct NeighborBlock
{
  int gid;     // global ID
  int lev;     // logical level
  int rank;    // MPI rank     
  int dest;    // index of recv buffer in target NeighborBlocks
};

//----------------------------------------------------------------------------------------
//! \struct LogicalLocation
//! \brief stores logical location and level of MeshBlock
//! lx1/2/3 = logical location in x1/2/3 = index in array of nodes at current level
//! WARNING: values of lx? can exceed the range of std::int32_t with >30 levels
//! of AMR, even if the root grid consists of a single MeshBlock, since the corresponding
//! max index = 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer types

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

// Forward declarations required due to recursive definitions amongst mesh classes
class MeshBlock;
class MeshBlockPack;
class MeshBlockTree;
class Mesh;

#include "parameter_input.hpp"
#include "meshblock.hpp"
#include "meshblock_pack.hpp"
#include "meshblock_tree.hpp"

//----------------------------------------------------------------------------------------
//! \class Mesh
//! \brief data/functions associated with the overall mesh

class Mesh
{
  // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
  friend class MeshBlock;
  friend class MeshBlockPack;
  friend class MeshBlockTree;

public:
  explicit Mesh(ParameterInput *pin);
  ~Mesh();

  // data
  RegionSize  mesh_size;      // (physical) size of mesh (physical root level)
  RegionIndcs mesh_indcs;     // indices of cells in mesh (physical root level)
  RegionIndcs mb_indcs;       // indices of cells in MeshBlocks (same for all MeshBlocks)
  BoundaryFlag mesh_bcs[6];   // physical boundary conditions at 6 faces of mesh
  bool strictly_periodic;     // true if all boundaries are periodic

  bool one_d, two_d, three_d; // flags to indicate 1D or 2D or 3D calculations
  bool multi_d;               // flag to indicate 2D and 3D calculations
  bool shearing_periodic;     // flag to indicate periodic x1/x2 boundaries are sheared
  bool multilevel;            // true for SMR and AMR 
  bool adaptive;              // true only for AMR

  int nmb_rootx1, nmb_rootx2, nmb_rootx3; // # of MeshBlocks at root level in each dir
  int nmb_total;           // total number of MeshBlocks across all levels/ranks
  int nmb_thisrank;        // number of MeshBlocks on this MPI rank (local)
  int nmb_created;         // number of MeshBlcoks created via AMR during run (per rank?)
  int nmb_deleted;         // number of MeshBlcoks deleted via AMR during run (per rank?)

  int root_level; // logical level of root (physical) grid (e.g. Fig. 3 of method paper)
  int max_level;  // logical level of maximum refinement grid in Mesh
  int gids, gide; // start/end of global IDs on this MPI rank

  // following 2x arrays allocated with length [nmbtotal]
  int *ranklist;              // rank of each MeshBlock
  float *costlist;            // cost of each MeshBlock
  LogicalLocation *lloclist;  // LogicalLocations for each MeshBlocks

  // following 2x arrays allocated with length [nranks]
  int *gidslist;       // starting global ID of MeshBlocks in each rank
  int *nmblist;        // number of MeshBlocks on each rank

  // following 8x arrays allocated with length [nranks] only with AMR
  int *nref, *nderef;
  int *rdisp, *ddisp;
  int *bnref, *bnderef;
  int *brdisp, *bddisp;

  Real time, dt, cfl_no;           
  int ncycle;

  MeshBlockPack* pmb_pack;                 // container for MeshBlocks on this rank
  std::unique_ptr<ProblemGenerator> pgen;  // class containing functions to set ICs

  // functions
  void BuildTreeFromScratch(ParameterInput *pin);
  void BuildTreeFromRestart(ParameterInput *pin, IOWrapper &resfile);
  void PrintMeshDiagnostics();
  void WriteMeshStructure();
  void NewTimeStep(const Real tlim);
  void RestrictCC(DvceArray5D<Real> a, DvceArray5D<Real> ca);
  void RestrictFC(DvceFaceFld4D<Real> a, DvceFaceFld4D<Real> ca);
  BoundaryFlag GetBoundaryFlag(const std::string& input_string);
  std::string GetBoundaryString(BoundaryFlag input_flag);

  // accessors
  int FindMeshBlockIndex(int tgid)
  {
    for (int m=0; m<pmb_pack->pmb->nmb; ++m) {
      if (pmb_pack->pmb->mb_gid.h_view(m) == tgid) return m;
    }
    return -1;
  }
  int NumberOfMeshBlockCells() const {
    return (mb_indcs.nx1)*(mb_indcs.nx2)*(mb_indcs.nx3);
  }

private:
  // variables for load balancing control (not yet implemented)
  bool lb_flag_, lb_automatic_;
  int lb_cyc_interval_;
  int cyc_since_lb_;

  std::unique_ptr<MeshBlockTree> ptree;  // pointer to root node in binary/quad/oct-tree

  // functions
  void LoadBalance(float *clist, int *rlist, int *slist, int *nlist, int nb);
  void ResetLoadBalanceCounters();
};

#endif  // MESH_MESH_HPP_
