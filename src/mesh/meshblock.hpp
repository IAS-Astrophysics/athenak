#ifndef MESH_MESHBLOCK_HPP_
#define MESH_MESHBLOCK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock.hpp
//  \brief defines MeshBlock class, and various structs used in them
//  The Mesh is the overall grid structure, and MeshBlocks are local patches of data
//  (potentially on different levels) that tile the entire domain.

#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

// Forward declarations
class BoundaryValues;
namespace hydro {class Hydro;}

//----------------------------------------------------------------------------------------
//! \class MeshBlock
//  \brief data/functions associated with a single block

class MeshBlock {
 // the three mesh classes (Mesh, MeshBlock, MeshBlockTree) like to play together
 friend class Mesh;
 friend class MeshBlockTree;
 public:
  MeshBlock(Mesh *pm, std::unique_ptr<ParameterInput> &pin, int igid,
    RegionSize isize, RegionCells icells, BoundaryFlag *input_bcs);
  ~MeshBlock();

  // data
  Mesh *pmesh_mb;         // ptr to Mesh containing this MeshBlock
  int mb_gid;             // grid ID, unique identifier for this MeshBlock
  RegionSize  mb_size;    // physical size of this MeshBlock
  RegionCells mb_cells;   // info about cells in this MeshBlock

  BoundaryValues *pbval;

//  BoundaryFlag mb_bcs[6]; // enums specifying BCs at all 6 faces of this MeshBlock
//  NeighborBlock neighbor[26];

  // cells on 1x coarser level MeshBlock (i.e. ncc2=nx2/2 + 2*nghost, if nx2>1)
  RegionCells cmb_cells;

  // physics modules (controlled by Mesh::SelectPhysics)
  hydro::Hydro *phydro;

  // task list (initialized by ... )
  TaskList tl_onestage;


  // functions
  int NumberOfMeshBlockCells() { return mb_cells.nx1 * mb_cells.nx2 * mb_cells.nx3; }
  int NumberOfCoarseMeshBlockCells() {return cmb_cells.nx1 *cmb_cells.nx2 *cmb_cells.nx3;}

 private:
  // data
  double lb_cost;  // cost of updating this MeshBlock for load balancing

  // functions
  void SetNeighbors(MeshBlockTree &tree, int *ranklist);

};

#endif // MESH_MESHBLOCK_HPP_
