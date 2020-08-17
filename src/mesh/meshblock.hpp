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
namespace hydro {class Hydro;}

//----------------------------------------------------------------------------------------
//! \class MeshBlock
//  \brief data/functions associated with a single block

class MeshBlock {
 // the three mesh classes (Mesh, MeshBlock, MeshBlockTree) like to play together
 friend class Mesh;
 friend class MeshBlockTree;
 public:
  MeshBlock(Mesh *pm, std::unique_ptr<ParameterInput> &pin, RegionSize in_size,
            RegionCells in_cells, int in_gid, BoundaryFlag *input_bcs);
  ~MeshBlock();

  // data
  Mesh *pmy_mesh;  // ptr to Mesh containing this MeshBlock
  int mblock_gid;      // grid ID, unique identifier for this MeshBlock
  RegionSize  mblock_size;    // physical size of this MeshBlock
  RegionCells mblock_cells;   // info about cells in this MeshBlock
  BoundaryFlag mblock_bcs[6]; // enums specifying BCs at all 6 faces of this MeshBlock

  // cells on 1x coarser level MeshBlock (i.e. ncc2=nx2/2 + 2*nghost, if nx2>1)
  RegionCells cmb_cells;

  // physics modules and task list (controlled by Mesh::SelectPhysics)
  hydro::Hydro *phydro;
  TaskList tl_onestage;


  // functions
  int GetNumberOfMeshBlockCells()
    { return mblock_cells.nx1 * mblock_cells.nx2 * mblock_cells.nx3; }

 private:
  // data
  double lb_cost;  // cost of updating this MeshBlock for load balancing

  // functions

};

#endif // MESH_MESHBLOCK_HPP_
