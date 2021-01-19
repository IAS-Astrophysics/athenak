#ifndef MESH_MESHBLOCK_PACK_HPP_
#define MESH_MESHBLOCK_PACK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock_pack.hpp
//  \brief defines MeshBlockPack class, a container for MeshBlocks

#include <memory>
#include <vector>
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

// Forward declarations
class MeshBlock;
namespace hydro {class Hydro;}
namespace mhd {class MHD;}

//----------------------------------------------------------------------------------------
//! \class MeshBlock
//  \brief data/functions associated with a single block

class MeshBlockPack
{
 // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
 friend class Mesh;
 friend class MeshBlock;
 friend class MeshBlockTree;

 public:
  MeshBlockPack(Mesh *pm, int igids, int igide, RegionCells icells);
  ~MeshBlockPack();

  // data
  Mesh *pmesh;            // ptr to Mesh containing this MeshBlockPack
  int gids, gide;         // start/end of global IDs in this MeshBlockPack
  int nmb_thispack;       // number of MBs in this pack
  // since all MeshBlocks are the same size, following data can be stored in the
  // MeshBlockPack container, and not the individual MeshBlocks themselves
  RegionCells mb_cells;   // info about cells in MeshBlock(s) in this MeshBlockPack 
  RegionCells cmb_cells;  // info about cells on next coarser level MBs

  MeshBlock* pmb;         // MeshBlocks in this MeshBlockPack

  // physics modules (controlled by InitPhysicsModules)
  hydro::Hydro *phydro;
  mhd::MHD *pmhd;

  // task lists for MeshBlocks in this MeshBlockPack
  TaskList tl_stagestart;
  TaskList tl_stagerun;
  TaskList tl_stageend;

  // functions
  void AddPhysicsModules(ParameterInput *pin);
  int NumberOfMeshBlockCells() { return mb_cells.nx1 * mb_cells.nx2 * mb_cells.nx3; }
  int NumberOfCoarseMeshBlockCells() {return cmb_cells.nx1 *cmb_cells.nx2 *cmb_cells.nx3;}

 private:
  // data

  // functions
  void SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist);

};

#endif // MESH_MESHBLOCK_PACK_HPP_
