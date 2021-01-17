#ifndef MESH_MESHBLOCK_HPP_
#define MESH_MESHBLOCK_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock.hpp
//  \brief defines MeshBlock class, a very lightweight class to store data about
//  MeshBlocks inside a given MeshBlockPack.
//  The Mesh is the overall grid structure, MeshBlocks are local patches of data
//  (potentially on different levels) that tile the entire domain and are stored in
//  containers called MashBlockPack.

#include <vector>
#include "bvals/bvals.hpp"
#include "meshblock_pack.hpp"

//----------------------------------------------------------------------------------------
//! \class MeshBlock
//  \brief data/functions associated with a single block

class MeshBlock
{
 // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
 friend class Mesh;
 friend class MeshBlockPack;
 friend class MeshBlockTree;

 public:
  MeshBlock(MeshBlockPack *pp, int igid);
  ~MeshBlock();

  // data
  int mb_gid;             // grid ID, unique identifier for this MeshBlock
  RegionSize  mb_size;    // physical size of this MeshBlock
  BoundaryFlag mb_bcs[6]; // boundary conditions at 6 faces of MeshBlock

  std::vector<NeighborBlock> nghbr;  // vector storing data about all neighboring MBs


 private:
  // data
  MeshBlockPack* pmy_pack;
  double lb_cost;  // cost of updating this MeshBlock for load balancing

  // functions
  void SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist);

};

#endif // MESH_MESHBLOCK_HPP_
