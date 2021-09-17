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
//
//  The Mesh is the overall grid structure, MeshBlocks are local patches of data
//  (potentially on different levels) that tile the entire domain and are stored in
//  containers called MashBlockPack.

#include "bvals/bvals.hpp"
#include "meshblock_pack.hpp"

//----------------------------------------------------------------------------------------
//! \class MeshBlock
//  \brief data/functions associated with each MeshBlock

class MeshBlock
{
 // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
 friend class Mesh;
 friend class MeshBlockPack;
 friend class MeshBlockTree;

 public:
  MeshBlock(Mesh *pm, int igids, int nmb);
  ~MeshBlock();

  // data
  int nmb;     // # of MeshBlocks
  int nnghbr;  // # of neighbors for each MB.  TODO: cannot be same for all MBs with AMR

  // AthenaDualArrays are used to store data used on both device and host 
  // First dimension of each array will be [# of MeshBlocks in this MeshBlockPack]

  DualArray1D<int> mbgid;            // grid ID, unique identifier for each MeshBlock
  HostArray2D<BoundaryFlag> mb_bcs;  // boundary conditions at 6 faces of each MeshBlock

  NeighborBlock nghbr[26];           // data on neighbors stored in fixed-length array

 private:
  // data
  Mesh* pmy_mesh;
  HostArray1D<double> lb_cost;  // cost of updating each MeshBlock for load balancing

  // functions
  void SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist);

};

#endif // MESH_MESHBLOCK_HPP_
