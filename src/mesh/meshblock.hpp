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
  MeshBlock(Mesh *pm, int igids, int igide);
  ~MeshBlock();

  // data
  int nmb;   // # of MeshBlocks
  int nnghbr;  // # of neighbors for each MB.  TODO: cannot be same for all MBs with AMR

  // AthenaDualArrays are used to store data used on both device and host 
  // STL vector is used for data only accessed on host
  // First dimension of each will be [# of MeshBlocks in this MeshBlockPack]

  AthenaArray1D<int> d_mbgid;     // grid ID, unique identifier for each MeshBlock
  HostArray1D<int> h_mbgid;     // grid ID, unique identifier for each MeshBlock
  AthenaArray2D<Real> d_mbsize;   // physical size of each MeshBlock
  HostArray2D<Real> h_mbsize;   // physical size of each MeshBlock
  HostArray2D<int> mb_bcs;  // boundary conditions at 6 faces of each MeshBlock

  AthenaArray3D<int> d_mbnghbr;  // vector storing data about all neighboring MBs
  HostArray3D<int> h_mbnghbr;  // vector storing data about all neighboring MBs


 private:
  // data
  Mesh* pmy_mesh;
  HostArray1D<double> lb_cost;  // cost of updating each MeshBlock for load balancing

  // functions
  void SetNeighbors(std::unique_ptr<MeshBlockTree> &ptree, int *ranklist);

};

#endif // MESH_MESHBLOCK_HPP_
