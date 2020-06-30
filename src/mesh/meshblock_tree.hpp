#ifndef MESH_MESHBLOCK_TREE_HPP_
#define MESH_MESHBLOCK_TREE_HPP_
//==================================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==================================================================================================
//! \file meshblock_tree.hpp
//  \brief defines the LogicalLocation structure and MeshBlockTree class

#include <unordered_map>
#include <vector>

#include "athena.hpp"

// Forward declarations
//class Mesh;

//--------------------------------------------------------------------------------------
//! \class MeshBlockTree
//  \brief Objects are nodes in an AMR MeshBlock tree structure

class MeshBlockTree {
 public:
  explicit MeshBlockTree(Mesh *pmesh);
  MeshBlockTree(MeshBlockTree *parent, int ox1, int ox2, int ox3);
  ~MeshBlockTree();

  // accessor
  MeshBlockTree* GetLeaf(int ox1, int ox2, int ox3) { return pleaf_[(ox1 + (ox2<<1) + (ox3<<2))]; }

  // functions
  void CreateRootGrid();
//  void AddMeshBlock(LogicalLocation rloc, int &nnew);
//  void AddMeshBlockWithoutRefine(LogicalLocation rloc);
//  void Refine(int &nnew);
//  void Derefine(int &ndel);
//  MeshBlockTree* FindMeshBlock(LogicalLocation tloc);
//  void CountMeshBlock(int& count);
//  void GetMeshBlockList(LogicalLocation *list, int *pglist, int& count);
//  MeshBlockTree* FindNeighbor(LogicalLocation myloc, int ox1, int ox2, int ox3,
//                              bool amrflag=false);
//  void CountMGOctets(int *noct);
//  void GetMGOctetList(std::vector<MGOctet> *oct,
//       std::unordered_map<LogicalLocation, int, LogicalLocationHash> *octmap, int *noct);

 private:
  // data
  MeshBlockTree** pleaf_;
  int gid_;
  LogicalLocation loc_;

  static Mesh* pmesh_;
  static MeshBlockTree* proot_;
  static int nleaf_;
};

#endif // MESH_MESHBLOCK_TREE_HPP_
