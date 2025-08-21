#ifndef MESH_MESHBLOCK_TREE_HPP_
#define MESH_MESHBLOCK_TREE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock_tree.hpp
//  \brief defines the MeshBlockTree class
// The MeshBlockTree stores the logical grid structure in a binary tree, and is used for
// neighbor searches, storing global IDs, etc. Levels refer to "logical level", where the
// logical root (single block) level is 0 (see Figs. 1 and 3 of method paper). Note the
// logical level of the physical root grid (user-specified root grid) will be greater than
// zero if it contains more than one MeshBlock.
//
// Original version of this foundational class written c2015-2016 by K. Tomida.

//--------------------------------------------------------------------------------------
//! \class MeshBlockTree
//  \brief Objects are nodes in a binary tree structure.  Thus, the class name does not
//  refer to the overall tree itself, but rather each MeshBlockTree is a node. A pointer
//  to the root node is stored in the Mesh class.

class MeshBlockTree {
  // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
  friend class Mesh;
  friend class MeshBlock;
  friend class MeshBlockPack;

 public:
  explicit MeshBlockTree(Mesh *pmesh);
  MeshBlockTree(MeshBlockTree *parent, int ox1, int ox2, int ox3);
  ~MeshBlockTree();

  // accessors
  MeshBlockTree* GetLeaf(int ox1, int ox2, int ox3)
    { return pleaf_[(ox1 + (ox2<<1) + (ox3<<2))]; }
  int GetGID() {return gid_;}
  std::int32_t GetLevel() {return lloc_.level;}

  // functions
  void CreateRootGrid();
  void AddNode(LogicalLocation rloc, int &nnew);
  void AddNodeWithoutRefinement(LogicalLocation rloc);
  void Refine(int &nnew);
  void Derefine(int &ndel);
  MeshBlockTree* FindMeshBlock(LogicalLocation tloc);
  void CountMeshBlocks(int& count);
  void CreateZOrderedLLList(LogicalLocation *list, int *pglist, int& count);
  MeshBlockTree* FindNeighbor(LogicalLocation myloc, int ox1, int ox2, int ox3,
                              bool amrflag=false);

 private:
  // data: note private variable names have trailing underscore for this class
  MeshBlockTree **pleaf_;  // 1D vector of pointers to leafs
  int gid_;                // grid ID
  LogicalLocation lloc_;   // stores logical x1/x2/x3 location, level for node in tree

  static Mesh *pmesh_;           // pointer to Mesh containing Tree
  static MeshBlockTree *proot_;  // pointer to leaf at root level
  static int nleaf_;             // number of leafs (2/4/8 for 1D/2D/3D)
};

#endif // MESH_MESHBLOCK_TREE_HPP_
