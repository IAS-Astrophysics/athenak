//==================================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//==================================================================================================
//! \file meshblocktree.cpp
//  \brief implementation of functions in the MeshBlockTree class
// The MeshBlockTree stores the logical grid structure, and is used for neighbor
// searches, assigning global IDs, etc.  Level is defined as "logical level", where the
// logical root (single block) level is 0.  Note the logical level of the physical root
// grid (user-specified root grid) will be greater than zero if it contains more than
// one MeshBlock

#include <cstdint>    // int64_t
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "meshblock.hpp"
#include "meshblock_tree.hpp"

// Define static member variables
Mesh* MeshBlockTree::pmesh_;
MeshBlockTree* MeshBlockTree::proot_;
int MeshBlockTree::nleaf_;


//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree::MeshBlockTree()
//  \brief constructor for the logical root

MeshBlockTree::MeshBlockTree(Mesh* pmesh) : pleaf_(nullptr), gid_(-1) {
  pmesh_ = pmesh;
  proot_ = this;
  loc_.lx1 = 0;
  loc_.lx2 = 0;
  loc_.lx3 = 0;
  loc_.level = 0;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree::MeshBlockTree(int gid, int ox1, int ox2, int ox3)
//  \brief constructor for a leaf

MeshBlockTree::MeshBlockTree(MeshBlockTree *parent, int ox1, int ox2, int ox3)
                           : pleaf_(nullptr), gid_(parent->gid_) {
  loc_.lx1 = (parent->loc_.lx1<<1)+ox1;
  loc_.lx2 = (parent->loc_.lx2<<1)+ox2;
  loc_.lx3 = (parent->loc_.lx3<<1)+ox3;
  loc_.level = parent->loc_.level+1;
}


//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree::~MeshBlockTree()
//  \brief destructor (for both root and leaves)

MeshBlockTree::~MeshBlockTree() {
  if (pleaf_ != nullptr) {
    for (int i=0; i<nleaf_; i++)
      delete pleaf_[i];
    delete [] pleaf_;
  }
}
