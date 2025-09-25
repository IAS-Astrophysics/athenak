//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file meshblock_tree.cpp
//  \brief implementation of constructor and functions in the MeshBlockTree class

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "meshblock_tree.hpp"

// Define static member variables
Mesh* MeshBlockTree::pmesh_;
MeshBlockTree* MeshBlockTree::proot_;
int MeshBlockTree::nleaf_;

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree::MeshBlockTree()
//! \brief constructor for the logical root level

MeshBlockTree::MeshBlockTree(Mesh *pmesh) :
  pleaf_(nullptr), gid_(-1) {
  pmesh_ = pmesh;
  lloc_.lx1 = 0;
  lloc_.lx2 = 0;
  lloc_.lx3 = 0;
  lloc_.level = 0;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree::MeshBlockTree(int gid, int ox1, int ox2, int ox3)
//! \brief constructor for a leaf

MeshBlockTree::MeshBlockTree(MeshBlockTree *parent, int ox1, int ox2, int ox3) :
  pleaf_(nullptr), gid_(parent->gid_) {
  lloc_.lx1 = (parent->lloc_.lx1<<1) + ox1;
  lloc_.lx2 = (parent->lloc_.lx2<<1) + ox2;
  lloc_.lx3 = (parent->lloc_.lx3<<1) + ox3;
  lloc_.level = parent->lloc_.level + 1;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree::~MeshBlockTree()
//! \brief destructor (for both root and leaves)

MeshBlockTree::~MeshBlockTree() {
  if (pleaf_ != nullptr) {
    for (int i=0; i<nleaf_; i++) { delete pleaf_[i]; }
    delete [] pleaf_;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::CreateRootGrid()
//! \brief create the root grid; note the root grid can be incomplete (less than 8 leaves)

void MeshBlockTree::CreateRootGrid() {
  if (lloc_.level == 0) {  // initialize base properties of tree (logical level = 0)
    proot_ = this;
    if (pmesh_->one_d)   nleaf_ = 2;
    if (pmesh_->two_d)   nleaf_ = 4;
    if (pmesh_->three_d) nleaf_ = 8;
  }
  // do not create any nodes beyond the logical level of root grid (which corresponds to
  // logical level = 0 only in the case of a grid containing a single MeshBlock)
  if (lloc_.level == pmesh_->root_level) return;

  // Otherwise create vector of leaf pointers
  pleaf_ = new MeshBlockTree*[nleaf_];
  for (int n=0; n<nleaf_; n++) {pleaf_[n] = nullptr;}

  // test if any of the leaves need to be refined further
  std::int32_t levfac = 1<<(pmesh_->root_level - lloc_.level-1);
  for (int n=0; n<nleaf_; n++) {
    // i,j,k values are 1st/2nd/3rd bit from end in binary representation of n=0,..,7
    int i = n&1, j = (n>>1)&1, k = (n>>2)&1;
    if ((lloc_.lx3*2 + k)*levfac < pmesh_->nmb_rootx3
     && (lloc_.lx2*2 + j)*levfac < pmesh_->nmb_rootx2
     && (lloc_.lx1*2 + i)*levfac < pmesh_->nmb_rootx1) {
      pleaf_[n] = new MeshBlockTree(this, i, j, k); // call leaf constructor
      pleaf_[n]->CreateRootGrid();
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::AddNode(LogicalLocation rloc, int &nnew)
//! \brief add a new node for a MeshBlock to the tree, also creates neighboring blocks

void MeshBlockTree::AddNode(LogicalLocation rloc, int &nnew) {
  if (lloc_.level == rloc.level) return; // done

  if (pleaf_ == nullptr) {
    Refine(nnew);             // leaf -> create the finer level
  }

  // get leaf index
  int sh = rloc.level-lloc_.level-1;
  int mx, my, mz;
  mx = ((rloc.lx1>>sh) & 1) == 1;
  my = ((rloc.lx2>>sh) & 1) == 1;
  mz = ((rloc.lx3>>sh) & 1) == 1;
  int n = mx + (my<<1) + (mz<<2);
  pleaf_[n]->AddNode(rloc, nnew);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::AddNodeWithoutRefinement(LogicalLocation rloc)
//! \brief add a node for a MeshBlock to the tree without refinement, used in restarting.
//! MeshBlockTree::CreateRootGrid must be called before this method

void MeshBlockTree::AddNodeWithoutRefinement(LogicalLocation rloc) {
  if (lloc_.level == rloc.level) // done
    return;

  if (pleaf_ == nullptr) {
    pleaf_ = new MeshBlockTree*[nleaf_];
    for (int n=0; n<nleaf_; n++) {
      pleaf_[n] = nullptr;
    }
  }

  // get leaf index
  int sh = rloc.level-lloc_.level-1;
  int mx, my, mz;
  mx = ((rloc.lx1>>sh) & 1) == 1;
  my = ((rloc.lx2>>sh) & 1) == 1;
  mz = ((rloc.lx3>>sh) & 1) == 1;
  int n = mx + (my<<1) + (mz<<2);
  if (pleaf_[n] == nullptr) {
    pleaf_[n] = new MeshBlockTree(this, mx, my, mz);
  }
  pleaf_[n]->AddNodeWithoutRefinement(rloc);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::Refine(int &nnew)
//! \brief make finer leaves

void MeshBlockTree::Refine(int &nnew) {
  if (pleaf_ != nullptr) return;

  pleaf_ = new MeshBlockTree*[nleaf_];
  for (int n=0; n<nleaf_; n++) {pleaf_[n] = nullptr;}

  for (int n=0; n<nleaf_; n++) {
    int i = n&1, j = (n>>1)&1, k = (n>>2)&1;
    pleaf_[n] = new MeshBlockTree(this, i, j, k);
  }

  std::int32_t nxmax,nymax,nzmax;
  std::int32_t oxmin, oxmax, oymin, oymax, ozmin, ozmax;
  LogicalLocation nloc;
  nloc.level=lloc_.level;

  oxmin = -1;
  oxmax = 1;
  nxmax = (pmesh_->nmb_rootx1<<(lloc_.level - pmesh_->root_level));
  oymin = 0;
  oymax = 0;
  nymax = 1;
  ozmin = 0;
  ozmax = 0;
  nzmax = 1;
  if (pmesh_->multi_d) {
    oymin = -1;
    oymax = 1;
    nymax = (pmesh_->nmb_rootx2<<(lloc_.level - pmesh_->root_level));
  }
  if (pmesh_->three_d) { // 3D
    ozmin = -1;
    ozmax = 1;
    nzmax = (pmesh_->nmb_rootx3<<(lloc_.level - pmesh_->root_level));
  }

  for (std::int32_t oz=ozmin; oz<=ozmax; oz++) {
    nloc.lx3 = lloc_.lx3 + oz;
    if (nloc.lx3<0) {
      if (pmesh_->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
        continue;
      } else {
        nloc.lx3 = nzmax - 1;
      }
    }
    if (nloc.lx3>=nzmax) {
      if (pmesh_->mesh_bcs[BoundaryFace::outer_x3] != BoundaryFlag::periodic) {
        continue;
      } else {
        nloc.lx3 = 0;
      }
    }

    for (std::int32_t oy=oymin; oy<=oymax; oy++) {
      nloc.lx2=lloc_.lx2+oy;
      if (nloc.lx2<0) {
        if (pmesh_->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
          continue;
        } else {
          nloc.lx2 = nymax - 1;
        }
      }
      if (nloc.lx2>=nymax) {
        if (pmesh_->mesh_bcs[BoundaryFace::outer_x2] != BoundaryFlag::periodic) {
          continue;
        } else {
          nloc.lx2=0;
        }
      }

      for (std::int32_t ox=oxmin; ox<=oxmax; ox++) {
        if (ox==0 && oy==0 && oz==0) continue;
        nloc.lx1 = lloc_.lx1 + ox;
        if (nloc.lx1<0) {
          if (pmesh_->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic
          && pmesh_->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic) {
            continue;
          } else {
            nloc.lx1 = nxmax - 1;
          }
        }
        if (nloc.lx1>=nxmax) {
          if (pmesh_->mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::periodic
          && pmesh_->mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::shear_periodic) {
            continue;
          } else {
            nloc.lx1 = 0;
          }
        }
        proot_->AddNode(nloc, nnew);
      }
    }
  }
  // this block is not a leaf anymore
  gid_ = -1;

  nnew += nleaf_-1;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::Derefine(int &ndel)
//! \brief destroy leaves and make this block a leaf

void MeshBlockTree::Derefine(int &ndel) {
  int s2=0, e2=0, s3=0, e3=0;
  if (pmesh_->multi_d) s2=-1, e2=1;
  if (pmesh_->three_d) s3=-1, e3=1;
  for (int ox3=s3; ox3<=e3; ox3++) {
    for (int ox2=s2; ox2<=e2; ox2++) {
      for (int ox1=-1; ox1<=1; ox1++) {
        MeshBlockTree *bt = proot_->FindNeighbor(lloc_, ox1, ox2, ox3, true);
        if (bt != nullptr) {
          if (bt->pleaf_ != nullptr) {
            int lis, lie, ljs, lje, lks, lke;
            if (ox1==-1)       lis=lie=1;
            else if (ox1==1)   lis=lie=0;
            else              lis=0, lie=1;
            if (pmesh_->multi_d) {
              if (ox2==-1)     ljs=lje=1;
              else if (ox2==1) ljs=lje=0;
              else            ljs=0, lje=1;
            } else {
              ljs=lje=0;
            }
            if (pmesh_->three_d) {
              if (ox3==-1)     lks=lke=1;
              else if (ox3==1) lks=lke=0;
              else            lks=0, lke=1;
            } else {
              lks=lke=0;
            }
            for (int lk=lks; lk<=lke; lk++) {
              for (int lj=ljs; lj<=lje; lj++) {
                for (int li=lis; li<=lie; li++) {
                  int n = li + (lj<<1) + (lk<<2);
                  if (bt->pleaf_[n]->pleaf_ != nullptr) return;
                }
              }
            }
          }
        }
      }
    }
  }

  gid_ = pleaf_[0]->gid_; // now this is a leaf; inherit the first leaf's GID
  for (int n=0; n<nleaf_; n++) {
    delete pleaf_[n];
  }
  delete [] pleaf_;
  pleaf_ = nullptr;
  ndel+=nleaf_-1;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::CountMeshBlocks(int& count)
//! \brief Returns the number of MeshBlocks in Tree

void MeshBlockTree::CountMeshBlocks(int& count) {
  if (lloc_.level == 0) {count = 0;}

  if (pleaf_ == nullptr) {
    count++;
  } else {
    for (int n=0; n<nleaf_; n++) {
      if (pleaf_[n] != nullptr) {pleaf_[n]->CountMeshBlocks(count);}
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlockTree::CreateZOrderedLLList(LogicalLocation *list, int *pg, int& cnt)
//! \brief Creates the Location list for tree sorted by Z-ordering, and creates new MB ids
//! based on this order.  Should be called from root of tree. Called in BuildTreeXXX()
//! functions when tree is constructed for first time, in which case second argument is
//! 'nullptr' and this function creates gids for all MBs based on Z-ordering. Also called
//! by ResdistributeAndRefineMeshBlocks() function with AMR with second argument pointing
//! to an integer array that is used to store old gid of node on old tree, before creating
//! a new gid based on Z-ordering in the new tree. Thus pglist[n] is a mapping of
//! (new gid n) --> (old gid). Also returns total number of MBs in tree in third argument.

void MeshBlockTree::CreateZOrderedLLList(LogicalLocation *list, int *pglist, int& count) {
  if (lloc_.level == 0) {count=0;}

  if (pleaf_ == nullptr) {
    list[count]=lloc_;
    if (pglist != nullptr) {pglist[count]=gid_;}
    gid_=count;
    count++;
  } else {
    for (int n=0; n<nleaf_; n++) {
      if (pleaf_[n] != nullptr) {pleaf_[n]->CreateZOrderedLLList(list, pglist, count);}
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree* MeshBlockTree::FindNeighbor(LogicalLocation myloc,
//!                                   int ox1, int ox2, int ox3, bool amrflag)
//! \brief find a neighboring block, called from the root of the tree
//!        If it is coarser or same level, return the pointer to that block.
//!        If it is a finer block, return the pointer to its parent.
//!        Note that this function must be called on a completed tree only

MeshBlockTree* MeshBlockTree::FindNeighbor(LogicalLocation myloc,
                                           int ox1, int ox2, int ox3, bool amrflag) {
  std::stringstream msg;
  std::int32_t lx, ly, lz;
  int ll;
  int ox, oy, oz;
  MeshBlockTree *bt = proot_;
  lx=myloc.lx1, ly=myloc.lx2, lz=myloc.lx3, ll=myloc.level;

  lx+=ox1; ly+=ox2; lz+=ox3;
  if (lx<0) {
    if (pmesh_->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic
        || pmesh_->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic) {
      lx = (pmesh_->nmb_rootx1<<(ll - pmesh_->root_level)) - 1;
    } else {
      return nullptr;
    }
  }
  if (lx>=pmesh_->nmb_rootx1<<(ll-pmesh_->root_level)) {
    if (pmesh_->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic
        || pmesh_->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::shear_periodic) {
      lx = 0;
    } else {
      return nullptr;
    }
  }
  if (ly<0) {
    if (pmesh_->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic) {
      ly = (pmesh_->nmb_rootx2<<(ll - pmesh_->root_level)) - 1;
    } else {
      return nullptr;
    }
  }
  if (ly>=pmesh_->nmb_rootx2<<(ll-pmesh_->root_level)) {
    if (pmesh_->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic) {
      ly = 0;
    } else {
      return nullptr;
    }
  }
  std::int32_t num_x3 = pmesh_->nmb_rootx3<<(ll - pmesh_->root_level);
  if (lz<0) {
    if (pmesh_->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic) {
      lz = num_x3 - 1;
    } else {
      return nullptr;
    }
  }
  if (lz>=num_x3) {
    if (pmesh_->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic) {
      lz = 0;
    } else {
      return nullptr;
    }
  }

  if (ll<1) return proot_; // single grid; return root

  for (int level=0; level<ll; level++) {
    if (bt->pleaf_ == nullptr) { // leaf
      if (level == ll-1) {
        return bt;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Neighbor search failed; MeshBlockTree broken." << std::endl;
        exit(EXIT_FAILURE);
        return nullptr;
      }
    }
    // find a leaf in the next level
    int sh=ll-level-1;
    ox = ((lx>>sh) & 1) == 1;
    oy = ((ly>>sh) & 1) == 1;
    oz = ((lz>>sh) & 1) == 1;
    bt=bt->GetLeaf(ox, oy, oz);
    if (bt == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Neighbor search failed; MeshBlockTree broken." << std::endl;
      exit(EXIT_FAILURE);
      return nullptr;
    }
  }
  if (bt->pleaf_ == nullptr) { // leaf on the same level
    return bt;
  }
  // one level finer: check if it is a leaf
  ox = oy = oz = 0;
  if (ox1 < 0) ox = 1;
  if (ox2 < 0) oy = 1;
  if (ox3 < 0) oz = 1;
  MeshBlockTree *btleaf = bt->GetLeaf(ox, oy, oz);
  if (btleaf->pleaf_ == nullptr) {
    return bt;  // return this block
  }
  if (!amrflag) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Neighbor search failed. The Block Tree is broken." << std::endl;
    exit(EXIT_FAILURE);
  }
  return bt;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlockTree* MeshBlockTree::FindMeshBlock(LogicalLocation tloc)
//! \brief find MeshBlock with LogicalLocation tloc and return a pointer

MeshBlockTree* MeshBlockTree::FindMeshBlock(LogicalLocation tloc) {
  if (tloc.level == lloc_.level) return this;
  if (pleaf_ == nullptr) return nullptr;
  // get leaf index
  int sh = tloc.level - lloc_.level - 1;
  int mx = (((tloc.lx1>>sh) & 1) == 1);
  int my = (((tloc.lx2>>sh) & 1) == 1);
  int mz = (((tloc.lx3>>sh) & 1) == 1);
  int n = mx + (my<<1) + (mz<<2);
  if (pleaf_[n] == nullptr) {
    return nullptr;
  }
  return pleaf_[n]->FindMeshBlock(tloc);
}
