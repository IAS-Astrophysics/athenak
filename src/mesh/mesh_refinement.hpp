#ifndef MESH_MESH_REFINEMENT_HPP_
#define MESH_MESH_REFINEMENT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement.hpp
//  \brief defines MeshRefinement class containing data and functions controlling SMR/AMR

//----------------------------------------------------------------------------------------
//! \class MeshRefinement
//  \brief data/functions associated with SMR/AMR

class MeshRefinement {
  // mesh classes (Mesh, MeshBlock, MeshBlockPack, MeshBlockTree) like to play together
//  friend class Mesh;
//  friend class MeshBlockPack;

 public:
  MeshRefinement(Mesh *pm, ParameterInput *pin);
  ~MeshRefinement();

  // data
  DualArray1D<int> refine_flag;   // refinement flag for each MeshBlock
  // following 8x arrays allocated with length [nranks] only with AMR
  int *nref, *nderef;
/**
  int *rdisp, *ddisp;
  int *bnref, *bnderef;
  int *brdisp, *bddisp;
**/

  // functions
  bool CheckForRefinement(MeshBlockPack* pmbp);
  void AdaptiveMeshRefinement();
  void UpdateMeshBlockTree(int &nnew, int &ndel);
  void RestrictCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void RestrictFC(DvceFaceFld4D<Real> &a, DvceFaceFld4D<Real> &ca);

 private:
  // data
  Mesh *pmy_mesh;
  Real d_threshold_, dd_threshold_, dv_threshold_;
  bool check_cons_;
};

#endif // MESH_MESH_REFINEMENT_HPP_
