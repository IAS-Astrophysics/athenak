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
  HostArray1D<int> cyc_since_ref; // # of cycles since MB last refined/derefined
  int max_nmb;                  // max number of MBs allowed in calculation (memory limit)
  int nmb_created;              // total number of MeshBlocks created via AMR on this rank
  int nmb_deleted;              // total number of MeshBlocks deleted via AMR on this rank
  int ncycle_check_amr;         // # of cycles between checking refinement/derefinement
  int ncycle_ref_inter;         // # of cycles between allowing refinement/derefinement

  // following 4x arrays allocated with length [nranks] only with AMR
  int *nref_eachrank;     // number of MBs refined per rank
  int *nderef_eachrank;   // number of MBs de-refined per rank
  int *nref_rsum;         // running sum of number of MBs refined per rank
  int *nderef_rsum;       // running sum of number of MBs de-refined per rank
  // following 2x arrays allocated with length [nmb_new] and [nmb_old]] only with AMR
  int *newtoold;          // mapping of new gid (index n) to old gid
  int *oldtonew;          // mapping of old gid (index n) to new gid

  // functions
  bool CheckForRefinement(MeshBlockPack* pmbp);
  void AdaptiveMeshRefinement(Driver *pdrive, ParameterInput *pin);
  void UpdateMeshBlockTree(int &nnew, int &ndel);
  void RedistAndRefineMeshBlocks(ParameterInput *pin, int nnew, int ndel);

  void RestrictCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void RefineCC(int nmb, DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void DerefineCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);

  void RestrictFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void RefineFC(int nmb, DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void DerefineFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);

 private:
  // data
  Mesh *pmy_mesh;
  Real d_threshold_, dd_threshold_, dp_threshold_, dv_threshold_;
  bool check_cons_;
};

#endif // MESH_MESH_REFINEMENT_HPP_
