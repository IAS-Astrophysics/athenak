#ifndef MESH_REFINEMENT_CRITERIA_HPP_
#define MESH_REFINEMENT_CRITERIA_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file refinement_criteria.hpp
//! \brief defines RefinementCriteria class containing data and functions controlling
//! how mesh is refined/derefined with AMR
//! This class implementes default refinement conditions:
//!   (1) min/max of selected variable
//!   (2) gradient of selected variable
//!   (3) second derivative of selected variable
//!   (4) region with specified radius of a selected point
//! Any number of refinement criteria can be specified using multiple
//! <refinement_criteriaN> blocks in the input file.  Each block can select a different
//! method and/or hydro/MHD/radiation variables can be selected.
//! TODO(@JMS): user-defined variables can also be selected
//!
//! User-defined refinement conditions can also be enrolled by setting the *usr_ref_func
//! pointer in the problem generator.

#include <string>
#include <vector>

#include "athena.hpp"

// identifiers for refinement criteria methods
enum class RefCritMethod {min_max, slope, second_deriv, location, user};

using DvceArray5DnSlice = Kokkos::Subview<DvceArray5D<Real>,
                          std::remove_const_t<decltype(Kokkos::ALL)>,
                          int,
                          std::remove_const_t<decltype(Kokkos::ALL)>,
                          std::remove_const_t<decltype(Kokkos::ALL)>,
                          std::remove_const_t<decltype(Kokkos::ALL)>>;

//----------------------------------------------------------------------------------------
//! \struct RefinementCriteriaData
//! \brief physical size in a Mesh or a MeshBlock

struct RefCritData {
  RefCritMethod rmethod;           // refinement method (min_max, slope, etc.)
  std::string rvariable;           // name of variable to be tested for refinement
  Real rvalue_min, rvalue_max;     // min/max criteria for refinement
  Real rloc_x1, rloc_x2, rloc_x3;  // x1-,x2-,x3-locations of point to refine around
  Real rloc_rad;                   // radius of region around point to be refined
  DvceArray5DnSlice rdata;         // slice of variable "n" in 5D array(m,n,k,j,i)
};

//----------------------------------------------------------------------------------------
//! \class RefinementCriteria
//! \brief data/functions associated with various refinement criteria for AMR

class RefinementCriteria {
 public:
  RefinementCriteria(Mesh *pm, ParameterInput *pin);
  ~RefinementCriteria();

  // data
  int ncriteria;
  int nderived;
  std::vector<RefCritData> rcrit;

  // functions
  void SetRefinementData(MeshBlockPack* pmbp, bool count, bool load);
  void CheckMinMax(MeshBlockPack* pmbp, RefCritData crit);
  void CheckSlope(MeshBlockPack* pmbp, RefCritData crit);
  void CheckSecondDeriv(MeshBlockPack* pmbp, RefCritData crit);
  void CheckLocation(MeshBlockPack* pmbp, RefCritData crit);

 private:
  // data
  Mesh *pmy_mesh;
  DvceArray5D<Real> dvars;  // derived variables
};
#endif // MESH_REFINEMENT_CRITERIA_HPP_
