#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.hpp
//  \brief definitions for ProblemGenerator class

#include <functional>
#include <memory>
#include <vector>

#include "geodesic-grid/spherical_grid.hpp"
#include "parameter_input.hpp"

using ProblemFinalizeFnPtr = void (*)(ParameterInput *pin, Mesh *pm);
using UserBoundaryFnPtr = void (*)(Mesh* pm);
using UserSrctermFnPtr = void (*)(Mesh* pm, const Real bdt);
using UserRefinementFnPtr = void (*)(MeshBlockPack* pmbp);
using UserHistoryFnPtr = void (*)(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \class ProblemGenerator

class ProblemGenerator {
 public:
  // constructor for new problems
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh);
  // constructor for restarts
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh, IOWrapper resfile,
                   bool single_file_per_rank=false);
  ~ProblemGenerator() = default;

  // true if user BCs are specified on any face
  bool user_bcs;

  // true if user srcterms are specified
  bool user_srcs;

  // true if user history outputs are specified
  bool user_hist;

  // vector of SphericalGrid objects for analysis
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;

  // function pointer for final work after main loop (e.g. compute errors).  Called by
  // Driver::Finalize()
  ProblemFinalizeFnPtr pgen_final_func=nullptr;
  // function pointer for user-enrolled BCs.  Called in ApplyPhysicalBCs in task list
  UserBoundaryFnPtr user_bcs_func=nullptr;
  UserSrctermFnPtr user_srcs_func=nullptr;
  UserRefinementFnPtr user_ref_func=nullptr;
  UserHistoryFnPtr user_hist_func=nullptr;

  // predefined problem generator functions (default test suite)
  void Advection(ParameterInput *pin, const bool restart);
  void AlfvenWave(ParameterInput *pin, const bool restart);
  void BondiAccretion(ParameterInput *pin, const bool restart);
  void CheckOrthonormalTetrad(ParameterInput *pin, const bool restart);
  void Hohlraum(ParameterInput *pin, const bool restart);
  void LinearWave(ParameterInput *pin, const bool restart);
  void LWImplode(ParameterInput *pin, const bool restart);
  void Monopole(ParameterInput *pin, const bool restart);
  void OrszagTang(ParameterInput *pin, const bool restart);
  void ShockTube(ParameterInput *pin, const bool restart);
  void RadiationLinearWave(ParameterInput *pin, const bool restart);
  void Z4cLinearWave(ParameterInput *pin, const bool restart);
  void SphericalCollapse(ParameterInput *pin, const bool restart);
  void Diffusion(ParameterInput *pin, const bool restart);

  // template for user-specified problem generator
  void UserProblem(ParameterInput *pin, const bool restart);

 private:
  bool single_file_per_rank; // for restart file naming
  Mesh* pmy_mesh_;
};

#endif // PGEN_PGEN_HPP_
