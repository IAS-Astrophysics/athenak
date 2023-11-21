#ifndef DYNGR_DYNGR_HPP_
#define DYNGR_DYNGR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr.hpp
//  \brief definitions for DynGR class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "driver/driver.hpp"
#include "eos/primitive_solver_hyd.hpp"

enum class DynGR_RSolver {llf_dyngr, hlle_dyngr};     // Riemann solvers for dynamical GR
enum class DynGR_EOS {eos_ideal, eos_piecewise_poly,
                      eos_poly}; // EOS policies for dynamical GR
enum class DynGR_Error {reset_floor};                 // Error policies for dynamical GR

//----------------------------------------------------------------------------------------
//! \struct DynGRTaskIDs
//  \brief container to hold TaskIDs of all dyngr tasks

struct DynGRTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID settmunu;
  TaskID sendf;
  TaskID recvf;
  TaskID expl;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID efld;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID bcs;
  TaskID c2p;
  TaskID newdt;
  TaskID clear;
  TaskID zrecv;
  TaskID zcopyu;
  TaskID zmattersrc;
  TaskID zcrhsdep;
  TaskID zcrhs;
  TaskID zsombc;
  TaskID zexpl;
  TaskID zsendu;
  TaskID zrecvu;
  TaskID znewdt;
  TaskID zbcs;
  TaskID zalgc;
  TaskID z4tad;
  TaskID zadmc;
  TaskID zclear;
  TaskID zrestu;
  TaskID zadep;
  TaskID c2pdep;
  TaskID rkdep;
};

namespace dyngr {

class DynGR {
 public:
  DynGR(MeshBlockPack *ppack, ParameterInput *pin);
  virtual ~DynGR();

  // container to hold names of TaskIDs
  DynGRTaskIDs id;

  TaskStatus SetTmunu(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver *d, int stage);

  // functions

  virtual void QueueDynGRTasks() = 0;

  virtual TaskStatus ConToPrim(Driver* pdrive, int stage) = 0;
  virtual void ConToPrimBC(int is, int ie, int js, int je, int ks, int ke) = 0;
  virtual void PrimToConInit(int is, int ie, int js, int je, int ks, int ke) = 0;
  virtual void ConvertInternalEnergyToPressure(int is, int ie,
                                               int js, int je, int ks, int ke) = 0;

  virtual void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                             const Real dt, DvceArray5D<Real> &u0, int nghost) = 0;

  // DynGR policies
  DynGR_RSolver rsolver_method;
  DynGR_RSolver fofc_method;
  DynGR_EOS eos_policy;
  DynGR_Error error_policy;

 protected:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  int scratch_level;        // GPU scratch level for flux and source calculations
  bool enforce_maximum;     // enforce local maximum principle during FOFC
  Real dmp_M;               // threshold multiplier for discrete maximum principle.
  bool fixed_evolution;     // Disable mhd evolution
};

template<class EOSPolicy, class ErrorPolicy>
class DynGRPS : public DynGR {
 public:
  DynGRPS(MeshBlockPack *ppack, ParameterInput *pin) :
      DynGR(ppack, pin), eos("mhd", ppack, pin) {}
  virtual ~DynGRPS() {}

  // Dynamical EOS
  PrimitiveSolverHydro<EOSPolicy, ErrorPolicy> eos;

  // CalculateFluxes function templated over Riemann Solvers
  template<DynGR_RSolver T>
  TaskStatus CalcFluxes(Driver *d, int stage);

  template<DynGR_RSolver T>
  void FOFC(Driver *d, int stage);

  // functions
  virtual void QueueDynGRTasks();

  virtual TaskStatus ConToPrim(Driver* pdrive, int stage);
  virtual void ConToPrimBC(int is, int ie, int js, int je, int ks, int ke);
  virtual void PrimToConInit(int is, int ie, int js, int je, int ks, int ke);
  virtual void ConvertInternalEnergyToPressure(int is, int ie,
                                               int js, int je, int ks, int ke);

  virtual void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                             const Real dt, DvceArray5D<Real> &u0, int nghost);

  template<int NGHOST>
  void AddCoordTermsEOS(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                        const Real dt, DvceArray5D<Real> &u0);
};

// Factory function for generating DynGR based on parameter input.
// Used to make the MeshBlockPack creation a little bit cleaner.
DynGR* BuildDynGR(MeshBlockPack *ppack, ParameterInput *pin);

} // namespace dyngr

#endif  // DYNGR_DYNGR_HPP_
