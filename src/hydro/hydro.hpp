#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Viscosity;
class Driver;

// constants that enumerate Hydro Riemann Solver options
enum class Hydro_RSolver {advect, llf, hllc, roe};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class HydroSourceTerm

class HydroSourceTerm
{
 public:
  HydroSourceTerm(Hydro *pmy_hydro, ParameterInput *pin);
  ~HydroSourceTerm();

  // objects for external forcing
  DvceArray5D<Real> force;  // forcing for driving hydro variables

  DvceArray3D<Real> x1sin;   // array for pre-computed sin(k x)
  DvceArray3D<Real> x1cos;   // array for pre-computed cos(k x)
  DvceArray3D<Real> x2sin;   // array for pre-computed sin(k y)
  DvceArray3D<Real> x2cos;   // array for pre-computed cos(k y)
  DvceArray3D<Real> x3sin;   // array for pre-computed sin(k z)
  DvceArray3D<Real> x3cos;   // array for pre-computed cos(k z)

  DvceArray3D<Real> amp1;
  DvceArray3D<Real> amp2;
  DvceArray3D<Real> amp3;
  // amplitudes for OU process
  DvceArray3D<Real> amp1_tmp;
  DvceArray3D<Real> amp2_tmp;
  DvceArray3D<Real> amp3_tmp;

  DvceArray2D<int64_t> seeds; // random seeds

  bool first_time_;
  int nlow,nhigh,ntot,nwave;
  Real tcorr,dedt;
  Real expo,exp_prl,exp_prp;
  std::string forcing_type;
  int forcing;

  void ApplyForcing();
  void ApplySourceTerms(Driver *d, int stage);
  KOKKOS_INLINE_FUNCTION Real RanGaussian(int64_t *idum);
  KOKKOS_INLINE_FUNCTION Real Ran2(int64_t *idum);

 private:
  Hydro* pmy_hydro;  // ptr to MeshBlockPack containing this Hydro
};



//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro
{
 friend class HydroSourceTerm; // might be not necessary
 public:
  Hydro(MeshBlockPack *ppack, ParameterInput *pin);
  ~Hydro();

  // data
  EquationOfState *peos;  // chosen EOS
  Viscosity *pvisc=nullptr;       // (optional) viscosity 

  int nhydro;             // number of hydro variables (5/4 for adiabatic/isothermal)
  int nscalars;           // number of passive scalars
  DvceArray5D<Real> u0;   // conserved variables
  DvceArray5D<Real> w0;   // primitive variables

  // Object containing boundary communication buffers and routines for u
  BoundaryValueCC *pbval_u;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables at intermediate step 
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  Real dtnew;

  // source terms
  HydroSourceTerm *hsrc;

  // functions
  void HydroStageStartTasks(TaskList &tl, TaskID start);
  void HydroStageRunTasks(TaskList &tl, TaskID start);
  void HydroStageEndTasks(TaskList &tl, TaskID start);
  TaskStatus HydroInitRecv(Driver *d, int stage);
  TaskStatus HydroClearRecv(Driver *d, int stage);
  TaskStatus HydroClearSend(Driver *d, int stage);
  TaskStatus HydroCopyCons(Driver *d, int stage);
  TaskStatus CalcFluxes(Driver *d, int stage);
  TaskStatus Update(Driver *d, int stage);
  TaskStatus HydroSendU(Driver *d, int stage); 
  TaskStatus HydroRecvU(Driver *d, int stage); 
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus ViscousFluxes(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus HydroApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus ApplySourceTerms(Driver *d, int stage);

  // functions to set physical BCs for Hydro conserved variables, applied to single MB
  // specified by argument 'm'. 
  void ReflectInnerX1(int m);
  void ReflectOuterX1(int m);
  void ReflectInnerX2(int m);
  void ReflectOuterX2(int m);
  void ReflectInnerX3(int m);
  void ReflectOuterX3(int m);
  void OutflowInnerX1(int m);
  void OutflowOuterX1(int m);
  void OutflowInnerX2(int m);
  void OutflowOuterX2(int m);
  void OutflowInnerX3(int m);
  void OutflowOuterX3(int m);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  ReconstructionMethod recon_method_;
  Hydro_RSolver rsolver_method_;
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
