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
class Coordinates;
class Viscosity;
class SourceTerms;
class Driver;

// constants that enumerate Hydro Riemann Solver options
enum class Hydro_RSolver {advect, llf, hlle, hllc, roe, llf_sr, hlle_sr, hllc_sr,
                          hlle_gr};

//----------------------------------------------------------------------------------------
//! \struct HydroTaskIDs
//  \brief container to hold TaskIDs of all hydro tasks
  
struct HydroTaskIDs
{   
  TaskID init_recv;
  TaskID copy_cons;
  TaskID calc_flux;
  TaskID update;
  TaskID sendu;
  TaskID recvu;
  TaskID phys_bcs;
  TaskID cons2prim;
  TaskID newdt;
  TaskID clear_send;
};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro
{
 public:
  Hydro(MeshBlockPack *ppack, ParameterInput *pin);
  ~Hydro();

  // data
  EquationOfState *peos;  // chosen EOS

  // flags to denote relativistic dynamics
  bool is_special_relativistic = false;
  bool is_general_relativistic = false;

  int nhydro;             // number of hydro variables (5/4 for adiabatic/isothermal)
  int nscalars;           // number of passive scalars
  DvceArray5D<Real> u0;   // conserved variables
  DvceArray5D<Real> w0;   // primitive variables

  // Object containing boundary communication buffers and routines for u
  BoundaryValueCC *pbval_u;

  // Object(s) for extra physics (viscosity, srcterms)
  Viscosity *pvisc = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables at intermediate step 
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  Real dtnew;

  // container to hold names of TaskIDs
  HydroTaskIDs id;

  // functions
  void AssembleHydroTasks(TaskList &start, TaskList &run, TaskList &end);
  TaskStatus InitRecv(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus CalcFluxes(Driver *d, int stage);
  TaskStatus ExpRKUpdate(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage); 
  TaskStatus RecvU(Driver *d, int stage); 
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);  // in file in hydro/bvals dir

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
  void ShearInnerX1(int m);
  void ShearOuterX1(int m);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
  ReconstructionMethod recon_method_;
  Hydro_RSolver rsolver_method_;
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
