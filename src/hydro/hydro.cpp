//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//  \brief implementation of functions in class Hydro

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "hydro/eos/eos.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(Mesh *pm, ParameterInput *pin, int gid) :
  pmesh_(pm), my_mbgid_(gid),
  u0("cons",1,1,1,1), w0("prim",1,1,1,1), u1("cons1",1,1,1,1), divf("divF",1,1,1,1),
  uflx_x1face("uflx_x1face",1,1,1),
  uflx_x2face("uflx_x2face",1,1,1),
  uflx_x3face("uflx_x3face",1,1,1)
{
  // construct EOS object (no default)
  peos = new EquationOfState(pmesh_, pin, my_mbgid_);
  if (peos->eos_data.is_adiabatic) {
    nhydro = 5;
  } else {
    nhydro = 4;
  }

  // set time-evolution option (no default)
  {std::string evolution_t = pin->GetString("hydro","evolution");
  if (evolution_t.compare("static") == 0) {
    hydro_evol = HydroEvolution::hydro_static;

  } else if (evolution_t.compare("kinematic") == 0) {
    hydro_evol = HydroEvolution::kinematic;

  } else if (evolution_t.compare("dynamic") == 0) {
    hydro_evol = HydroEvolution::hydro_dynamic;

  } else if (evolution_t.compare("none") == 0) {
    hydro_evol = HydroEvolution::no_evolution;

  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> evolution = '" << evolution_t << "' not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }} // extra brace to limit scope of string

  // allocate memory for conserved and primitive variables
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ncells1 = pmb->mb_cells.nx1 + 2*(pmb->mb_cells.ng);
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*(pmb->mb_cells.ng)) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*(pmb->mb_cells.ng)) : 1;
  Kokkos::realloc(u0,nhydro, ncells3, ncells2, ncells1);
  Kokkos::realloc(w0,nhydro, ncells3, ncells2, ncells1);

  // allocate memory for boundary buffers
  pmb->pbvals->AllocateBuffers(bbuf, nhydro);

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (hydro_evol != HydroEvolution::no_evolution) {

    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");
  
    if (xorder.compare("dc") == 0) {
      recon_method_ = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method_ = ReconstructionMethod::plm;
    } else if (xorder.compare("ppm") == 0) {
      // check that nghost > 2
      if (pmb->mb_cells.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "PPM reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << pmb->mb_cells.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method_ = ReconstructionMethod::ppm;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }}

    // select Riemann solver (no default).  Test for compatibility of options
    {std::string rsolver = pin->GetString("hydro","rsolver");

    if (rsolver.compare("advection") == 0) {
      if (hydro_evol == HydroEvolution::hydro_dynamic) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro>/rsolver = '" << rsolver
                  << "' cannot be used with hydrodynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      } else {
        rsolver_method_ = RiemannSolver::advect;
      }
    } else if (hydro_evol != HydroEvolution::hydro_dynamic) { 
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/rsolver = '" << rsolver
                << "' cannot be used with non-hydrodynamic problems" << std::endl;
      std::exit(EXIT_FAILURE);
    } else if (rsolver.compare("llf") == 0) {
      rsolver_method_ = RiemannSolver::llf;
    } else if (rsolver.compare("hlle") == 0) {
      rsolver_method_ = RiemannSolver::hlle;
    } else if (rsolver.compare("hllc") == 0) {
      if (peos->eos_data.is_adiabatic) {
        rsolver_method_ = RiemannSolver::hllc;
      } else { 
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro>/rsolver = '" << rsolver
                  << "' cannot be used with isothermal EOS" << std::endl;
        std::exit(EXIT_FAILURE); 
        }  
    } else if (rsolver.compare("roe") == 0) {
      rsolver_method_ = RiemannSolver::roe;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE); 
    }}

    // allocate registers, flux divergence, scratch arrays for time-dep probs
    Kokkos::realloc(u1,nhydro, ncells3, ncells2, ncells1);
    Kokkos::realloc(divf,nhydro, ncells3, ncells2, ncells1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageStartTasks
//  \brief adds Hydro tasks to stage start TaskList
//  These are taks that must be cmpleted (such as posting MPI receives, setting 
//  BoundaryRecvStatus flags, etc) over all MeshBlocks before stage can be run.

void Hydro::HydroStageStartTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added)
{
  auto hydro_init = tl.AddTask(&Hydro::HydroInitStage, this, start);
  added.emplace_back(hydro_init);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageRunTasks
//  \brief adds Hydro tasks to stage run TaskList

void Hydro::HydroStageRunTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added)
{
  auto hydro_copycons = tl.AddTask(&Hydro::HydroCopyCons, this, start);
  auto hydro_divflux  = tl.AddTask(&Hydro::HydroDivFlux, this, hydro_copycons);
  auto hydro_update  = tl.AddTask(&Hydro::HydroUpdate, this, hydro_divflux);
  auto hydro_send  = tl.AddTask(&Hydro::HydroSend, this, hydro_update);
  auto hydro_newdt  = tl.AddTask(&Hydro::NewTimeStep, this, hydro_send);
  auto hydro_recv  = tl.AddTask(&Hydro::HydroReceive, this, hydro_newdt);
  auto hydro_con2prim  = tl.AddTask(&Hydro::ConToPrim, this, hydro_recv);

  added.emplace_back(hydro_copycons);
  added.emplace_back(hydro_divflux);
  added.emplace_back(hydro_update);
  added.emplace_back(hydro_send);
  added.emplace_back(hydro_newdt);
  added.emplace_back(hydro_recv);
  added.emplace_back(hydro_con2prim);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageEndTasks
//  \brief adds Hydro tasks to stage end TaskList
//  These are tasks that can only be cmpleted after all the stage run tasks are finished
//  over all MeshBlocks.  Current NoOp.

void Hydro::HydroStageEndTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added)
{
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroCopyCons
//  \brief

TaskStatus Hydro::HydroInitStage(Driver *pdrive, int stage)
{
  BoundaryValues* pbval = pmesh_->FindMeshBlock(my_mbgid_)->pbvals;
  // initialize all boundary status arrays to waiting
  for (int n=0; n<2; ++n) {
//    if (pbval->bndry_flag[n]==BoundaryFlag::block ||
//        pbval->bndry_flag[n]==BoundaryFlag::periodic) {
    if (pbval->nghbr_x1face[1-n].gid >= 0) {
      bbuf.bstat_x1face[n] = BoundaryRecvStatus::waiting;
    }
  }
  if (pmesh_->nx2gt1) {
    for (int n=0; n<2; ++n) {
//      if (pbval->bndry_flag[n+2]==BoundaryFlag::block ||
//          pbval->bndry_flag[n+2]==BoundaryFlag::periodic) {
      if (pbval->nghbr_x2face[1-n].gid >= 0) {
        bbuf.bstat_x2face[n] = BoundaryRecvStatus::waiting;
      }
    }
    for (int n=0; n<4; ++n) {
//      if (pbval->bndry_flag[(n/2)+2]==BoundaryFlag::block ||
//          pbval->bndry_flag[(n/2)+2]==BoundaryFlag::periodic){
      if (pbval->nghbr_x1x2ed[3-n].gid >= 0) {
        bbuf.bstat_x1x2ed[n] = BoundaryRecvStatus::waiting;
      }
    }
  }
  if (pmesh_->nx3gt1) {
    for (int n=0; n<2; ++n) {
//      if (pbval->bndry_flag[n+4]==BoundaryFlag::block ||
//          pbval->bndry_flag[n+4]==BoundaryFlag::periodic) {
      if (pbval->nghbr_x3face[1-n].gid >= 0) {
        bbuf.bstat_x3face[n] = BoundaryRecvStatus::waiting;
      }
    }
    for (int n=0; n<4; ++n) {
//      if (pbval->bndry_flag[(n/2)+4]==BoundaryFlag::block ||
//          pbval->bndry_flag[(n/2)+4]==BoundaryFlag::periodic){
      if (pbval->nghbr_x3x1ed[3-n].gid >= 0) {
        bbuf.bstat_x3x1ed[n] = BoundaryRecvStatus::waiting;
      }
      if (pbval->nghbr_x2x3ed[3-n].gid >= 0) {
        bbuf.bstat_x2x3ed[n] = BoundaryRecvStatus::waiting;
      }
    }
    for (int n=0; n<8; ++n) {
//      if (pbval->bndry_flag[(n/4)+4]==BoundaryFlag::block ||
//          pbval->bndry_flag[(n/4)+4]==BoundaryFlag::periodic){
      if (pbval->nghbr_corner[7-n].gid >= 0) {
        bbuf.bstat_corner[n] = BoundaryRecvStatus::waiting;
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroCopyCons
//  \brief

TaskStatus Hydro::HydroCopyCons(Driver *pdrive, int stage)
{
  MeshBlock* pmb = pmesh_->FindMeshBlock(my_mbgid_);
  // copy u0 --> u1 in first stage
  if (stage == 1) {
    Kokkos::deep_copy(pmb->exe_space, u1, u0);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroSend
//  \brief

TaskStatus Hydro::HydroSend(Driver *pdrive, int stage) 
{
  MeshBlock* pmb = pmesh_->FindMeshBlock(my_mbgid_);
  TaskStatus tstat;
  tstat = pmb->pbvals->SendCellCenteredVars(u0, nhydro, "hydro");
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroReceive
//  \brief

TaskStatus Hydro::HydroReceive(Driver *pdrive, int stage)
{
  MeshBlock* pmb = pmesh_->FindMeshBlock(my_mbgid_);
  TaskStatus tstat;
  tstat = pmb->pbvals->RecvCellCenteredVars(u0, nhydro, "hydro");
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ConToPrim
//  \brief

TaskStatus Hydro::ConToPrim(Driver *pdrive, int stage)
{
  peos->ConservedToPrimitive(u0, w0);
  return TaskStatus::complete;
}

} // namespace hydro
