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
#include "hydro/eos/hydro_eos.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u0("cons",1,1,1,1,1),
  w0("prim",1,1,1,1,1),
  u1("cons1",1,1,1,1,1),
  divf("divF",1,1,1,1,1),
  uflx_x1face("uflx_x1face",1,1,1),
  uflx_x2face("uflx_x2face",1,1,1),
  uflx_x3face("uflx_x3face",1,1,1)
{
  // construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("hydro","eos");
  if (eqn_of_state.compare("adiabatic") == 0) {
    peos = new AdiabaticHydro(ppack, pin);
    nhydro = 5;
  } else if (eqn_of_state.compare("isothermal") == 0) {
    peos = new IsothermalHydro(ppack, pin);
    nhydro = 4;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Initialize number of scalars
  nscalars = pin->GetOrAddInteger("hydro","nscalars",0);

  // set time-evolution option (default=dynamic) [error checked in driver constructor]
  std::string evolution_t = pin->GetOrAddString("hydro","evolution","dynamic");

  // allocate memory for conserved and primitive variables
  int nmb = ppack->nmb_thispack;
  auto &ncells = ppack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;
  Kokkos::realloc(u0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(w0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);

  // allocate memory for boundary buffers
  pbvals = new BoundaryValues(ppack, pin);
  for (int i=0; i<nmb; ++i) {
    std::vector<BoundaryBuffer> snd, rcv;
    pbvals->AllocateBuffersCCVars((nhydro+nscalars), ppack->mb_cells, snd, rcv);
    pbvals->send_buf.push_back(snd);
    pbvals->recv_buf.push_back(rcv);
  }

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {

    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method_ = ReconstructionMethod::dc;

    } else if (xorder.compare("plm") == 0) {
      recon_method_ = ReconstructionMethod::plm;

    } else if (xorder.compare("ppm") == 0) {
      // check that nghost > 2
      if (ncells.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "PPM reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << ncells.ng << std::endl;
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
      if (evolution_t.compare("dynamic") == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro>/rsolver = '" << rsolver
                  << "' cannot be used with hydrodynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      } else {
        rsolver_method_ = Hydro_RSolver::advect;
      }

    } else  if (evolution_t.compare("dynamic") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/rsolver = '" << rsolver
                << "' cannot be used with non-hydrodynamic problems" << std::endl;
      std::exit(EXIT_FAILURE);

    } else if (rsolver.compare("llf") == 0) {
      rsolver_method_ = Hydro_RSolver::llf;

    } else if (rsolver.compare("hllc") == 0) {
      if (peos->eos_data.is_adiabatic) {
        rsolver_method_ = Hydro_RSolver::hllc;
      } else { 
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro>/rsolver = '" << rsolver
                  << "' cannot be used with isothermal EOS" << std::endl;
        std::exit(EXIT_FAILURE); 
        }  

    } else if (rsolver.compare("roe") == 0) {
      rsolver_method_ = Hydro_RSolver::roe;

    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE); 
    }}

    // allocate registers, flux divergence, scratch arrays for time-dep probs
    Kokkos::realloc(u1,   nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(divf, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageStartTasks
//  \brief adds Hydro tasks to stage start TaskList
//  These are taks that must be cmpleted (such as posting MPI receives, setting 
//  BoundaryCommStatus flags, etc) over all MeshBlocks before stage can be run.

void Hydro::HydroStageStartTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added)
{
  auto hydro_init = tl.AddTask(&Hydro::HydroInitRecv, this, start);
  added.emplace_back(hydro_init);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageRunTasks
//  \brief adds Hydro tasks to stage run TaskList

void Hydro::HydroStageRunTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added)
{
  // WARNING: If number or order of Hydro tasks below is changed then index of hydro_recv
  // in Mesh::InitPhysicsModules may need to be changed 

  auto hydro_copycons = tl.AddTask(&Hydro::HydroCopyCons, this, start);
  auto hydro_divflux  = tl.AddTask(&Hydro::HydroDivFlux, this, hydro_copycons);
  auto hydro_update  = tl.AddTask(&Hydro::HydroUpdate, this, hydro_divflux);
  auto hydro_send  = tl.AddTask(&Hydro::HydroSend, this, hydro_update);
  auto hydro_recv  = tl.AddTask(&Hydro::HydroReceive, this, hydro_send);
  auto hydro_phybcs  = tl.AddTask(&Hydro::HydroApplyPhysicalBCs, this, hydro_recv);
  auto hydro_con2prim  = tl.AddTask(&Hydro::ConToPrim, this, hydro_phybcs);
  auto hydro_newdt  = tl.AddTask(&Hydro::NewTimeStep, this, hydro_con2prim);

  added.emplace_back(hydro_copycons);
  added.emplace_back(hydro_divflux);
  added.emplace_back(hydro_update);
  added.emplace_back(hydro_send);
  added.emplace_back(hydro_recv);
  added.emplace_back(hydro_phybcs);
  added.emplace_back(hydro_con2prim);
  added.emplace_back(hydro_newdt);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageEndTasks
//  \brief adds Hydro tasks to stage end TaskList
//  These are tasks that can only be cmpleted after all the stage run tasks are finished
//  over all MeshBlocks, such as clearing all MPI non-blocking sends, etc.

void Hydro::HydroStageEndTasks(TaskList &tl, TaskID start, std::vector<TaskID> &added)
{
  auto hydro_clear = tl.AddTask(&Hydro::HydroClearSend, this, start);
  added.emplace_back(hydro_clear);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroInitRecv
//  \brief
// initialize all boundary receive status flags to waiting, post non-blocking receives

TaskStatus Hydro::HydroInitRecv(Driver *pdrive, int stage)
{
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;

  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (pmy_pack->pmb->nghbr[n].gid.h_view(m) >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (pbvals->nghbr_x1face[n].rank != global_variable::my_rank) {
          using Kokkos::ALL;
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = pbvals->CreateMPItag(lid, n, PhysicsID::Hydro_ID);
          auto recvbuf = Kokkos::subview(bbuf.recv_x1face,n,ALL,ALL,ALL,ALL);
          void* recv_ptr = recvbuf.data();
          int ierr = MPI_Irecv(recv_ptr, recvbuf.size(), MPI_ATHENA_REAL,
            pbvals->nghbr_x1face[n].rank, tag, MPI_COMM_WORLD, &(bbuf.recv_rq_x1face[n]));
        }
#endif
        pbvals->recv_buf[m][n].bcomm_stat = BoundaryCommStatus::waiting;
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroClearRecv
//  \brief

TaskStatus Hydro::HydroClearRecv(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  // wait for all non-blocking receives to finish before continuing 
  BoundaryValues* pbvals = pmesh_->FindMeshBlock(my_mbgid_)->pbvals;

  for (int n=0; n<nnghbr; ++n) {
    if (pbvals->nghbr_x1face[n].gid >= 0) {
      if (pbvals->nghbr_x1face[n].rank != global_variable::my_rank) {
        MPI_Wait(&(bbuf.recv_rq_x1face[n]), MPI_STATUS_IGNORE);
      }
    }
  }

#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroClearSend
//  \brief

TaskStatus Hydro::HydroClearSend(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  // wait for all non-blocking sends to finish before continuing 
  BoundaryValues* pbvals = pmesh_->FindMeshBlock(my_mbgid_)->pbvals;

  for (int n=0; n<nnghbr; ++n) {
    if (pbvals->nghbr_x1face[n].gid >= 0) {
      if (pbvals->nghbr_x1face[n].rank != global_variable::my_rank) {
        MPI_Wait(&(bbuf.send_rq_x1face[n]), MPI_STATUS_IGNORE);
      }
    }
  }

#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroCopyCons
//  \brief  copy u0 --> u1 in first stage

TaskStatus Hydro::HydroCopyCons(Driver *pdrive, int stage)
{
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroSend
//  \brief

TaskStatus Hydro::HydroSend(Driver *pdrive, int stage) 
{
  TaskStatus tstat = pbvals->SendBuffers(u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroReceive
//  \brief

TaskStatus Hydro::HydroReceive(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbvals->RecvBuffers(u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ConToPrim
//  \brief

TaskStatus Hydro::ConToPrim(Driver *pdrive, int stage)
{
  peos->ConsToPrim(u0, w0);
  return TaskStatus::complete;
}

} // namespace hydro
