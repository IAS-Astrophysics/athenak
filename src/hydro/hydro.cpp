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
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "utils/create_mpitag.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u0("cons",1,1,1,1,1),
  w0("prim",1,1,1,1,1),
  u1("cons1",1,1,1,1,1),
  uflx("uflx",1,1,1,1,1)
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

  // read time-evolution option [already error checked in driver constructor]
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for conserved and primitive variables
  int nmb = ppack->nmb_thispack;
  auto &ncells = ppack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;
  Kokkos::realloc(u0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(w0, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_u = new BoundaryValueCC(ppack, pin);
  pbval_u->AllocateBuffersCC((nhydro+nscalars));

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

    } else if (xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      if (ncells.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "WENOZ reconstruction requires at least 3 ghost zones, "
            << "but <mesh>/nghost=" << ncells.ng << std::endl;
        std::exit(EXIT_FAILURE); 
      }                
      recon_method_ = ReconstructionMethod::wenoz;

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

//    } else if (rsolver.compare("hllc") == 0) {
//      if (peos->eos_data.is_adiabatic) {
//        rsolver_method_ = Hydro_RSolver::hllc;
//      } else { 
//        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
//                  << std::endl << "<hydro>/rsolver = '" << rsolver
//                  << "' cannot be used with isothermal EOS" << std::endl;
//        std::exit(EXIT_FAILURE); 
//        }  

//    } else if (rsolver.compare("roe") == 0) {
//      rsolver_method_ = Hydro_RSolver::roe;

    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE); 
    }}

    // allocate second registers, fluxes
    Kokkos::realloc(u1,       nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(uflx.x1f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(uflx.x2f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(uflx.x3f, nmb, (nhydro+nscalars), ncells3, ncells2, ncells1);
  }
}

//----------------------------------------------------------------------------------------
// destructor
  
Hydro::~Hydro()
{
  delete peos;
  delete pbval_u;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageStartTasks
//  \brief adds Hydro tasks to stage start TaskList
//  These are taks that must be cmpleted (such as posting MPI receives, setting 
//  BoundaryCommStatus flags, etc) over all MeshBlocks before stage can be run.

void Hydro::HydroStageStartTasks(TaskList &tl, TaskID start)
{
  auto hydro_init = tl.AddTask(&Hydro::HydroInitRecv, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageRunTasks
//  \brief adds Hydro tasks to stage run TaskList

void Hydro::HydroStageRunTasks(TaskList &tl, TaskID start)
{
  auto hydro_copycons = tl.AddTask(&Hydro::HydroCopyCons, this, start);
  auto hydro_fluxes = tl.AddTask(&Hydro::CalcFluxes, this, hydro_copycons);
  auto hydro_update = tl.AddTask(&Hydro::Update, this, hydro_fluxes);
  auto hydro_send = tl.AddTask(&Hydro::HydroSendU, this, hydro_update);
  auto hydro_recv = tl.AddTask(&Hydro::HydroRecvU, this, hydro_send);
  auto hydro_phybcs = tl.AddTask(&Hydro::HydroApplyPhysicalBCs, this, hydro_recv);
  auto hydro_con2prim = tl.AddTask(&Hydro::ConToPrim, this, hydro_phybcs);
  auto hydro_newdt = tl.AddTask(&Hydro::NewTimeStep, this, hydro_con2prim);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageEndTasks
//  \brief adds Hydro tasks to stage end TaskList
//  These are tasks that can only be cmpleted after all the stage run tasks are finished
//  over all MeshBlocks, such as clearing all MPI non-blocking sends, etc.

void Hydro::HydroStageEndTasks(TaskList &tl, TaskID start)
{
  auto hydro_clear = tl.AddTask(&Hydro::HydroClearSend, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroInitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI).

TaskStatus Hydro::HydroInitRecv(Driver *pdrive, int stage)
{
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications for cell-centered conserved variables
  auto &rbufu = pbval_u->recv_buf;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n, VariablesID::FluidCons_ID);
          auto recv_data = Kokkos::subview(rbufu[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int ierr = MPI_Irecv(recv_ptr, recv_data.size(), MPI_ATHENA_REAL,
            nghbr[n].rank.h_view(m), tag, MPI_COMM_WORLD, &(rbufu[n].comm_req[m]));
        }
#endif
        // initialize boundary receive status flag
        rbufu[n].bcomm_stat(m) = BoundaryCommStatus::waiting;
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Hydro::HydroClearRecv(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for U to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Hydro::HydroClearSend(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for U to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
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
//! \fn  void Hydro::HydroSendU
//  \brief sends cell-centered conserved variables

TaskStatus Hydro::HydroSendU(Driver *pdrive, int stage) 
{
  TaskStatus tstat = pbval_u->SendBuffersCC(u0, VariablesID::FluidCons_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroRecvU
//  \brief receives cell-centered conserved variables

TaskStatus Hydro::HydroRecvU(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_u->RecvBuffersCC(u0);
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
