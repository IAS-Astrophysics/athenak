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
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(Mesh *pm, ParameterInput *pin, int gid) :
  pmesh_(pm), my_mbgid_(gid),
  u0("cons",1,1,1,1),
  w0("prim",1,1,1,1),
  u1("cons1",1,1,1,1),
  divf("divF",1,1,1,1),
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

  // set time-evolution option (default=dynamic)
  {std::string evolution_t = pin->GetOrAddString("hydro","evolution","dynamic");
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

  // Initialize number of scalars
  nscalars = pin->GetOrAddInteger("hydro","nscalars",0);

  // allocate memory for conserved and primitive variables
  MeshBlock *pmb = pmesh_->FindMeshBlock(my_mbgid_);
  int ncells1 = pmb->mb_cells.nx1 + 2*(pmb->mb_cells.ng);
  int ncells2 = (pmb->mb_cells.nx2 > 1)? (pmb->mb_cells.nx2 + 2*(pmb->mb_cells.ng)) : 1;
  int ncells3 = (pmb->mb_cells.nx3 > 1)? (pmb->mb_cells.nx3 + 2*(pmb->mb_cells.ng)) : 1;
  Kokkos::realloc(u0, (nhydro+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(w0, (nhydro+nscalars), ncells3, ncells2, ncells1);

  // allocate memory for boundary buffers
  pmb->pbvals->AllocateBuffers(bbuf, (nhydro+nscalars));

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
    Kokkos::realloc(u1,   (nhydro+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(divf, (nhydro+nscalars), ncells3, ncells2, ncells1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroStageStartTasks
//  \brief adds Hydro tasks to stage start TaskList
//  These are taks that must be cmpleted (such as posting MPI receives, setting 
//  BoundaryRecvStatus flags, etc) over all MeshBlocks before stage can be run.

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
  auto hydro_con2prim  = tl.AddTask(&Hydro::ConToPrim, this, hydro_recv);
  auto hydro_newdt  = tl.AddTask(&Hydro::NewTimeStep, this, hydro_con2prim);

  added.emplace_back(hydro_copycons);
  added.emplace_back(hydro_divflux);
  added.emplace_back(hydro_update);
  added.emplace_back(hydro_send);
  added.emplace_back(hydro_recv);
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

TaskStatus Hydro::HydroInitRecv(Driver *pdrive, int stage)
{
  BoundaryValues* pbvals = pmesh_->FindMeshBlock(my_mbgid_)->pbvals;
  int lid = my_mbgid_ - pmesh_->gids; // local ID for creating MPI tags

  // initialize all boundary receive status flags to waiting, post non-blocking receives
  // x1 faces
  for (int n=0; n<2; ++n) {
    if (pbvals->nghbr_x1face[n].gid >= 0) {
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
      bbuf.bstat_x1face[n] = BoundaryRecvStatus::waiting;
    }
  }

  // x2faces and x1x2 edges
  if (pmesh_->nx2gt1) {
    for (int n=0; n<2; ++n) {
      if (pbvals->nghbr_x2face[n].gid >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (pbvals->nghbr_x2face[n].rank != global_variable::my_rank) {
          using Kokkos::ALL;
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = pbvals->CreateMPItag(lid, (2+n), PhysicsID::Hydro_ID);
          auto recvbuf = Kokkos::subview(bbuf.recv_x2face,n,ALL,ALL,ALL,ALL);
          void* recv_ptr = recvbuf.data();
          int ierr = MPI_Irecv(recv_ptr, recvbuf.size(), MPI_ATHENA_REAL,
            pbvals->nghbr_x2face[n].rank, tag, MPI_COMM_WORLD, &(bbuf.recv_rq_x2face[n]));
        }
#endif
        bbuf.bstat_x2face[n] = BoundaryRecvStatus::waiting;
      }
    }
    for (int n=0; n<4; ++n) {
      if (pbvals->nghbr_x1x2ed[n].gid >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (pbvals->nghbr_x1x2ed[n].rank != global_variable::my_rank) {
          using Kokkos::ALL;
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = pbvals->CreateMPItag(lid, (4+n), PhysicsID::Hydro_ID);
          auto recvbuf = Kokkos::subview(bbuf.recv_x1x2ed,n,ALL,ALL,ALL,ALL);
          void* recv_ptr = recvbuf.data();
          int ierr = MPI_Irecv(recv_ptr, recvbuf.size(), MPI_ATHENA_REAL,
            pbvals->nghbr_x1x2ed[n].rank, tag, MPI_COMM_WORLD, &(bbuf.recv_rq_x1x2ed[n]));
        }
#endif
        bbuf.bstat_x1x2ed[n] = BoundaryRecvStatus::waiting;
      }
    }
  }

  // x3faces, x3x1 and x2x3 edges, and corners
  if (pmesh_->nx3gt1) {
    for (int n=0; n<2; ++n) {
      if (pbvals->nghbr_x3face[n].gid >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (pbvals->nghbr_x3face[n].rank != global_variable::my_rank) {
          using Kokkos::ALL;
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = pbvals->CreateMPItag(lid, (8+n), PhysicsID::Hydro_ID);
          auto recvbuf = Kokkos::subview(bbuf.recv_x3face,n,ALL,ALL,ALL,ALL);
          void* recv_ptr = recvbuf.data();
          int ierr = MPI_Irecv(recv_ptr, recvbuf.size(), MPI_ATHENA_REAL,
            pbvals->nghbr_x3face[n].rank, tag, MPI_COMM_WORLD, &(bbuf.recv_rq_x3face[n]));
        }
#endif
        bbuf.bstat_x3face[n] = BoundaryRecvStatus::waiting;
      }
    }
    for (int n=0; n<4; ++n) {
      if (pbvals->nghbr_x3x1ed[n].gid >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (pbvals->nghbr_x3x1ed[n].rank != global_variable::my_rank) {
          using Kokkos::ALL;
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = pbvals->CreateMPItag(lid, (10+n), PhysicsID::Hydro_ID);
          auto recvbuf = Kokkos::subview(bbuf.recv_x3x1ed,n,ALL,ALL,ALL,ALL);
          void* recv_ptr = recvbuf.data();
          int ierr = MPI_Irecv(recv_ptr, recvbuf.size(), MPI_ATHENA_REAL,
            pbvals->nghbr_x3x1ed[n].rank, tag, MPI_COMM_WORLD, &(bbuf.recv_rq_x3x1ed[n]));
        }
#endif
        bbuf.bstat_x3x1ed[n] = BoundaryRecvStatus::waiting;
      }
      if (pbvals->nghbr_x2x3ed[n].gid >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (pbvals->nghbr_x2x3ed[n].rank != global_variable::my_rank) {
          using Kokkos::ALL;
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = pbvals->CreateMPItag(lid, (14+n), PhysicsID::Hydro_ID);
          auto recvbuf = Kokkos::subview(bbuf.recv_x2x3ed,n,ALL,ALL,ALL,ALL);
          void* recv_ptr = recvbuf.data();
          int ierr = MPI_Irecv(recv_ptr, recvbuf.size(), MPI_ATHENA_REAL,
            pbvals->nghbr_x2x3ed[n].rank, tag, MPI_COMM_WORLD, &(bbuf.recv_rq_x2x3ed[n]));
        }
#endif
        bbuf.bstat_x2x3ed[n] = BoundaryRecvStatus::waiting;
      }
    }
    for (int n=0; n<8; ++n) {
      if (pbvals->nghbr_corner[n].gid >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (pbvals->nghbr_corner[n].rank != global_variable::my_rank) {
          using Kokkos::ALL;
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = pbvals->CreateMPItag(lid, (18+n), PhysicsID::Hydro_ID);
          auto recvbuf = Kokkos::subview(bbuf.recv_corner,n,ALL,ALL,ALL,ALL);
          void* recv_ptr = recvbuf.data();
          int ierr = MPI_Irecv(recv_ptr, recvbuf.size(), MPI_ATHENA_REAL,
            pbvals->nghbr_corner[n].rank, tag, MPI_COMM_WORLD, &(bbuf.recv_rq_corner[n]));
        }
#endif
        bbuf.bstat_corner[n] = BoundaryRecvStatus::waiting;
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

  // x1 faces
  for (int n=0; n<2; ++n) {
    if (pbvals->nghbr_x1face[n].gid >= 0) {
      if (pbvals->nghbr_x1face[n].rank != global_variable::my_rank) {
        MPI_Wait(&(bbuf.recv_rq_x1face[n]), MPI_STATUS_IGNORE);
      }
    }
  }

  // x2faces and x1x2 edges
  if (pmesh_->nx2gt1) {
    for (int n=0; n<2; ++n) {
      if (pbvals->nghbr_x2face[n].gid >= 0) {
        if (pbvals->nghbr_x2face[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.recv_rq_x2face[n]), MPI_STATUS_IGNORE);
        }
      }
    }
    for (int n=0; n<4; ++n) {
      if (pbvals->nghbr_x1x2ed[n].gid >= 0) {
        if (pbvals->nghbr_x1x2ed[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.recv_rq_x1x2ed[n]), MPI_STATUS_IGNORE);
        }
      }
    }
  }

  // x3faces, x3x1 and x2x3 edges, and corners
  if (pmesh_->nx3gt1) {
    for (int n=0; n<2; ++n) {
      if (pbvals->nghbr_x3face[n].gid >= 0) {
        if (pbvals->nghbr_x3face[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.recv_rq_x3face[n]), MPI_STATUS_IGNORE);
        }
      }
    }
    for (int n=0; n<4; ++n) {
      if (pbvals->nghbr_x3x1ed[n].gid >= 0) {
        if (pbvals->nghbr_x3x1ed[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.recv_rq_x3x1ed[n]), MPI_STATUS_IGNORE);
        }
      }
      if (pbvals->nghbr_x2x3ed[n].gid >= 0) {
        if (pbvals->nghbr_x2x3ed[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.recv_rq_x2x3ed[n]), MPI_STATUS_IGNORE);
        }
      }
    }
    for (int n=0; n<8; ++n) {
      if (pbvals->nghbr_corner[n].gid >= 0) {
        if (pbvals->nghbr_corner[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.recv_rq_corner[n]), MPI_STATUS_IGNORE);
        }
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

  // x1 faces
  for (int n=0; n<2; ++n) {
    if (pbvals->nghbr_x1face[n].gid >= 0) {
      if (pbvals->nghbr_x1face[n].rank != global_variable::my_rank) {
        MPI_Wait(&(bbuf.send_rq_x1face[n]), MPI_STATUS_IGNORE);
      }
    }
  }

  // x2faces and x1x2 edges
  if (pmesh_->nx2gt1) {
    for (int n=0; n<2; ++n) {
      if (pbvals->nghbr_x2face[n].gid >= 0) {
        if (pbvals->nghbr_x2face[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.send_rq_x2face[n]), MPI_STATUS_IGNORE);
        }
      }
    }
    for (int n=0; n<4; ++n) {
      if (pbvals->nghbr_x1x2ed[n].gid >= 0) {
        if (pbvals->nghbr_x1x2ed[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.send_rq_x1x2ed[n]), MPI_STATUS_IGNORE);
        }
      }
    }
  }

  // x3faces, x3x1 and x2x3 edges, and corners
  if (pmesh_->nx3gt1) {
    for (int n=0; n<2; ++n) {
      if (pbvals->nghbr_x3face[n].gid >= 0) {
        if (pbvals->nghbr_x3face[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.send_rq_x3face[n]), MPI_STATUS_IGNORE);
        }
      }
    }
    for (int n=0; n<4; ++n) {
      if (pbvals->nghbr_x3x1ed[n].gid >= 0) {
        if (pbvals->nghbr_x3x1ed[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.send_rq_x3x1ed[n]), MPI_STATUS_IGNORE);
        }
      }
      if (pbvals->nghbr_x2x3ed[n].gid >= 0) {
        if (pbvals->nghbr_x2x3ed[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.send_rq_x2x3ed[n]), MPI_STATUS_IGNORE);
        }
      }
    }
    for (int n=0; n<8; ++n) {
      if (pbvals->nghbr_corner[n].gid >= 0) {
        if (pbvals->nghbr_corner[n].rank != global_variable::my_rank) {
          MPI_Wait(&(bbuf.send_rq_corner[n]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
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
  tstat = pmb->pbvals->SendCellCenteredVars(u0, (nhydro+nscalars), PhysicsID::Hydro_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::HydroReceive
//  \brief

TaskStatus Hydro::HydroReceive(Driver *pdrive, int stage)
{
  MeshBlock* pmb = pmesh_->FindMeshBlock(my_mbgid_);
  TaskStatus tstat;
  tstat = pmb->pbvals->RecvCellCenteredVars(u0, (nhydro+nscalars), PhysicsID::Hydro_ID);
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
