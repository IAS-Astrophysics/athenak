//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.cpp
//  \brief implementation of MHD class constructor and assorted functions

#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"
#include "utils/create_mpitag.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

MHD::MHD(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u0("cons",1,1,1,1,1),
  w0("prim",1,1,1,1,1),
  b0("B_fc",1,1,1,1),
  bcc0("B_cc",1,1,1,1,1),
  u1("cons1",1,1,1,1,1),
  b1("B_fc1",1,1,1,1),
  uflx("uflx",1,1,1,1,1),
  efld("efld",1,1,1,1),
  e3x1("e3x1",1,1,1,1),
  e2x1("e2x1",1,1,1,1),
  e1x2("e1x2",1,1,1,1),
  e3x2("e3x2",1,1,1,1),
  e2x3("e2x3",1,1,1,1),
  e1x3("e1x3",1,1,1,1),
  e1_cc("e1_cc",1,1,1,1),
  e2_cc("e2_cc",1,1,1,1),
  e3_cc("e3_cc",1,1,1,1)

{
  // construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("mhd","eos");
  if (eqn_of_state.compare("adiabatic") == 0) {
    peos = new AdiabaticMHD(ppack, pin);
    nmhd = 5;
  } else if (eqn_of_state.compare("isothermal") == 0) {
    peos = new IsothermalMHD(ppack, pin);
    nmhd = 4;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<mhd> eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Initialize number of scalars
  nscalars = pin->GetOrAddInteger("mhd","nscalars",0);

  // read time-evolution option [already error checked in driver constructor]
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for conserved and primitive variables
  int nmb = ppack->nmb_thispack;
  auto &ncells = ppack->mb_cells;
  int ncells1 = ncells.nx1 + 2*(ncells.ng);
  int ncells2 = (ncells.nx2 > 1)? (ncells.nx2 + 2*(ncells.ng)) : 1;
  int ncells3 = (ncells.nx3 > 1)? (ncells.nx3 + 2*(ncells.ng)) : 1;
  Kokkos::realloc(u0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(w0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);

  // allocate memory for face-centered and cell-centered magnetic fields
  Kokkos::realloc(bcc0,   nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(b0.x1f, nmb, ncells3, ncells2, ncells1+1);
  Kokkos::realloc(b0.x2f, nmb, ncells3, ncells2+1, ncells1);
  Kokkos::realloc(b0.x3f, nmb, ncells3+1, ncells2, ncells1);
  
  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_u = new BoundaryValueCC(ppack, pin);
  pbval_u->AllocateBuffersCC((nmhd+nscalars));

  // allocate boundary buffers for face-centered magnetic field
  pbval_b = new BoundaryValueFC(ppack, pin);
  pbval_b->AllocateBuffersFC();

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {

    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("mhd","reconstruct","plm");
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
                << std::endl << "<mhd>/recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }}

    // select Riemann solver (no default).  Test for compatibility of options
    {std::string rsolver = pin->GetString("mhd","rsolver");
    if (rsolver.compare("advection") == 0) {
      if (evolution_t.compare("dynamic") == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver
                  << "' cannot be used with dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      } else {
        rsolver_method_ = MHD_RSolver::advect;
      }

    } else  if (evolution_t.compare("dynamic") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/rsolver = '" << rsolver
                << "' cannot be used with non-dynamic problems" << std::endl;
      std::exit(EXIT_FAILURE);

    } else if (rsolver.compare("llf") == 0) {
      rsolver_method_ = MHD_RSolver::llf;

//    } else if (rsolver.compare("hlld") == 0) {
//      if (peos->eos_data.is_adiabatic) {
//        rsolver_method_ = MHD_RSolver::hlld;
//      } else { 
//        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
//                  << std::endl << "<mhd>/rsolver = '" << rsolver
//                  << "' cannot be used with isothermal EOS" << std::endl;
//        std::exit(EXIT_FAILURE); 
//        }  

//    } else if (rsolver.compare("roe") == 0) {
//      rsolver_method_ = MHD_RSolver::roe;

    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE); 
    }}

    // allocate second registers
    Kokkos::realloc(u1,     nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(b1.x1f, nmb, ncells3, ncells2, ncells1+1);
    Kokkos::realloc(b1.x2f, nmb, ncells3, ncells2+1, ncells1);
    Kokkos::realloc(b1.x3f, nmb, ncells3+1, ncells2, ncells1);

    // allocate fluxes, electric fields
    Kokkos::realloc(uflx.x1f, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1+1);
    Kokkos::realloc(uflx.x2f, nmb, (nmhd+nscalars), ncells3, ncells2+1, ncells1);
    Kokkos::realloc(uflx.x3f, nmb, (nmhd+nscalars), ncells3+1, ncells2, ncells1);
    Kokkos::realloc(efld.x1e, nmb, ncells3+1, ncells2+1, ncells1);
    Kokkos::realloc(efld.x2e, nmb, ncells3+1, ncells2, ncells1+1);
    Kokkos::realloc(efld.x3e, nmb, ncells3, ncells2+1, ncells1+1);

    // allocate scratch arrays for face- and cell-centered E used in CornerE
    Kokkos::realloc(e3x1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e2x1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e1x2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e3x2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e2x3, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e1x3, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e1_cc, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e2_cc, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(e3_cc, nmb, ncells3, ncells2, ncells1);
  }
}

//----------------------------------------------------------------------------------------
// destructor
  
MHD::~MHD()
{
  delete peos;
  delete pbval_u;
  delete pbval_b;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDStageStartTasks
//  \brief adds MHD tasks to stage start TaskList
//  These are taks that must be cmpleted (such as posting MPI receives, setting 
//  BoundaryCommStatus flags, etc) over all MeshBlocks before stage can be run.

void MHD::MHDStageStartTasks(TaskList &tl, TaskID start)
{
  auto mhd_init = tl.AddTask(&MHD::MHDInitRecv, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDStageRunTasks
//  \brief adds MHD tasks to stage run TaskList

void MHD::MHDStageRunTasks(TaskList &tl, TaskID start)
{
  auto mhd_copycons = tl.AddTask(&MHD::MHDCopyCons, this, start);
  auto mhd_fluxes = tl.AddTask(&MHD::CalcFluxes, this, mhd_copycons);
  auto mhd_update = tl.AddTask(&MHD::Update, this, mhd_fluxes);
  auto mhd_sendu = tl.AddTask(&MHD::MHDSendU, this, mhd_update);
  auto mhd_recvu = tl.AddTask(&MHD::MHDRecvU, this, mhd_sendu);
  auto mhd_emf = tl.AddTask(&MHD::CornerE, this, mhd_recvu);
  auto mhd_ct = tl.AddTask(&MHD::CT, this, mhd_emf);
  auto mhd_sendb = tl.AddTask(&MHD::MHDSendB, this, mhd_ct);
  auto mhd_recvb = tl.AddTask(&MHD::MHDRecvB, this, mhd_sendb);
  auto mhd_phybcs = tl.AddTask(&MHD::MHDApplyPhysicalBCs, this, mhd_recvb);
  auto mhd_con2prim = tl.AddTask(&MHD::ConToPrim, this, mhd_phybcs);
  auto mhd_newdt = tl.AddTask(&MHD::NewTimeStep, this, mhd_con2prim);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDStageEndTasks
//  \brief adds MHD tasks to stage end TaskList
//  These are tasks that can only be cmpleted after all the stage run tasks are finished
//  over all MeshBlocks, such as clearing all MPI non-blocking sends, etc.

void MHD::MHDStageEndTasks(TaskList &tl, TaskID start)
{
  auto mhd_clear = tl.AddTask(&MHD::MHDClearSend, this, start);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDInitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI).  Note this must be done for
//  communication of BOTH conserved (cell-centered) and face-centered fields

TaskStatus MHD::MHDInitRecv(Driver *pdrive, int stage)
{
  int &nmb = pmy_pack->pmb->nmb;
  int &nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications for both cell-centered conserved variables and 
  // face-centered magnetic fields
  auto &rbufu = pbval_u->recv_buf;
  auto &rbufb = pbval_b->recv_buf;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
#if MPI_PARALLEL_ENABLED
        // post non-blocking receive if neighboring MeshBlock on a different rank 
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          {
          // Receive requests for U
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n, VariablesID::FluidCons_ID);
          auto recv_data = Kokkos::subview(rbufu[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int ierr = MPI_Irecv(recv_ptr, recv_data.size(), MPI_ATHENA_REAL,
            nghbr[n].rank.h_view(m), tag, MPI_COMM_WORLD, &(rbufu[n].comm_req[m]));
          }

          {
          // Receive requests for B
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateMPITag(m, n, VariablesID::BField_ID);
          auto recv_data = Kokkos::subview(rbufb[n].data, m, Kokkos::ALL, Kokkos::ALL);
          void* recv_ptr = recv_data.data();
          int ierr = MPI_Irecv(recv_ptr, recv_data.size(), MPI_ATHENA_REAL,
            nghbr[n].rank.h_view(m), tag, MPI_COMM_WORLD, &(rbufb[n].comm_req[m]));
          }
        }
#endif
        // initialize boundary receive status flag
        rbufu[n].bcomm_stat(m) = BoundaryCommStatus::waiting;
        rbufb[n].bcomm_stat(m) = BoundaryCommStatus::waiting;
      }
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue
//  With MHD, clears both receives of U and B

TaskStatus MHD::MHDClearRecv(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for U and B to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
          MPI_Wait(&(pbval_b->recv_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue
//  With MHD, clears both sends of U and B

TaskStatus MHD::MHDClearSend(Driver *pdrive, int stage)
{
#if MPI_PARALLEL_ENABLED
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for U and B to finish before continuing 
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr[n].gid.h_view(m) >= 0) {
        if (nghbr[n].rank.h_view(m) != global_variable::my_rank) {
          MPI_Wait(&(pbval_u->send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
          MPI_Wait(&(pbval_b->send_buf[n].comm_req[m]), MPI_STATUS_IGNORE);
        }
      }
    }
  }
#endif
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDCopyCons
//  \brief  copy u0 --> u1, and b0 --> b1 in first stage

TaskStatus MHD::MHDCopyCons(Driver *pdrive, int stage)
{
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
    Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
    Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
    Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDSendU
//  \brief sends cell-centered conserved variables

TaskStatus MHD::MHDSendU(Driver *pdrive, int stage) 
{
  TaskStatus tstat = pbval_u->SendBuffersCC(u0, VariablesID::FluidCons_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDSendB
//  \brief sends face-centered magnetic fields

TaskStatus MHD::MHDSendB(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_b->SendBuffersFC(b0, VariablesID::BField_ID);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDRecvU
//  \brief receives cell-centered conserved variables

TaskStatus MHD::MHDRecvU(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_u->RecvBuffersCC(u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::MHDRecvB
//  \brief receives face-centered magnetic fields

TaskStatus MHD::MHDRecvB(Driver *pdrive, int stage)
{
  TaskStatus tstat = pbval_b->RecvBuffersFC(b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void MHD::ConToPrim
//  \brief

TaskStatus MHD::ConToPrim(Driver *pdrive, int stage)
{
  peos->ConsToPrim(u0, b0, w0, bcc0);
  return TaskStatus::complete;
}

} // namespace mhd
