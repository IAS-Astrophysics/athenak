//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shearing_box_tasks.cpp
//! \brief functions included in task lists to post/clear non-blocking MPI calls for
//! shearing box, and flux correction steps with shearing box boundaries.

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "shearing_box.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::InitRecv
//! \brief Calculates x2-distance that x1-boundaries have sheared.  With MPI, posts
//! non-blocking receives for boundary communications for shearing box boundaries

TaskStatus ShearingBox::InitRecv(Real time) {
  // figure out distance boundaries are sheared
  const auto &mesh_size = pmy_pack->pmesh->mesh_size;
  Real lx = (mesh_size.x1max - mesh_size.x1min);
  yshear = (qshear*omega0)*lx*time;

#if MPI_PARALLEL_ENABLED
  // post non-blocking receives
  bool no_errors=true;
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &js = indcs.js, &je = indcs.je;
  const int &ng = indcs.ng;
  const int &nx2 = indcs.nx2;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      // Find integer and fractional number of grids over which offset extends.
      // This assumes every grid has same number of cells in x2-direction!
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;

      if (jr < ng) {               //--- CASE 1 (in my nomenclature)
        std::pair<int,int> jdst[3];
        if (n==0) {
          jdst[0] = std::make_pair(je+1+jr,je+ng+1);
          jdst[1] = std::make_pair(js+jr,je+jr+1);
          jdst[2] = std::make_pair(js-ng,js+jr);
        } else {
          jdst[0] = std::make_pair(je+1-jr,je+ng+1);
          jdst[1] = std::make_pair(js-jr,je-jr+1);
          jdst[2] = std::make_pair(js-ng,js-jr);
        }
        // ix1 boundary: receive from (target+1) through (target-1)
        // ox1 boundary: receive from (target+1) through (target-1)
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l-1);} else {jshift = -(l-1-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            // create tag using local ID of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(gid, ((n<<2) | l));

            // get pointer to variables
            using Kokkos::ALL;
            auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, jdst[l], ALL, ALL, ALL);
            int data_size = recv_ptr.size();

            // Post non-blocking receive for this buffer on this MeshBlock
            int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, srank, tag,
                                 comm_sbox, &(recvbuf[n].vars_req[3*m + l]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      } else if (jr < (nx2-ng)) {  //--- CASE 2
        std::pair<int,int> jdst[2];
        if (n==0) {
          jdst[0] = std::make_pair(js+jr,je+ng+1);
          jdst[1] = std::make_pair(js-ng,js+jr);
        } else {
          jdst[0] = std::make_pair(je-jr+1,je+ng+1);
          jdst[1] = std::make_pair(js-ng,je-jr+1);
        }
        // ix1 boundary: receive from (target  ) through (target-1)
        // ox1 boundary: receive from (target+1) through (target  )
        for (int l=0; l<2; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l);} else {jshift = -(l-1-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            // create tag using local ID of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(gid, ((n<<2) | l));

            // get pointer to variables
            using Kokkos::ALL;
            auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, jdst[l], ALL, ALL, ALL);
            int data_size = recv_ptr.size();

            // Post non-blocking receive for this buffer on this MeshBlock
            int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, srank, tag,
                                 comm_sbox, &(recvbuf[n].vars_req[3*m + l]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      } else {                      //--- CASE 3
        std::pair<int,int> jdst[3];
        if (n==0) {
          jdst[0] = std::make_pair(je+1-(nx2-jr),je+ng+1);
          jdst[1] = std::make_pair(js-(nx2-jr),je-(nx2-jr)+1);
          jdst[2] = std::make_pair(js-ng,js-(nx2-jr));
        } else {
          jdst[0] = std::make_pair(je+1+(nx2-jr),je+ng+1);
          jdst[1] = std::make_pair(js+(nx2-jr),je+(nx2-jr)+1);
          jdst[2] = std::make_pair(js-ng,js+(nx2-jr));
        }
        // ix1 boundary: send to (target  ) through (target+2)
        // ox1 boundary: send to (target-2) through (target  )
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l);} else {jshift = -(l-2-ji);}
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            // create tag using local ID of *receiving* MeshBlock
            int tag = CreateBvals_MPI_Tag(gid, ((n<<2) | l));

            // get pointer to variables
            using Kokkos::ALL;
            auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, jdst[l], ALL, ALL, ALL);
            int data_size = recv_ptr.size();

            // Post non-blocking receive for this buffer on this MeshBlock
            int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, srank, tag,
                                 comm_sbox, &(recvbuf[n].vars_req[3*m + l]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void ShearingBox::ClearRecv
//! \brief Waits for all MPI receives associated with communication for shearing box
//! boundaries to complete before allowing execution to continue

TaskStatus ShearingBox::ClearRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  const int &ng = pmy_pack->pmesh->mb_indcs.ng;
  const int &nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  // wait for all non-blocking receives for vars to finish before continuing
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      // Find integer and fractional number of grids over which offset extends.
      // This assumes every grid has same number of cells in x2-direction!
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;

      if (jr < ng) {               //--- CASE 1 (in my nomenclature)
        // ix1 boundary: receive from (target+1) through (target-1)
        // ox1 boundary: receive from (target+1) through (target-1)
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l-1);} else {jshift = -(l-1-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int ierr = MPI_Wait(&(recvbuf[n].vars_req[3*m + l]), MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      } else if (jr < (nx2-ng)) {  //--- CASE 2
        // ix1 boundary: receive from (target  ) through (target-1)
        // ox1 boundary: receive from (target+1) through (target  )
        for (int l=0; l<2; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l);} else {jshift = -(l-1-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int ierr = MPI_Wait(&(recvbuf[n].vars_req[3*m + l]), MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      } else {                     //--- CASE 3
        // ix1 boundary: send to (target  ) through (target+2)
        // ox1 boundary: send to (target-2) through (target  )
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = -(ji+l);} else {jshift = -(l-2-ji);} // offset of sender
          int sgid, srank;
          FindTargetMB(gid,jshift,sgid,srank);
          if (srank != global_variable::my_rank) {
            int ierr = MPI_Wait(&(recvbuf[n].vars_req[3*m + l]), MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void ShearingBox::ClearSend
//! \brief Waits for all MPI sends associated with communcation for shearing box
//! boundaries to complete before allowing execution to continue

TaskStatus ShearingBox::ClearSend() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  const int &ng = pmy_pack->pmesh->mb_indcs.ng;
  const int &nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  // wait for all non-blocking sends for vars to finish before continuing
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      int mm = gid - pmy_pack->gids;
      // Find integer and fractional number of grids over which offset extends.
      // This assumes every grid has same number of cells in x2-direction!
      int joffset  = static_cast<int>(yshear/(pmy_pack->pmb->mb_size.h_view(mm).dx2));
      int ji = joffset/nx2;
      int jr = joffset - ji*nx2;

      if (jr < ng) {               //--- CASE 1 (in my nomenclature)
        // ix1 boundary: send to (target-1) through (target+1)
        // ox1 boundary: send to (target-1) through (target+1)
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = ji+l-1;} else {jshift = l-1-ji;} // offset of target
          int tgid, trank;
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank != global_variable::my_rank) {
            int ierr = MPI_Wait(&(sendbuf[n].vars_req[3*m + l]), MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      } else if (jr < (nx2-ng)) {  //--- CASE 2
        // ix1 boundary: send to (target  ) through (target+1)
        // ox1 boundary: send to (target-1) through (target  )
        for (int l=0; l<2; ++l) {
          int jshift;
          if (n==0) {jshift = ji+l;} else {jshift = l-1-ji;}
          int tgid, trank;
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank != global_variable::my_rank) {
            int ierr = MPI_Wait(&(sendbuf[n].vars_req[3*m + l]), MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      } else {                     //--- CASE 3
        // ix1 boundary: send to (target  ) through (target+2)
        // ox1 boundary: send to (target-2) through (target  )
        for (int l=0; l<3; ++l) {
          int jshift;
          if (n==0) {jshift = ji+l;} else {jshift = l-2-ji;}
          int tgid, trank;
          FindTargetMB(gid,jshift,tgid,trank);
          if (trank != global_variable::my_rank) {
            int ierr = MPI_Wait(&(sendbuf[n].vars_req[3*m + l]), MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}
