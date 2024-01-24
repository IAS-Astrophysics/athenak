//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_part.cpp
//! \brief

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValuesCC::PackAndSendCC()
//! \brief

namespace particles {

KOKKOS_INLINE_FUNCTION
void FlagForSendOrUpdateGID(NeighborBlock &nghbr, int myrank, int &mp) {
  if (nghbr.rank == myrank) {
    mp = nghbr.gid;
  }
  // Store data for sends with MPI
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::PackAndSendParticles()
//! \brief

TaskStatus ParticlesBoundaryValues::PackAndSendParticles() {
  // create local references for variables in kernel
  auto ppos = pmy_part->prtcl_pos;
  auto pgid = pmy_part->prtcl_gid;
  int &npart = pmy_part->nparticles_thispack;
  auto &mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto &meshsize = pmy_part->pmy_pack->pmesh->mesh_size;
  auto &myrank = global_variable::my_rank;
  auto &nghbr = pmy_part->pmy_pack->pmb->nghbr;

  par_for("part_update",DevExeSpace(),0,npart, KOKKOS_LAMBDA(const int p) {
    int &mp = pgid.d_view(p);
    Real &x1 = ppos(p,IPX);
    Real &x2 = ppos(p,IPY);
    Real &x3 = ppos(p,IPZ);

    if (x1 < mbsize.d_view(mp).x1min) {
      if (x2 < mbsize.d_view(mp).x2min) {
        if (x3 < mbsize.d_view(mp).x3min) {
          // corner
          FlagForSendOrUpdateGID(nghbr.d_view(mp,48), myrank, mp);
        } else if (x3 > mbsize.d_view(mp).x3max) {
          // corner
          FlagForSendOrUpdateGID(nghbr.d_view(mp,49), myrank, mp);
        } else {
          // x3x1 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,32), myrank, mp);
        }
      } else if (x2 > mbsize.d_view(mp).x2max) {
        if (x3 < mbsize.d_view(mp).x3min) {
          // corner
          FlagForSendOrUpdateGID(nghbr.d_view(mp,52), myrank, mp);
        } else if (x3 > mbsize.d_view(mp).x3max) {
          // corner
          FlagForSendOrUpdateGID(nghbr.d_view(mp,53), myrank, mp);
        } else {
          // x3x1 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,36), myrank, mp);
        }
      } else {
        if (x2 < mbsize.d_view(mp).x3min) {
          // x1x2 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,16), myrank, mp);
        } else if (x3 > mbsize.d_view(mp).x3max) {
          // x1x2 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,20), myrank, mp);
        } else {
          // x1 face
          FlagForSendOrUpdateGID(nghbr.d_view(mp,0), myrank, mp);
        }
      }

    } else if (x1 > mbsize.d_view(mp).x1max) {
      if (x2 < mbsize.d_view(mp).x2min) {
        if (x3 < mbsize.d_view(mp).x3min) {
          // corner
          FlagForSendOrUpdateGID(nghbr.d_view(mp,50), myrank, mp);
        } else if (x3 > mbsize.d_view(mp).x3max) {
          // corner
          FlagForSendOrUpdateGID(nghbr.d_view(mp,51), myrank, mp);
        } else {
          // x3x1 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,34), myrank, mp);
        }
      } else if (x2 > mbsize.d_view(mp).x2max) {
        if (x3 < mbsize.d_view(mp).x3min) {
          // corner
          FlagForSendOrUpdateGID(nghbr.d_view(mp,54), myrank, mp);
        } else if (x3 > mbsize.d_view(mp).x3max) {
          // corner
          FlagForSendOrUpdateGID(nghbr.d_view(mp,55), myrank, mp);
        } else {
          // x3x1 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,38), myrank, mp);
        }
      } else {
        if (x2 < mbsize.d_view(mp).x3min) {
          // x1x2 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,18), myrank, mp);
        } else if (x3 > mbsize.d_view(mp).x3max) {
          // x1x2 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,22), myrank, mp);
        } else {
          // x1 face
          FlagForSendOrUpdateGID(nghbr.d_view(mp,4), myrank, mp);
        }
      }

    } else {
      if (x2 < mbsize.d_view(mp).x2min) {
        if (x3 < mbsize.d_view(mp).x3min) {
          // x2x3 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,40), myrank, mp);
        } else if (x3 > mbsize.d_view(mp).x3max) {
          // x2x3 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,44), myrank, mp);
        } else {
          // x2 face
          FlagForSendOrUpdateGID(nghbr.d_view(mp,8), myrank, mp);
        }
      } else if (x2 > mbsize.d_view(mp).x2max) {
        if (x3 < mbsize.d_view(mp).x3min) {
          // x2x3 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,42), myrank, mp);
        } else if (x3 > mbsize.d_view(mp).x3max) {
          // x2x3 edge
          FlagForSendOrUpdateGID(nghbr.d_view(mp,46), myrank, mp);
        } else {
          // x2 face
          FlagForSendOrUpdateGID(nghbr.d_view(mp,12), myrank, mp);
        }
      } else {
        if (x2 < mbsize.d_view(mp).x3min) {
          // x3 face
          FlagForSendOrUpdateGID(nghbr.d_view(mp,24), myrank, mp);
        } else if (x3 > mbsize.d_view(mp).x3max) {
          // x3 face
          FlagForSendOrUpdateGID(nghbr.d_view(mp,28), myrank, mp);
        }
      }
    }

    // reset x,y,z positions if particle crosses Mesh boundary using periodic BCs
    if (x1 < meshsize.x1min) {
      ppos(p,IPX) += (meshsize.x1max - meshsize.x1min);
    } else if (x1 > meshsize.x1max) {
      ppos(p,IPX) -= (meshsize.x1max - meshsize.x1min);
    }
  });

  return TaskStatus::complete;
}
}
