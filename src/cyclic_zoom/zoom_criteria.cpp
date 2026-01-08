//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_refinement.cpp
//! \brief Functions to handle cyclic zoom mesh refinement

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::CheckRefinement()
//! \brief Main function for CyclicZoom Adaptive Mesh Refinement

void CyclicZoom::CheckRefinement() {
  if (!zoom_ref) return;
  // zamr.dump_rst = (zstate.zone == 0);
  if (pmesh->time >= zstate.next_time) {
    if (global_variable::my_rank == 0) {
      std::cout << "CyclicZoom AMR: old level = " << zamr.level << std::endl;
    }
    SetRefinementFlags();
    UpdateState();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateState()
//! \brief Update zoom state before/after refinement/coarsening

void CyclicZoom::UpdateState() {
  // TODO(@mhguo): may clean the logic here a bit, perhaps using zone directly?
  // Update flags
  if (zstate.direction > 0) {zamr.zooming_out = true;}
  if (zstate.direction < 0) {zamr.zooming_in = true;}
  if (zamr.zooming_out && zamr.zooming_in) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CyclicZoom AMR: zooming_in and zooming_out both true!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (zamr.zooming_out) {
    if (fix_efield && emf_flag >=1) {
      zamr.first_emf = true;
    }
  }
  // Update zoom state
  // TODO(@mhguo): this may not be a deep copy if ZoomRegion contains pointers but now it does not
  old_zregion = zregion; // store previous zoom region
  zstate.last_zone = zstate.zone;
  zstate.zone += zstate.direction;
  if (zstate.zone == 0) {zstate.direction = 1;}
  if (zstate.zone == zamr.nlevels - 1 ) {zstate.direction = -1;}
  zamr.level = zamr.max_level - zstate.zone;
  zamr.refine_flag = -zstate.direction;
  SetRegionAndInterval();
  zstate.id++;
  zstate.next_time = pmesh->time + zint.runtime;
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom AMR:"
              << " new id = " << zstate.id
              << " zone = " << zstate.zone 
              << " level = " << zamr.level
              << std::endl;
    std::cout << "CyclicZoom AMR: old region radius = " << old_zregion.radius << std::endl;
    std::cout << "CyclicZoom AMR: region radius = " << zregion.radius << std::endl;
    std::cout << "CyclicZoom AMR: runtime = " << zint.runtime 
              << " next time = " << zstate.next_time << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::SetRegionAndInterval()
//! \brief Set the time interval for the next zoom

void CyclicZoom::SetRegionAndInterval() {
  zregion.radius = zregion.r_0 * std::pow(2.0,static_cast<Real>(zstate.zone));
  Real timescale = pow(zregion.radius,zint.t_run_pow);
  // zint.runtime = zint.t_run_fac*timescale;
  zint.runtime = zint.t_run_fac_zones[zstate.zone]*timescale;
  if (zint.runtime > zint.t_run_max) {zint.runtime = zint.t_run_max;}
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::SetRefinementFlags()
//! \brief User-defined refinement condition(s)

void CyclicZoom::SetRefinementFlags() {
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  auto &ms = pmesh->mesh_size;

  // check (on device) Hydro/MHD refinement conditions over all MeshBlocks
  auto &refine_flag = pmesh->pmr->refine_flag;
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];

  int old_level = zamr.level;
  int ref_flag = zamr.refine_flag;
  Real r_zoom = zregion.radius;
  Real x1c = zregion.x1c, x2c = zregion.x2c, x3c = zregion.x3c;
  if(global_variable::my_rank == 0) {
    std::cout << "CyclicZoom AMR: Refine/derefine to level " << old_level + ref_flag
              << " (refine_flag=" << ref_flag << ")" << std::endl;
  }
  // Check whether the MeshBlock is overlapping with the zoom region
  for (int m=0; m<nmb; ++m) {
    if (pmesh->lloc_eachmb[m+mbs].level == old_level) {
      // extract bounds of MeshBlock
      Real x1min, x1max, x2min, x2max, x3min, x3max;
      if (ref_flag > 0) {
        // For refinement, the bounds are simply those of the current MeshBlock
        x1min = size.h_view(m).x1min;
        x1max = size.h_view(m).x1max;
        x2min = size.h_view(m).x2min;
        x2max = size.h_view(m).x2max;
        x3min = size.h_view(m).x3min;
        x3max = size.h_view(m).x3max;
      } else  {
        // For coarsening, need to compute the bounds of the parent MeshBlock
        // if (refine_flag < 0) {
        std::int32_t plev = pmesh->lloc_eachmb[m+mbs].level - 1;
        std::int32_t plx1 = pmesh->lloc_eachmb[m+mbs].lx1 >> 1;
        std::int32_t nmbx1 = pmesh->nmb_rootx1 << (plev - pmesh->root_level);
        x1min = LeftEdgeX(plx1, nmbx1, ms.x1min, ms.x1max);
        x1max = LeftEdgeX(plx1+1, nmbx1, ms.x1min, ms.x1max);
        std::int32_t plx2 = pmesh->lloc_eachmb[m+mbs].lx2 >> 1;
        std::int32_t nmbx2 = pmesh->nmb_rootx2 << (plev - pmesh->root_level);
        x2min = LeftEdgeX(plx2, nmbx2, ms.x2min, ms.x2max);
        x2max = LeftEdgeX(plx2+1, nmbx2, ms.x2min, ms.x2max);
        std::int32_t plx3 = pmesh->lloc_eachmb[m+mbs].lx3 >> 1;
        std::int32_t nmbx3 = pmesh->nmb_rootx3 << (plev - pmesh->root_level);
        x3min = LeftEdgeX(plx3, nmbx3, ms.x3min, ms.x3max);
        x3max = LeftEdgeX(plx3+1, nmbx3, ms.x3min, ms.x3max);
      }
      // levels_thisrank.h_view(m) = pmesh->lloc_eachmb[m+mbs].level;
      // Find the closest point on the box to the sphere center
      Real closest_x1 = fmax(x1min, fmin(x1c, x1max));
      Real closest_x2 = fmax(x2min, fmin(x2c, x2max));
      Real closest_x3 = fmax(x3min, fmin(x3c, x3max));
      // Calculate the distance from sphere center to this closest point
      Real r_sq = SQR(x1c - closest_x1) + SQR(x2c - closest_x2) + SQR(x3c - closest_x3);
      if (r_sq < SQR(r_zoom)) {
        // std::cerr << "CyclicZoom Error: MeshBlock " << m+mbs 
        //           << " at level " << pmesh->lloc_eachmb[m+mbs].level
        //           << " cannot be refined/coarsened to level " << old_level + ref_flag
        //           << "!" << std::endl;
        // Kokkos::abort("CyclicZoom AMR level mismatch");
        // printf("MB %d: dist^2 = %e, r_zoom^2 = %e\n", m+mbs, r_sq, SQR(r_zoom));
        refine_flag.h_view(m+mbs) = ref_flag;
      }
    }
  }
  // sync host array with device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  return;
}
