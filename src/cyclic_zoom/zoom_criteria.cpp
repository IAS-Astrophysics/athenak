//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_criteria.cpp
//! \brief Functions to handle cyclic zoom mesh refinement criteria

#include <algorithm>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/z4c.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
//! \fn ParentBounds()
//! \brief Bounds {x1min,x1max,x2min,x2max,x3min,x3max} of the parent of MeshBlock gid

static void ParentBounds(Mesh *pmesh, int gid, Real b[6]) {
  auto &ms = pmesh->mesh_size;
  auto &lloc = pmesh->lloc_eachmb[gid];
  std::int32_t plev = lloc.level - 1;
  std::int32_t plx1 = lloc.lx1 >> 1;
  std::int32_t nmbx1 = pmesh->nmb_rootx1 << (plev - pmesh->root_level);
  b[0] = LeftEdgeX(plx1, nmbx1, ms.x1min, ms.x1max);
  b[1] = LeftEdgeX(plx1+1, nmbx1, ms.x1min, ms.x1max);
  std::int32_t plx2 = lloc.lx2 >> 1;
  std::int32_t nmbx2 = pmesh->nmb_rootx2 << (plev - pmesh->root_level);
  b[2] = LeftEdgeX(plx2, nmbx2, ms.x2min, ms.x2max);
  b[3] = LeftEdgeX(plx2+1, nmbx2, ms.x2min, ms.x2max);
  std::int32_t plx3 = lloc.lx3 >> 1;
  std::int32_t nmbx3 = pmesh->nmb_rootx3 << (plev - pmesh->root_level);
  b[4] = LeftEdgeX(plx3, nmbx3, ms.x3min, ms.x3max);
  b[5] = LeftEdgeX(plx3+1, nmbx3, ms.x3min, ms.x3max);
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::CheckRefinement()
//! \brief Main function for CyclicZoom adaptive mesh refinement criteria

void CyclicZoom::CheckRefinement() {
  if (pmesh->time >= zstate.next_time) {
    if (verbose && global_variable::my_rank == 0) {
      std::cout << "CyclicZoom: old level = " << zamr.level << std::endl;
    }
    UpdateState();
    SetRefinementFlags();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::UpdateState()
//! \brief Update zoom state before/after refinement/coarsening

void CyclicZoom::UpdateState() {
  // update flags
  if (zstate.direction > 0) {zamr.zooming_out = true;}
  if (zstate.direction < 0) {zamr.zooming_in = true;}
  if (zamr.zooming_out && zamr.zooming_in) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "CyclicZoom: zooming_in and zooming_out both true!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // update zoom state
  zstate.last_zone = zstate.zone;
  zstate.zone += zstate.direction;
  if (zstate.zone == 0) {zstate.direction = 1;}
  if (zstate.zone == zamr.nlevels - 1 ) {zstate.direction = -1;}
  zstate.id++;
  // update zoom AMR parameters
  zamr.level = zamr.max_level - zstate.zone;
  zamr.refine_flag = -zstate.direction;
  // update zoom interval
  SetRegionAndInterval();
  zstate.next_time = pmesh->time + zint.runtime;
  // print verbose output
  if (verbose && global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: new id = " << zstate.id
              << " zone = " << zstate.zone
              << " level = " << zamr.level
              << std::endl;
    std::cout << "CyclicZoom: old radius = " << old_zregion.radius
              << " r_ref = " << old_zregion.ref.r << std::endl;
    std::cout << "CyclicZoom: new radius = " << zregion.radius
              << " r_ref = " << zregion.ref.r << std::endl;
    std::cout << "CyclicZoom: r_exc = " << zregion.exc.r
              << " r_cut = " << zregion.cut.r
              << " r_flx = " << zregion.flx.r
              << std::endl;
    std::cout << "CyclicZoom: time = " << pmesh->time
              << " runtime = " << zint.runtime
              << " next time = " << zstate.next_time
              << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::SetRegionAndInterval()
//! \brief Set the time interval for the next zoom

void CyclicZoom::SetRegionAndInterval() {
  // TODO(@mhguo): may add more flexible and robust region settings later
  auto set_scales = [](ZoomRegion &zr) {
    zr.ref.r = std::min(zr.ref.f * zr.radius, zr.ref.r_max);
    zr.exc.r = std::min(zr.exc.f * zr.radius, zr.exc.r_max);
    zr.cut.r = std::min(zr.cut.f * zr.radius, zr.cut.r_max);
    zr.flx.r = std::min(zr.flx.f * zr.radius, zr.flx.r_max);
  };
  old_zregion.radius = zregion.r_0 * std::pow(2.0,static_cast<Real>(zstate.last_zone));
  set_scales(old_zregion);

  zregion.radius = zregion.r_0 * std::pow(2.0,static_cast<Real>(zstate.zone));
  set_scales(zregion);

  Real timescale = pow(zregion.radius,zint.trun_pow);
  zint.runtime = zint.trun_facs[zstate.zone]*timescale;
  if (zint.runtime > zint.trun_max) {zint.runtime = zint.trun_max;}
}

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::SetRefinementFlags()
//! \brief User-defined refinement condition(s)

void CyclicZoom::SetRefinementFlags() {
  auto &size = pmesh->pmb_pack->pmb->mb_size;

  // check (on device) Hydro/MHD refinement conditions over all MeshBlocks
  auto &refine_flag = pmesh->pmr->refine_flag;
  int nmb = pmesh->pmb_pack->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];

  int old_level = zamr.max_level - zstate.last_zone;
  int ref_flag = - (zstate.zone - zstate.last_zone);
  Real r_ref = (ref_flag < 0) ? zregion.ref.r : old_zregion.ref.r;
  Real x1c = zregion.x1c, x2c = zregion.x2c, x3c = zregion.x3c;
  if(verbose && global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: " << (ref_flag > 0 ? "refines" : "derefines")
              << " to level " << old_level + ref_flag
              << " (refine_flag=" << ref_flag << ")" << std::endl;
  }
  // Check whether the MeshBlock is overlapping with the refinement radius
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
      } else {
        // For coarsening, need to compute the bounds of the parent MeshBlock
        Real pb[6];
        ParentBounds(pmesh, m+mbs, pb);
        x1min = pb[0]; x1max = pb[1];
        x2min = pb[2]; x2max = pb[3];
        x3min = pb[4]; x3max = pb[5];
      }
      // levels_thisrank.h_view(m) = pmesh->lloc_eachmb[m+mbs].level;
      // zoom in where the trackers refine (Z4c_AMR::RefineTracker), so the original
      // tracker mesh and its 2:1 nest are recovered instead of a larger sphere
      if (ref_flag > 0 && zamr.zoom_in_tracker) {
        int plevel = old_level - pmesh->root_level;
        for (auto &pt : pmesh->pmb_pack->pz4c->ptracker) {
          Real cx = fmax(x1min, fmin(pt->GetPos(0), x1max));
          Real cy = fmax(x2min, fmin(pt->GetPos(1), x2max));
          Real cz = fmax(x3min, fmin(pt->GetPos(2), x3max));
          Real d2 = SQR(pt->GetPos(0) - cx) + SQR(pt->GetPos(1) - cy)
                  + SQR(pt->GetPos(2) - cz);
          bool contained = (d2 == 0.0);
          if ((d2 < SQR(pt->GetRadius()) || contained) &&
              (pt->GetReflevel() < 0 || plevel < pt->GetReflevel())) {
            refine_flag.h_view(m+mbs) = ref_flag;
          }
        }
        continue;
      }
      // Find the closest point on the box to the sphere center
      Real closest_x1 = fmax(x1min, fmin(x1c, x1max));
      Real closest_x2 = fmax(x2min, fmin(x2c, x2max));
      Real closest_x3 = fmax(x3min, fmin(x3c, x3max));
      // Calculate the distance from sphere center to this closest point
      Real r_sq = SQR(x1c - closest_x1) + SQR(x2c - closest_x2) + SQR(x3c - closest_x3);
      if (r_sq < SQR(r_ref)) {
        refine_flag.h_view(m+mbs) = ref_flag;
      }
    }
  }
  // sync host array with device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  return;
}


//----------------------------------------------------------------------------------------
//! \fn bool CyclicZoom::CheckStoreFlag(int m)
//! \brief Check whether to store variables for MeshBlock with given global ID

bool CyclicZoom::CheckStoreFlag(int m) {
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  // note that now the zoom state has been updated
  int old_level = zamr.max_level - zstate.last_zone;
  // use the updated zoom region parameters
  Real r_zoom = zregion.radius;
  Real x1c = zregion.x1c, x2c = zregion.x2c, x3c = zregion.x3c;
  // check previous level (finer level)
  if (pmesh->lloc_eachmb[m+mbs].level == old_level) {
    // with z4c, store every MeshBlock the zoom out derefines (same parent-box test as
    // SetRefinementFlags), so the metric is restored exactly on zooming in
    if (pmesh->pmb_pack->pz4c != nullptr) {
      Real pb[6];
      ParentBounds(pmesh, m+mbs, pb);
      Real x1c = zregion.x1c, x2c = zregion.x2c, x3c = zregion.x3c;
      Real r_sq = SQR(x1c - fmax(pb[0], fmin(x1c, pb[1])))
                + SQR(x2c - fmax(pb[2], fmin(x2c, pb[3])))
                + SQR(x3c - fmax(pb[4], fmin(x3c, pb[5])));
      return (r_sq < SQR(zregion.ref.r));
    }
    // extract bounds of MeshBlock
    Real x1min = size.h_view(m).x1min;
    Real x1max = size.h_view(m).x1max;
    Real x2min = size.h_view(m).x2min;
    Real x2max = size.h_view(m).x2max;
    Real x3min = size.h_view(m).x3min;
    Real x3max = size.h_view(m).x3max;
    // Find the closest point on the box to the sphere center
    Real closest_x1 = fmax(x1min, fmin(x1c, x1max));
    Real closest_x2 = fmax(x2min, fmin(x2c, x2max));
    Real closest_x3 = fmax(x3min, fmin(x3c, x3max));
    // Calculate the distance from sphere center to this closest point
    Real r_sq = SQR(x1c - closest_x1) + SQR(x2c - closest_x2) + SQR(x3c - closest_x3);
    if (r_sq < SQR(r_zoom)) {
      return true;
    }
  }
  return false;
}
