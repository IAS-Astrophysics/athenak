//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_criteria.cpp
//! \brief Functions to handle cyclic zoom mesh refinement criteria

#include <algorithm>
#include <iostream>
#include <set>
#include <tuple>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
//! \fn void CyclicZoom::CheckRefinement()
//! \brief Main function for CyclicZoom adaptive mesh refinement criteria

void CyclicZoom::CheckRefinement() {
  if (pmesh->time >= zstate.next_time && pmesh->ncycle > zstate.last_cycle) {
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
  zstate.last_cycle = pmesh->ncycle;
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
  old_zregion = base_zregion;
  zregion = base_zregion;
  old_zregion.radius = zregion.r_0 * std::pow(2.0,static_cast<Real>(zstate.last_zone));
  set_scales(old_zregion);

  zregion.radius = zregion.r_0 * std::pow(2.0,static_cast<Real>(zstate.zone));
  set_scales(zregion);

  // Merge only once the COM sphere contains the previous pair of spheres.
  auto merge_centers = [&](ZoomRegion &zr) {
    if ((bondi_hydro_schedule || bondi_mhd_schedule) && merge_radius > 0.0 && zr.radius >= merge_radius) {
      zr.ncenters = 1;
      zr.x1c = zr.x2c = zr.x3c = 0.0;
    }
  };
  merge_centers(old_zregion);
  merge_centers(zregion);
  Real timescale = pow(zregion.radius,zint.trun_pow);
  zint.runtime = zint.trun_facs[zstate.zone]*timescale;
  if (bondi_hydro_schedule) {
    Real mass = zregion.ncenters == 2 ? zoom_mass : total_mass;
    zint.runtime = zstate.zone == 0 ? 100.0*zoom_mass :
        (zstate.zone == 1 || zstate.zone == 2 ? 0.0 :
         10.0*std::sqrt(zregion.radius*zregion.radius*zregion.radius/mass));
    zint.runtime *= residence_scale;
  }
  if (bondi_mhd_schedule) {
    Real mass=zregion.ncenters == 2 ? zoom_mass : total_mass;
    zint.runtime=zstate.zone == 0 ? 500.0*zoom_mass :
        (zstate.zone == 1 || zstate.zone == 2 ? 0.0 :
         std::min(50.0*std::sqrt(std::pow(zregion.radius,3)/mass),0.15*bondi_time));
    zint.runtime *= residence_scale;
  }
  if (zint.runtime > zint.trun_max) {zint.runtime = zint.trun_max;}
  if (!std::isfinite(zint.runtime)) Kokkos::abort("Nonfinite cyclic zoom runtime");
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

  int old_level = zamr.max_level - zstate.last_zone;
  int ref_flag = - (zstate.zone - zstate.last_zone);
  // Restore the hierarchy that was actually saved. Geometric refinement on the
  // return sweep can refine unrelated siblings, multiplying the mesh each cycle.
  if (ref_flag > 0) {
    std::set<std::tuple<int,int,int>> parents;
    int first = pzmesh->gzms_eachlevel[zstate.zone];
    int count = pzmesh->nzmb_eachlevel[zstate.zone];
    for (int zm=first; zm<first+count; ++zm) {
      const auto &loc = pzmesh->lloc_eachzmb[zm];
      if (loc.level == old_level+1) {
        parents.emplace(loc.lx1 >> 1, loc.lx2 >> 1, loc.lx3 >> 1);
      }
    }
    for (int m=0; m<nmb; ++m) {
      const auto &loc = pmesh->lloc_eachmb[m+mbs];
      if (loc.level == old_level && parents.count({loc.lx1,loc.lx2,loc.lx3})) {
        refine_flag.h_view(m+mbs) = 1;
      }
    }
    refine_flag.template modify<HostMemSpace>();
    refine_flag.template sync<DevExeSpace>();
    return;
  }
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
      if (zregion.Overlaps(x1min,x1max,x2min,x2max,x3min,x3max,r_ref)) {
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
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  // note that now the zoom state has been updated
  int old_level = zamr.max_level - zstate.last_zone;
  // Save complete coarsening families for every solver, including siblings
  // outside the mask. These locations define the exact return hierarchy.
  return pmesh->lloc_eachmb[m+mbs].level == old_level &&
         pmesh->pmr->refine_flag.h_view(m+mbs) < 0;
}
