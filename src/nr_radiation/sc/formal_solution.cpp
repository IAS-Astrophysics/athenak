//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file formal_solution.cpp
//! \brief Short-characteristics formal solution (Davis, Stone & Jiang 2012 Eq. 20): one
//! ordered upwind sweep of every meshblock, for every discrete ray, packaged into Kokkos
//! kernels by one of two schemes (<nr_radiation>/sweep_kernel). Both produce
//! bit-identical intensities; they differ only in how the work is split into launches
//! and teams.
//!
//!   plane      one launch per plane along each ray's OWN dominant axis, a flat par_for
//!              over the transverse cells x rays. n launches per meshblock per sweep in
//!              3D, every one of them the same full size; no tiles, teams or table.
//!   tiled      the meshblock is cut into tiles of tile_size^3 cells (tile_size = 0: one
//!              tile). One launch per TILE hyperplane; inside it one team per (tile,
//!              ray), each team walking its tile's cell hyperplanes with a team barrier
//!              between them. Every footpoint of a tile lies in the same tile or on an
//!              earlier tile-plane, so the launch boundary is the cross-tile barrier
//!              (Koch-Baker-Alcouffe).
//!
//! plane is the default and has no parameters; tiled trades that for tile_size and
//! team_size. Only tiled uses HyperplaneOrder tables, built once from the meshblock and
//! tile dims.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "nr_radiation/nr_radiation.hpp"
#include "nr_radiation/sc/sc_interp.hpp"

namespace nr_radiation {

namespace {

//----------------------------------------------------------------------------------------
//! \fn HyperplaneOrder IndexHyperplanes
//! \brief Upwind hyperplane ordering of a box of dims (n1,n2,n3) with ndim active
//! dimensions. For each plane h enumerates exactly the items with l1+l2+l3 == h and
//! stores the packed linear index lin = (l3*n2+l2)*n1+l1, grouped by plane
//! (plane h == [start[h], start[h+1])). Within a plane the items are causally
//! independent (every footpoint sits on a strictly lower plane), so any order within a
//! plane yields bit-identical results.

HyperplaneOrder IndexHyperplanes(int n1, int n2, int n3, int ndim) {
  const int d2 = (ndim >= 2) ? n2 : 1;
  const int d3 = (ndim == 3) ? n3 : 1;
  const int nitems = n1 * d2 * d3;
  const int hmax = (n1 - 1) + (d2 - 1) + (d3 - 1);

  HyperplaneOrder ord;
  ord.nplanes = hmax + 1;
  HostArray1D<int> h_items("hyperplane_items_host", nitems);
  ord.start.assign(hmax + 2, 0);
  int idx = 0;
  for (int h = 0; h <= hmax; ++h) {
    ord.start[h] = idx;
    for (int l3 = 0; l3 < d3; ++l3) {
      for (int l2 = 0; l2 < d2; ++l2) {
        int l1 = h - l2 - l3;
        if (l1 < 0 || l1 >= n1) continue;
        h_items(idx++) = (l3 * n2 + l2) * n1 + l1;
      }
    }
  }
  ord.start[hmax + 1] = idx;   // == nitems

  Kokkos::realloc(ord.items, nitems);
  Kokkos::deep_copy(ord.items, h_items);
  const int nstart = hmax + 2;
  HostArray1D<int> h_start("hyperplane_start_host", nstart);
  for (int h = 0; h < nstart; ++h) { h_start(h) = ord.start[h]; }
  Kokkos::realloc(ord.start_dev, nstart);
  Kokkos::deep_copy(ord.start_dev, h_start);
  return ord;
}


//----------------------------------------------------------------------------------------
//! \fn int MarchAxis
//! \brief The grid axis a ray marches along, given the face it crosses.
//!
//! Named rather than inlined because the 2D case is not the identity: there
//! SCRayInv::axis 0 is the ray that crosses the y-face, so that ray marches along y. In
//! 3D the two coincide. Reached only through RayGeometry, which is what keeps the host
//! and device answers the same.

KOKKOS_INLINE_FUNCTION
int MarchAxis(int inv_axis, int ndim) {
  if (ndim == 1) { return 0; }
  if (ndim == 2) { return (inv_axis == 0) ? 1 : 0; }
  return inv_axis;
}

//----------------------------------------------------------------------------------------
//! \struct SweepRay
//! \brief Everything a plane launch needs to know about one ray in one meshblock.

struct SweepRay {
  int sx, sy, sz;    // octant signs: which corner of the block the ray enters from
  int march_axis;    // grid axis it marches along
  SCRayInv inv;      // inverse lengths and crossed face, consumed by GatherSolveSC
};

//----------------------------------------------------------------------------------------
//! \fn SweepRay RayGeometry()
//! \brief Resolve one ray against one cell size.
//!
//! The single definition of this arithmetic: both plane kernels call it on the device and
//! BuildRayGroups calls it twice on the host, so a ray cannot be classified one way when
//! the groups are built and marched another way when the sweep runs. Templated on the
//! view type only so that h_view and d_view can share it.

template <class MuView>
KOKKOS_INLINE_FUNCTION
SweepRay RayGeometry(const MuView &mu, int angg, int nang, int ndim,
                     Real dx1, Real dx2, Real dx3) {
  const int oct = angg / nang;
  const int a = angg - oct*nang;
  const Real mux = mu(oct,a,0);
  const Real muy = (ndim >= 2) ? mu(oct,a,1) : 0.0;
  const Real muz = (ndim == 3) ? mu(oct,a,2) : 0.0;
  SweepRay g;
  g.sx = (mux > 0.0) ? 1 : -1;
  g.sy = (ndim >= 2) ? ((muy > 0.0) ? 1 : -1) : 0;
  g.sz = (ndim == 3) ? ((muz > 0.0) ? 1 : -1) : 0;
  g.inv = ComputeSCAngleInv(mux,muy,muz,dx1,dx2,dx3,ndim,g.sx,g.sy,g.sz);
  g.march_axis = MarchAxis(g.inv.axis, ndim);
  return g;
}

//----------------------------------------------------------------------------------------
//! \fn void AxisExtents
//! \brief Planes along a marching axis, and the transverse extents spanning one plane.

KOKKOS_INLINE_FUNCTION
void AxisExtents(int march_axis, int nx1, int nx2, int nx3,
                 int &nplanes, int &nu, int &nv) {
  if (march_axis == 0) {
    nplanes = nx1; nu = nx2; nv = nx3;
  } else if (march_axis == 1) {
    nplanes = nx2; nu = nx1; nv = nx3;
  } else {
    nplanes = nx3; nu = nx1; nv = nx2;
  }
}


}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void SC::BuildIndices
//! \brief Build the tiled kernel's hyperplane orderings once: the cells of one tile, and
//! the tiles of the meshblock. The plane kernel derives its cell coordinates
//! arithmetically and needs no cell table: over a meshblock such a table would be one int
//! per cell, 28 MB at 192^3. On a non-cubic block it does need the ray grouping, which is
//! one int a ray.

void SC::BuildIndices() {
  if (sweep_kernel == SweepKernel::plane) {
    if (!(pmy_pack->pmesh->equal_block_nx)) { BuildRayGroups(); }
    return;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ndim = pang->ndim;
  tile_nx1 = (tile_size > 0) ? tile_size : indcs.nx1;
  tile_nx2 = (ndim >= 2) ? ((tile_size > 0) ? tile_size : indcs.nx2) : 1;
  tile_nx3 = (ndim == 3) ? ((tile_size > 0) ? tile_size : indcs.nx3) : 1;
  ntile1 = indcs.nx1 / tile_nx1;
  ntile2 = (ndim >= 2) ? indcs.nx2 / tile_nx2 : 1;
  ntile3 = (ndim == 3) ? indcs.nx3 / tile_nx3 : 1;
  tile_cells  = IndexHyperplanes(tile_nx1, tile_nx2, tile_nx3, ndim);
  block_tiles = IndexHyperplanes(ntile1, ntile2, ntile3, ndim);
}

//----------------------------------------------------------------------------------------
//! \fn void SC::FormalSolution
//! \brief Dispatch on sweep_kernel.

void SC::FormalSolution() {
  switch (sweep_kernel) {
    case SweepKernel::tiled: FormalSolutionTiled(); break;
    case SweepKernel::plane: FormalSolutionPlane(); break;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::FormalSolutionTiled
//! \brief One launch per tile hyperplane; one team per (tile, meshblock, ray); the team
//! walks the tile's cell hyperplanes with a team barrier between planes. tile_size = 0 is
//! one tile per meshblock: a single launch with nmb x nang_tot teams.

void SC::FormalSolutionTiled() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;
  int ndim = pang->ndim;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &mu = pang->mu;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto ir_ = ir;
  auto chi_ = chi;
  auto srad_ = srad;

  auto tcell_ = tile_cells.items;
  auto tstart_ = tile_cells.start_dev;
  auto tpc_ = block_tiles.items;
  const int tx1 = tile_nx1, tx2 = tile_nx2, tx3 = tile_nx3;
  const int nt1 = ntile1;   // nt2 is implicit in nt12 below; see the packing comment
  const int nt12 = ntile1 * ntile2;
  const int tx12 = tile_nx1 * tile_nx2;
  const int nplanes_cell = tile_cells.nplanes;
  const int ts = team_size;

  for (int H = 0; H < block_tiles.nplanes; ++H) {
    const int tlo = block_tiles.start[H];
    const int ntile = block_tiles.start[H+1] - tlo;
    if (ntile <= 0) continue;

    const int league_size = ntile * nmb * nang_tot_;
    Kokkos::TeamPolicy<> policy = (ts > 0)
        ? Kokkos::TeamPolicy<>(DevExeSpace(), league_size, ts)
        : Kokkos::TeamPolicy<>(DevExeSpace(), league_size, Kokkos::AUTO);
    Kokkos::parallel_for("sc_sweep_tiled", policy,
    KOKKOS_LAMBDA(const TeamMember_t &tmember) {
      const int lid = tmember.league_rank();
      const int t = lid / (nmb * nang_tot_);
      const int rem = lid - t * (nmb * nang_tot_);
      const int m = rem / nang_tot_;
      const int angg = rem - m * nang_tot_;

      const int tlin = tpc_(tlo + t);              // packed tile index (tc*nt2+tb)*nt1+ta
      const int tc = tlin / nt12;
      const int tr = tlin - tc * nt12;
      const int tb = tr / nt1;
      const int ta = tr - tb * nt1;

      const int oct = angg / nang;
      const int a = angg - oct * nang;
      Real mux = mu.d_view(oct,a,0);
      Real muy = (ndim >= 2) ? mu.d_view(oct,a,1) : 0.0;
      Real muz = (ndim == 3) ? mu.d_view(oct,a,2) : 0.0;
      int sx = (mux > 0.0) ? 1 : -1;
      int sy = (ndim >= 2) ? ((muy > 0.0) ? 1 : -1) : 0;
      int sz = (ndim == 3) ? ((muz > 0.0) ? 1 : -1) : 0;
      Real dx1v = mbsize.d_view(m).dx1;
      Real dx2v = mbsize.d_view(m).dx2;
      Real dx3v = mbsize.d_view(m).dx3;

      const int i0 = (sx > 0) ? (is + ta*tx1) : (ie - ta*tx1);
      const int j0 = (sy > 0) ? (js + tb*tx2) : (je - tb*tx2);
      const int k0 = (sz > 0) ? (ks + tc*tx3) : (ke - tc*tx3);

      for (int h = 0; h < nplanes_cell; ++h) {
        const int lo = tstart_(h);
        const int cnt = tstart_(h + 1) - lo;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tmember, cnt),
        [&](const int c) {
          // packed tile-local index (l3*tx2+l2)*tx1+l1
          const int lin = tcell_(lo + c);
          const int l3 = lin / tx12;
          const int r = lin - l3 * tx12;
          const int l2 = r / tx1;
          const int l1 = r - l2 * tx1;
          const int i = (sx > 0) ? (i0 + l1) : (i0 - l1);
          const int j = (sy > 0) ? (j0 + l2) : (j0 - l2);
          const int k = (sz > 0) ? (k0 + l3) : (k0 - l3);
          Real a1 = 0.0;
          Real I = UpdateCellSC(chi_,srad_,ir_,m,angg,
                                i,j,k, sx,sy,sz,
                                mux,muy,muz,
                                dx1v,dx2v,dx3v, ndim,ks,js, &a1);
          ir_(m,angg,k,j,i) = I;
        });
        tmember.team_barrier();
      }
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::FormalSolutionPlane()
//! \brief Sweep by planes taken along each ray's own marching axis.
//!
//! The 3D gather in GatherSolveSC offsets all four upwind cells one step along inv.axis
//! and varies only the transverse indices (axis 0: every cell sits at i-sx; axis 1: at
//! j-sy; axis 2: at k-sz), and the 2D branch does the same about its own crossed face. So
//! the intensity on plane p of a ray depends only on plane p-1 of that ray and on nothing
//! within plane p, so a diagonal i+j+k ordering would be stricter than the stencil
//! requires. For a 192^3 block that is 192 launches of 36864 cells, every one of them
//! full, instead of the 574 averaging 12335 and ramping from 1 up to 27648 that the
//! diagonal costs. Flat par_for, no tiles, teams or barriers, and no cell table.
//!
//! Two launch loops implement it, and this picks between them by block shape. They sweep
//! the same planes in the same order and differ only in thread assignment, so they
//! are bit-identical to each other and to tiled:
//!
//!   PlaneSweepAllRays     one launch per plane, every ray in it, sized to axis 0's
//!                         transverse face. Cubic blocks only: sizing one launch for
//!                         every ray costs nothing when the faces match, and would waste
//!                         longest/shortest of the threads when they do not.
//!   PlaneSweepRaysByAxis  rays grouped by marching axis, one launch per (axis, plane),
//!                         sized to that axis's own extents, so no thread exits early.
//!                         Costs nx1+nx2+nx3 launches instead of max(nx1,nx2,nx3), and a
//!                         ray-index load per thread.
//!
//! The test is Mesh::equal_block_nx, exact equality of the active cell counts. Exact is
//! load-bearing here and not merely a convention: PlaneSweepAllRays sizes every launch
//! from axis 0's extents and carries no per-thread range guard, so a block that was only
//! approximately cubic would index past the block rather than run slowly.
//!
//! On a cubic block every transverse face is the same size, so the all-rays loop wastes
//! nothing and the grouping can only add overhead -- that much is geometry, not tuning,
//! and it is why the test is exact equality of the cell counts rather than a tuned
//! threshold. Off cubic the waste grows without bound (longest/shortest extent) while the
//! grouping's overhead does not, so the grouped loop takes over. Measured on 36
//! configurations across V100 and A100 MIG; the ungrouped loop alone falls to 0.46x on
//! elongated blocks on the A100.
//!
//! PRECONDITION: a cell may read the INTENSITY only at a strictly smaller index along its
//! ray's marching axis. chi and srad are built before the sweep and are read-only during
//! it, so the stencil may read those anywhere, as it already does downwind. The present
//! parabolic scheme satisfies this -- all four of its intensity reads sit one step upwind
//! along the marching axis. What would violate it is an intensity read at the SAME index
//! along that axis: transverse coupling within a plane, such as a limiter or a diffusion
//! term applied to the intensity being computed, which would be a data race rather than a
//! merely different answer. Widening the transverse stencil on the upwind face is
//! safe here, every cell still being one step upwind, but would break the i+j+k ordering
//! tiled relies on. The verification suite pins the two kernels against each other bit
//! for bit.

void SC::FormalSolutionPlane() {
  // the same flag BuildIndices tested when it decided whether to build the ray grouping
  // that PlaneSweepRaysByAxis reads. Mesh sets it once at construction and never changes
  // it, so the two cannot disagree and leave the grouped loop with no groups to sweep.
  if (pmy_pack->pmesh->equal_block_nx) {
    PlaneSweepAllRays();
  } else {
    PlaneSweepRaysByAxis();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::PlaneSweepAllRays()
//! \brief One launch per plane carrying every ray, sized to the largest transverse face.
//!
//! Used when the meshblock is cubic in cells, where that size is every ray's exact size.

void SC::PlaneSweepAllRays() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int ndim = pang->ndim;
  const int nang = pang->nang;
  const int nangt1 = nang_tot - 1;
  auto &mu = pang->mu;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto ir_ = ir;
  auto chi_ = chi;
  auto srad_ = srad;

  // The block is cubic in cells, so every axis carries the same number of planes and
  // every transverse face the same number of cells: no maximum to take over the three
  // axes, and no ray runs out of planes before the others. Axis 0's extents are also
  // right in 2D and 1D without special-casing ndim: an absent dimension already has
  // nx = 1 (mesh.cpp rejects nx < 1, and nx2 == 1 with nx3 > 1), so 2D gives nx2*1 and
  // 1D gives 1*1, which are the face sizes a ray actually needs there.
  const int nplanes = nx1;
  const int ntrans = nx2*nx3;

  for (int p = 0; p < nplanes; ++p) {
    par_for("sc_sweep_plane", DevExeSpace(), 0, nmb1, 0, nangt1, 0, ntrans-1,
    KOKKOS_LAMBDA(const int m, const int angg, const int t) {
      const SweepRay g = RayGeometry(mu.d_view, angg, nang, ndim, mbsize.d_view(m).dx1,
                                     mbsize.d_view(m).dx2, mbsize.d_view(m).dx3);
      // cubic, so the transverse fast extent is nx2 whichever axis the ray marches along
      const int u = t % nx2, v = t / nx2;
      // Inlined on purpose: as a helper this costs 10 registers and up to 26% on V100
      int i, j, k;
      if (g.march_axis == 0) {
        i = (g.sx > 0) ? (is + p) : (ie - p);
        j = (ndim >= 2) ? ((g.sy > 0) ? (js + u) : (je - u)) : js;
        k = (ndim == 3) ? ((g.sz > 0) ? (ks + v) : (ke - v)) : ks;
      } else if (g.march_axis == 1) {
        j = (g.sy > 0) ? (js + p) : (je - p);
        i = (g.sx > 0) ? (is + u) : (ie - u);
        k = (ndim == 3) ? ((g.sz > 0) ? (ks + v) : (ke - v)) : ks;
      } else {
        k = (g.sz > 0) ? (ks + p) : (ke - p);
        i = (g.sx > 0) ? (is + u) : (ie - u);
        j = (g.sy > 0) ? (js + v) : (je - v);
      }
      Real a1 = 0.0;
      ir_(m,angg,k,j,i) = GatherSolveSC(chi_,srad_,ir_,m,angg,i,j,k,g.inv,
                                        ndim,ks,js,&a1);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::PlaneSweepRaysByAxis()
//! \brief One launch per (marching axis, plane), each sized to that axis's own extents.
//!
//! Used when the meshblock is not cubic in cells. Every launched thread computes a cell;
//! the price is nx1+nx2+nx3 launches per sweep instead of nx1=nx2=nx3, and one
//! ray-index load per thread. Same per-cell gather and same plane ordering as
//! PlaneSweepAllRays, hence bit-identical to it.

void SC::PlaneSweepRaysByAxis() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int ndim = pang->ndim;
  const int nang = pang->nang;
  auto &mu = pang->mu;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto ir_ = ir;
  auto chi_ = chi;
  auto srad_ = srad;
  auto rays_ = ray_by_axis;

  for (int march_axis = 0; march_axis < 3; ++march_axis) {
    const int r0 = ray_axis_start[march_axis];
    const int nr = ray_axis_start[march_axis+1] - r0;
    if (nr == 0) continue;
    int nplanes, nu, nv;
    AxisExtents(march_axis, nx1, nx2, nx3, nplanes, nu, nv);
    for (int p = 0; p < nplanes; ++p) {
      par_for("sc_sweep_plane_byaxis", DevExeSpace(), 0, nmb1, 0, nr-1, 0, nu*nv-1,
      KOKKOS_LAMBDA(const int m, const int r, const int t) {
        const int angg = rays_(r0 + r);
        const SweepRay g = RayGeometry(mu.d_view, angg, nang, ndim, mbsize.d_view(m).dx1,
                                       mbsize.d_view(m).dx2, mbsize.d_view(m).dx3);
        const int u = t % nu, v = t / nu;
        // the group's axis, not g.march_axis: the two agree by the invariant
        // BuildRayGroups verifies, and it is the group's axis that sized this launch.
        // Inlined for the reason given in PlaneSweepAllRays above.
        int i, j, k;
        if (march_axis == 0) {
          i = (g.sx > 0) ? (is + p) : (ie - p);
          j = (ndim >= 2) ? ((g.sy > 0) ? (js + u) : (je - u)) : js;
          k = (ndim == 3) ? ((g.sz > 0) ? (ks + v) : (ke - v)) : ks;
        } else if (march_axis == 1) {
          j = (g.sy > 0) ? (js + p) : (je - p);
          i = (g.sx > 0) ? (is + u) : (ie - u);
          k = (ndim == 3) ? ((g.sz > 0) ? (ks + v) : (ke - v)) : ks;
        } else {
          k = (g.sz > 0) ? (ks + p) : (ke - p);
          i = (g.sx > 0) ? (is + u) : (ie - u);
          j = (g.sy > 0) ? (js + v) : (je - v);
        }
        Real a1 = 0.0;
        ir_(m,angg,k,j,i) = GatherSolveSC(chi_,srad_,ir_,m,angg,i,j,k,g.inv,
                                          ndim,ks,js,&a1);
      });
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SC::BuildRayGroups()
//! \brief Sort the rays by the grid axis they march along (host, once).
//!
//! Group g holds ray_by_axis[ray_axis_start[g] .. ray_axis_start[g+1]) and marches along
//! axis g. Classification uses the same MarchAxis and ComputeSCAngleInv the kernel calls,
//! so it cannot disagree with the kernel, ties included.
//!
//! PlaneSweepRaysByAxis takes the marching axis from the group while still computing inv
//! per thread for the gather, so one grouping is only valid if every meshblock would
//! classify a ray the same way. It does: the axis follows the RATIOS dx1:dx2:dx3, and
//! uniform refinement halves all three together. That is checked below, not assumed,
//! because a mesh that broke it would desync the march axis from the stencil offsets and
//! turn the sweep into a silent data race.

void SC::BuildRayGroups() {
  const int ndim = pang->ndim;
  const int nang = pang->nang;
  auto &ms = pmy_pack->pmesh->mesh_size;
  auto &mu = pang->mu;

  std::vector<int> axis_of(nang_tot);
  for (int angg = 0; angg < nang_tot; ++angg) {
    axis_of[angg] = RayGeometry(mu.h_view, angg, nang, ndim,
                                ms.dx1, ms.dx2, ms.dx3).march_axis;
  }

  // every meshblock must classify every ray the same way as the root mesh (see above)
  auto &mbsize = pmy_pack->pmb->mb_size;
  for (int m = 0; m < pmy_pack->nmb_thispack; ++m) {
    for (int angg = 0; angg < nang_tot; ++angg) {
      const int march_axis = RayGeometry(mu.h_view, angg, nang, ndim,
                                         mbsize.h_view(m).dx1, mbsize.h_view(m).dx2,
                                         mbsize.h_view(m).dx3).march_axis;
      if (march_axis != axis_of[angg]) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "sweep_kernel = 'plane' needs every meshblock to agree on "
          << "each ray's marching axis, but block " << m << " puts ray " << angg
          << " on axis " << march_axis << ", the root mesh on " << axis_of[angg]
          << ". Use sweep_kernel = 'tiled', which resolves the axis per meshblock."
          << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  HostArray1D<int> h_rays("ray_by_axis_host", nang_tot);
  int idx = 0;
  for (int g = 0; g < 3; ++g) {
    ray_axis_start[g] = idx;
    for (int angg = 0; angg < nang_tot; ++angg) {
      if (axis_of[angg] == g) { h_rays(idx++) = angg; }
    }
  }
  ray_axis_start[3] = idx;
  // the three groups must partition the rays. They do as long as MarchAxis returns 0, 1
  // or 2; a fourth value would drop those rays from every group, and the sweep would
  // skip them with no other symptom.
  if (idx != nang_tot) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "ray grouping covers " << idx << " of " << nang_tot
      << " rays; MarchAxis returned an axis outside 0-2." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Kokkos::realloc(ray_by_axis, nang_tot);
  Kokkos::deep_copy(ray_by_axis, h_rays);
}


}  // namespace nr_radiation
