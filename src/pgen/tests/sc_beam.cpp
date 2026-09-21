//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sc_beam.cpp
//! \brief Pencil-beam test for the LTE short-characteristics SC solver.
//! Ported from Athena-C beam2d.c. Injects unit intensity at selected boundary cells along
//! a chosen discrete-ordinate angle through a uniform absorber with no source (Davis 2012
//! Fig. 6, linear interpolation). There is no fluid: the pgen enrolls hooks that write
//! sigma_a = const and brad = 0 at every solve (see sc_hooks.hpp).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace {
struct BeamVars {
  int iang;
  int ihor1, ihor2;
  int ivert1, ivert2;
  Real sigma_a;
  Real offbeam_margin;   // root cells from every exit point beyond which J must vanish
};
BeamVars beamvars;

// uniform absorber, no source, over all cells including ghosts
void BeamOpacity(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto sigma_a = pmbp->pnrrad->sigma_a;
  Real sig = beamvars.sigma_a;
  par_for("sc_beam_sigma", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    sigma_a(m,0,k,j,i) = sig;
  });
}

void BeamEmission(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto brad = pmbp->pnrrad->brad;
  par_for("sc_beam_brad", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    brad(m,k,j,i) = 0.0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SCBeamBCs()
//! \brief User boundary function for the intensity on the four `user` faces: every
//! ghost ray is zero (vacuum) except the beam cells, which carry unit intensity along ray
//! `iang` of the octant that points into the domain. Beams are one ROOT cell wide:
//! bottom-face beams occupy root column ihor1 (octant 0) and ihor2 (octant 1), left-face
//! beams root row ivert1 (octant 0) and ivert2 (octant 2). Cells are matched by physical
//! position, so a refined block on the face injects the beam into every fine cell inside
//! that root cell. Runs on the device with no host<->device traffic.

void SCBeamBCs(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pnrrad == nullptr) return;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks;
  int ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng, n2 = indcs.nx2 + 2*ng;
  int nmb = pmbp->nmb_thispack;
  int nang = pmbp->pnrrad->pang->nang, nang_tot = pmbp->pnrrad->nang_tot;
  const BeamVars bv = beamvars;
  auto ir = pmbp->pnrrad->ir;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &size = pmbp->pmb->mb_size;

  // root-cell bounds of each beam
  Real dxr = (pm->mesh_size.x1max - pm->mesh_size.x1min) / pm->mesh_indcs.nx1;
  Real dyr = (pm->mesh_size.x2max - pm->mesh_size.x2min) / pm->mesh_indcs.nx2;
  Real xh1_lo = pm->mesh_size.x1min + bv.ihor1 * dxr, xh1_hi = xh1_lo + dxr;
  Real xh2_lo = pm->mesh_size.x1min + bv.ihor2 * dxr, xh2_hi = xh2_lo + dxr;
  Real yv1_lo = pm->mesh_size.x2min + bv.ivert1 * dyr, yv1_hi = yv1_lo + dyr;
  Real yv2_lo = pm->mesh_size.x2min + bv.ivert2 * dyr, yv2_hi = yv2_lo + dyr;
  bool h1 = bv.ihor1 >= 0, h2 = bv.ihor2 >= 0, v1 = bv.ivert1 >= 0, v2 = bv.ivert2 >= 0;

  // x2 faces: bottom carries the horizontal-entry beams, top is vacuum
  par_for("beam_bc_x2", DevExeSpace(), 0, nmb-1, 0, nang_tot-1, 0, ng-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int a, int ig, int i) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x2) == BoundaryFlag::user) {
      Real v = 0.0;
      if (i >= is && i <= ie) {
        Real dx = size.d_view(m).dx1;
        Real xc = size.d_view(m).x1min + (i - is + 0.5) * dx;   // cell centre
        if (h1 && xc > xh1_lo && xc < xh1_hi && a == 0*nang + bv.iang) v = 1.0;
        if (h2 && xc > xh2_lo && xc < xh2_hi && a == 1*nang + bv.iang) v = 1.0;
      }
      ir(m, a, ks, js-ig-1, i) = v;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x2) == BoundaryFlag::user) {
      ir(m, a, ks, je+ig+1, i) = 0.0;
    }
  });
  // x1 faces: left carries the vertical-entry beams, right is vacuum
  par_for("beam_bc_x1", DevExeSpace(), 0, nmb-1, 0, nang_tot-1, 0, ng-1, 0, n2-1,
  KOKKOS_LAMBDA(int m, int a, int ig, int j) {
    if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::user) {
      Real v = 0.0;
      if (j >= js && j <= je) {
        Real dy = size.d_view(m).dx2;
        Real yc = size.d_view(m).x2min + (j - js + 0.5) * dy;
        if (v1 && yc > yv1_lo && yc < yv1_hi && a == 0*nang + bv.iang) v = 1.0;
        if (v2 && yc > yv2_lo && yc < yv2_hi && a == 2*nang + bv.iang) v = 1.0;
      }
      ir(m, a, ks, j, is-ig-1) = v;
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x1) == BoundaryFlag::user) {
      ir(m, a, ks, j, ie+ig+1) = 0.0;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SCBeamErrors()
//! \brief Diagnostics of the beams, written to sc_beam-errs.dat as
//!   Nx1 Nx2 Ncycle  [integral/expected  centroid_err/dx  rms_width/dx] per beam
//!   offbeam/peak  niter
//! Beam 1 enters at global column ihor1 along the octant-0 ray iang (mu_x > 0), beam 2
//! at column ihor2 along the mirror ray of octant 1 (mu_x < 0); a beam whose column is -1
//! is absent. Along a ray the medium is transparent (tau ~ 1e-6), so the beam that leaves
//! the box carries what entered it: the integral of J across the exit face equals
//! w_iang * dx_root times a width factor (1 for exit through the top, mu_y/|mu_x| for
//! exit through a side). Its centroid sits where the ray from the first ghost cell's
//! centre meets the exit slice, the last interior root-level cells. Off the beams, more
//! than `offbeam_margin` root cells from every exit point, J must be dark. Interpolation
//! smears a beam (Davis 2012 Fig. 6) but conserves its integral and centroid; refinement
//! sharpens it; a second beam crossing the first must not disturb it (rays of different
//! octants are independent).

struct BeamExit {
  bool present = false, side = false;   // side: exits through a vertical face
  // exit coordinate along the slice (y for side, x for top)
  Real pos = 0.0;
  Real expected = 0.0;
  Real integral = 0.0, first = 0.0, second = 0.0, peak = 0.0;
};

void SCBeamErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  nr_radiation::SC *psc = pmbp->pnrrad;
  if (psc == nullptr) return;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2;
  int nmb = pmbp->nmb_thispack;
  auto j_h = Kokkos::create_mirror_view(psc->moments);
  Kokkos::deep_copy(j_h, psc->moments);
  auto &size = pmbp->pmb->mb_size;
  auto mu_h = psc->pang->mu.h_view;
  auto wmu_h = psc->pang->wmu.h_view;
  const BeamVars bv = beamvars;

  Real mux = mu_h(0, bv.iang, 0), muy = mu_h(0, bv.iang, 1);
  Real w = wmu_h(bv.iang);
  Real x1min = pm->mesh_size.x1min, x1max = pm->mesh_size.x1max;
  Real x2min = pm->mesh_size.x2min, x2max = pm->mesh_size.x2max;
  Real dx = (x1max - x1min) / pm->mesh_indcs.nx1;   // root cell
  Real margin = bv.offbeam_margin * dx;

  // geometry of each beam: origin at the centre of the first ghost cell below the bottom
  // face; exit through whichever of the top slice (y = x2max - dx/2) or the side slice
  // (x = x1max - dx/2 or x1min + dx/2) the ray reaches first
  BeamExit beam[2];
  const int cols[2] = {bv.ihor1, bv.ihor2};
  const Real sgn[2] = {1.0, -1.0};
  for (int b = 0; b < 2; ++b) {
    if (cols[b] < 0) continue;
    beam[b].present = true;
    Real x0 = x1min + (cols[b] + 0.5) * dx, y0 = x2min - 0.5 * dx;
    Real ux = sgn[b] * mux;
    Real t_top  = (x2max - 0.5*dx - y0) / muy;
    Real xface  = (ux > 0.0) ? (x1max - 0.5*dx) : (x1min + 0.5*dx);
    Real t_side = (xface - x0) / ux;
    if (t_side < t_top) {
      beam[b].side = true;  beam[b].pos = y0 + t_side * muy;
      beam[b].expected = w * dx * muy / std::fabs(ux);
    } else {
      beam[b].side = false; beam[b].pos = x0 + t_top * ux;
      beam[b].expected = w * dx;
    }
  }

  // walk the exit slices: the last interior root-level row (top) and columns (sides)
  Real offbeam = 0.0, peak_all = 0.0;
  for (int m = 0; m < nmb; ++m) {
    auto &sz = size.h_view(m);
    bool at_top   = std::fabs(sz.x2max - x2max) < 1.0e-12 * (x2max - x2min);
    bool at_right = std::fabs(sz.x1max - x1max) < 1.0e-12 * (x1max - x1min);
    bool at_left  = std::fabs(sz.x1min - x1min) < 1.0e-12 * (x1max - x1min);
    Real ddx = (sz.x1max - sz.x1min) / nx1, ddy = (sz.x2max - sz.x2min) / nx2;
    if (at_top) {
      for (int i = is; i <= ie; ++i) {
        Real x = sz.x1min + (i - is + 0.5) * ddx, J = j_h(m, 0, ks, je, i);
        // each top cell belongs to the beam whose exit point is nearer
        int bn = -1; Real dn = 1.0e30;
        for (int b = 0; b < 2; ++b) {
          if (!beam[b].present || beam[b].side) continue;
          Real d = std::fabs(x - beam[b].pos);
          if (d < dn) { dn = d; bn = b; }
        }
        if (bn >= 0) {
          Real d = x - beam[bn].pos;
          beam[bn].integral += J * ddx; beam[bn].first += x * J * ddx;
          beam[bn].second += d * d * J * ddx; beam[bn].peak = std::max(beam[bn].peak, J);
        }
        peak_all = std::max(peak_all, J);
        if (bn < 0 || dn > margin) offbeam = std::max(offbeam, J);
      }
    }
    for (int b = 0; b < 2; ++b) {
      if (!beam[b].present || !beam[b].side) continue;
      bool mine = (sgn[b] > 0.0) ? at_right : at_left;
      if (!mine) continue;
      int i = (sgn[b] > 0.0) ? ie : is;
      for (int j = js; j <= je; ++j) {
        Real y = sz.x2min + (j - js + 0.5) * ddy, J = j_h(m, 0, ks, j, i);
        Real d = y - beam[b].pos;
        beam[b].integral += J * ddy; beam[b].first += y * J * ddy;
        beam[b].second += d * d * J * ddy; beam[b].peak = std::max(beam[b].peak, J);
        peak_all = std::max(peak_all, J);
        if (std::fabs(d) > margin) offbeam = std::max(offbeam, J);
      }
    }
  }
  Real offbeam_ratio = (peak_all > 0.0) ? offbeam / peak_all : 0.0;

  if (global_variable::my_rank == 0) {
    std::string fname = "sc_beam-errs.dat";
    FILE *pf = std::fopen(fname.c_str(), "r");
    if (pf != nullptr) {
      pf = std::freopen(fname.c_str(), "a", pf);
    } else {
      pf = std::fopen(fname.c_str(), "w");
      std::fprintf(pf, "# Nx1  Nx2   Ncycle   [integral/expected  centroid_err/dx  "
                       "rms_width/dx]"
                       " x2 beams  offbeam/peak  niter\n");
    }
    std::fprintf(pf, "%04d  %04d  %05d ", pm->mesh_indcs.nx1, pm->mesh_indcs.nx2,
                 pm->ncycle);
    for (int b = 0; b < 2; ++b) {
      Real ratio = 0.0, cen = 0.0, wid = 0.0;
      if (beam[b].present && beam[b].integral > 0.0) {
        ratio = beam[b].integral / beam[b].expected;
        cen = (beam[b].first / beam[b].integral - beam[b].pos) / dx;
        wid = std::sqrt(beam[b].second / beam[b].integral) / dx;
      }
      std::fprintf(pf, " %e %e %e", ratio, cen, wid);
      if (beam[b].present) {
        std::cout << "sc_beam: beam " << b+1 << " exits through the "
                  << (beam[b].side ? "side" : "top") << " at " << beam[b].pos
                  << ": integral/expected = " << ratio << ", centroid error = " << cen
                  << " root cells, rms width = " << wid << " root cells" << std::endl;
      }
    }
    std::fprintf(pf, " %e %d\n", offbeam_ratio, psc->niter_last);
    std::fclose(pf);
    std::cout << "sc_beam: off-beam max/peak = " << offbeam_ratio << " ("
              << psc->niter_last
              << " sweeps)" << std::endl;
  }
}
}  // namespace

void ProblemGenerator::SCBeam(ParameterInput *pin, const bool restart) {
  (void)restart;   // hooks are enrolled on the restart path too
  user_bcs_func = SCBeamBCs;
  pgen_final_func = SCBeamErrors;

  beamvars.iang    = pin->GetOrAddInteger("problem", "iang", 2);
  beamvars.ihor1   = pin->GetOrAddInteger("problem", "ihor1", 96);
  beamvars.ihor2   = pin->GetOrAddInteger("problem", "ihor2", -1);
  beamvars.ivert1  = pin->GetOrAddInteger("problem", "ivert1", -1);
  beamvars.ivert2  = pin->GetOrAddInteger("problem", "ivert2", -1);
  beamvars.sigma_a = pin->GetReal("problem", "sigma_a");
  beamvars.offbeam_margin = pin->GetOrAddReal("problem", "offbeam_margin", 16.0);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pnrrad == nullptr) {
    std::cout << "### FATAL ERROR: sc_beam needs <nr_radiation>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->phydro != nullptr || pmbp->pmhd != nullptr) {
    std::cout << "### FATAL ERROR: sc_beam takes no fluid (the hooks supply sigma_a and "
              << "emission)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &pang = *pmbp->pnrrad->pang;
  if (beamvars.iang < 0 || beamvars.iang >= pang.nang) {
    std::cout << "### FATAL ERROR: problem/iang out of range" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "sc_beam: iang=" << beamvars.iang
            << " mu0=(" << pang.mu.h_view(0,beamvars.iang,0) << ","
            << pang.mu.h_view(0,beamvars.iang,1) << ")" << std::endl;

  pmbp->pnrrad->EnrollOpacityFunction(BeamOpacity);
  pmbp->pnrrad->EnrollEmissionFunction(BeamEmission);
}
