//========================================================================================
// surface_grid.cpp — Implementation for the SphericalSurfaceGrid class
//----------------------------------------------------------------------------------------

#include "surface_grid.hpp"

#include "mesh/mesh.hpp"

#include "coordinates/cell_locations.hpp"  // host utilities for cell centers
#include <cmath>
#include <iostream>
#include <cstdlib> // For std::exit, EXIT_FAILURE

//----------------------------------------------------------------------------------------
// Constructor
SphericalSurfaceGrid::SphericalSurfaceGrid(MeshBlockPack* pack,
                                           int ntheta, int nphi,
                                           RFunc r_of_thph,
                                           const std::string& name,
                                           const Real* center)
    : pmy_pack(pack), tag(name) {
  // --- basic validation ---
  if (pmy_pack == nullptr || pmy_pack->pmesh == nullptr || pmy_pack->pmb == nullptr) {
    std::cerr << "### FATAL: SphericalSurfaceGrid requires a valid MeshBlockPack/mesh"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ntheta < 3 || nphi < 3) {
    std::cerr << "### FATAL: SphericalSurfaceGrid needs ntheta>=3 and nphi>=3 (got "
              << ntheta << "," << nphi << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Store center
  center_[0] = center ? center[0] : 0.0;
  center_[1] = center ? center[1] : 0.0;
  center_[2] = center ? center[2] : 0.0;

  // Grid in parameter space
  n_th = ntheta; n_ph = nphi; npts = n_th * n_ph;
  dth = (n_th > 1) ? (M_PI / (n_th - 1)) : 0.0;     // θ ∈ [0,π]
  dph = (n_ph > 1) ? (2.0 * M_PI / n_ph) : 0.0;     // φ ∈ [0,2π)

  // Allocate geometry arrays
  Kokkos::realloc(theta,  npts);
  Kokkos::realloc(phi,    npts);
  Kokkos::realloc(radius, npts);
  Kokkos::realloc(coords, npts, 3);
  Kokkos::realloc(tan_th, npts, 3);
  Kokkos::realloc(tan_ph, npts, 3);
  Kokkos::realloc(weights, npts);

  // Interpolation maps: allocate BEFORE building them
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  Kokkos::realloc(interp_indcs, npts, 4);
  Kokkos::realloc(interp_wghts, npts, 2 * indcs.ng, 3);

  // Initialize parameter grid + r(θ,φ) on host
  int p = 0;
  for (int it = 0; it < n_th; ++it) {
    const Real th = it * dth;
    for (int ip = 0; ip < n_ph; ++ip, ++p) {
      const Real ph = ip * dph;
      theta.h_view(p) = th;
      phi.h_view(p)   = ph;
      radius.h_view(p) = r_of_thph(th, ph);
    }
  }

  // Build everything
  RebuildAll();
}

//----------------------------------------------------------------------------------------
// Public Methods

void SphericalSurfaceGrid::SetCenter(const Real new_center[3]) {
  center_[0] = new_center[0];
  center_[1] = new_center[1];
  center_[2] = new_center[2];
  BuildCoordinates();
  SetInterpolationIndices();
  SetInterpolationWeights();
}

void SphericalSurfaceGrid::InterpolateToSurface(int nvars, DvceArray5D<Real> &val) {
  if (pmy_pack->pmesh->adaptive) {
    SetInterpolationIndices();
    SetInterpolationWeights();
  }
  Kokkos::realloc(interp_vals, npts, nvars);

  auto &iindcs = interp_indcs;
  auto &iwghts = interp_wghts;
  auto &ivals  = interp_vals;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;

  par_for("int2surf_block", DevExeSpace(), 0, npts-1, 0, nvars-1,
    KOKKOS_LAMBDA(const int p, const int v) {
      const int mb_id = iindcs.d_view(p,0);
      if (mb_id == -1) {
        ivals.d_view(p,v) = 0.0;
        return;
      }
      // Base indices are logical (0-based in active zone)
      const int i0 = iindcs.d_view(p,1);
      const int j0 = iindcs.d_view(p,2);
      const int k0 = iindcs.d_view(p,3);
      Real accum = 0.0;

      for (int k_sten = 0; k_sten < 2*ng; ++k_sten) {
        for (int j_sten = 0; j_sten < 2*ng; ++j_sten) {
          for (int i_sten = 0; i_sten < 2*ng; ++i_sten) {
            const Real w = iwghts.d_view(p, i_sten, 0)
                         * iwghts.d_view(p, j_sten, 1)
                         * iwghts.d_view(p, k_sten, 2);

            // Adapted index calculation from GaussLegendreGrid example
            const int I = i0 - (ng - i_sten - is) + 1;
            const int J = j0 - (ng - j_sten - js) + 1;
            const int K = k0 - (ng - k_sten - ks) + 1;

            accum += w * val(mb_id, v, K, J, I);
          }
        }
      }
      ivals.d_view(p,v) = accum;
    });
  interp_vals.template modify<DevExeSpace>();
}


void SphericalSurfaceGrid::BuildSurfaceCovectors(const DualArray1D<Real>& sqrtg,
                                                 DualArray2D<Real>& dSigma) const {
  const int np = npts;
  Kokkos::realloc(dSigma, np, 3);
  auto &eTh = tan_th; auto &ePh = tan_ph; auto &wq = weights;

  Kokkos::parallel_for("surf_cov_curved",
    Kokkos::RangePolicy<DevExeSpace>(0, np),
    KOKKOS_LAMBDA(const int p) {
      // The scaling factor is sqrt(g) * dθ * dφ
      const Real s = sqrtg.d_view(p) * wq.d_view(p);

      // The cross product computes [ijk] * e_θ^j * e_φ^k
      const Real e1x = eTh.d_view(p,0), e1y = eTh.d_view(p,1), e1z = eTh.d_view(p,2);
      const Real e2x = ePh.d_view(p,0), e2y = ePh.d_view(p,1), e2z = ePh.d_view(p,2);
      dSigma.d_view(p,0) = s * (e1y*e2z - e1z*e2y);
      dSigma.d_view(p,1) = s * (e1z*e2x - e1x*e2z);
      dSigma.d_view(p,2) = s * (e1x*e2y - e1y*e2x);
    }
  );
  dSigma.template modify<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
// Private Helper Methods

void SphericalSurfaceGrid::RebuildAll() {
  BuildCoordinates();
  // MUST determine point ownership before assigning weights
  SetInterpolationIndices();
  BuildTangentsFD();
  // MUST assign quad weights after indices are known for MPI-safety
  BuildQuadWeights();
  SetInterpolationWeights();
}

void SphericalSurfaceGrid::BuildCoordinates() {
  auto h_th = theta.h_view;
  auto h_ph = phi.h_view;
  auto h_r = radius.h_view;
  auto h_x = coords.h_view;
  for (int p = 0; p < npts; ++p) {
    Real Xl, Yl, Zl;
    const Real st = sin(h_th(p)), ct = cos(h_th(p));
    const Real sp = sin(h_ph(p)), cp = cos(h_ph(p));
    Xl = h_r(p) * st * cp;
    Yl = h_r(p) * st * sp;
    Zl = h_r(p) * ct;
    h_x(p,0) = Xl + center_[0];
    h_x(p,1) = Yl + center_[1];
    h_x(p,2) = Zl + center_[2];
  }
  coords.template modify<HostMemSpace>();
  coords.template sync<DevExeSpace>();
}

void SphericalSurfaceGrid::BuildQuadWeights() {
  auto h_w = weights.h_view;
  // This must be called after SetInterpolationIndices to work correctly with MPI
  auto h_iind = interp_indcs.h_view;

  for (int p = 0; p < npts; ++p) {
    // A point has a non-zero integration weight only if it is owned by this rank
    if (h_iind(p, 0) != -1) {
      h_w(p) = dth * dph;
    } else {
      h_w(p) = 0.0;
    }
  }
  weights.template modify<HostMemSpace>();
  weights.template sync<DevExeSpace>();
}

//---- ADAPTED INTERPOLATION ROUTINES ----

void SphericalSurfaceGrid::SetInterpolationIndices() {
  auto &size = pmy_pack->pmb->mb_size;
  const int nmb = pmy_pack->nmb_thispack;

  auto h_coords = coords.h_view;
  auto h_iind = interp_indcs.h_view;

  for (int p = 0; p < npts; ++p) {
    // indices default to -1 if point does not reside in this MeshBlockPack
    h_iind(p,0) = -1;
    h_iind(p,1) = -1;
    h_iind(p,2) = -1;
    h_iind(p,3) = -1;
    for (int m = 0; m < nmb; ++m) {
      const auto &mb = size.h_view(m);
      const Real x1p = h_coords(p,0), x2p = h_coords(p,1), x3p = h_coords(p,2);

      // check if this point resides in this MeshBlock
      if ((x1p >= mb.x1min && x1p < mb.x1max) &&
          (x2p >= mb.x2min && x2p < mb.x2max) &&
          (x3p >= mb.x3min && x3p < mb.x3max)) {
        h_iind(p,0) = m;
        // calculate logical indices using floor logic
        h_iind(p,1) = static_cast<int>(std::floor((x1p - (mb.x1min + mb.dx1/2.0)) / mb.dx1));
        h_iind(p,2) = static_cast<int>(std::floor((x2p - (mb.x2min + mb.dx2/2.0)) / mb.dx2));
        h_iind(p,3) = static_cast<int>(std::floor((x3p - (mb.x3min + mb.dx3/2.0)) / mb.dx3));
        break; // Found the meshblock, move to next point
      }
    }
  }
  interp_indcs.template modify<HostMemSpace>();
  interp_indcs.template sync<DevExeSpace>();
}

void SphericalSurfaceGrid::SetInterpolationWeights() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int ng = indcs.ng;

  auto h_iind = interp_indcs.h_view;
  auto h_iw = interp_wghts.h_view;
  auto h_coords = coords.h_view;

  for (int p = 0; p < npts; ++p) {
    const int mb_id = h_iind(p,0);
    if (mb_id == -1) {  // point not on this rank
      for (int i=0; i<2*ng; ++i) {
        h_iw(p,i,0) = 0.0;
        h_iw(p,i,1) = 0.0;
        h_iw(p,i,2) = 0.0;
      }
    } else {
      const int i0 = h_iind(p,1), j0 = h_iind(p,2), k0 = h_iind(p,3);
      const Real x0 = h_coords(p,0), y0 = h_coords(p,1), z0 = h_coords(p,2);

      const auto &mb = size.h_view(mb_id);

      // set interpolation weights
      for (int i=0; i<2*ng; ++i) {
        h_iw(p,i,0) = 1.0;
        h_iw(p,i,1) = 1.0;
        h_iw(p,i,2) = 1.0;
        for (int j=0; j<2*ng; ++j) {
          if (j != i) {
            Real x1i = CellCenterX(i0 - ng + i + 1, indcs.nx1, mb.x1min, mb.x1max);
            Real x1j = CellCenterX(i0 - ng + j + 1, indcs.nx1, mb.x1min, mb.x1max);
            h_iw(p,i,0) *= (x0 - x1j) / (x1i - x1j);

            Real x2i = CellCenterX(j0 - ng + i + 1, indcs.nx2, mb.x2min, mb.x2max);
            Real x2j = CellCenterX(j0 - ng + j + 1, indcs.nx2, mb.x2min, mb.x2max);
            h_iw(p,i,1) *= (y0 - x2j) / (x2i - x2j);

            Real x3i = CellCenterX(k0 - ng + i + 1, indcs.nx3, mb.x3min, mb.x3max);
            Real x3j = CellCenterX(k0 - ng + j + 1, indcs.nx3, mb.x3min, mb.x3max);
            h_iw(p,i,2) *= (z0 - x3j) / (x3i - x3j);
          }
        }
      }
    }
  }
  interp_wghts.template modify<HostMemSpace>();
  interp_wghts.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------

void SphericalSurfaceGrid::BuildTangentsFD() {
  auto h_x = coords.h_view;
  auto h_tan_th = tan_th.h_view;
  auto h_tan_ph = tan_ph.h_view;

  for (int it = 0; it < n_th; ++it) {
    for (int ip = 0; ip < n_ph; ++ip) {
      const int p = it * n_ph + ip;

      // θ-direction
      if (it > 0 && it < n_th - 1) { // central
        const int pp = p + n_ph;
        const int pm = p - n_ph;
        const Real fac = 0.5 / dth;
        for (int d=0; d<3; ++d) h_tan_th(p,d) = fac*(h_x(pp,d)-h_x(pm,d));
      } else if (it == 0) { // forward
        const int p1 = p + n_ph, p2 = p + 2*n_ph;
        const Real fac = 0.5 / dth;
        for (int d=0; d<3; ++d) h_tan_th(p,d) = fac*(-3.0*h_x(p,d) + 4.0*h_x(p1,d) - h_x(p2,d));
      } else { // backward (it == n_th - 1)
        const int p1 = p - n_ph, p2 = p - 2*n_ph;
        const Real fac = 0.5 / dth;
        for (int d=0; d<3; ++d) h_tan_th(p,d) = fac*(3.0*h_x(p,d) - 4.0*h_x(p1,d) + h_x(p2,d));
      }

      // φ-direction (periodic central)
      const int pr = it * n_ph + ((ip + 1) % n_ph);
      const int pl = it * n_ph + ((ip - 1 + n_ph) % n_ph);
      const Real facp = 0.5 / dph;
      for (int d=0; d<3; ++d) h_tan_ph(p,d) = facp*(h_x(pr,d)-h_x(pl,d));
    }
  }
  tan_th.template modify<HostMemSpace>();
  tan_ph.template modify<HostMemSpace>();
  tan_th.template sync<DevExeSpace>();
  tan_ph.template sync<DevExeSpace>();
}