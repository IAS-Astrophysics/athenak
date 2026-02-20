//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file binary_gravity.cpp
//  \brief Problem generator for binary potential test (two spherical mass blobs)
//
//  Sets up two uniform-density spheres with masses m1, m2 at (x1,y1,z1) and (x2,y2,z2).
//  Used to validate the multigrid Poisson solver: the numerical potential can be
//  compared to the analytical solution for two spheres (interior: quadratic in r,
//  exterior: -G*m/r). Domain should contain both spheres; typical setup uses
//  a small domain around the origin (e.g. [-0.5, 0.5]^3) with sphere radius r
//  and positions scaled accordingly (e.g. Athena++ defaults: x1=6/1024, x2=-12/1024).
//
//  Reference: Athena++ pgen binary_gravity.cpp

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "gravity/gravity.hpp"
#include "gravity/mg_gravity.hpp"
#include "pgen/pgen.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace {
constexpr Real floor_den = 1.0e-300;
constexpr Real pi = 3.14159265358979323846;
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BinaryGravity()
//  \brief Problem generator for binary potential test (two spheres, multigrid gravity)
void ProblemGenerator::BinaryGravity(ParameterInput *pin, const bool restart) {
  if (restart) return;

  Real four_pi_G = pin->GetOrAddReal("problem", "four_pi_G", 1.0);
  pin->SetReal("gravity", "four_pi_G", four_pi_G);
  if (pmy_mesh_->pmb_pack->pgrav != nullptr) {
    pmy_mesh_->pmb_pack->pgrav->four_pi_G = four_pi_G;
    if (pmy_mesh_->pmb_pack->pgrav->pmgd != nullptr) {
      pmy_mesh_->pmb_pack->pgrav->pmgd->SetFourPiG(four_pi_G);
    }
  }

  // Sphere positions and properties (Athena++ defaults)
  Real x1 = pin->GetOrAddReal("problem", "x1", 6.0 / 1024.0);
  Real y1 = pin->GetOrAddReal("problem", "y1", 0.0);
  Real z1 = pin->GetOrAddReal("problem", "z1", 0.0);
  Real x2 = pin->GetOrAddReal("problem", "x2", -12.0 / 1024.0);
  Real y2 = pin->GetOrAddReal("problem", "y2", 0.0);
  Real z2 = pin->GetOrAddReal("problem", "z2", 0.0);
  Real r = pin->GetOrAddReal("problem", "radius", 6.0 / 1024.0);
  Real m1 = pin->GetOrAddReal("problem", "m1", 2.0);
  Real m2 = pin->GetOrAddReal("problem", "m2", 1.0);

  Real den1 = m1 / ((4.0 * pi / 3.0) * r * r * r);
  Real den2 = m2 / ((4.0 * pi / 3.0) * r * r * r);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  if (pmbp->phydro == nullptr) return;

  auto &u0 = pmbp->phydro->u0;
  bool is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
  int nmb = pmbp->nmb_thispack;

  // Initialize density to match Athena++ exactly: plain Euclidean distance,
  // same formulas (cell center, left edge for sub-sampling, dd/dr from block dx).
  par_for(
      "binary_gravity_init", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
        Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
        Real x3min = size.d_view(m).x3min, x3max = size.d_view(m).x3max;
        Real dx = size.d_view(m).dx1;
        Real dd = 0.1 * dx;
        Real dv = 1.0e-3;
        Real dr = 0.6 * static_cast<Real>(1.7320508075688772) * dx;  // sqrt(3)

        Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
        Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
        Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

        Real xf = LeftEdgeX(i - is, indcs.nx1, x1min, x1max);
        Real yf = LeftEdgeX(j - js, indcs.nx2, x2min, x2max);
        Real zf = LeftEdgeX(k - ks, indcs.nx3, x3min, x3max);

        Real r1 = std::sqrt(SQR(x - x1) + SQR(y - y1) + SQR(z - z1));
        Real r2 = std::sqrt(SQR(x - x2) + SQR(y - y2) + SQR(z - z2));

        Real rho = floor_den;

        if (r1 < r + dr) {
          if (r1 < r - dr) {
            rho = den1;
          } 
          else {
            int n_in = 0;
            for (int kk = 0; kk < 10; ++kk) {
              Real zz = zf + (kk + 0.5) * dd;
              for (int jj = 0; jj < 10; ++jj) {
                Real yy = yf + (jj + 0.5) * dd;
                for (int ii = 0; ii < 10; ++ii) {
                  Real xx = xf + (ii + 0.5) * dd;
                  Real rr = std::sqrt(SQR(xx - x1) + SQR(yy - y1) + SQR(zz - z1));
                  if (rr < r) n_in++;
                }
              }
            }
            rho = floor_den + dv * den1 * static_cast<Real>(n_in);
          }
        }
        if (r2 < r + dr) {
          if (r2 < r - dr) {
            rho = den2;  // interior of sphere 2 (overwrites if also in sphere 1)
          } 
          else {
            int n_in = 0;
            for (int kk = 0; kk < 10; ++kk) {
              Real zz = zf + (kk + 0.5) * dd;
              for (int jj = 0; jj < 10; ++jj) {
                Real yy = yf + (jj + 0.5) * dd;
                for (int ii = 0; ii < 10; ++ii) {
                  Real xx = xf + (ii + 0.5) * dd;
                  Real rr = std::sqrt(SQR(xx - x2) + SQR(yy - y2) + SQR(zz - z2));
                  if (rr < r) n_in++;
                }
              }
            }
            rho += dv * den2 * static_cast<Real>(n_in);
          }
        }

        u0(m, IDN, k, j, i) = rho;
        u0(m, IM1, k, j, i) = 0.0;
        u0(m, IM2, k, j, i) = 0.0;
        u0(m, IM3, k, j, i) = 0.0;
        if (is_ideal) {
          u0(m, IEN, k, j, i) = rho;
        }
      });

  // Sum total mass (rho * vol) over this pack
  int nk = ke - ks + 1, nj = je - js + 1, ni = ie - is + 1;
  int nmkji = nmb * nk * nj * ni;
  Real local_mass = 0.0;
  Kokkos::parallel_reduce(
      "binary_gravity_mass",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(int idx, Real &lm) {
        int ii = idx % ni;
        int jj = (idx / ni) % nj;
        int kk = (idx / (ni * nj)) % nk;
        int mm = idx / (ni * nj * nk);
        int i = is + ii, j = js + jj, k = ks + kk;
        Real vol = size.d_view(mm).dx1 * size.d_view(mm).dx2 * size.d_view(mm).dx3;
        lm += u0(mm, IDN, k, j, i) * vol;
      },
      local_mass);

#ifdef MPI_PARALLEL
  Real total_mass = 0.0;
  MPI_Allreduce(&local_mass, &total_mass, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
  Real total_mass = local_mass;
#endif

  Real fac = (total_mass > 0.0) ? ((m1 + m2) / total_mass) : 1.0;
  if (global_variable::my_rank == 0 && (total_mass < (m1 + m2) * 0.7 || total_mass > (m1 + m2) * 1.3)) {
    std::cout << "binary_gravity: total mass " << total_mass << " far from m1+m2=" << (m1 + m2)
              << "; resolution may be too low." << std::endl;
  }

  par_for(
      "binary_gravity_rescale", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        u0(m, IDN, k, j, i) *= fac;
        if (is_ideal) {
          u0(m, IEN, k, j, i) *= fac;
        }
      });
}
