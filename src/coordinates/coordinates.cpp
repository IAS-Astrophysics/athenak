//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.cpp
//! \brief 

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "cartesian_ks.hpp"
#include "coordinates.hpp"
#include "cell_locations.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures describing MeshBlocks

Coordinates::Coordinates(Mesh *pm, RegionIndcs indcs, int igids, int nmb)
  : pmy_mesh(pm),
    coord_data(nmb)
{
  // initialize MeshBlock cell indices
  coord_data.mb_indcs.ng  = indcs.ng;
  coord_data.mb_indcs.nx1 = indcs.nx1;
  coord_data.mb_indcs.nx2 = indcs.nx2;
  coord_data.mb_indcs.nx3 = indcs.nx3;

  coord_data.mb_indcs.is = coord_data.mb_indcs.ng;
  coord_data.mb_indcs.ie = coord_data.mb_indcs.is + coord_data.mb_indcs.nx1 - 1;
    
  if (coord_data.mb_indcs.nx2 > 1) {
    coord_data.mb_indcs.js = coord_data.mb_indcs.ng;
    coord_data.mb_indcs.je = coord_data.mb_indcs.js + coord_data.mb_indcs.nx2 - 1; 
  } else {
    coord_data.mb_indcs.js = 0;
    coord_data.mb_indcs.je = 0;
  } 
      
  if (coord_data.mb_indcs.nx3 > 1) {
    coord_data.mb_indcs.ks = coord_data.mb_indcs.ng;
    coord_data.mb_indcs.ke = coord_data.mb_indcs.ks + coord_data.mb_indcs.nx3 - 1;
  } else {
    coord_data.mb_indcs.ks = 0;
    coord_data.mb_indcs.ke = 0;
  } 

  // calculate physical size of MeshBlocks.  Note only host array is initialized
  auto &ms = pm->mesh_size;
  auto &mb_size = coord_data.mb_size;
  for (int m=0; m<nmb; ++m) {

    // x1-direction
    std::int32_t &lx1 = pm->loclist[igids+m].lx1;
    std::int32_t &lev = pm->loclist[igids+m].level;
    std::int32_t nmbx1 = pm->nmb_rootx1 << (lev - pm->root_level);
    if (lx1 == 0) {
      mb_size.h_view(m).x1min = ms.x1min;
    } else {
      mb_size.h_view(m).x1min = LeftEdgeX(lx1, nmbx1, ms.x1min, ms.x1max);
    }

    if (lx1 == nmbx1 - 1) {
      mb_size.h_view(m).x1max = ms.x1max;
    } else {
      mb_size.h_view(m).x1max = LeftEdgeX(lx1+1, nmbx1, ms.x1min, ms.x1max);
    }

    // x2-direction
    if (pm->mesh_indcs.nx2 == 1) {
      mb_size.h_view(m).x2min = ms.x2min;
      mb_size.h_view(m).x2max = ms.x2max;
    } else {

      std::int32_t &lx2 = pm->loclist[igids+m].lx2;
      std::int32_t nmbx2 = pm->nmb_rootx2 << (lev - pm->root_level);
      if (lx2 == 0) {
        mb_size.h_view(m).x2min = ms.x2min;
      } else {
        mb_size.h_view(m).x2min = LeftEdgeX(lx2, nmbx2, ms.x2min, ms.x2max);
      }

      if (lx2 == (nmbx2) - 1) {
        mb_size.h_view(m).x2max = ms.x2max;
      } else {
        mb_size.h_view(m).x2max = LeftEdgeX(lx2+1, nmbx2, ms.x2min, ms.x2max);
      }

    }

    // x3-direction
    if (pm->mesh_indcs.nx3 == 1) {
      mb_size.h_view(m).x3min = ms.x3min;
      mb_size.h_view(m).x3max = ms.x3max;
    } else {
      std::int32_t &lx3 = pm->loclist[igids+m].lx3;
      std::int32_t nmbx3 = pm->nmb_rootx3 << (lev - pm->root_level);
      if (lx3 == 0) {
        mb_size.h_view(m).x3min = ms.x3min;
      } else {
        mb_size.h_view(m).x3min = LeftEdgeX(lx3, nmbx3, ms.x3min, ms.x3max);
      }
      if (lx3 == (nmbx3) - 1) {
        mb_size.h_view(m).x3max = ms.x3max;
      } else {
        mb_size.h_view(m).x3max = LeftEdgeX(lx3+1, nmbx3, ms.x3min, ms.x3max);
      }
    }

    // grid spacing at this level.  Ensure all MeshBlocks at same level have same dx
    mb_size.h_view(m).dx1 = ms.dx1*static_cast<Real>(1<<(lev - pm->root_level));
    mb_size.h_view(m).dx2 = ms.dx2*static_cast<Real>(1<<(lev - pm->root_level));
    mb_size.h_view(m).dx3 = ms.dx3*static_cast<Real>(1<<(lev - pm->root_level));
  }

  // mark DualArray as modified, and then sync device with host
  mb_size.template modify<HostMemSpace>();
  mb_size.template sync<DevExeSpace>();

}

//----------------------------------------------------------------------------------------
//! \fn
// Read properties of metric from input file for GR.  This function called from Hydro
// and MHD constructors, but only when GR is specified

void Coordinates::InitMetric(ParameterInput *pin)
{
  coord_data.is_minkowski = pin->GetOrAddBoolean("coord","minkowski",false);
  coord_data.bh_mass = pin->GetReal("coord","m");
  coord_data.bh_spin = pin->GetReal("coord","a");
  coord_data.bh_rmin = pin->GetOrAddReal("coord","rmin",0.0);
}

//----------------------------------------------------------------------------------------
//! \fn
// Coordinate (geometric) source term function for GR hydrodynamics

void Coordinates::AddCoordTerms(const DvceArray5D<Real> &prim, const EOS_Data &eos,
                                const Real dt, DvceArray5D<Real> &cons)
{
  // capture variables for kernel
  auto &indcs = coord_data.mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  auto &coord = coord_data;
  int nmb1 = pmy_mesh->pmb_pack->nmb_thispack - 1;

  Real gamma_prime = eos.gamma / (eos.gamma - 1.0);

  par_for("coord_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
    {
      // Extract components of metric
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, false,
                              coord.bh_spin, g_, gi_);

      // Extract primitives
      const Real &rho  = prim(m,IDN,k,j,i);
      const Real &pgas = prim(m,IEN,k,j,i);
      const Real &uu1  = prim(m,IVX,k,j,i);
      const Real &uu2  = prim(m,IVY,k,j,i);
      const Real &uu3  = prim(m,IVZ,k,j,i);

      // Calculate 4-velocity
      Real uu_sq = g_[I11]*uu1*uu1 + 2.0*g_[I12]*uu1*uu2 + 2.0*g_[I13]*uu1*uu3
                 + g_[I22]*uu2*uu2 + 2.0*g_[I23]*uu2*uu3
                 + g_[I33]*uu3*uu3;
      Real alpha = sqrt(-1.0/gi_[I00]);
      Real gamma = sqrt(1.0 + uu_sq);
      Real u0 = gamma / alpha;
      Real u1 = uu1 - alpha * gamma * gi_[I01];
      Real u2 = uu2 - alpha * gamma * gi_[I02];
      Real u3 = uu3 - alpha * gamma * gi_[I03];

      // Calculate stress-energy tensor
      Real wtot = rho + gamma_prime * pgas;
      Real ptot = pgas;
      Real tt[NMETRIC];
      tt[I00] = wtot * u0 * u0 + ptot * gi_[I00];
      tt[I01] = wtot * u0 * u1 + ptot * gi_[I01];
      tt[I02] = wtot * u0 * u2 + ptot * gi_[I02];
      tt[I03] = wtot * u0 * u3 + ptot * gi_[I03];
      tt[I11] = wtot * u1 * u1 + ptot * gi_[I11];
      tt[I12] = wtot * u1 * u2 + ptot * gi_[I12];
      tt[I13] = wtot * u1 * u3 + ptot * gi_[I13];
      tt[I22] = wtot * u2 * u2 + ptot * gi_[I22];
      tt[I23] = wtot * u2 * u3 + ptot * gi_[I23];
      tt[I33] = wtot * u3 * u3 + ptot * gi_[I33];

      // Calculate source terms
      Real dg_dx1[NMETRIC], dg_dx2[NMETRIC], dg_dx3[NMETRIC];
      ComputeMetricDerivatives(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                               dg_dx1, dg_dx2, dg_dx3);

      Real s_1 = 0.0, s_2 = 0.0, s_3 = 0.0;
      s_1 += dg_dx1[I00] * tt[I00];
      s_1 += dg_dx1[I01] * tt[I01];
      s_1 += dg_dx1[I02] * tt[I02];
      s_1 += dg_dx1[I03] * tt[I03];
      s_1 += dg_dx1[I11] * tt[I11];
      s_1 += dg_dx1[I12] * tt[I12];
      s_1 += dg_dx1[I13] * tt[I13];
      s_1 += dg_dx1[I22] * tt[I22];
      s_1 += dg_dx1[I23] * tt[I23];
      s_1 += dg_dx1[I33] * tt[I33];

      s_2 += dg_dx2[I00] * tt[I00];
      s_2 += dg_dx2[I01] * tt[I01];
      s_2 += dg_dx2[I02] * tt[I02];
      s_2 += dg_dx2[I03] * tt[I03];
      s_2 += dg_dx2[I11] * tt[I11];
      s_2 += dg_dx2[I12] * tt[I12];
      s_2 += dg_dx2[I13] * tt[I13];
      s_2 += dg_dx2[I22] * tt[I22];
      s_2 += dg_dx2[I23] * tt[I23];
      s_2 += dg_dx2[I33] * tt[I33];

      s_3 += dg_dx3[I00] * tt[I00];
      s_3 += dg_dx3[I01] * tt[I01];
      s_3 += dg_dx3[I02] * tt[I02];
      s_3 += dg_dx3[I03] * tt[I03];
      s_3 += dg_dx3[I11] * tt[I11];
      s_3 += dg_dx3[I12] * tt[I12];
      s_3 += dg_dx3[I13] * tt[I13];
      s_3 += dg_dx3[I22] * tt[I22];
      s_3 += dg_dx3[I23] * tt[I23];
      s_3 += dg_dx3[I33] * tt[I33];

      s_1 -= 0.5 * (dg_dx1[I00] * tt[I00]
                  + dg_dx1[I11] * tt[I11]
                  + dg_dx1[I22] * tt[I22]
                  + dg_dx1[I33] * tt[I33]);
      s_2 -= 0.5 * (dg_dx2[I00] * tt[I00]
                  + dg_dx2[I11] * tt[I11]
                  + dg_dx2[I22] * tt[I22]
                  + dg_dx2[I33] * tt[I33]);
      s_3 -= 0.5 * (dg_dx3[I00] * tt[I00]
                  + dg_dx3[I11] * tt[I11]
                  + dg_dx3[I22] * tt[I22]
                  + dg_dx3[I33] * tt[I33]);

      // Add source terms to conserved quantities
      cons(m,IM1,k,j,i) += dt * s_1;
      cons(m,IM2,k,j,i) += dt * s_2;
      cons(m,IM3,k,j,i) += dt * s_3;
    }
  );

  return;
}
  
//----------------------------------------------------------------------------------------
//! \fn
// Coordinate (geometric) source term function for GR MHD
//
// TODO: Most of this function just copies the Hydro version.  Only difference is the
// inclusion of the magnetic field in computing the stress-energy tensor.  There must be
// a smarter way to generalize these two functions and avoid duplicated code.
// Functions distinguished only by argument list.

void Coordinates::AddCoordTerms(const DvceArray5D<Real> &prim,
                                const DvceArray5D<Real> &bcc, const EOS_Data &eos,
                                const Real dt, DvceArray5D<Real> &cons)
{
  // capture variables for kernel
  auto &indcs = coord_data.mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  auto &coord = coord_data;
  int nmb1 = pmy_mesh->pmb_pack->nmb_thispack - 1;

  Real gamma_prime = eos.gamma / (eos.gamma - 1.0);

  par_for("coord_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
    {
      // Extract components of metric
      Real &x1min = coord.mb_size.d_view(m).x1min;
      Real &x1max = coord.mb_size.d_view(m).x1max;
      int nx1 = coord.mb_indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = coord.mb_size.d_view(m).x2min;
      Real &x2max = coord.mb_size.d_view(m).x2max;
      int nx2 = coord.mb_indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = coord.mb_size.d_view(m).x3min;
      Real &x3max = coord.mb_size.d_view(m).x3max;
      int nx3 = coord.mb_indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real g_[NMETRIC], gi_[NMETRIC];
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, false,
                              coord.bh_spin, g_, gi_);

      // create references to components of metric; formatting reflects structure
      const Real
        &g_00 = g_[I00], &g_01 = g_[I01], &g_02 = g_[I02], &g_03 = g_[I03],
                         &g_11 = g_[I11], &g_12 = g_[I12], &g_13 = g_[I13],
                                          &g_22 = g_[I22], &g_23 = g_[I23],
                                                           &g_33 = g_[I33];
      const Real
        &g00 = gi_[I00], &g01 = gi_[I01], &g02 = gi_[I02], &g03 = gi_[I03],
                         &g11 = gi_[I11], &g12 = gi_[I12], &g13 = gi_[I13],
                                          &g22 = gi_[I22], &g23 = gi_[I23],
                                                           &g33 = gi_[I33];
      Real alpha = sqrt(-1.0/g00);

      // Extract primitives
      const Real &rho  = prim(m,IDN,k,j,i);
      const Real &pgas = prim(m,IEN,k,j,i);
      const Real &uu1  = prim(m,IVX,k,j,i);
      const Real &uu2  = prim(m,IVY,k,j,i);
      const Real &uu3  = prim(m,IVZ,k,j,i);

      // Calculate 4-velocity
      Real uu_sq = g_11*uu1*uu1 + 2.0*g_12*uu1*uu2 + 2.0*g_13*uu1*uu3
                 + g_22*uu2*uu2 + 2.0*g_23*uu2*uu3
                 + g_33*uu3*uu3;
      Real gamma = sqrt(1.0 + uu_sq);
      Real u0 = gamma / alpha;
      Real u1 = uu1 - alpha * gamma * g01;
      Real u2 = uu2 - alpha * gamma * g02;
      Real u3 = uu3 - alpha * gamma * g03;

      // Extract and calculate magnetic field
      Real u_1 = g_01*u0 + g_11*u1 + g_12*u2 + g_13*u3;
      Real u_2 = g_02*u0 + g_12*u1 + g_22*u2 + g_23*u3;
      Real u_3 = g_03*u0 + g_13*u1 + g_23*u2 + g_33*u3;
      const Real &bb1 = bcc(m,IBX,k,j,i);
      const Real &bb2 = bcc(m,IBY,k,j,i);
      const Real &bb3 = bcc(m,IBZ,k,j,i);
      Real b0 = u_1*bb1 + u_2*bb2 + u_3*bb3;
      Real b1 = (bb1 + b0 * u1) / u0;
      Real b2 = (bb2 + b0 * u2) / u0;
      Real b3 = (bb3 + b0 * u3) / u0;
      Real b_0 = g_00*b0 + g_01*b1 + g_02*b2 + g_03*b3;
      Real b_1 = g_01*b0 + g_11*b1 + g_12*b2 + g_13*b3;
      Real b_2 = g_02*b0 + g_12*b1 + g_22*b2 + g_23*b3;
      Real b_3 = g_03*b0 + g_13*b1 + g_23*b2 + g_33*b3;
      Real b_sq = b_0*b0 + b_1*b1 + b_2*b2 + b_3*b3;

      // Calculate stress-energy tensor
      Real wtot = rho + gamma_prime * pgas + b_sq;
      Real ptot = pgas + 0.5*b_sq;
      Real tt[NMETRIC];
      tt[I00] = wtot * u0 * u0 + ptot * g00 - b0 * b0;
      tt[I01] = wtot * u0 * u1 + ptot * g01 - b0 * b1;
      tt[I02] = wtot * u0 * u2 + ptot * g02 - b0 * b2;
      tt[I03] = wtot * u0 * u3 + ptot * g03 - b0 * b3;
      tt[I11] = wtot * u1 * u1 + ptot * g11 - b1 * b1;
      tt[I12] = wtot * u1 * u2 + ptot * g12 - b1 * b2;
      tt[I13] = wtot * u1 * u3 + ptot * g13 - b1 * b3;
      tt[I22] = wtot * u2 * u2 + ptot * g22 - b2 * b2;
      tt[I23] = wtot * u2 * u3 + ptot * g23 - b2 * b3;
      tt[I33] = wtot * u3 * u3 + ptot * g33 - b3 * b3;

      // Calculate source terms
      Real dg_dx1[NMETRIC], dg_dx2[NMETRIC], dg_dx3[NMETRIC];
      ComputeMetricDerivatives(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                               dg_dx1, dg_dx2, dg_dx3);

      Real s_1 = 0.0, s_2 = 0.0, s_3 = 0.0;
      s_1 += dg_dx1[I00] * tt[I00];
      s_1 += dg_dx1[I01] * tt[I01];
      s_1 += dg_dx1[I02] * tt[I02];
      s_1 += dg_dx1[I03] * tt[I03];
      s_1 += dg_dx1[I11] * tt[I11];
      s_1 += dg_dx1[I12] * tt[I12];
      s_1 += dg_dx1[I13] * tt[I13];
      s_1 += dg_dx1[I22] * tt[I22];
      s_1 += dg_dx1[I23] * tt[I23];
      s_1 += dg_dx1[I33] * tt[I33];

      s_2 += dg_dx2[I00] * tt[I00];
      s_2 += dg_dx2[I01] * tt[I01];
      s_2 += dg_dx2[I02] * tt[I02];
      s_2 += dg_dx2[I03] * tt[I03];
      s_2 += dg_dx2[I11] * tt[I11];
      s_2 += dg_dx2[I12] * tt[I12];
      s_2 += dg_dx2[I13] * tt[I13];
      s_2 += dg_dx2[I22] * tt[I22];
      s_2 += dg_dx2[I23] * tt[I23];
      s_2 += dg_dx2[I33] * tt[I33];

      s_3 += dg_dx3[I00] * tt[I00];
      s_3 += dg_dx3[I01] * tt[I01];
      s_3 += dg_dx3[I02] * tt[I02];
      s_3 += dg_dx3[I03] * tt[I03];
      s_3 += dg_dx3[I11] * tt[I11];
      s_3 += dg_dx3[I12] * tt[I12];
      s_3 += dg_dx3[I13] * tt[I13];
      s_3 += dg_dx3[I22] * tt[I22];
      s_3 += dg_dx3[I23] * tt[I23];
      s_3 += dg_dx3[I33] * tt[I33];

      s_1 -= 0.5 * (dg_dx1[I00] * tt[I00]
                  + dg_dx1[I11] * tt[I11]
                  + dg_dx1[I22] * tt[I22]
                  + dg_dx1[I33] * tt[I33]);
      s_2 -= 0.5 * (dg_dx2[I00] * tt[I00]
                  + dg_dx2[I11] * tt[I11]
                  + dg_dx2[I22] * tt[I22]
                  + dg_dx2[I33] * tt[I33]);
      s_3 -= 0.5 * (dg_dx3[I00] * tt[I00]
                  + dg_dx3[I11] * tt[I11]
                  + dg_dx3[I22] * tt[I22]
                  + dg_dx3[I33] * tt[I33]);

      // Add source terms to conserved quantities
      cons(m,IM1,k,j,i) += dt * s_1;
      cons(m,IM2,k,j,i) += dt * s_2;
      cons(m,IM3,k,j,i) += dt * s_3;
    }
  );

  return;
}
