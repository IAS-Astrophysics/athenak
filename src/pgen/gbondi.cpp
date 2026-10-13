//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gbondi.cpp
//! \brief Problem generator to initialize MHD disk from GIZMO simulation data.
//!
//! Reads gridded HDF5 file containing density, internal energy, velocities, and
//! vector potential (Ax, Ay, Az) interpolated onto a uniform Cartesian grid.
//! Uses trilinear interpolation to map data to the simulation mesh.
//! Computes magnetic field from curl of the vector potential to ensure div(B)=0.
//!
//! The HDF5 file is expected to have the following structure (from notebook output):
//!   /GridData/Coordinates/x, y, z  - 1D coordinate arrays
//!   /GridData/Fields/Density       - 3D density field
//!   /GridData/Fields/InternalEnergy or Temperature - 3D internal energy/temperature
//!   /GridData/Fields/Velocity_x, Velocity_y, Velocity_z - 3D velocity components
//!   /GridData/Fields/VectorPotential_x, VectorPotential_y, VectorPotential_z - 3D A-field
//!
//! Input parameters allow rescaling of the GIZMO disk to different physical scales.
//! Compile with '-D PROBLEM=gbondi' and ensure HDF5OUTPUT is enabled.

#include <stdio.h>
#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp" // Added for dyngr support

// HDF5 support
#include <hdf5.h>

namespace {

//----------------------------------------------------------------------------------------
// Structure to hold GIZMO disk data and rescaling parameters
struct GizmoDiskData {
  // Grid dimensions
  int nx, ny, nz;
  
  // Grid coordinates (1D arrays for uniform grid)
  std::vector<Real> x_coords;
  std::vector<Real> y_coords;
  std::vector<Real> z_coords;
  
  // Grid spacing
  Real dx, dy, dz;
  
  // Grid extent
  Real x_min, x_max;
  Real y_min, y_max;
  Real z_min, z_max;
  
  // Field data (3D arrays stored as 1D, indexed as [i + nx*(j + ny*k)])
  std::vector<Real> density;
  std::vector<Real> internal_energy;
  std::vector<Real> vx, vy, vz;
  std::vector<Real> Ax, Ay, Az;
  
  // Rescaling parameters
  Real length_scale;
  Real density_scale;
  Real velocity_scale;
  Real pressure_scale;
  Real bfield_scale;
  
  // Binary separation (physical, e.g., in pc)
  // GIZMO data is in units where a_bin = 1
  // To get physical coordinates: x_phys = x_gizmo * binary_separation
  Real binary_separation;
  
  // Black hole positions (from HDF5, in a_bin=1 units)
  Real bh1_pos[3];
  Real bh2_pos[3];
  Real bh1_mass, bh2_mass;
  bool binary_data_available;
  
  // Offset to center the disk
  Real x_offset, y_offset, z_offset;
  
  // Background values
  Real rho_floor;
  Real pgas_floor;
  
  // EOS parameters
  Real gamma_adi;
  
  // Flag for whether data has been loaded
  bool data_loaded;
  
  // Flag for whether temperature was read instead of internal energy
  // If true, need to convert: eint = temp * rho / (gamma - 1)
  bool temperature_as_eint;
};

// Global instance
GizmoDiskData gizmo_disk;

//----------------------------------------------------------------------------------------
// Function prototypes
void ReadGizmoHDF5(const std::string &filename, GizmoDiskData &data);

} // anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions from GIZMO simulation data

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  
  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int ng = indcs.ng;
  
  // Check if we are running with DynGR
  bool use_dyngr = (pmbp->pdyngr != nullptr);

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  bool is_mhd = false;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
    gizmo_disk.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
    gizmo_disk.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
    is_mhd = true;
  }
  Real gm1 = gizmo_disk.gamma_adi - 1.0;

  // Read problem parameters
  std::string gizmo_file = pin->GetString("problem", "gizmo_file");
  
  // Binary separation: physical separation of the binary (e.g., in pc or code units)
  // GIZMO remapped data is in units where a_bin = 1
  // The length_scale will be set to binary_separation if specified
  gizmo_disk.binary_separation = pin->GetOrAddReal("problem", "binary_separation", 1.0);
  
  // If binary_separation is specified, use it as the length scale
  // This converts from a_bin=1 units to physical/code units
  gizmo_disk.length_scale = pin->GetOrAddReal("problem", "length_scale", 
                                               gizmo_disk.binary_separation);
  gizmo_disk.density_scale = pin->GetOrAddReal("problem", "density_scale", 1.0);
  gizmo_disk.velocity_scale = pin->GetOrAddReal("problem", "velocity_scale", 1.0);
  gizmo_disk.pressure_scale = pin->GetOrAddReal("problem", "pressure_scale", 1.0);
  gizmo_disk.bfield_scale = pin->GetOrAddReal("problem", "bfield_scale", 1.0);
  
  gizmo_disk.x_offset = pin->GetOrAddReal("problem", "x_offset", 0.0);
  gizmo_disk.y_offset = pin->GetOrAddReal("problem", "y_offset", 0.0);
  gizmo_disk.z_offset = pin->GetOrAddReal("problem", "z_offset", 0.0);
  
  gizmo_disk.rho_floor = pin->GetOrAddReal("problem", "rho_floor", 1.0e-10);
  gizmo_disk.pgas_floor = pin->GetOrAddReal("problem", "pgas_floor", 1.0e-12);

  int myrank = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif

  if (myrank == 0) {
    std::cout << "Reading GIZMO data file: " << gizmo_file << std::endl;
  }
  
  ReadGizmoHDF5(gizmo_file, gizmo_disk);

  // Print binary information if available
  if (myrank == 0 && gizmo_disk.binary_data_available) {
    std::cout << "Binary data from GIZMO file:" << std::endl;
    std::cout << "  BH1 position (a_bin=1): (" << gizmo_disk.bh1_pos[0] << ", " 
              << gizmo_disk.bh1_pos[1] << ", " << gizmo_disk.bh1_pos[2] << ")" << std::endl;
    std::cout << "  BH2 position (a_bin=1): (" << gizmo_disk.bh2_pos[0] << ", " 
              << gizmo_disk.bh2_pos[1] << ", " << gizmo_disk.bh2_pos[2] << ")" << std::endl;
    std::cout << "  Binary separation (input): " << gizmo_disk.binary_separation << std::endl;
    std::cout << "  BH1 position (scaled): (" << gizmo_disk.bh1_pos[0] * gizmo_disk.length_scale << ", " 
              << gizmo_disk.bh1_pos[1] * gizmo_disk.length_scale << ", " 
              << gizmo_disk.bh1_pos[2] * gizmo_disk.length_scale << ")" << std::endl;
    std::cout << "  BH2 position (scaled): (" << gizmo_disk.bh2_pos[0] * gizmo_disk.length_scale << ", " 
              << gizmo_disk.bh2_pos[1] * gizmo_disk.length_scale << ", " 
              << gizmo_disk.bh2_pos[2] * gizmo_disk.length_scale << ")" << std::endl;
  }

  // Apply length scaling to coordinates
  for (int i = 0; i < gizmo_disk.nx; ++i) {
    gizmo_disk.x_coords[i] = gizmo_disk.x_coords[i] * gizmo_disk.length_scale 
                            + gizmo_disk.x_offset;
  }
  for (int j = 0; j < gizmo_disk.ny; ++j) {
    gizmo_disk.y_coords[j] = gizmo_disk.y_coords[j] * gizmo_disk.length_scale 
                            + gizmo_disk.y_offset;
  }
  for (int k = 0; k < gizmo_disk.nz; ++k) {
    gizmo_disk.z_coords[k] = gizmo_disk.z_coords[k] * gizmo_disk.length_scale 
                            + gizmo_disk.z_offset;
  }
  
  gizmo_disk.x_min = gizmo_disk.x_coords[0];
  gizmo_disk.x_max = gizmo_disk.x_coords[gizmo_disk.nx - 1];
  gizmo_disk.y_min = gizmo_disk.y_coords[0];
  gizmo_disk.y_max = gizmo_disk.y_coords[gizmo_disk.ny - 1];
  gizmo_disk.z_min = gizmo_disk.z_coords[0];
  gizmo_disk.z_max = gizmo_disk.z_coords[gizmo_disk.nz - 1];
  gizmo_disk.dx = (gizmo_disk.x_max - gizmo_disk.x_min) / (gizmo_disk.nx - 1);
  gizmo_disk.dy = (gizmo_disk.y_max - gizmo_disk.y_min) / (gizmo_disk.ny - 1);
  gizmo_disk.dz = (gizmo_disk.z_max - gizmo_disk.z_min) / (gizmo_disk.nz - 1);

  // Apply scaling to field data
  int ntot = gizmo_disk.nx * gizmo_disk.ny * gizmo_disk.nz;
  Real gm1_inv = 1.0 / gm1;  // 1 / (gamma - 1)
  
  for (int idx = 0; idx < ntot; ++idx) {
    gizmo_disk.density[idx] *= gizmo_disk.density_scale;
    gizmo_disk.vx[idx] *= gizmo_disk.velocity_scale;
    gizmo_disk.vy[idx] *= gizmo_disk.velocity_scale;
    gizmo_disk.vz[idx] *= gizmo_disk.velocity_scale;
    gizmo_disk.Ax[idx] *= gizmo_disk.bfield_scale * gizmo_disk.length_scale;
    gizmo_disk.Ay[idx] *= gizmo_disk.bfield_scale * gizmo_disk.length_scale;
    gizmo_disk.Az[idx] *= gizmo_disk.bfield_scale * gizmo_disk.length_scale;
    
    // Convert temperature to internal energy if needed
    // For ideal gas: eint = rho * temp / (gamma - 1) = P / (gamma - 1)
    // where P = rho * T for normalized units (k_B/mu = 1)
    if (gizmo_disk.temperature_as_eint) {
      Real temp = gizmo_disk.internal_energy[idx];
      Real rho = gizmo_disk.density[idx];
      gizmo_disk.internal_energy[idx] = rho * temp * gm1_inv * gizmo_disk.pressure_scale;
    } else {
      gizmo_disk.internal_energy[idx] *= gizmo_disk.density[idx] * gizmo_disk.pressure_scale;
    }
  }

  if (myrank == 0) {
    std::cout << "GIZMO data: " << gizmo_disk.nx << " x " << gizmo_disk.ny 
              << " x " << gizmo_disk.nz << std::endl;
    std::cout << "Domain: [" << gizmo_disk.x_min << ", " << gizmo_disk.x_max 
              << "] x [" << gizmo_disk.y_min << ", " << gizmo_disk.y_max 
              << "] x [" << gizmo_disk.z_min << ", " << gizmo_disk.z_max << "]" << std::endl;
  }

  // Copy data to device
  int nx_g = gizmo_disk.nx;
  int ny_g = gizmo_disk.ny;
  int nz_g = gizmo_disk.nz;
  
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_x("gizmo_x", nx_g);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_y("gizmo_y", ny_g);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_z("gizmo_z", nz_g);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_rho("gizmo_rho", ntot);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_eint("gizmo_eint", ntot);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_vx("gizmo_vx", ntot);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_vy("gizmo_vy", ntot);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_vz("gizmo_vz", ntot);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_Ax("gizmo_Ax", ntot);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_Ay("gizmo_Ay", ntot);
  Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace> d_Az("gizmo_Az", ntot);
  
  auto h_x = Kokkos::create_mirror_view(d_x);
  auto h_y = Kokkos::create_mirror_view(d_y);
  auto h_z = Kokkos::create_mirror_view(d_z);
  auto h_rho = Kokkos::create_mirror_view(d_rho);
  auto h_eint = Kokkos::create_mirror_view(d_eint);
  auto h_vx = Kokkos::create_mirror_view(d_vx);
  auto h_vy = Kokkos::create_mirror_view(d_vy);
  auto h_vz = Kokkos::create_mirror_view(d_vz);
  auto h_Ax = Kokkos::create_mirror_view(d_Ax);
  auto h_Ay = Kokkos::create_mirror_view(d_Ay);
  auto h_Az = Kokkos::create_mirror_view(d_Az);
  
  for (int i = 0; i < nx_g; ++i) h_x(i) = gizmo_disk.x_coords[i];
  for (int j = 0; j < ny_g; ++j) h_y(j) = gizmo_disk.y_coords[j];
  for (int k = 0; k < nz_g; ++k) h_z(k) = gizmo_disk.z_coords[k];
  for (int idx = 0; idx < ntot; ++idx) {
    h_rho(idx) = gizmo_disk.density[idx];
    h_eint(idx) = gizmo_disk.internal_energy[idx];
    h_vx(idx) = gizmo_disk.vx[idx];
    h_vy(idx) = gizmo_disk.vy[idx];
    h_vz(idx) = gizmo_disk.vz[idx];
    h_Ax(idx) = gizmo_disk.Ax[idx];
    h_Ay(idx) = gizmo_disk.Ay[idx];
    h_Az(idx) = gizmo_disk.Az[idx];
  }
  
  Kokkos::deep_copy(d_x, h_x);
  Kokkos::deep_copy(d_y, h_y);
  Kokkos::deep_copy(d_z, h_z);
  Kokkos::deep_copy(d_rho, h_rho);
  Kokkos::deep_copy(d_eint, h_eint);
  Kokkos::deep_copy(d_vx, h_vx);
  Kokkos::deep_copy(d_vy, h_vy);
  Kokkos::deep_copy(d_vz, h_vz);
  Kokkos::deep_copy(d_Ax, h_Ax);
  Kokkos::deep_copy(d_Ay, h_Ay);
  Kokkos::deep_copy(d_Az, h_Az);

  Real x_min = gizmo_disk.x_min, x_max = gizmo_disk.x_max;
  Real y_min = gizmo_disk.y_min, y_max = gizmo_disk.y_max;
  Real z_min = gizmo_disk.z_min, z_max = gizmo_disk.z_max;
  Real dx_g = gizmo_disk.dx, dy_g = gizmo_disk.dy, dz_g = gizmo_disk.dz;
  Real rho_floor = gizmo_disk.rho_floor;
  Real pgas_floor = gizmo_disk.pgas_floor;
  Real gamma = gizmo_disk.gamma_adi;
  
  auto &size = pmbp->pmb->mb_size;

  // initialize ADM variables -----------------------------------------

  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }

  // Initialize primitive variables
  par_for("pgen_gizmo_hydro", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    bool in_domain = (x1v >= x_min && x1v <= x_max &&
                      x2v >= y_min && x2v <= y_max &&
                      x3v >= z_min && x3v <= z_max);

    Real rho, eint, ux, uy, uz;
    
    if (in_domain) {
      int ix = static_cast<int>((x1v - x_min) / dx_g);
      int iy = static_cast<int>((x2v - y_min) / dy_g);
      int iz = static_cast<int>((x3v - z_min) / dz_g);
      
      ix = (ix < 0) ? 0 : ((ix >= nx_g - 1) ? nx_g - 2 : ix);
      iy = (iy < 0) ? 0 : ((iy >= ny_g - 1) ? ny_g - 2 : iy);
      iz = (iz < 0) ? 0 : ((iz >= nz_g - 1) ? nz_g - 2 : iz);
      
      Real x0 = d_x(ix), x1_c = d_x(ix + 1);
      Real y0 = d_y(iy), y1_c = d_y(iy + 1);
      Real z0 = d_z(iz), z1_c = d_z(iz + 1);
      
      Real xd = fmax(0.0, fmin(1.0, (x1v - x0) / (x1_c - x0)));
      Real yd = fmax(0.0, fmin(1.0, (x2v - y0) / (y1_c - y0)));
      Real zd = fmax(0.0, fmin(1.0, (x3v - z0) / (z1_c - z0)));
      
      // C-style row-major indexing for HDF5/numpy arrays: idx = i*ny*nz + j*nz + k
      // Array stored as arr[ix, iy, iz] with shape (nx, ny, nz)
      int idx000 = ix * ny_g * nz_g + iy * nz_g + iz;
      int idx100 = (ix + 1) * ny_g * nz_g + iy * nz_g + iz;
      int idx010 = ix * ny_g * nz_g + (iy + 1) * nz_g + iz;
      int idx110 = (ix + 1) * ny_g * nz_g + (iy + 1) * nz_g + iz;
      int idx001 = ix * ny_g * nz_g + iy * nz_g + (iz + 1);
      int idx101 = (ix + 1) * ny_g * nz_g + iy * nz_g + (iz + 1);
      int idx011 = ix * ny_g * nz_g + (iy + 1) * nz_g + (iz + 1);
      int idx111 = (ix + 1) * ny_g * nz_g + (iy + 1) * nz_g + (iz + 1);
      
      // Trilinear interpolation
      Real c00, c01, c10, c11, c0, c1;
      
      c00 = d_rho(idx000)*(1.0-xd) + d_rho(idx100)*xd;
      c01 = d_rho(idx001)*(1.0-xd) + d_rho(idx101)*xd;
      c10 = d_rho(idx010)*(1.0-xd) + d_rho(idx110)*xd;
      c11 = d_rho(idx011)*(1.0-xd) + d_rho(idx111)*xd;
      c0 = c00*(1.0-yd) + c10*yd;
      c1 = c01*(1.0-yd) + c11*yd;
      rho = c0*(1.0-zd) + c1*zd;
      
      c00 = d_eint(idx000)*(1.0-xd) + d_eint(idx100)*xd;
      c01 = d_eint(idx001)*(1.0-xd) + d_eint(idx101)*xd;
      c10 = d_eint(idx010)*(1.0-xd) + d_eint(idx110)*xd;
      c11 = d_eint(idx011)*(1.0-xd) + d_eint(idx111)*xd;
      c0 = c00*(1.0-yd) + c10*yd;
      c1 = c01*(1.0-yd) + c11*yd;
      eint = c0*(1.0-zd) + c1*zd;
      
      c00 = d_vx(idx000)*(1.0-xd) + d_vx(idx100)*xd;
      c01 = d_vx(idx001)*(1.0-xd) + d_vx(idx101)*xd;
      c10 = d_vx(idx010)*(1.0-xd) + d_vx(idx110)*xd;
      c11 = d_vx(idx011)*(1.0-xd) + d_vx(idx111)*xd;
      c0 = c00*(1.0-yd) + c10*yd;
      c1 = c01*(1.0-yd) + c11*yd;
      ux = c0*(1.0-zd) + c1*zd;
      
      c00 = d_vy(idx000)*(1.0-xd) + d_vy(idx100)*xd;
      c01 = d_vy(idx001)*(1.0-xd) + d_vy(idx101)*xd;
      c10 = d_vy(idx010)*(1.0-xd) + d_vy(idx110)*xd;
      c11 = d_vy(idx011)*(1.0-xd) + d_vy(idx111)*xd;
      c0 = c00*(1.0-yd) + c10*yd;
      c1 = c01*(1.0-yd) + c11*yd;
      uy = c0*(1.0-zd) + c1*zd;
      
      c00 = d_vz(idx000)*(1.0-xd) + d_vz(idx100)*xd;
      c01 = d_vz(idx001)*(1.0-xd) + d_vz(idx101)*xd;
      c10 = d_vz(idx010)*(1.0-xd) + d_vz(idx110)*xd;
      c11 = d_vz(idx011)*(1.0-xd) + d_vz(idx111)*xd;
      c0 = c00*(1.0-yd) + c10*yd;
      c1 = c01*(1.0-yd) + c11*yd;
      uz = c0*(1.0-zd) + c1*zd;
      
      rho = fmax(rho, rho_floor);
      eint = fmax(eint, pgas_floor / (gamma - 1.0));
    } else {
      rho = rho_floor;
      eint = pgas_floor / (gamma - 1.0);
      ux = 0.0;
      uy = 0.0;
      uz = 0.0;
    }
    
    w0_(m, IDN, k, j, i) = rho;
    
    // DynGR expects Pressure in IPR, Standard Hydro expects Internal Energy in IEN
    if (use_dyngr) {
      w0_(m, IPR, k, j, i) = eint * (gamma - 1.0);
    } else {
      w0_(m, IEN, k, j, i) = eint;
    }

    w0_(m, IVX, k, j, i) = ux;
    w0_(m, IVY, k, j, i) = uy;
    w0_(m, IVZ, k, j, i) = uz;
  });

  // Initialize magnetic fields (MHD only)
  if (is_mhd) {
    int ncells1 = indcs.nx1 + 2*ng;
    int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
    int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
    
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);
    
    par_for("pgen_gizmo_vpot", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      Real x1f = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);
      
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      Real x2f = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);
      
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      Real x3f = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);
      
      auto interp_A = [&](Real px, Real py, Real pz, 
                          const Kokkos::View<Real*, Kokkos::LayoutRight, DevMemSpace>& A_f) -> Real {
        if (px < x_min || px > x_max || py < y_min || py > y_max || 
            pz < z_min || pz > z_max) {
          return 0.0;
        }
        
        int ix = static_cast<int>((px - x_min) / dx_g);
        int iy = static_cast<int>((py - y_min) / dy_g);
        int iz = static_cast<int>((pz - z_min) / dz_g);
        
        ix = (ix < 0) ? 0 : ((ix >= nx_g - 1) ? nx_g - 2 : ix);
        iy = (iy < 0) ? 0 : ((iy >= ny_g - 1) ? ny_g - 2 : iy);
        iz = (iz < 0) ? 0 : ((iz >= nz_g - 1) ? nz_g - 2 : iz);
        
        Real x0_l = d_x(ix), x1_l = d_x(ix + 1);
        Real y0_l = d_y(iy), y1_l = d_y(iy + 1);
        Real z0_l = d_z(iz), z1_l = d_z(iz + 1);
        
        Real xd = fmax(0.0, fmin(1.0, (px - x0_l) / (x1_l - x0_l)));
        Real yd = fmax(0.0, fmin(1.0, (py - y0_l) / (y1_l - y0_l)));
        Real zd = fmax(0.0, fmin(1.0, (pz - z0_l) / (z1_l - z0_l)));
        
        // C-style row-major indexing for HDF5/numpy arrays: idx = i*ny*nz + j*nz + k
        int idx000 = ix * ny_g * nz_g + iy * nz_g + iz;
        int idx100 = (ix + 1) * ny_g * nz_g + iy * nz_g + iz;
        int idx010 = ix * ny_g * nz_g + (iy + 1) * nz_g + iz;
        int idx110 = (ix + 1) * ny_g * nz_g + (iy + 1) * nz_g + iz;
        int idx001 = ix * ny_g * nz_g + iy * nz_g + (iz + 1);
        int idx101 = (ix + 1) * ny_g * nz_g + iy * nz_g + (iz + 1);
        int idx011 = ix * ny_g * nz_g + (iy + 1) * nz_g + (iz + 1);
        int idx111 = (ix + 1) * ny_g * nz_g + (iy + 1) * nz_g + (iz + 1);
        
        Real c00 = A_f(idx000)*(1.0-xd) + A_f(idx100)*xd;
        Real c01 = A_f(idx001)*(1.0-xd) + A_f(idx101)*xd;
        Real c10 = A_f(idx010)*(1.0-xd) + A_f(idx110)*xd;
        Real c11 = A_f(idx011)*(1.0-xd) + A_f(idx111)*xd;
        Real c0 = c00*(1.0-yd) + c10*yd;
        Real c1 = c01*(1.0-yd) + c11*yd;
        return c0*(1.0-zd) + c1*zd;
      };
      
      a1(m, k, j, i) = interp_A(x1v, x2f, x3f, d_Ax);
      a2(m, k, j, i) = interp_A(x1f, x2v, x3f, d_Ay);
      a3(m, k, j, i) = interp_A(x1f, x2f, x3v, d_Az);
    });
    
    auto &b0 = pmbp->pmhd->b0;
    
    par_for("pgen_gizmo_bfield", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      
      b0.x1f(m, k, j, i) = (a3(m, k, j+1, i) - a3(m, k, j, i)) / dx2 -
                          (a2(m, k+1, j, i) - a2(m, k, j, i)) / dx3;
      b0.x2f(m, k, j, i) = (a1(m, k+1, j, i) - a1(m, k, j, i)) / dx3 -
                          (a3(m, k, j, i+1) - a3(m, k, j, i)) / dx1;
      b0.x3f(m, k, j, i) = (a2(m, k, j, i+1) - a2(m, k, j, i)) / dx1 -
                          (a1(m, k, j+1, i) - a1(m, k, j, i)) / dx2;
      
      if (i == ie) {
        b0.x1f(m, k, j, i+1) = (a3(m, k, j+1, i+1) - a3(m, k, j, i+1)) / dx2 -
                              (a2(m, k+1, j, i+1) - a2(m, k, j, i+1)) / dx3;
      }
      if (j == je) {
        b0.x2f(m, k, j+1, i) = (a1(m, k+1, j+1, i) - a1(m, k, j+1, i)) / dx3 -
                              (a3(m, k, j+1, i+1) - a3(m, k, j+1, i)) / dx1;
      }
      if (k == ke) {
        b0.x3f(m, k+1, j, i) = (a2(m, k+1, j, i+1) - a2(m, k+1, j, i)) / dx1 -
                              (a1(m, k+1, j+1, i) - a1(m, k+1, j, i)) / dx2;
      }
    });
    
    auto &bcc_ = pmbp->pmhd->bcc0;
    
    par_for("pgen_gizmo_bcc", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bcc_(m, IBX, k, j, i) = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i+1));
      bcc_(m, IBY, k, j, i) = 0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j+1, i));
      bcc_(m, IBZ, k, j, i) = 0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k+1, j, i));
    });
  }

  // Convert primitives to conserved
  if (use_dyngr) {
    // DynGR specific initialization
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  } else {
    // Standard Hydro/MHD initialization
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
    } else if (pmbp->pmhd != nullptr) {
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
    }
  }

  if (myrank == 0) {
    std::cout << "GIZMO disk initialization complete." << std::endl;
  }

  return;
}

namespace {

void ReadGizmoHDF5(const std::string &filename, GizmoDiskData &data) {
  int myrank = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif

  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    std::cout << "### FATAL ERROR: Cannot open " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  hid_t coord_group = H5Gopen(file_id, "/GridData/Coordinates", H5P_DEFAULT);
  
  hid_t x_dset = H5Dopen(coord_group, "x", H5P_DEFAULT);
  hid_t x_space = H5Dget_space(x_dset);
  hsize_t x_dims[1];
  H5Sget_simple_extent_dims(x_space, x_dims, NULL);
  data.nx = static_cast<int>(x_dims[0]);
  
  hid_t y_dset = H5Dopen(coord_group, "y", H5P_DEFAULT);
  hid_t y_space = H5Dget_space(y_dset);
  hsize_t y_dims[1];
  H5Sget_simple_extent_dims(y_space, y_dims, NULL);
  data.ny = static_cast<int>(y_dims[0]);
  
  hid_t z_dset = H5Dopen(coord_group, "z", H5P_DEFAULT);
  hid_t z_space = H5Dget_space(z_dset);
  hsize_t z_dims[1];
  H5Sget_simple_extent_dims(z_space, z_dims, NULL);
  data.nz = static_cast<int>(z_dims[0]);
  
  data.x_coords.resize(data.nx);
  data.y_coords.resize(data.ny);
  data.z_coords.resize(data.nz);
  
  std::vector<double> x_temp(data.nx), y_temp(data.ny), z_temp(data.nz);
  H5Dread(x_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x_temp.data());
  H5Dread(y_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, y_temp.data());
  H5Dread(z_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, z_temp.data());
  
  for (int i = 0; i < data.nx; ++i) data.x_coords[i] = static_cast<Real>(x_temp[i]);
  for (int j = 0; j < data.ny; ++j) data.y_coords[j] = static_cast<Real>(y_temp[j]);
  for (int k = 0; k < data.nz; ++k) data.z_coords[k] = static_cast<Real>(z_temp[k]);
  
  H5Dclose(x_dset); H5Dclose(y_dset); H5Dclose(z_dset);
  H5Sclose(x_space); H5Sclose(y_space); H5Sclose(z_space);
  H5Gclose(coord_group);
  
  data.x_min = data.x_coords[0];
  data.x_max = data.x_coords[data.nx - 1];
  data.y_min = data.y_coords[0];
  data.y_max = data.y_coords[data.ny - 1];
  data.z_min = data.z_coords[0];
  data.z_max = data.z_coords[data.nz - 1];
  data.dx = (data.x_max - data.x_min) / (data.nx - 1);
  data.dy = (data.y_max - data.y_min) / (data.ny - 1);
  data.dz = (data.z_max - data.z_min) / (data.nz - 1);
  
  int ntot = data.nx * data.ny * data.nz;
  data.density.resize(ntot);
  data.internal_energy.resize(ntot);
  data.vx.resize(ntot);
  data.vy.resize(ntot);
  data.vz.resize(ntot);
  data.Ax.resize(ntot);
  data.Ay.resize(ntot);
  data.Az.resize(ntot);
  
  hid_t fields_group = H5Gopen(file_id, "/GridData/Fields", H5P_DEFAULT);
  
  auto read_field = [&](const char* name, std::vector<Real>& field, bool required) {
    hid_t dset = H5Dopen(fields_group, name, H5P_DEFAULT);
    if (dset < 0) {
      if (required) {
        std::cout << "### FATAL ERROR: Cannot find " << name << std::endl;
        exit(EXIT_FAILURE);
      }
      std::fill(field.begin(), field.end(), 0.0);
      return;
    }
    
    hid_t dtype = H5Dget_type(dset);
    size_t type_size = H5Tget_size(dtype);
    
    if (type_size == 4) {
      std::vector<float> temp(ntot);
      H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp.data());
      for (int idx = 0; idx < ntot; ++idx) field[idx] = static_cast<Real>(temp[idx]);
    } else {
      std::vector<double> temp(ntot);
      H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, temp.data());
      for (int idx = 0; idx < ntot; ++idx) field[idx] = static_cast<Real>(temp[idx]);
    }
    
    H5Tclose(dtype);
    H5Dclose(dset);
  };
  
  read_field("Density", data.density, true);
  
  hid_t eint_test = H5Dopen(fields_group, "InternalEnergy", H5P_DEFAULT);
  if (eint_test >= 0) {
    H5Dclose(eint_test);
    read_field("InternalEnergy", data.internal_energy, true);
    data.temperature_as_eint = false;
  } else {
    // Read temperature - will be converted to internal energy later
    // using eint = temp * rho / (gamma - 1)
    read_field("Temperature", data.internal_energy, true);
    data.temperature_as_eint = true;
    if (myrank == 0) {
      std::cout << "  Note: Temperature data found, will convert to internal energy" << std::endl;
    }
  }
  
  read_field("Velocity_x", data.vx, true);
  read_field("Velocity_y", data.vy, true);
  read_field("Velocity_z", data.vz, true);
  
  read_field("VectorPotential_x", data.Ax, false);
  read_field("VectorPotential_y", data.Ay, false);
  read_field("VectorPotential_z", data.Az, false);
  
  H5Gclose(fields_group);
  
  // Read binary black hole data if available
  data.binary_data_available = false;
  htri_t binary_exists = H5Lexists(file_id, "/Binary", H5P_DEFAULT);
  if (binary_exists > 0) {
    hid_t binary_group = H5Gopen(file_id, "/Binary", H5P_DEFAULT);
    if (binary_group >= 0) {
      // Read BH1 position
      hid_t bh1_dset = H5Dopen(binary_group, "bh1_position", H5P_DEFAULT);
      if (bh1_dset >= 0) {
        double bh1_temp[3];
        H5Dread(bh1_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bh1_temp);
        data.bh1_pos[0] = static_cast<Real>(bh1_temp[0]);
        data.bh1_pos[1] = static_cast<Real>(bh1_temp[1]);
        data.bh1_pos[2] = static_cast<Real>(bh1_temp[2]);
        H5Dclose(bh1_dset);
        data.binary_data_available = true;
      }
      
      // Read BH2 position
      hid_t bh2_dset = H5Dopen(binary_group, "bh2_position", H5P_DEFAULT);
      if (bh2_dset >= 0) {
        double bh2_temp[3];
        H5Dread(bh2_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bh2_temp);
        data.bh2_pos[0] = static_cast<Real>(bh2_temp[0]);
        data.bh2_pos[1] = static_cast<Real>(bh2_temp[1]);
        data.bh2_pos[2] = static_cast<Real>(bh2_temp[2]);
        H5Dclose(bh2_dset);
      }
      
      // Read BH masses from attributes if available
      if (H5Aexists(binary_group, "bh1_mass_code") > 0) {
        hid_t attr = H5Aopen(binary_group, "bh1_mass_code", H5P_DEFAULT);
        double mass_temp;
        H5Aread(attr, H5T_NATIVE_DOUBLE, &mass_temp);
        data.bh1_mass = static_cast<Real>(mass_temp);
        H5Aclose(attr);
      }
      if (H5Aexists(binary_group, "bh2_mass_code") > 0) {
        hid_t attr = H5Aopen(binary_group, "bh2_mass_code", H5P_DEFAULT);
        double mass_temp;
        H5Aread(attr, H5T_NATIVE_DOUBLE, &mass_temp);
        data.bh2_mass = static_cast<Real>(mass_temp);
        H5Aclose(attr);
      }
      
      H5Gclose(binary_group);
      
      if (myrank == 0) {
        std::cout << "  Binary data loaded from HDF5" << std::endl;
      }
    }
  }
  
  H5Fclose(file_id);
  
  data.data_loaded = true;
  
  if (myrank == 0) {
    std::cout << "  Loaded " << data.nx << "x" << data.ny << "x" << data.nz << " grid" << std::endl;
  }
}

} // namespace