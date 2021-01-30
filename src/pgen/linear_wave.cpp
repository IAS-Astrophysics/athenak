//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file linear_wave.c
//  \brief Linear wave problem generator for 1D/2D/3D problems. Initializes both hydro and
//  MHD problems. Direction of the wavevector is set to be along the x? axis by using the
//  along_x? input flags, else it is automatically set along the grid diagonal in 2D/3D
//

// C/C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "utils/grid_locations.hpp"
#include "pgen.hpp"

// function to compute eigenvectors of linear waves in hydrodynamics
void HydroEigensystem(const Real d, const Real p, const Real v1, const Real v2,
                      const Real v3, const EOS_Data &eos, Real right_eigenmatrix[5][5]);
// function to compute eigenvectors of linear waves in mhd
void MHDEigensystem(const Real d, const Real p, const Real v1, const Real v2,
                    const Real v3, const EOS_Data &eos, Real right_eigenmatrix[5][5]);

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for linear wave tests

void ProblemGenerator::LinearWave_(MeshBlockPack *pmbp, ParameterInput *pin)
{
  // read global parameters
  int wave_flag = pin->GetInteger("problem", "wave_flag");
  Real amp = pin->GetReal("problem", "amp");
  Real vflow = pin->GetOrAddReal("problem", "vflow", 0.0);
  bool along_x1 = pin->GetOrAddBoolean("problem", "along_x1", false);
  bool along_x2 = pin->GetOrAddBoolean("problem", "along_x2", false);
  bool along_x3 = pin->GetOrAddBoolean("problem", "along_x3", false);
  // error check input flags
  if ((along_x1 && (along_x2 || along_x3)) || (along_x2 && along_x3)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Can only specify one of along_x1/2/3 to be true" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((along_x2 || along_x3) && !(pmesh_->nx2gt1)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x2 or x3 axis in 1D" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (along_x3 && !(pmesh_->nx2gt1)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot specify waves along x3 axis in 2D" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmesh_->mesh_size.x1max - pmesh_->mesh_size.x1min;
  Real x2size = pmesh_->mesh_size.x2max - pmesh_->mesh_size.x2min;
  Real x3size = pmesh_->mesh_size.x3max - pmesh_->mesh_size.x3min;

  // start with wavevector along x1 axis
  Real cos_a3 = 1.0;
  Real sin_a3 = 0.0;
  Real cos_a2 = 1.0;
  Real sin_a2 = 0.0;
  if (pmesh_->nx2gt1 && !(along_x1)) {
    Real ang_3 = std::atan(x1size/x2size);
    sin_a3 = std::sin(ang_3);
    cos_a3 = std::cos(ang_3);
  }
  if (pmesh_->nx3gt1 && !(along_x1)) {
    Real ang_2 = std::atan(0.5*(x1size*cos_a3 + x2size*sin_a3)/x3size);
    sin_a2 = std::sin(ang_2);
    cos_a2 = std::cos(ang_2);
  }

  // hardcode wavevector along x2 axis, override ang_2, ang_3
  if (along_x2) {
    cos_a3 = 0.0;
    sin_a3 = 1.0;
    cos_a2 = 1.0;
    sin_a2 = 0.0;
  }

  // hardcode wavevector along x3 axis, override ang_2, ang_3
  if (along_x3) {
    cos_a3 = 0.0;
    sin_a3 = 1.0;
    cos_a2 = 0.0;
    sin_a2 = 1.0;
  }

  // choose the smallest projection of the wavelength in each direction that is > 0
  Real lambda = std::numeric_limits<float>::max();
  if (cos_a2*cos_a3 > 0.0) lambda = std::min(lambda, x1size*cos_a2*cos_a3);
  if (cos_a2*sin_a3 > 0.0) lambda = std::min(lambda, x2size*cos_a2*sin_a3);
  if (sin_a2 > 0.0) lambda = std::min(lambda, x3size*sin_a2);

  // Initialize k_parallel
  Real k_par = 2.0*(M_PI)/lambda;

  // Set background state: u0 is parallel to the wavevector, and v0/w0 are perpendicular
  Real d0 = 1.0;
  Real v1_0 = vflow;
  Real v2_0 = 0.0;
  Real v3_0 = 0.0;

  // capture variables for kernel
  int &nx1 = pmbp->mb_cells.nx1;
  int &nx2 = pmbp->mb_cells.nx2;
  int &nx3 = pmbp->mb_cells.nx3;
  int &is = pmbp->mb_cells.is, &ie = pmbp->mb_cells.ie;
  int &js = pmbp->mb_cells.js, &je = pmbp->mb_cells.je;
  int &ks = pmbp->mb_cells.ks, &ke = pmbp->mb_cells.ke;
  auto &size = pmbp->pmb->mbsize;

  // initialize Hydro variables --------------------------------
  if (pmbp->phydro != nullptr) {
    using namespace hydro;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    auto &u0 = pmbp->phydro->u0; 
    Real rem[5][5];
    // Compute eigenvectors in hydrodynamics
    HydroEigensystem(d0, p0, v1_0, v2_0, v3_0, eos, rem);
    par_for("pgen_linwave", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        Real x1 = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
        Real x2 = CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m));
        Real x3 = CellCenterX(k-ks, nx3, size.x3min.d_view(m), size.x3max.d_view(m));
        Real x = cos_a2*(x1*cos_a3 + x2*sin_a3) + x3*sin_a2;
        Real sn = std::sin(k_par*x);
        Real mx = d0*vflow + amp*sn*rem[1][wave_flag];
        Real my = amp*sn*rem[2][wave_flag];
        Real mz = amp*sn*rem[3][wave_flag];
  
        u0(m,IDN,k,j,i) = d0 + amp*sn*rem[0][wave_flag];
        u0(m,IM1,k,j,i) = mx*cos_a2*cos_a3 - my*sin_a3 - mz*sin_a2*cos_a3;
        u0(m,IM2,k,j,i) = mx*cos_a2*sin_a3 + my*cos_a3 - mz*sin_a2*sin_a3;
        u0(m,IM3,k,j,i) = mx*sin_a2                    + mz*cos_a2;

        if (eos.is_adiabatic) {
          u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*v1_0*v1_0 + amp*sn*rem[4][wave_flag];
        }
      }
    );
  }  // End initialization Hydro variables

  // initialize MHD variables --------------------------------
  if (pmbp->pmhd != nullptr) {
    using namespace hydro;
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = 1.0/eos.gamma;
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    Real rem[5][5];
    // Compute eigenvectors in mhd
    MHDEigensystem(d0, p0, v1_0, v2_0, v3_0, eos, rem);
    par_for("pgen_linwave", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i)
      {
        Real x1 = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));
        Real x2 = CellCenterX(j-js, nx2, size.x2min.d_view(m), size.x2max.d_view(m));
        Real x3 = CellCenterX(k-ks, nx3, size.x3min.d_view(m), size.x3max.d_view(m));
        Real x = cos_a2*(x1*cos_a3 + x2*sin_a3) + x3*sin_a2;
        Real sn = std::sin(k_par*x);
        Real mx = d0*vflow + amp*sn*rem[1][wave_flag];
        Real my = amp*sn*rem[2][wave_flag];
        Real mz = amp*sn*rem[3][wave_flag];
 
        u0(m,IDN,k,j,i) = d0 + amp*sn*rem[0][wave_flag];
        u0(m,IM1,k,j,i) = mx*cos_a2*cos_a3 - my*sin_a3 - mz*sin_a2*cos_a3;
        u0(m,IM2,k,j,i) = mx*cos_a2*sin_a3 + my*cos_a3 - mz*sin_a2*sin_a3;
        u0(m,IM3,k,j,i) = mx*sin_a2                    + mz*cos_a2;

        if (eos.is_adiabatic) {
          u0(m,IEN,k,j,i) = p0/gm1 + 0.5*d0*v1_0*v1_0 + amp*sn*rem[4][wave_flag];
        }
      }
    );
  }  // End initialization MHD variables

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HydroEigensystem()
//  \brief computes eigenvectors of linear waves in adiabatic/isothermal hydrodynamics

void HydroEigensystem(const Real d, const Real p, const Real v1, const Real v2,
                      const Real v3, const EOS_Data &eos, Real right_eigenmatrix[5][5])
{
  //--- Adiabatic Hydrodynamics ---
  if (eos.is_adiabatic) {
    Real vsq = v1*v1 + v2*v2 + v3*v3;
    Real h = ((p/(eos.gamma - 1.0) + 0.5*d*(v1*v1 + v2*v2+v3*v3)) + p)/d;
    Real a = std::sqrt(eos.gamma*p/d);

    // Right-eigenvectors, stored as COLUMNS (eq. B3)
    right_eigenmatrix[0][0] = 1.0;
    right_eigenmatrix[1][0] = v1 - a;
    right_eigenmatrix[2][0] = v2;
    right_eigenmatrix[3][0] = v3;
    right_eigenmatrix[4][0] = h - v1*a;

    right_eigenmatrix[0][1] = 0.0;
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[2][1] = 1.0;
    right_eigenmatrix[3][1] = 0.0;
    right_eigenmatrix[4][1] = v2;

    right_eigenmatrix[0][2] = 0.0;
    right_eigenmatrix[1][2] = 0.0;
    right_eigenmatrix[2][2] = 0.0;
    right_eigenmatrix[3][2] = 1.0;
    right_eigenmatrix[4][2] = v3;

    right_eigenmatrix[0][3] = 1.0;
    right_eigenmatrix[1][3] = v1;
    right_eigenmatrix[2][3] = v2;
    right_eigenmatrix[3][3] = v3;
    right_eigenmatrix[4][3] = 0.5*vsq;

    right_eigenmatrix[0][4] = 1.0;
    right_eigenmatrix[1][4] = v1 + a;
    right_eigenmatrix[2][4] = v2;
    right_eigenmatrix[3][4] = v3;
    right_eigenmatrix[4][4] = h + v1*a;

  //--- Isothermal Hydrodynamics ---
  } else {
    // Right-eigenvectors, stored as COLUMNS (eq. B3)
    right_eigenmatrix[0][0] = 1.0;
    right_eigenmatrix[1][0] = v1 - eos.iso_cs;
    right_eigenmatrix[2][0] = v2;
    right_eigenmatrix[3][0] = v3;

    right_eigenmatrix[0][1] = 0.0;
    right_eigenmatrix[1][1] = 0.0;
    right_eigenmatrix[2][1] = 1.0;
    right_eigenmatrix[3][1] = 0.0;

    right_eigenmatrix[0][2] = 0.0;
    right_eigenmatrix[1][2] = 0.0;
    right_eigenmatrix[2][2] = 0.0;
    right_eigenmatrix[3][2] = 1.0;

    right_eigenmatrix[0][3] = 1.0;
    right_eigenmatrix[1][3] = v1 + eos.iso_cs;
    right_eigenmatrix[2][3] = v2;
    right_eigenmatrix[3][3] = v3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MHDEigensystem()
//  \brief computes eigenvectors of linear waves in adiabatic/isothermal mhd

void MHDEigensystem(const Real d, const Real p, const Real v1, const Real v2,
                      const Real v3, const EOS_Data &eos, Real right_eigenmatrix[5][5])
{
}

