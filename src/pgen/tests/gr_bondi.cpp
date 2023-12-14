//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_bondi.cpp
//! \brief Problem generator for spherically symmetric black hole accretion.

#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()
#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bondi_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitives(struct bondi_pgen pgen, Real r,
                                Real *prho, Real *ppgas, Real *pur);

KOKKOS_INLINE_FUNCTION
static Real TemperatureMin(struct bondi_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureBisect(struct bondi_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureResidual(struct bondi_pgen pgen, Real t, Real r);

struct bondi_pgen {
  Real spin;                // black hole spin
  Real dexcise, pexcise;    // excision parameters
  Real n_adi, k_adi, gm;    // hydro EOS parameters
  Real r_crit;              // sonic point radius
  Real c1, c2;              // useful constants
  Real temp_min, temp_max;  // bounds for temperature root find
  bool reset_ic = false;    // reset initial conditions after run
};

  bondi_pgen bondi;

} // namespace

// prototypes for user-defined BCs and error functions
void FixedBondiInflow(Mesh *pm);
void BondiErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BondiAccretion_()
//! \brief set initial conditions for Bondi accretion test
//  Compile with '-D PROBLEM=gr_bondi' to enroll as user-specific problem generator
//    reference: Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void ProblemGenerator::BondiAccretion(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (!(pmbp->phydro->peos->eos_data.use_e)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "gr_bondi test requires hydro/use_e=true" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!pmbp->pcoord->is_general_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GR bondi problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // set user-defined BCs and error function pointers
  pgen_final_func = BondiErrors;
  user_bcs_func = FixedBondiInflow;
  if (restart) return;

  // Read problem-specific parameters from input file
  // global parameters
  bondi.k_adi = pin->GetReal("problem", "k_adi");
  bondi.r_crit = pin->GetReal("problem", "r_crit");

  // Get ideal gas EOS data
  bondi.gm = pmbp->phydro->peos->eos_data.gamma;
  Real gm1 = bondi.gm - 1.0;

  // Parameters
  bondi.temp_min = 1.0e-2;  // lesser temperature root must be greater than this
  bondi.temp_max = 1.0e1;   // greater temperature root must be less than this

  // Get spin of black hole
  bondi.spin = pmbp->pcoord->coord_data.bh_spin;

  // Get excision parameters
  bondi.dexcise = pmbp->pcoord->coord_data.dexcise;
  bondi.pexcise = pmbp->pcoord->coord_data.pexcise;

  // Get ratio of specific heats
  bondi.n_adi = 1.0/(bondi.gm - 1.0);

  // Prepare various constants for determining primitives
  Real u_crit_sq = 1.0/(2.0*bondi.r_crit);                           // (HSW 71)
  Real u_crit = -sqrt(u_crit_sq);
  Real t_crit = (bondi.n_adi/(bondi.n_adi+1.0)
                 * u_crit_sq/(1.0-(bondi.n_adi+3.0)*u_crit_sq));     // (HSW 74)
  bondi.c1 = pow(t_crit, bondi.n_adi) * u_crit * SQR(bondi.r_crit);  // (HSW 68)
  bondi.c2 = (SQR(1.0 + (bondi.n_adi+1.0) * t_crit)
              * (1.0 - 3.0/(2.0*bondi.r_crit)));                     // (HSW 69)

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  auto bondi_ = bondi;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  auto w0_ = pmbp->phydro->w0;

  // Initialize primitive values (HYDRO ONLY)
  par_for("pgen_bondi", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho, pgas, uu1, uu2, uu3;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas/gm1;
    w0_(m,IM1,k,j,i) = uu1;
    w0_(m,IM2,k,j,i) = uu2;
    w0_(m,IM3,k,j,i) = uu3;
  });

  // Convert primitives to conserved
  auto &u0_ = pmbp->phydro->u0;
  auto &u1_ = pmbp->phydro->u1;
  if (bondi.reset_ic) {
    pmbp->phydro->peos->PrimToCons(w0_, u1_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  } else {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::LinearWaveErrors_()
//  \brief Computes errors in linear wave solution and outputs to file.

void BondiErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  bondi.reset_ic=true;
  pm->pgen->BondiAccretion(pin, false);

  Real l1_err[8];
  int nvars=0;

  // capture class variables for kernel
  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // compute errors for Hydro  -----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro;

    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    auto &u0_ = pmbp->phydro->u0;
    auto &u1_ = pmbp->phydro->u1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("Bondi-err-Sums",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // Hydro conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      if (eos.is_ideal) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
      }

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
  }

#if MPI_PARALLEL_ENABLED
  // sum over all ranks
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, l1_err, 8, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(l1_err, l1_err, 8, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
  }
#endif

  // normalize errors by number of cells
  Real vol=  (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min)
            *(pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min)
            *(pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int i=0; i<nvars; ++i) l1_err[i] = l1_err[i]/vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i=0; i<nvars; ++i) {
    rms_err += SQR(l1_err[i]);
  }
  rms_err = std::sqrt(rms_err);

  // open output file and write out errors
  if (global_variable::my_rank==0) {
    // open output file and write out errors
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // The file es not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle  RMS-L1-err       ");
      if (pmbp->phydro != nullptr) {
        std::fprintf(pfile, "d_L1         M1_L1         M2_L1");
        std::fprintf(pfile, "         M3_L1         E_L1 ");
      }
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e", pmbp->pmesh->ncycle, rms_err);
    for (int i=0; i<nvars; ++i) {
      std::fprintf(pfile, "  %e", l1_err[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3) {
  // Calculate Boyer-Lindquist coordinates of cell
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1v, x2v, x3v, &r, &theta, &phi);

  // Compute primitive in BL coordinates, transform to Cartesian KS
  Real my_rho, my_pgas, my_ur;
  CalculatePrimitives(pgen, r, &my_rho, &my_pgas, &my_ur);
  Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
  TransformVector(pgen, my_ur, 0.0, 0.0, x1v, x2v, x3v, &u1, &u2, &u3);

  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1v,x2v,x3v, coord.is_minkowski, coord.bh_spin, glower, gupper);

  Real tmp = glower[1][1]*u1*u1 + 2.0*glower[1][2]*u1*u2 + 2.0*glower[1][3]*u1*u3
           + glower[2][2]*u2*u2 + 2.0*glower[2][3]*u2*u3
           + glower[3][3]*u3*u3;
  Real gammasq = 1.0 + tmp;
  Real b = glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
  u0 = (-b - sqrt(fmax(SQR(b) - glower[0][0]*gammasq, 0.0)))/glower[0][0];

  if (r > 1.0) {
    rho = my_rho;
    pgas = my_pgas;
    uu1 = u1 - gupper[0][1]/gupper[0][0] * u0;
    uu2 = u2 - gupper[0][2]/gupper[0][0] * u0;
    uu3 = u3 - gupper[0][3]/gupper[0][0] * u0;
  } else {
    rho = pgen.dexcise;
    pgas = pgen.pexcise;
    uu1 = 0.0;
    uu2 = 0.0;
    uu3 = 0.0;
  }

  return;
}


//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                        + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
    *pr = r;
    *ptheta = acos(x3/r);
    *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
            pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bondi_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3,
                            Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r + SQR(pgen.spin);
  *pa1 = a1_bl * ( (r*x1+pgen.spin*x2)/(SQR(r) + SQR(pgen.spin)) - x2*pgen.spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-pgen.spin*x1)/(SQR(r) + SQR(pgen.spin)) + x1*pgen.spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(pgen.spin)));
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating primitives given radius
// Inputs:
//   r: Schwarzschild radius
//   temp_min,temp_max: bounds on temperature
// Outputs:
//   prho: value set to density
//   ppgas: value set to gas pressure
//   put: value set to u^t in Schwarzschild coordinates
//   pur: value set to u^r in Schwarzschild coordinates
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitives(struct bondi_pgen pgen, Real r,
                                Real *prho, Real *ppgas, Real *pur) {
  // Calculate solution to (HSW 76)
  Real temp_neg_res = TemperatureMin(pgen, r, pgen.temp_min, pgen.temp_max);
  Real temp;
  if (r <= pgen.r_crit) {  // use lesser of two roots
    temp = TemperatureBisect(pgen, r, pgen.temp_min, temp_neg_res);
  } else {  // user greater of two roots
    temp = TemperatureBisect(pgen, r, temp_neg_res, pgen.temp_max);
  }

  // Calculate primitives
  Real rho = pow(temp/pgen.k_adi, pgen.n_adi);             // not same K as HSW
  Real pgas = temp * rho;
  Real ur = pgen.c1 / (SQR(r) * pow(temp, pgen.n_adi));    // (HSW 75)

  // Set primitives
  *prho = rho;
  *ppgas = pgas;
  *pur = ur;
  return;
}

//----------------------------------------------------------------------------------------
// Function for finding temperature at which residual is minimized
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which minimum must occur
// Outputs:
//   returned value: some temperature for which residual of (HSW 76) is negative
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs golden section search (cf. Numerical Recipes, 3rd ed., 10.2)

KOKKOS_INLINE_FUNCTION
static Real TemperatureMin(struct bondi_pgen pgen, Real r, Real t_min, Real t_max) {
  // Parameters
  const Real ratio = 0.3819660112501051;  // (3+\sqrt{5})/2
  const int max_iterations = 40;          // maximum number of iterations

  // Initialize values
  Real t_mid = t_min + ratio * (t_max - t_min);
  Real res_mid = TemperatureResidual(pgen, t_mid, r);

  // Apply golden section method
  bool larger_to_right = true;  // flag indicating larger subinterval is on right
  for (int n = 0; n < max_iterations; ++n) {
    if (res_mid < 0.0) {
      return t_mid;
    }
    Real t_new;
    if (larger_to_right) {
      t_new = t_mid + ratio * (t_max - t_mid);
      Real res_new = TemperatureResidual(pgen, t_new, r);
      if (res_new < res_mid) {
        t_min = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_max = t_new;
        larger_to_right = false;
      }
    } else {
      t_new = t_mid - ratio * (t_mid - t_min);
      Real res_new = TemperatureResidual(pgen, t_new, r);
      if (res_new < res_mid) {
        t_max = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_min = t_new;
        larger_to_right = true;
      }
    }
  }
  return NAN;
}

//----------------------------------------------------------------------------------------
// Bisection root finder
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which root must occur
// Outputs:
//   returned value: temperature that satisfies (HSW 76)
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs bisection search

KOKKOS_INLINE_FUNCTION
static Real TemperatureBisect(struct bondi_pgen pgen, Real r, Real t_min, Real t_max) {
  // Parameters
  const int max_iterations = 40;
  const Real tol_residual = 1.0e-12;
  const Real tol_temperature = 1.0e-12;

  // Find initial residuals
  Real res_min = TemperatureResidual(pgen, t_min, r);
  Real res_max = TemperatureResidual(pgen, t_max, r);
  if (std::abs(res_min) < tol_residual) {
    return t_min;
  }
  if (std::abs(res_max) < tol_residual) {
    return t_max;
  }
  if ((res_min < 0.0 && res_max < 0.0) || (res_min > 0.0 && res_max > 0.0)) {
    return NAN;
  }

  // Iterate to find root
  Real t_mid;
  for (int i = 0; i < max_iterations; ++i) {
    t_mid = (t_min + t_max) / 2.0;
    if (t_max - t_min < tol_temperature) {
      return t_mid;
    }
    Real res_mid = TemperatureResidual(pgen, t_mid, r);
    if (std::abs(res_mid) < tol_residual) {
      return t_mid;
    }
    if ((res_mid < 0.0 && res_min < 0.0) || (res_mid > 0.0 && res_min > 0.0)) {
      t_min = t_mid;
      res_min = res_mid;
    } else {
      t_max = t_mid;
      res_max = res_mid;
    }
  }
  return t_mid;
}

//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct temperature
// Inputs:
//   t: temperature
//   r: Schwarzschild radius
// Outputs:
//   returned value: residual that should vanish for correct temperature
// Notes:
//   implements (76) from Hawley, Smarr, & Wilson 1984, ApJ 277 296

KOKKOS_INLINE_FUNCTION
static Real TemperatureResidual(struct bondi_pgen pgen, Real t, Real r) {
  return SQR(1.0 + (pgen.n_adi+1.0) * t)
      * (1.0 - 2.0/r + SQR(pgen.c1)
         / (SQR(SQR(r)) * pow(t, 2.0*pgen.n_adi))) - pgen.c2;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn FixedBondiInflow
//  \brief Sets boundary condition on surfaces of computational domain
// Note quantities at boundaryies are held fixed to initial condition values

void FixedBondiInflow(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &coord = pm->pmb_pack->pcoord->coord_data;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto bondi_ = bondi;

  int nmb = pm->pmb_pack->nmb_thispack;
  auto u0_ = pm->pmb_pack->phydro->u0;
  auto w0_ = pm->pmb_pack->phydro->w0;

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is-1,0,(n2-1),0,(n3-1));
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie+1,ie+ng,0,(n2-1),0,(n3-1));
  par_for("fixed_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x1 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x1 boundary
    x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_, rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,(ie+i+1)) = rho;
      w0_(m,IEN,k,j,(ie+i+1)) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,(ie+i+1)) = uu1;
      w0_(m,IM2,k,j,(ie+i+1)) = uu2;
      w0_(m,IM3,k,j,(ie+i+1)) = uu3;
    }
  });
  // PrimToCons on X1 physical boundary ghost zones
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js-1,0,(n3-1));
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je+1,je+ng,0,(n3-1));
  par_for("fixed_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x2 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x2 boundary
    x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,(je+j+1),i) = rho;
      w0_(m,IEN,k,(je+j+1),i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,(je+j+1),i) = uu1;
      w0_(m,IM2,k,(je+j+1),i) = uu2;
      w0_(m,IM3,k,(je+j+1),i) = uu3;
    }
  });
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));

  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke+1,ke+ng);
  par_for("fixed_ix3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x3 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x3 boundary
    x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,(ke+k+1),j,i) = rho;
      w0_(m,IEN,(ke+k+1),j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,(ke+k+1),j,i) = uu1;
      w0_(m,IM2,(ke+k+1),j,i) = uu2;
      w0_(m,IM3,(ke+k+1),j,i) = uu3;
    }
  });
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
  pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);

  return;
}
