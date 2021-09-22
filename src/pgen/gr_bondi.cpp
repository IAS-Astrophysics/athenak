//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_bondi.cpp
//! \brief Problem generator for spherically symmetric black hole accretion.

#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(int m, int k, int j, int i,
                                   CoordData coord, Real g_[], Real gi_[],
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
  Real mass, spin;          // black hole mass and spin
  Real n_adi, k_adi, gm;    // hydro EOS parameters
  Real r_crit;              // sonic point radius
  Real c1, c2;              // useful constants
  Real temp_min, temp_max;  // bounds for temperature root find
  bool reset_ic = false;    // reset initial conditions after run
};

  bondi_pgen bondi;

} // namespace

void FixedInnerX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u);
void FixedOuterX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u);
void FixedInnerX2(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u);
void FixedOuterX2(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u);
void FixedInnerX3(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u);
void FixedOuterX3(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BondiAccretion_()
//! \brief set initial conditions for Bondi accretion test
//  Compile with '-D PROBLEM=gr_bondi' to enroll as user-specific problem generator 
//    reference: Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void ProblemGenerator::BondiAccretion_(MeshBlockPack *pmbp, ParameterInput *pin)
{
  // Read problem-specific parameters from input file
  // global parameters
  bondi.k_adi = pin->GetReal("problem", "k_adi");
  bondi.r_crit = pin->GetReal("problem", "r_crit");
  bondi.gm = pin->GetReal("eos", "gamma");

  // Parameters
  bondi.temp_min = 1.0e-2;  // lesser temperature root must be greater than this
  bondi.temp_max = 1.0e1;   // greater temperature root must be less than this

  // Get mass and spin of black hole
  bondi.mass = pmbp->coord.coord_data.bh_mass;
  bondi.spin = pmbp->coord.coord_data.bh_spin;

  // Get ratio of specific heats
  bondi.n_adi = 1.0/(bondi.gm - 1.0);

  // Prepare various constants for determining primitives
  Real u_crit_sq = bondi.mass/(2.0*bondi.r_crit);                    // (HSW 71)
  Real u_crit = -sqrt(u_crit_sq);
  Real t_crit = (bondi.n_adi/(bondi.n_adi+1.0)
                 * u_crit_sq/(1.0-(bondi.n_adi+3.0)*u_crit_sq));     // (HSW 74)
  bondi.c1 = pow(t_crit, bondi.n_adi) * u_crit * SQR(bondi.r_crit);  // (HSW 68)
  bondi.c2 = (SQR(1.0 + (bondi.n_adi+1.0) * t_crit)
              * (1.0 - 3.0*bondi.mass/(2.0*bondi.r_crit)));          // (HSW 69)

  // capture variables for the kernel
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto coord = pmbp->coord.coord_data;
  auto w0_ = pmbp->phydro->w0;
  auto bondi_ = bondi;

  // Initialize primitive values (HYDRO ONLY)
  par_for("pgen_bondi", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      Real rho, pgas, uu1, uu2, uu3, g_[NMETRIC], gi_[NMETRIC];
      ComputePrimitiveSingle(m,k,j,i,coord,g_,gi_,bondi_,
                             rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IPR,k,j,i) = pgas;
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }
  );

  // Convert primitives to conserved
  auto &u0_ = pmbp->phydro->u0;
  auto &u1_ = pmbp->phydro->u1;
  if (bondi.reset_ic) {
    pmbp->phydro->peos->PrimToCons(w0_, u1_);
  } else {
    pmbp->phydro->peos->PrimToCons(w0_, u0_);
  }

  // Enroll boundary function
  if (pin->GetString("mesh", "ix1_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::inner_x1, FixedInnerX1);
  }
  if (pin->GetString("mesh", "ox1_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::outer_x1, FixedOuterX1);
  }
  if (pin->GetString("mesh", "ix2_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::inner_x2, FixedInnerX2);
  }
  if (pin->GetString("mesh", "ox2_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::outer_x2, FixedOuterX2);
  }
  if (pin->GetString("mesh", "ix3_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::inner_x3, FixedInnerX3);
  }
  if (pin->GetString("mesh", "ox3_bc")=="user") {
    pmbp->pmesh->EnrollBoundaryFunction(BoundaryFace::outer_x3, FixedOuterX3);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::LinearWaveErrors_()
//  \brief Computes errors in linear wave solution and outputs to file.

void ProblemGenerator::BondiErrors_(MeshBlockPack *pmbp, ParameterInput *pin)
{
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  bondi.reset_ic=true;
  BondiAccretion_(pmbp, pin);

  Real l1_err[8];
  int nvars=0;

  // capture class variables for kernel
  auto &indcs = pmbp->coord.coord_data.mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  auto &size = pmbp->coord.coord_data.mb_size;

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
      KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum)
      {
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

      }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb)
    );

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
static void ComputePrimitiveSingle(int m, int k, int j, int i,
                                   CoordData coord, Real g_[], Real gi_[],
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas, Real& uu1, Real& uu2, Real& uu3)
{
  auto &indcs = coord.mb_indcs;
  int &is = indcs.is; int &js = indcs.js; int &ks = indcs.ks;

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

  // Calculate Boyer-Lindquist coordinates of cell
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1v, x2v, x3v, &r, &theta, &phi);

  // Compute primitive in BL coordinates, transform to Cartesian KS
  Real my_rho, my_pgas, my_ur;
  CalculatePrimitives(pgen, r, &my_rho, &my_pgas, &my_ur);
  Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
  TransformVector(pgen, my_ur, 0.0, 0.0, x1v, x2v, x3v, &u1, &u2, &u3);

  ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, true,
                          coord.bh_spin, g_, gi_);

  Real tmp = g_[I11]*u1*u1 + 2.0*g_[I12]*u1*u2 + 2.0*g_[I13]*u1*u3
           + g_[I22]*u2*u2 + 2.0*g_[I23]*u2*u3
           + g_[I33]*u3*u3;
  Real gammasq = 1.0 + tmp;
  Real b = g_[I01]*u1+g_[I02]*u2+g_[I03]*u3;
  u0 = (-b - sqrt(SQR(b) - g_[I00]*gammasq))/g_[I00];

  rho = my_rho;
  pgas = my_pgas;
  uu1 = u1 - gi_[I01]/gi_[I00] * u0;
  uu2 = u2 - gi_[I02]/gi_[I00] * u0;
  uu3 = u3 - gi_[I03]/gi_[I00] * u0;

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
                                         Real *pr, Real *ptheta, Real *pphi)
{
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                   + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0);
    *pr = r;
    *ptheta = acos(x3/r);
    *pphi = atan2( (r*x2-pgen.spin*x1)/(SQR(r)+SQR(pgen.spin)),
                   (pgen.spin*x2+r*x1)/(SQR(r)+SQR(pgen.spin)) );
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
                            Real *pa1, Real *pa2, Real *pa3)
{ 
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = sqrt( SQR(rad) - SQR(pgen.spin) + sqrt( SQR(SQR(rad) - SQR(pgen.spin))
               + 4.0*SQR(pgen.spin)*SQR(x3) ) )/ sqrt(2.0);
  Real delta = SQR(r) - 2.0*pgen.mass*r + SQR(pgen.spin);
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
                                Real *prho, Real *ppgas, Real *pur)
{
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
      * (1.0 - 2.0*pgen.mass/r + SQR(pgen.c1)
         / (SQR(SQR(r)) * pow(t, 2.0*pgen.n_adi))) - pgen.c2;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn FixedInnerX1
//  \brief Sets boundary condition on inner X1 boundary
// Note quantities at this boundary are held fixed to initial condition values

void FixedInnerX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u)
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  auto bondi_ = bondi;

  par_for("fixed_ix1", DevExeSpace(),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real rho, pgas, uu1, uu2, uu3, g_[NMETRIC], gi_[NMETRIC];
      ComputePrimitiveSingle(m,k,j,(is-i-1),coord,g_,gi_,bondi_,
                             rho,pgas,uu1,uu2,uu3);
      Real ud, ue, um1, um2, um3;
      eos.PrimToConsSingleGR(g_, gi_, rho, pgas, uu1, uu2, uu3,
                             ud, ue, um1, um2, um3);
      u(m,IDN,k,j,(is-i-1)) = ud;
      u(m,IEN,k,j,(is-i-1)) = ue;
      u(m,IM1,k,j,(is-i-1)) = um1;
      u(m,IM2,k,j,(is-i-1)) = um2;
      u(m,IM3,k,j,(is-i-1)) = um3;
    }
  );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn FixedOuterrX1
//  \brief Sets boundary condition on outer X1 boundary
// Note quantities at this boundary are held fixed to initial condition values

void FixedOuterX1(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u)
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &ie = indcs.ie;
  auto bondi_ = bondi;

  par_for("fixed_ox1", DevExeSpace(),0,(n3-1),0,(n2-1),0,(ng-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real rho, pgas, uu1, uu2, uu3, g_[NMETRIC], gi_[NMETRIC];
      ComputePrimitiveSingle(m,k,j,(ie+i+1),coord,g_,gi_,bondi_,
                             rho,pgas,uu1,uu2,uu3);
      Real ud, ue, um1, um2, um3;
      eos.PrimToConsSingleGR(g_, gi_, rho, pgas, uu1, uu2, uu3,
                             ud, ue, um1, um2, um3);
      u(m,IDN,k,j,(ie+i+1)) = ud;
      u(m,IEN,k,j,(ie+i+1)) = ue;
      u(m,IM1,k,j,(ie+i+1)) = um1;
      u(m,IM2,k,j,(ie+i+1)) = um2;
      u(m,IM3,k,j,(ie+i+1)) = um3;
    }
  );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn FixedInnerX2
//  \brief Sets boundary condition on inner X2 boundary
// Note quantities at this boundary are held fixed to initial condition values

void FixedInnerX2(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u)
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &js = indcs.js;
  auto bondi_ = bondi;

  par_for("fixed_ix2", DevExeSpace(),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real rho, pgas, uu1, uu2, uu3, g_[NMETRIC], gi_[NMETRIC];
      ComputePrimitiveSingle(m,k,(js-j-1),i,coord,g_,gi_,bondi_,
                             rho,pgas,uu1,uu2,uu3);
      Real ud, ue, um1, um2, um3;
      eos.PrimToConsSingleGR(g_, gi_, rho, pgas, uu1, uu2, uu3,
                             ud, ue, um1, um2, um3);
      u(m,IDN,k,(js-j-1),i) = ud;
      u(m,IEN,k,(js-j-1),i) = ue;
      u(m,IM1,k,(js-j-1),i) = um1;
      u(m,IM2,k,(js-j-1),i) = um2;
      u(m,IM3,k,(js-j-1),i) = um3;
    }
  );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn FixedOuterX2
//  \brief Sets boundary condition on outer X2 boundary
// Note quantities at this boundary are held fixed to initial condition values

void FixedOuterX2(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u)
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &je = indcs.je;
  auto bondi_ = bondi;

  par_for("fixed_ox2", DevExeSpace(),0,(n3-1),0,(ng-1),0,(n1-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real rho, pgas, uu1, uu2, uu3, g_[NMETRIC], gi_[NMETRIC];
      ComputePrimitiveSingle(m,k,(je+j+1),i,coord,g_,gi_,bondi_,
                             rho,pgas,uu1,uu2,uu3);
      Real ud, ue, um1, um2, um3;
      eos.PrimToConsSingleGR(g_, gi_, rho, pgas, uu1, uu2, uu3,
                             ud, ue, um1, um2, um3);
      u(m,IDN,k,(je+j+1),i) = ud;
      u(m,IEN,k,(je+j+1),i) = ue;
      u(m,IM1,k,(je+j+1),i) = um1;
      u(m,IM2,k,(je+j+1),i) = um2;
      u(m,IM3,k,(je+j+1),i) = um3;
    }
  );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn FixedInnerX3
//  \brief Sets boundary condition on inner X3 boundary
// Note quantities at this boundary are held fixed to initial condition values

void FixedInnerX3(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u)
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ks = indcs.ks;
  auto bondi_ = bondi;

  par_for("fixed_ix3", DevExeSpace(),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real rho, pgas, uu1, uu2, uu3, g_[NMETRIC], gi_[NMETRIC];
      ComputePrimitiveSingle(m,(ks-k-1),j,i,coord,g_,gi_,bondi_,
                             rho,pgas,uu1,uu2,uu3);
      Real ud, ue, um1, um2, um3;
      eos.PrimToConsSingleGR(g_, gi_, rho, pgas, uu1, uu2, uu3,
                             ud, ue, um1, um2, um3);
      u(m,IDN,(ks-k-1),j,i) = ud;
      u(m,IEN,(ks-k-1),j,i) = ue;
      u(m,IM1,(ks-k-1),j,i) = um1;
      u(m,IM2,(ks-k-1),j,i) = um2;
      u(m,IM3,(ks-k-1),j,i) = um3;
    }
  );
  return;
}

//----------------------------------------------------------------------------------------
//! \fn FixedOuterX3
//  \brief Sets boundary condition on outer X3 boundary
// Note quantities at this boundary are held fixed to initial condition values

void FixedOuterX3(int m, CoordData &coord, EOS_Data &eos, DvceArray5D<Real> &u)
{
  auto &indcs = coord.mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = indcs.nx2 + 2*ng;
  int &ke = indcs.ke;
  auto bondi_ = bondi;

  par_for("fixed_ox3", DevExeSpace(),0,(ng-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int k, int j, int i)
    {
      Real rho, pgas, uu1, uu2, uu3, g_[NMETRIC], gi_[NMETRIC];
      ComputePrimitiveSingle(m,(ke+k+1),j,i,coord,g_,gi_,bondi_,
                             rho,pgas,uu1,uu2,uu3);
      Real ud, ue, um1, um2, um3;
      eos.PrimToConsSingleGR(g_, gi_, rho, pgas, uu1, uu2, uu3,
                             ud, ue, um1, um2, um3);
      u(m,IDN,(ke+k+1),j,i) = ud;
      u(m,IEN,(ke+k+1),j,i) = ue;
      u(m,IM1,(ke+k+1),j,i) = um1;
      u(m,IM2,(ke+k+1),j,i) = um2;
      u(m,IM3,(ke+k+1),j,i) = um3;
    }
  );
  return;
}
