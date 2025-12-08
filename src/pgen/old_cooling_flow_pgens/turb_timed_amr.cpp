#include <cmath>
#include <iostream>

#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "globals.hpp"

// User-defined history functions
void TurbulentHistory(HistoryData *pdata, Mesh *pm);

// User-defined refinement condition
void RefinementCondition(MeshBlockPack* pmbp);

// Static variable to store refinement time
static Real t_refine = -1.0;

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//  \brief Problem Generator for turbulence with time-dependent AMR

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Read refinement time from input (default to a large value if not specified)
  // Add to input file: <problem> t_refine = <time_value>
  // Set to a large value to disable refinement
  t_refine = pin->GetOrAddReal("problem","t_refine",1.0e10);

  // Register the refinement condition function
  user_ref_func = RefinementCondition;
  // enroll user history function
  user_hist_func = TurbulentHistory;

  if (restart) {
    return;
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Turbulence problem generator can only be run with Hydro and/or MHD, but no "
       << "<hydro> or <mhd> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  // Read problem parameters
  Real cs = pin->GetOrAddReal("eos","iso_sound_speed",1.0);
  Real beta = pin->GetOrAddReal("problem","beta",1.0);

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
    auto &u0 = pmbp->phydro->u0;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = rho0*cs*cs/eos.gamma;

    // Set initial conditions
    par_for("pgen_turb", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = rho0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) {
        u0(m,IEN,k,j,i) = p0/gm1 +
           0.5*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) +
           SQR(u0(m,IM3,k,j,i)))/u0(m,IDN,k,j,i);
      }
    });
  }

  // Initialize MHD variables ---------------------------------
  if (pmbp->pmhd != nullptr) {
    Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real p0 = rho0*cs*cs/eos.gamma;
    Real B0 = cs*std::sqrt(2.0*p0/beta);


    // Set initial conditions
    par_for("pgen_turb", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = rho0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;

      // initialize B
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = B0;
      if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
      if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
      if (k==ke) {b0.x3f(m,k+1,j,i) = B0;}

      if (eos.is_ideal) {
        u0(m,IEN,k,j,i) = p0/gm1 + 0.5*B0*B0 +
           0.5*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i)) +
           SQR(u0(m,IM3,k,j,i)))/u0(m,IDN,k,j,i);
      }
    });
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCondition()
//  \brief Simple time-dependent refinement condition
//  Refines everywhere when t > t_refine

void RefinementCondition(MeshBlockPack* pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;

  // Get current simulation time
  Real time = pmesh->time;
  Real t_ref = t_refine;  // Capture for lambda

  // Diagnostic output when refinement is about to trigger
  static bool refinement_triggered = false;
  if (time > t_ref && !refinement_triggered && global_variable::my_rank == 0) {
    std::cout << "### REFINEMENT TRIGGERED at t=" << time
              << " (t_refine=" << t_ref << "), nmb=" << nmb << std::endl;
    refinement_triggered = true;
  }

  // Simple time-based refinement: refine all blocks when t > t_refine
  if (time > t_ref) {
    // Use host-side loop to avoid potential GPU memory issues
    for (int m = 0; m < nmb; m++) {
      refine_flag.h_view(m+mbs) = 1;  // Flag for refinement
    }
    // Mark as modified on host and sync to device
    refine_flag.template modify<HostMemSpace>();
    refine_flag.template sync<DevExeSpace>();
  }
}

//----------------------------------------------------------------------------------------
// Function for computing history variables (same as turb.cpp)
// 0-2 = <Bx>, <By>, <Bz>
// 3 = <B^2>
// 4 = <B^4>
// 5 = <(d_j B_i)(d_j B_i)>
// 6 = <(B_j d_j B_i)(B_k d_k B_i)>
// 7 = <|BxJ|^2>
// 8 = <|B.J|^2>
// 9 = <U^2>
// 10 = <(d_j U_i)(d_j U_i)>

void TurbulentHistory(HistoryData *pdata, Mesh *pm) {
  // Only compute if MHD is enabled
  if (pm->pmb_pack->pmhd == nullptr) {
    pdata->nhist = 0;
    return;
  }

  pdata->nhist = 12;
  pdata->label[0] = "Bx";
  pdata->label[1] = "By";
  pdata->label[2] = "Bz";
  pdata->label[3] = "B^2";
  pdata->label[4] = "B^4";
  pdata->label[5] = "dB^2";
  pdata->label[6] = "BdB^2";
  pdata->label[7] = "|BxJ|^2";
  pdata->label[8] = "|B.J|^2";
  pdata->label[9] = "U^2";
  pdata->label[10] = "dU";
  pdata->label[11] = "|divB|";

  // capture class variables for kernel
  auto &bcc = pm->pmb_pack->pmhd->bcc0;
  auto &b = pm->pmb_pack->pmhd->b0;
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;
  bool &multi_d = pm->multi_d;
  bool &three_d = pm->three_d;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;
  Real total_vol = 0.0;
  Kokkos::parallel_reduce("HistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real dx_squared = size.d_view(m).dx1 * size.d_view(m).dx1;

    // MHD conserved variables:
    array_sum::GlobalSum hvars;

    // calculate mean B
    hvars.the_array[0] = bcc(m,IBX,k,j,i);
    hvars.the_array[1] = bcc(m,IBY,k,j,i);
    hvars.the_array[2] = bcc(m,IBZ,k,j,i);

    // 0 = < B^2 >
    Real B_mag_sq = bcc(m,IBX,k,j,i)*bcc(m,IBX,k,j,i)
                  + bcc(m,IBY,k,j,i)*bcc(m,IBY,k,j,i)
                  + bcc(m,IBZ,k,j,i)*bcc(m,IBZ,k,j,i);
    hvars.the_array[3] = B_mag_sq*vol;
    // 0 = < B^4 >
    Real B_fourth = B_mag_sq*B_mag_sq;
    hvars.the_array[4] = B_fourth*vol;
    // 1 = < (d_j B_i)(d_j B_i) >
    hvars.the_array[5] = (
      ((b.x1f(m,k,j,i+1)-b.x1f(m,k,j,i))*(b.x1f(m,k,j,i+1)-b.x1f(m,k,j,i))
     + (b.x2f(m,k,j+1,i)-b.x2f(m,k,j,i))*(b.x2f(m,k,j+1,i)-b.x2f(m,k,j,i))
     + (b.x3f(m,k+1,j,i)-b.x3f(m,k,j,i))*(b.x3f(m,k+1,j,i)-b.x3f(m,k,j,i))
     + 0.25*(bcc(m,IBX,k,j+1,i)-bcc(m,IBX,k,j-1,i))
           *(bcc(m,IBX,k,j+1,i)-bcc(m,IBX,k,j-1,i))
     + 0.25*(bcc(m,IBX,k+1,j,i)-bcc(m,IBX,k-1,j,i))
           *(bcc(m,IBX,k+1,j,i)-bcc(m,IBX,k-1,j,i))
     + 0.25*(bcc(m,IBY,k,j,i+1)-bcc(m,IBY,k,j,i-1))
           *(bcc(m,IBY,k,j,i+1)-bcc(m,IBY,k,j,i-1))
     + 0.25*(bcc(m,IBY,k+1,j,i)-bcc(m,IBY,k-1,j,i))
           *(bcc(m,IBY,k+1,j,i)-bcc(m,IBY,k-1,j,i))
     + 0.25*(bcc(m,IBZ,k,j,i+1)-bcc(m,IBZ,k,j,i-1))
           *(bcc(m,IBZ,k,j,i+1)-bcc(m,IBZ,k,j,i-1))
     + 0.25*(bcc(m,IBZ,k,j+1,i)-bcc(m,IBZ,i,j-1,i))
           *(bcc(m,IBZ,k,j+1,i)-bcc(m,IBZ,i,j-1,i)))
       / dx_squared)*vol;
    // 2 = < (B_j d_j B_i)(B_k d_k B_i) >
    Real bdb1 = bcc(m,IBX,k,j,i)*(b.x1f(m,k,j,i+1)-b.x1f(m,k,j,i))
                +0.5*bcc(m,IBY,k,j,i)*(bcc(m,IBX,k,j+1,i)-bcc(m,IBX,k,j-1,i))
                +0.5*bcc(m,IBZ,k,j,i)*(bcc(m,IBX,k+1,j,i)-bcc(m,IBX,k-1,j,i));
    Real bdb2 = bcc(m,IBY,k,j,i)*(b.x2f(m,k,j+1,i)-b.x2f(m,k,j,i))
                +0.5*bcc(m,IBZ,k,j,i)*(bcc(m,IBY,k+1,j,i)-bcc(m,IBY,k-1,j,i))
                +0.5*bcc(m,IBX,k,j,i)*(bcc(m,IBY,k,j,i+1)-bcc(m,IBY,k,j,i-1));
    Real bdb3 = bcc(m,IBZ,k,j,i)*(b.x3f(m,k+1,j,i)-b.x3f(m,k,j,i))
                +0.5*bcc(m,IBX,k,j,i)*(bcc(m,IBZ,k,j,i+1)-bcc(m,IBZ,k,j,i-1))
                +0.5*bcc(m,IBY,k,j,i)*(bcc(m,IBZ,k,j+1,i)-bcc(m,IBZ,k,j-1,i));
    hvars.the_array[6] = ((bdb1*bdb1 + bdb2*bdb2 + bdb3*bdb3) / dx_squared)*vol;
    // 3 = < |BxJ|^2 >
    Real Jx = 0.5*(bcc(m,IBZ,k,j+1,i)-bcc(m,IBZ,k,j-1,i))
             -0.5*(bcc(m,IBY,k+1,j,i)-bcc(m,IBY,k-1,j,i));
    Real Jy = 0.5*(bcc(m,IBX,k+1,j,i)-bcc(m,IBX,k-1,j,i))
             -0.5*(bcc(m,IBZ,k,j,i+1)-bcc(m,IBZ,k,j,i-1));
    Real Jz = 0.5*(bcc(m,IBY,k,j,i+1)-bcc(m,IBY,k,j,i-1))
             -0.5*(bcc(m,IBX,k,j+1,i)-bcc(m,IBX,k,j-1,i));
    hvars.the_array[7] =((
       (bcc(m,IBY,k,j,i)*Jz - bcc(m,IBZ,k,j,i)*Jy)
      *(bcc(m,IBY,k,j,i)*Jz - bcc(m,IBZ,k,j,i)*Jy)
      +(bcc(m,IBZ,k,j,i)*Jx - bcc(m,IBX,k,j,i)*Jz)
      *(bcc(m,IBZ,k,j,i)*Jx - bcc(m,IBX,k,j,i)*Jz)
      +(bcc(m,IBX,k,j,i)*Jy - bcc(m,IBY,k,j,i)*Jx)
      *(bcc(m,IBX,k,j,i)*Jy - bcc(m,IBY,k,j,i)*Jx))
                    / dx_squared)*vol;
    // 4 = < |B.J|^2 >
    hvars.the_array[8] = (
      ((bcc(m,IBX,k,j,i)*Jx + bcc(m,IBY,k,j,i)*Jy + bcc(m,IBZ,k,j,i)*Jz)
      *(bcc(m,IBX,k,j,i)*Jx + bcc(m,IBY,k,j,i)*Jy + bcc(m,IBZ,k,j,i)*Jz)
                          )/dx_squared)*vol;
    // 5 = < U^2 >
    hvars.the_array[9] += ((w0_(m,IVX,k,j,i)*w0_(m,IVX,k,j,i))
                        + (w0_(m,IVY,k,j,i)*w0_(m,IVY,k,j,i))
                        + (w0_(m,IVZ,k,j,i)*w0_(m,IVZ,k,j,i)))*vol;
    // 6 = < (d_j U_i)(d_j U_i) >
    hvars.the_array[10] +=
    (((0.25*(w0_(m,IVX,k,j,i+1)-w0_(m,IVX,k,j,i-1))
           *(w0_(m,IVX,k,j,i+1)-w0_(m,IVX,k,j,i-1))
     + 0.25*(w0_(m,IVY,k,j+1,i)-w0_(m,IVY,k,j-1,i))
           *(w0_(m,IVY,k,j+1,i)-w0_(m,IVY,k,j-1,i))
     + 0.25*(w0_(m,IVZ,k+1,j,i)-w0_(m,IVZ,k-1,j,i))
           *(w0_(m,IVZ,k+1,j,i)-w0_(m,IVZ,k-1,j,i))
     + 0.25*(w0_(m,IVX,k,j+1,i)-w0_(m,IVX,k,j-1,i))
           *(w0_(m,IVX,k,j+1,i)-w0_(m,IVX,k,j-1,i))
     + 0.25*(w0_(m,IVX,k+1,j,i)-w0_(m,IVX,k-1,j,i))
           *(w0_(m,IVX,k+1,j,i)-w0_(m,IVX,k-1,j,i))
     + 0.25*(w0_(m,IVY,k,j,i+1)-w0_(m,IVY,k,j,i-1))
           *(w0_(m,IVY,k,j,i+1)-w0_(m,IVY,k,j,i-1))
     + 0.25*(w0_(m,IVY,k+1,j,i)-w0_(m,IVY,k-1,j,i))
           *(w0_(m,IVY,k+1,j,i)-w0_(m,IVY,k-1,j,i))
     + 0.25*(w0_(m,IVZ,k,j,i+1)-w0_(m,IVZ,k,j,i-1))
           *(w0_(m,IVZ,k,j,i+1)-w0_(m,IVZ,k,j,i-1))
     + 0.25*(w0_(m,IVZ,k,j+1,i)-w0_(m,IVZ,k,j-1,i))
           *(w0_(m,IVZ,k,j+1,i)-w0_(m,IVZ,k,j-1,i))))
     / dx_squared)*vol;

    // Compute divergence of B
    Real divb = (b.x1f(m,k,j,i+1) - b.x1f(m,k,j,i))/size.d_view(m).dx1;
    if (multi_d) {
      divb += (b.x2f(m,k,j+1,i) - b.x2f(m,k,j,i))/size.d_view(m).dx2;
    }
    if (three_d) {
      divb += (b.x3f(m,k+1,j,i) - b.x3f(m,k,j,i))/size.d_view(m).dx3;
    }

    // Sum of divergence of B
    hvars.the_array[11] += fabs(divb)*vol;
    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
  // Kokkos::fence();

  // store data in HistoryData array
  for (int n=0; n<nhist_; n++) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }

  return;
}
