//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sgrid_bns.cpp
//  \brief Initial data reader for binary neutron star data with SGRID

#include <math.h>
#include <sys/stat.h> // mkdir

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"

// libsgrid
// Functions protoypes are "SGRID_*"

extern "C" {
int libsgrid_main(int argc, char **argv);
extern int SGRID_memory_persists;
int SGRID_grid_exists(void);

void SGRID_errorexits(char *file, int line, char *s, char *t);
#define SGRID_errorexits(s, t) SGRID_errorexits(__FILE__, __LINE__, (s), (t))

int SGRID_system2(char *s1, char *s2);
int SGRID_lock_curr_til_EOF(FILE *out);
int SGRID_construct_argv(char *str, char ***argv);

int SGRID_fgotonext(FILE *in, const char *label);
int SGRID_fgetparameter(FILE *in, const char *par, char *str);
int SGRID_extract_after_EQ(char *str);
int SGRID_extrstr_before_after_EQ(const char *str, char *before, char *after);
int SGRID_fscanline(FILE *in, char *str);
int SGRID_extrstr_before_after(const char *str, char *before, char *after,
                               char z);
int SGRID_find_before_after(const char *str, char *before, char *after,
                            const char *z);
int SGRID_pfind_before_after(const char *str, int p, char *before, char *after,
                             const char *z);
int SGRID_sscan_word_at_p(const char *str, int p, char *word);
int SGRID_fscan_str_using_getc(FILE *in, char *str);
int SGRID_fscanf1(FILE *in, char *fmt, char *str);
void SGRID_free_everything();

void SGRID_EoS_T0_rho0_P_rhoE_from_hm1(double hm1, double *rho0, double *P,
                                       double *rhoE);
double SGRID_epsl_of_rho0_rhoE(double rho0, double rhoE);

int SGRID_DNSdata_Interpolate_ADMvars_to_xyz(double xyz[3], double *vars,
                                             int init);
}

#define STRLEN (16384)

// Indexes for Initial data variables (IDVars)
enum {
  idvar_alpha,
  idvar_Bx,
  idvar_By,
  idvar_Bz,
  idvar_gxx,
  idvar_gxy,
  idvar_gxz,
  idvar_gyy,
  idvar_gyz,
  idvar_gzz,
  idvar_Kxx,
  idvar_Kxy,
  idvar_Kxz,
  idvar_Kyy,
  idvar_Kyz,
  idvar_Kzz,
  idvar_q,
  idvar_VRx,
  idvar_VRy,
  idvar_VRz,
  idvar_NDATAMAX, // TODO(DR) in NMESH this is 23, but they are 20, and only 20
                  // used...
};

namespace {
// Utilities wrapping various SGRID DNS calls (DNS_*)
void DNS_init_sgrid(ParameterInput *pin);
int DNS_position_fileptr_after_str(FILE *in, const char *str);
int DNS_parameters(ParameterInput *pin);
int DNS_call_sgrid(const char *command);
} // namespace

void SGRIDHistory(HistoryData *pdata, Mesh *pm);
void SGRIDRefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for BNS with Elliptica
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Set the custom history function
  user_hist_func = &SGRIDHistory;
  user_ref_func  = &SGRIDRefinementCondition;

  if (restart)
    return;

  if (global_variable::my_rank == 0) {
    mkdir("SGRID", 0775);
  }
#if MPI_PARALLEL_ENABLED
  (void)MPI_Barrier(MPI_COMM_WORLD);
#endif

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;

  if (pmbp->pdyngr == nullptr || pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "BNS data requires <mhd> and <z4c> blocks." << std::endl;
    exit(EXIT_FAILURE);
  }

  //
  // Initialize the data reader
  DNS_init_sgrid(pin);
  DNS_parameters(pin);

  //
  // Read parameters
  const bool verbose = pin->GetOrAddBoolean("problem", "verbose", false);
  const Real sgrid_x_CM = pin->GetReal("problem", "x_CM");
  const Real Omega = pin->GetReal("problem", "Omega");
  const Real ecc = pin->GetReal("problem", "ecc");
  const Real xmax1 = pin->GetReal("problem", "xmax1");
  const Real xmax2 = pin->GetReal("problem", "xmax2");

  const Real rdot = pin->GetReal("problem", "rdot");
  const Real rdotor = rdot / (xmax1 - xmax2);

  const int rotation180 = pin->GetOrAddInteger("problem", "180rotation", 0);
  const Real s180 = (1 - 2 * rotation180);

  // Prepare SGRID interpolator
  if (verbose) {
    std::cout << "Initializing SGRID_DNSdata_Interpolate_ADMvars_to_xyz"
              << std::endl;
  }
  SGRID_DNSdata_Interpolate_ADMvars_to_xyz(NULL, NULL, 1);

  //
  // Capture variables for kernel; note that when Z4c is enabled, the gauge
  // variables are part of the Z4c class.
  auto &u_adm = pmbp->padm->u_adm;
  auto &w0 = pmbp->pmhd->w0;
  auto &u_z4c = pmbp->pz4c->u0;
  // Because SGRID only operates on the CPU, we can't construct the data on the
  // GPU. Instead, we create a mirror guaranteed to be on the CPU, populate the
  // data there, then move it back to the GPU.
  // TODO(JMF): This needs to be tested on CPUs to ensure that it functions
  // properly; In theory, create_mirror_view shouldn't copy the data unless it's
  // in a different memory space.
  HostArray5D<Real>::HostMirror host_u_adm = create_mirror_view(u_adm);
  HostArray5D<Real>::HostMirror host_w0 = create_mirror_view(w0);
  HostArray5D<Real>::HostMirror host_u_z4c = create_mirror_view(u_z4c);
  adm::ADM::ADMhost_vars host_adm;
  host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
  host_adm.beta_u.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_BETAX,
                                       z4c::Z4c::I_Z4C_BETAZ);
  host_adm.g_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_GXX,
                                     adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_KXX,
                                      adm::ADM::I_ADM_KZZ);

  if (verbose)
    std::cout << "Host mirrors created." << std::endl;

  // REMARK: you need to compile SGRID without OpenMP support!
  int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  int ncells2 = indcs.nx2 + 2 * (indcs.ng);
  int ncells3 = indcs.nx3 + 2 * (indcs.ng);
  int nmb = pmbp->nmb_thispack;

  // TODO(DR): Use a Kokkos loop to improve performance
  for (int m = 0; m < nmb; m++)
    for (int k = 0; k < ncells3; k++)
      for (int j = 0; j < ncells2; j++)
        for (int i = 0; i < ncells1; i++) {
          Real &x1min = size.h_view(m).x1min;
          // not to be confused by xmax1, which is used by SGRID
          Real &x1max = size.h_view(m).x1max;
          int nx1 = indcs.nx1;

          Real &x2min = size.h_view(m).x2min;
          // not to be confused by xmax2, which is used by SGRID
          Real &x2max = size.h_view(m).x2max;
          int nx2 = indcs.nx2;

          Real &x3min = size.h_view(m).x3min;
          Real &x3max = size.h_view(m).x3max;
          int nx3 = indcs.nx3;

          const Real z = CellCenterX(k - ks, nx3, x3min, x3max);
          const Real y = CellCenterX(j - js, nx2, x2min, x2max);
          const Real x = CellCenterX(i - is, nx1, x1min, x1max);

          Real zb = z;
          Real yb = y * s180; // multiply by -1 if 180 degree rotation
          Real xb = x * s180;
          Real xs = xb + sgrid_x_CM; // shift x-coord
          Real xyz[3] = {xs, yb, zb};

          // Initial data variables at one point
          // 20 values for the fields at (x_i,y_j,z_k) ordered as:
          //  alpha DNSdata_Bx DNSdata_By DNSdata_Bz
          //  gxx gxy gxz gyy gyz gzz
          //  Kxx Kxy Kxz Kyy Kyz Kzz
          //  q VRx VRy VRz
          Real IDvars[idvar_NDATAMAX];

          // Interpolate
          // This call is supposed to be threadsafe, it contains an OMP Critical
          SGRID_DNSdata_Interpolate_ADMvars_to_xyz(xyz, IDvars, 0);

          // transform some tensor components, if we have a 180 degree rotation
          IDvars[idvar_Bx] *= s180;
          IDvars[idvar_By] *= s180;
          IDvars[idvar_gxz] *= s180;
          IDvars[idvar_gyz] *= s180;
          IDvars[idvar_Kxz] *= s180;
          IDvars[idvar_Kyz] *= s180;
          IDvars[idvar_VRx] *= s180;
          IDvars[idvar_VRy] *= s180;

          // Extract metric quantities
          host_adm.alpha(m, k, j, i) = IDvars[idvar_alpha];
          host_adm.beta_u(m, 0, k, j, i) = IDvars[idvar_Bx];
          host_adm.beta_u(m, 1, k, j, i) = IDvars[idvar_By];
          host_adm.beta_u(m, 2, k, j, i) = IDvars[idvar_Bz];

          Real g3d[NSPMETRIC];
          host_adm.g_dd(m, 0, 0, k, j, i) = g3d[S11] = IDvars[idvar_gxx];
          host_adm.g_dd(m, 0, 1, k, j, i) = g3d[S12] = IDvars[idvar_gxy];
          host_adm.g_dd(m, 0, 2, k, j, i) = g3d[S13] = IDvars[idvar_gxz];
          host_adm.g_dd(m, 1, 1, k, j, i) = g3d[S22] = IDvars[idvar_gyy];
          host_adm.g_dd(m, 1, 2, k, j, i) = g3d[S23] = IDvars[idvar_gyz];
          host_adm.g_dd(m, 2, 2, k, j, i) = g3d[S33] = IDvars[idvar_gzz];

          host_adm.vK_dd(m, 0, 0, k, j, i) = IDvars[idvar_Kxx];
          host_adm.vK_dd(m, 0, 1, k, j, i) = IDvars[idvar_Kxy];
          host_adm.vK_dd(m, 0, 2, k, j, i) = IDvars[idvar_Kxz];
          host_adm.vK_dd(m, 1, 1, k, j, i) = IDvars[idvar_Kyy];
          host_adm.vK_dd(m, 1, 2, k, j, i) = IDvars[idvar_Kyz];
          host_adm.vK_dd(m, 2, 2, k, j, i) = IDvars[idvar_Kzz];

          // Extract hydro quantities
          Real rho = 0;
          Real press = 0;
          Real eps = 0;
          Real v_u_x = 0, v_u_y = 0, v_u_z = 0;

          // if we are in matter region, convert q, VR to rho, press, eps, v^i :
          if (IDvars[idvar_q] > 0.0) {
            SGRID_EoS_T0_rho0_P_rhoE_from_hm1(IDvars[idvar_q], &rho, &press,
                                              &eps);

            // 3-velocity  v^i
            Real xmax = (xb > 0) ? xmax1 : xmax2;

            // construct KV xi from Omega, ecc, rdot, xmax1-xmax2
            Real xix = -Omega * yb + xb * rdotor; // CM is at (0,0,0) in bam
            Real xiy = Omega * (xb - ecc * xmax) + yb * rdotor;
            Real xiz = zb * rdotor;

            // vI^i = VR^i + xi^i
            Real vIx = IDvars[idvar_VRx] + xix;
            Real vIy = IDvars[idvar_VRy] + xiy;
            Real vIz = IDvars[idvar_VRz] + xiz;

            // Note: vI^i = u^i/u^0 in DNSdata,
            //       while matter_v^i = u^i/(alpha u^0) + beta^i / alpha
            //   ==> matter_v^i = (vI^i + beta^i)/alpha
            v_u_x = (vIx + IDvars[idvar_Bx]) / IDvars[idvar_alpha];
            v_u_y = (vIy + IDvars[idvar_By]) / IDvars[idvar_alpha];
            v_u_z = (vIz + IDvars[idvar_Bz]) / IDvars[idvar_alpha];
          }

          // Store fluid quantities
          host_w0(m, IDN, k, j, i) = rho;
          host_w0(m, IPR, k, j, i) = press;
          Real vu[3] = {v_u_x, v_u_y, v_u_z};

          // Before we store the velocity, we need to make sure it's physical
          // and calculate the Lorentz factor. If the velocity is superluminal,
          // we make a last-ditch attempt to salvage the solution by rescaling
          // it to vsq = 1.0 - 1e-15
          Real vsq = Primitive::SquareVector(vu, g3d);
          if (1.0 - vsq <= 0) {
            std::cout << "The velocity is superluminal!" << std::endl
                      << "Attempting to adjust..." << std::endl;
            Real fac = sqrt((1.0 - 1e-15) / vsq);
            vu[0] *= fac;
            vu[1] *= fac;
            vu[2] *= fac;
            vsq = 1.0 - 1.0e-15;
          }
          Real W = sqrt(1.0 / (1.0 - vsq));

          host_w0(m, IVX, k, j, i) = W * vu[0];
          host_w0(m, IVY, k, j, i) = W * vu[1];
          host_w0(m, IVZ, k, j, i) = W * vu[2];
        }

  if (verbose)
    std::cout << "Host mirrors filled." << std::endl;

  // Cleanup
  SGRID_free_everything();
  if (verbose)
    std::cout << "SGRID freed." << std::endl;

  // Copy the data to the GPU.
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(w0, host_w0);
  Kokkos::deep_copy(u_z4c, host_u_z4c);

  if (verbose)
    std::cout << "Data copied." << std::endl;

  // TODO(JMF): Add magnetic fields
  auto &b0 = pmbp->pmhd->b0;
  par_for(
      "pgen_Bfc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        b0.x1f(m, k, j, i) = 0.0;
        b0.x2f(m, k, j, i) = 0.0;
        b0.x3f(m, k, j, i) = 0.0;

        if (i == ie) {
          b0.x1f(m, k, j, i + 1) = 0.0;
        }
        if (j == je) {
          b0.x2f(m, k, j + 1, i) = 0.0;
        }
        if (k == ke) {
          b0.x3f(m, k + 1, j, i) = 0.0;
        }
      });

  if (verbose)
    std::cout << "Face-centered fields zeroed." << std::endl;

  // Compute cell-centered fields
  auto &bcc0 = pmbp->pmhd->bcc0;
  par_for(
      "pgen_bcc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        bcc0(m, IBX, k, j, i) =
            0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i + 1));
        bcc0(m, IBY, k, j, i) =
            0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j + 1, i));
        bcc0(m, IBZ, k, j, i) =
            0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k + 1, j, i));
      });

  if (verbose)
    std::cout << "Cell-centered fields calculated." << std::endl;

  pmbp->pdyngr->PrimToConInit(0, (ncells1 - 1), 0, (ncells2 - 1), 0,
                              (ncells3 - 1));
  switch (indcs.ng) {
  case 2:
    pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
    break;
  case 3:
    pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
    break;
  case 4:
    pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
    break;
  }

  return;
}

// History function
void SGRIDHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  pdata->nhist = 2;
  pdata->label[0] = "rho-max";
  pdata->label[1] = "alpha-min";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int nx1 = indcs.nx1;
  int js = indcs.js;
  int nx2 = indcs.nx2;
  int ks = indcs.ks;
  int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack) * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  Real rho_max = std::numeric_limits<Real>::max();
  Real alpha_min = -rho_max;
  Kokkos::parallel_reduce(
      "BNSHistSums", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &mb_max, Real &mb_alp_min) {
        // coompute n,k,j,i indices of thread
        int m = (idx) / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;

        mb_max = fmax(mb_max, w0_(m, IDN, k, j, i));
        mb_alp_min = fmin(mb_alp_min, adm.alpha(m, k, j, i));
      },
      Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we
  // need MPI_MAX and MPI_MIN operations instead. This is a cheap hack to make
  // it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&rho_max, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    rho_max = 0.;
    alpha_min = 0.;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
}

void SGRIDRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn void DNS_init_sgrid(ParameterInput *pin)
//  \brief Initialize libsgrid: alloc mem, build the command and read checkpoint
//  This code is adapted from W.Tichy NMESH example
void DNS_init_sgrid(ParameterInput *pin) {
  const int level_l = 0;
  const int myrank = global_variable::my_rank;

  const bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);
  std::string const sgrid_datadir_const = pin->GetString("problem", "datadir");
  const bool keep_sgrid_output =
      pin->GetOrAddBoolean("problem", "keep_sgrid_output", false);
  const bool Interpolate_verbose =
      pin->GetOrAddBoolean("problem", "Interpolate_verbose", false);
  const bool Interpolate_make_finer_grid2 =
      pin->GetOrAddBoolean("problem", "Interpolate_make_finer_grid2", false);
  const Real Interpolate_max_xyz_diff =
      pin->GetOrAddReal("problem", "Interpolate_max_xyz_diff", 0.0);
  std::string const outdir = pin->GetOrAddString("problem", "outdir", "SGRID");

  char *sgrid_datadir =
      reinterpret_cast<char *>(malloc(sgrid_datadir_const.length() + 1));
  std::strcpy(sgrid_datadir, sgrid_datadir_const.c_str()); // NOLINT

  char command[STRLEN + 65676];
  char sgrid_exe[] = "sgrid"; // name is not important
  char sgridoutdir[STRLEN], sgridoutdir_previous[STRLEN];
  char sgridcheckpoint_indir[STRLEN];
  char sgridparfile[STRLEN];
  char *stringptr;

  // initialize file names
  // std::sprintf(gridfile, "%s/grid_level_%d_proc_%d.dat", outdir, level_l,
  // MPIrank);
  std::snprintf(sgridoutdir, STRLEN - 1, "%s/sgrid_level_%d_proc_%d",
                outdir.c_str(), level_l, myrank);
  std::snprintf(sgridoutdir_previous, STRLEN - 1,
                "%s/sgrid_level_%d_proc_%d_previous", outdir.c_str(), level_l,
                myrank);
  std::snprintf(sgridcheckpoint_indir, STRLEN - 1, "%s", sgrid_datadir);
  stringptr = strrchr(sgrid_datadir, '/'); // find last /
  if (stringptr == NULL) { // no / found in DNSdataReader_sgrid_datadir
    std::snprintf(sgridparfile, STRLEN - 1, "%s.par", sgrid_datadir);
  } else {
    std::snprintf(sgridparfile, STRLEN - 1, "%s%s", stringptr + 1, ".par");
  }

  // IMPORTANT: Put sgrid in a mode where it does not free its memory before
  // returning from libsgrid_main. So later we need to explicitly call
  //   SGRID_free_everything();
  // Done in UserWorkAfterLoop
  SGRID_memory_persists = 1;

  // init sgrid if needed, so that we can call funcs in it
  if (!SGRID_grid_exists()) {
    if (verbose)
      std::printf("Init sgrid\n");

    // call sgrid without running interpolator
    std::snprintf(command,
                  STRLEN + 65675, "%s %s/%s "
                             "--modify-par:BNSdata_Interpolate_pointsfile=%s "
                             "--modify-par:BNSdata_Interpolate_output=%s "
                             "--modify-par:outdir=%s "
                             "--modify-par:checkpoint_indir=%s",
                  sgrid_exe, sgrid_datadir, sgridparfile, "****NONE****",
                  "<NONE>", sgridoutdir, sgridcheckpoint_indir);

    // low verbosity
    std::strcat(command, " --modify-par:Coordinates_set_bfaces=no" // NOLINT
                         " --modify-par:verbose=no"
                         " --modify-par:Coordinates_verbose=no");

    // add other pars
    if (Interpolate_verbose)
      std::strcat(command, // NOLINT
                  " --modify-par:BNSdata_Interpolate_verbose=yes");
    if (Interpolate_max_xyz_diff > 0.0) {
      char str[STRLEN];
      std::snprintf(str, STRLEN - 1,
                    " --modify-par:BNSdata_Interpolate_max_xyz_diff=%g",
                    Interpolate_max_xyz_diff);
      std::strcat(command, str); // NOLINT
    }
    if (!Interpolate_make_finer_grid2)
      std::strcat( // NOLINT
          command,
          " --modify-par:BNSdata_Interpolate_make_finer_grid2_forXYZguess=no");
    if (!keep_sgrid_output)
      std::strcat(command, " > /dev/null"); // NOLINT

    int ret = DNS_call_sgrid(command);
    if (verbose)
      std::printf("DNS_call_sgrid returned: %d\n", ret);
  }
}

//----------------------------------------------------------------------------------------
//! \fn int DNS_call_sgrid(const char *command)
//  \brief Utility for SGRID DNS files: call libsgrid
//  This code is minimally changed from W.Tichy NMESH example
int DNS_call_sgrid(const char *command) {
  char *com =
      strdup(command); /* duplicate since construct_argv modifies its args */
  char **argv;
  int argc, status = -911;

  // cleanup in case we have called this already before
  if (SGRID_grid_exists())
    SGRID_free_everything();

  std::printf("calling libsgrid_main with these arguments:\n%s\n", command);

  // argc = construct_argv(com, &argv);
  argc = SGRID_construct_argv(com, &argv);

  status = libsgrid_main(argc, argv);

  if (status != 0) {
    std::printf("WARNING: Return value = %d\n", status);
  }

  free(argv); // free since construct_argv allocates argv
  free(com);

  return status;
}

//----------------------------------------------------------------------------------------
//! \fn int DNS_parameters(ParameterInput *pin)
//  \brief Utility for SGRID DNS files: read SGRID BNSdata_properties.txt and
//  get pars This code is minimally changed from W.Tichy NMESH example
int DNS_parameters(ParameterInput *pin) {
  bool verbose = pin->GetOrAddBoolean("problem", "verbose", 0);

  FILE *fp1;
  char str[STRLEN];
  char strn[STRLEN], strrho0[STRLEN], strkappa[STRLEN];
  char EoS_type[STRLEN], EoS_file[STRLEN];
  char datadir[STRLEN];

  // put empty string in some strings
  strn[0] = strrho0[0] = strkappa[0] = EoS_type[0] = EoS_file[0] = 0;

  // Get datadir and remove any trailing "/"
  std::snprintf(datadir, STRLEN - 1, "%s",
                pin->GetString("problem", "datadir").c_str());
  int j = strlen(datadir);
  if (datadir[j - 1] == '/') {
    datadir[j - 1] = 0;
    pin->SetString("problem", "datadir", datadir);
  }
  std::strcat(datadir, "/BNSdata_properties.txt"); // NOLINT

  //
  // Open file
  //

  fp1 = fopen(datadir, "r");
  if (fp1 == NULL) {
    std::cout << "### FATAL ERROR datadir: " << datadir << " "
              << " could not be accessed." << std::endl;
    exit(EXIT_FAILURE);
  }

  // move fp1 to place where time = 0 is
  j = DNS_position_fileptr_after_str(fp1, "NS data properties (time = 0):\n");
  if (j == EOF) {
    std::cout << "### FATAL ERROR could not find (time = 0) in: " << datadir
              << std::endl;
    exit(EXIT_FAILURE);
  }

  //
  // Get SGRID pars
  //

  // EOS
  Real ret = SGRID_fgetparameter(fp1, "EoS_type", EoS_type);
  if (ret == EOF) {
    // if we can't find EoS_type default to PwP
    std::snprintf(EoS_type, STRLEN - 1, "%s", "PwP");
    rewind(fp1);
    j = DNS_position_fileptr_after_str(fp1, "NS data properties (time = 0):\n");
    if (verbose)
      std::printf("Cannot find EoS, use default ...\n");
  }
  if (verbose)
    std::printf("EoS_type = %s\n", EoS_type);

  // Check if we need to read piecewise poly (PwP) pars
  if ((strcmp(EoS_type, "PwP") == 0) || (strcmp(EoS_type, "pwp") == 0)) {
    SGRID_fgotonext(fp1, "n_list");
    SGRID_fscanline(fp1, strn);
    for (j = 0; strn[j] == ' ' || strn[j] == '\t'; j++) {
    }
    if (j)
      std::memmove(strn, strn + j, strlen(strn) + 1);

    SGRID_fgotonext(fp1, "rho0_list");
    SGRID_fscanline(fp1, strrho0);
    for (j = 0; strrho0[j] == ' ' || strrho0[j] == '\t'; j++) {
    }
    if (j)
      std::memmove(strrho0, strrho0 + j, strlen(strrho0) + 1);

    SGRID_fgetparameter(fp1, "kappa", strkappa);

    if (verbose) {
      std::printf("initial data uses PwP EoS with:\n");
      std::printf("n_list    = %s\n", strn);
      std::printf("rho0_list = %s\n", strrho0);
      std::printf("kappa     = %s\n", strkappa);
      std::printf("Note: n_list contains the polytropic indices n,\n"
                  "      compute each Gamma using:  Gamma = 1 + 1/n\n");
    }
  }

  // Check if EoS is in sgrid table
  if (strcmp(EoS_type, "tab1d_AtT0") == 0) {
    SGRID_fgetparameter(fp1, "EoS_file", EoS_file);
    if (verbose) {
      std::printf("initial data uses T=0 EoS table:\n");
      std::printf("EoS_file = %s\n", EoS_file);
    }
  }

  // Other parameters
  SGRID_fgetparameter(fp1, "x_CM", str);
  Real sgrid_x_CM = atof(str);
  SGRID_fgetparameter(fp1, "Omega", str);
  Real Omega = atof(str);
  SGRID_fgetparameter(fp1, "ecc", str);
  Real ecc = atof(str);
  SGRID_fgetparameter(fp1, "rdot", str);
  Real rdot = atof(str);
  SGRID_fgetparameter(fp1, "m01", str);
  Real m01 = atof(str);
  SGRID_fgetparameter(fp1, "m02", str);
  Real m02 = atof(str);

  // Shift xmax1/2 such that CM is at 0, also read qmax1/2
  SGRID_fgetparameter(fp1, "xin1", str);
  Real xin1 = atof(str) - sgrid_x_CM;
  SGRID_fgetparameter(fp1, "xmax1", str);
  Real xmax1 = atof(str) - sgrid_x_CM;
  SGRID_fgetparameter(fp1, "xout1", str);
  Real xout1 = atof(str) - sgrid_x_CM;
  SGRID_fgetparameter(fp1, "qmax1", str);
  Real qmax1 = atof(str);
  SGRID_fgetparameter(fp1, "xin2", str);
  Real xin2 = atof(str) - sgrid_x_CM;
  SGRID_fgetparameter(fp1, "xmax2", str);
  Real xmax2 = atof(str) - sgrid_x_CM;
  SGRID_fgetparameter(fp1, "xout2", str);
  Real xout2 = atof(str) - sgrid_x_CM;
  SGRID_fgetparameter(fp1, "qmax2", str);
  Real qmax2 = atof(str);

  //
  // Set AthenaK parameters for later
  //
  pin->SetReal("problem", "x_CM", sgrid_x_CM);
  pin->SetReal("problem", "Omega", Omega);
  pin->SetReal("problem", "ecc", ecc);
  pin->SetReal("problem", "rdot", rdot);
  pin->SetReal("problem", "m01", m01);
  pin->SetReal("problem", "m02", m02);

  pin->SetReal("problem", "xin1", xin1);
  pin->SetReal("problem", "xmax1", xmax1);
  pin->SetReal("problem", "qmax1", qmax1);
  pin->SetReal("problem", "xin2", xin2);
  pin->SetReal("problem", "xmax2", xmax2);
  pin->SetReal("problem", "xout2", xout2);
  pin->SetReal("problem", "qmax2", qmax2);

  pin->SetReal("problem", "center1_mass", m01);
  pin->SetReal("problem", "center2_mass", m02);
  pin->SetReal("problem", "center0_x", 0.);
  pin->SetReal("problem", "center0_y", 0.);
  pin->SetReal("problem", "center0_z", 0.);
  pin->SetReal("problem", "center1_x", xmax1);
  pin->SetReal("problem", "center1_y", 0.);
  pin->SetReal("problem", "center1_z", 0.);
  pin->SetReal("problem", "center2_x", xmax2);
  pin->SetReal("problem", "center2_y", 0.);
  pin->SetReal("problem", "center2_z", 0.);

  //
  // Close file
  //
  fclose(fp1);

  if (verbose) {
    printf("Done with reading SGRID parameters:\n");
    printf("Omega = %g\n", Omega);
    printf("ecc = %g\n", ecc);
    printf("rdot = %g\n", rdot);
    printf("m01 = %g\n", m01);
    printf("m02 = %g\n", m02);
    printf("sgrid_x_CM = %g\n", sgrid_x_CM);
    printf("xmax1 - sgrid_x_CM = %g\n", xmax1);
    printf("xmax2 - sgrid_x_CM = %g\n", xmax2);
    printf("Make sure to center the mesh on the latter two!!!\n");
  }

  return 0;
}

//----------------------------------------------------------------------------------------
//! \fn int DNS_position_fileptr_after_str(FILE *in, const char *str)
//  \brief Utility for SGRID DNS files: position filepointer after the string
//  str This code is minimally changed from W.Tichy NMESH example
int DNS_position_fileptr_after_str(FILE *in, const char *str) {
  char line[STRLEN];
  while (fgets(line, STRLEN - 1, in) != NULL) {
    if (strstr(line, str) != NULL)
      return 1; // break;
  }
  return EOF;
}

} // namespace
