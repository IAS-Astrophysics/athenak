//========================================================================================
// Athena++ astrophysical MHD code, Kokkos version
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file snr.cpp
//! \brief Problem generator for supernova blast wave (supernova remnant).
//!
//! REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//!   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

#include <algorithm>
#include <cmath>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/ismcooling.hpp"
#include "globals.hpp"
#include "units/units.hpp"

namespace {
struct pgen_snr {
  int ndiag;
  Real t_cold;
  Real t_warm;
  Real t_hot;
  Real v_shell;
  Real v_bubble;
};
  pgen_snr psnr;

void Diagnostic(Mesh *pm, const Real bdt);
void UserHistOutput(HistoryData *pdata, Mesh *pm);
} // namespace


//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_srcs_func = Diagnostic;
  user_hist_func = UserHistOutput;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  Real rout = pin->GetReal("problem", "radius");
  Real rin  = rout - pin->GetOrAddReal("problem", "ramp", 0.0);
  //Real pa   = pin->GetOrAddReal("problem", "pamb", 1.0);
  //Real da   = pin->GetOrAddReal("problem", "damb", 1.0);
  //Real prat = pin->GetReal("problem", "prat");
  //Real drat = pin->GetOrAddReal("problem", "drat", 1.0);

  Real tamb   = pin->GetOrAddReal("problem", "tamb", 1.0);
  Real damb   = pin->GetOrAddReal("problem", "damb", 1.0);

  Real mass_snr = pin->GetOrAddReal("problem", "mass_snr", 1.0);
  Real etot_snr = pin->GetOrAddReal("problem", "etot_snr", 1.0);
  Real rout_snr = rout;
  psnr.ndiag = pin->GetOrAddInteger("problem","ndiag",-1);

  // Get temperature in Kelvin
  //Real temp_0 = pin->GetOrAddReal("problem","temp",1.0);
  Real hrate = pin->GetOrAddReal("hydro","hrate",2.0e-26);
  Real t_cold = psnr.t_cold = pin->GetOrAddReal("problem","t_cold",2.58);
  Real t_warm = psnr.t_warm = pin->GetOrAddReal("problem","t_warm",71.0);
  Real t_hot = psnr.t_hot = pin->GetOrAddReal("problem","t_hot",2.81e2);
  psnr.v_shell = pin->GetOrAddReal("problem","v_shell",1.0);
  psnr.v_bubble = pin->GetOrAddReal("problem","v_bubble",psnr.v_shell);

  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie; int nx1 = indcs.nx1;
  int &js = indcs.js; int &je = indcs.je; int nx2 = indcs.nx2;
  int &ks = indcs.ks; int &ke = indcs.ke; int nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  
  // sum the volume of the cells of SNR
  // 3 cell is the minimum
  Real m0 = 0.0, m1 = 0.0;
  Kokkos::parallel_reduce("pgen_snr_sum",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum_m0, Real &sum_m1) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    if (rad < rout) {
      sum_m0 += vol;
    }
  }, Kokkos::Sum<Real>(m0), Kokkos::Sum<Real>(m1));

#if MPI_PARALLEL_ENABLED
    Real m_sum2[2] = {m0,m1};
    Real gm_sum2[2];
    MPI_Allreduce(m_sum2, gm_sum2, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    m0 = gm_sum2[0];
    m1 = gm_sum2[1];
#endif

  Real vol_snr  = m0;
  Real dens_snr = mass_snr/vol_snr;
  Real eint_snr = etot_snr/vol_snr;

  // Print info
  if (global_variable::my_rank == 0) {
    std::cout << "============== Check Initialization ===============" << std::endl;
    //std::cout << "  rho_eq (code) = " << rho_eq << std::endl;
    std::cout << "  rout_snr = " << rout_snr << std::endl;
    std::cout << "  vol_snr  = " << vol_snr  << std::endl;
    std::cout << "  dens_snr = " << dens_snr << std::endl;
    std::cout << "  eint_snr = " << eint_snr << std::endl;
    //std::cout << "  isothermal sound speed (code) = " << cs_iso << std::endl;
    std::cout << "  mu = " << pmbp->punit->mu() << std::endl;
    //std::cout << "  temperature (c.g.s) = " << temp_0 << std::endl;
    //std::cout << "  cooling function (c.g.s) = " << ISMCoolFn(temp_0) << std::endl;
    std::cout << "  hrate = " << hrate << std::endl;
    std::cout << "  t_cold = " << t_cold << std::endl;
    std::cout << "  t_warm = " << t_warm << std::endl;
    std::cout << "  t_hot = " << t_hot << std::endl;
    std::cout << "  v_shell = " << psnr.v_shell << std::endl;
    std::cout << "  v_bubble = " << psnr.v_bubble << std::endl;
    std::cout << "  user_hist = " << user_hist << std::endl;
  }
  // End print info

  // TODO(@mhguo): set snr if it is the fisrt restarting, else return
  // if (restart) return;

  // setup uniform ambient medium with spherical over-pressured region in Hydro
  if (pmbp->phydro != nullptr) {
    auto &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    //bool use_e = eos.use_e;
    Real pres_snr = gm1*eint_snr;
    Real temp_snr = pres_snr/dens_snr;
    auto &w0 = pmbp->phydro->w0;
    auto &u0 = pmbp->phydro->u0;
    int &nhydro = pmbp->phydro->nhydro;
    int &nscalars = pmbp->phydro->nscalars;
    par_for("pgen_snr", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

      Real rad = std::sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

      if (rad < rout) {
        Real den = dens_snr;
        Real pres = pres_snr;
        //if (rad < rin) {
        //  den = drat*da;
        //  pres = prat*pa;
        //} else {   // add smooth ramp in density
        //  Real f = (rad-rin) / (rout-rin);
        //  Real log_den = (1.0-f) * log(drat*da) + f * log(da);
        //  den = exp(log_den);
        //  Real log_pres = (1.0-f) * log(prat*pa) + f * log(pa);
        //  pres = exp(log_pres);
        //}
        Real prim;
        prim = pres/gm1;
        /*if (eos.use_e) {
          prim = pres/(eos.gamma - 1.0);
        } else {
          prim = pres/((eos.gamma - 1.0)*den);
        }*/
        u0(m,IDN,k,j,i) = den;
        u0(m,IM1,k,j,i) = 0.0;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        u0(m,IEN,k,j,i) = prim;
        // add passive scalars
        if (nscalars > 3) u0(m,nhydro+3,k,j,i) = u0(m,IDN,k,j,i);
      } else {
        if (!restart) {
          u0(m,IDN,k,j,i) = damb;
          u0(m,IM1,k,j,i) = 0.0;
          u0(m,IM2,k,j,i) = 0.0;
          u0(m,IM3,k,j,i) = 0.0;
          u0(m,IEN,k,j,i) = damb*tamb/gm1;
        }
        // add passive scalars
        if (nscalars > 1) {
          Real dens = u0(m,IDN,k,j,i);
          Real e_k = 0.5*(SQR(u0(m,IM1,k,j,i))+SQR(u0(m,IM2,k,j,i))+SQR(u0(m,IM3,k,j,i)))
                    /dens;
          Real temp = gm1*(u0(m,IEN,k,j,i) - e_k)/dens;
          if (temp < t_cold) {
            u0(m,nhydro,k,j,i) = dens;
          } else if (temp < t_warm) {
            u0(m,nhydro+1,k,j,i) = dens;
          } else {
            u0(m,nhydro+2,k,j,i) = dens;
          }
        }
      }


      // set either internal energy density or temparature as primitive
      /*
      Real prim;
      if (eos.use_e) {
        prim = pres/(eos.gamma - 1.0);
      } else {
        prim = pres/((eos.gamma - 1.0)*den);
      }

      w0(m,IDN,k,j,i) = den;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IEN,k,j,i) = prim;
      */
    });

    // Convert primitives to conserved
    // auto &u0 = pmbp->phydro->u0;
    // pmbp->phydro->peos->PrimToCons(w0, u0);
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void Diagnostic()
//! \brief Diagnostics.

void Diagnostic(Mesh *pm, const Real bdt) {
  if (psnr.ndiag<=0 || pm->ncycle % psnr.ndiag != 0) return;
  MeshBlockPack *pmbp = pm->pmb_pack;
  DvceArray5D<Real> &u0 = pmbp->phydro->u0;
  const DvceArray5D<Real> &w0 = pmbp->phydro->w0;
  const EOS_Data &eos_data = pmbp->phydro->peos->eos_data;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  Real use_e = eos_data.use_e;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;

  Real dtnew = std::numeric_limits<Real>::max();

  Real min_dens = std::numeric_limits<Real>::max();
  Real min_vtot = std::numeric_limits<Real>::max();
  Real min_temp = std::numeric_limits<Real>::max();
  Real min_eint = std::numeric_limits<Real>::max();
  Real max_dens = std::numeric_limits<Real>::min();
  Real max_vtot = std::numeric_limits<Real>::min();
  Real max_temp = std::numeric_limits<Real>::min();
  Real max_eint = std::numeric_limits<Real>::min();

  // find smallest (e/cooling_rate) in each cell
  Kokkos::parallel_reduce("diagnostic", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt, Real &min_d, Real &min_v, Real &min_t,
  Real &min_e, Real &max_d, Real &max_v, Real &max_t, Real &max_e) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real dx = fmin(fmin(size.d_view(m).dx1,size.d_view(m).dx2),size.d_view(m).dx3);

    // temperature in cgs unit
    Real temp = 1.0;
    Real eint = 1.0;
    if (use_e) {
      temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      eint = w0(m,IEN,k,j,i);
    } else {
      temp = w0(m,ITM,k,j,i);
      eint = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i)/gm1;
    }

    Real vtot = sqrt(SQR(w0(m,IVX,k,j,i))+SQR(w0(m,IVY,k,j,i))+SQR(w0(m,IVZ,k,j,i)));
    min_dt = fmin(dx/sqrt(gamma*temp), min_dt);
    min_d = fmin(w0(m,IDN,k,j,i), min_d);
    min_v = fmin(vtot,min_v);
    min_t = fmin(temp, min_t);
    min_e = fmin(eint, min_e);
    max_d = fmax(w0(m,IDN,k,j,i), max_d);
    max_v = fmax(vtot,max_v);
    max_t = fmax(temp, max_t);
    max_e = fmax(eint, max_e);
  }, Kokkos::Min<Real>(dtnew),
     Kokkos::Min<Real>(min_dens),
     Kokkos::Min<Real>(min_vtot),
     Kokkos::Min<Real>(min_temp),
     Kokkos::Min<Real>(min_eint),
     Kokkos::Max<Real>(max_dens),
     Kokkos::Max<Real>(max_vtot),
     Kokkos::Max<Real>(max_temp),
     Kokkos::Max<Real>(max_eint));
#if MPI_PARALLEL_ENABLED
  Real m_min[5] = {dtnew,min_dens,min_vtot,min_temp,min_eint};
  Real m_max[4] = {max_dens,max_vtot,max_temp,max_eint};
  Real gm_min[5];
  Real gm_max[4];
  //MPI_Allreduce(MPI_IN_PLACE, &dtnew, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(m_min, gm_min, 5, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(m_max, gm_max, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  dtnew = gm_min[0];
  min_dens = gm_min[1];
  min_vtot = gm_min[2];
  min_temp = gm_min[3];
  min_eint = gm_min[4];
  max_dens = gm_max[0];
  max_vtot = gm_max[1];
  max_temp = gm_max[2];
  max_eint = gm_max[3];
#endif
  if (global_variable::my_rank == 0) {
    std::cout << " min_d=" << min_dens << " max_d=" << max_dens << std::endl
              << " min_v=" << min_vtot << " max_v=" << max_vtot << std::endl
              << " min_t=" << min_temp << " max_t=" << max_temp << std::endl
              << " min_e=" << min_eint << " max_e=" << max_eint << std::endl
              << " dt_cs=" << dtnew << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void UserHistOutput()
//! \brief Sets user-defined history output

void UserHistOutput(HistoryData *pdata, Mesh *pm) {
  int n0 = pdata->nhist;
  int nuser = 65;
  pdata->nhist += nuser;
  const char *data_label[nuser] = {
    "Vcnm",  "Vunm",  "Vwnm",  "Vhot",  "Vsh",
    "Mcnm",  "Munm",  "Mwnm",  "Mhot",  "Msh",
    "eicnm", "eiunm", "eiwnm", "eihot", "eish",
    "ekcnm", "ekunm", "ekwnm", "ekhot", "eksh",
    "ecnm",  "eunm",  "ewnm",  "ehot",  "esh",
    "prcnm", "prunm", "prwnm", "prhot", "prsh",
    "MRsh",  "Ms0",   "Ms1",   "Ms2",   "Ms3",
    "Mcs0",  "Mcs1",  "Mcs2",  "Mcs3",  "Mus0",
    "Mus1",  "Mus2",  "Mus3",  "Mws0",  "Mws1",
    "Mws2",  "Mws3",  "Mhs0",  "Mhs1",  "Mhs2",
    "Mhs3",  "Mshs0", "Mshs1", "Mshs2", "Mshs3",
    "Vhv",   "Mhv",   "eihv",  "ekhv",  "ehv",
    "prhv",  "Mhvs0", "Mhvs1", "Mhvs2", "Mhvs3",
  };
  for (int n=0; n<nuser; ++n) {
    pdata->label[n0+n] = data_label[n];
  }
  //pdata->label[n0+1] = "Mdot ";
  EOS_Data &eos = pm->pmb_pack->phydro->peos->eos_data;
  Real use_e = eos.use_e;
  Real gm1 = eos.gamma - 1.0;
  Real t_cold = psnr.t_cold;
  Real t_warm = psnr.t_warm;
  Real t_hot = psnr.t_hot;
  Real v_shell = psnr.v_shell;
  Real v_bubble = psnr.v_bubble;
  // capture class variabels for kernel
  auto &u0 = pm->pmb_pack->phydro->u0;
  auto &w0 = pm->pmb_pack->phydro->w0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhydro = pm->pmb_pack->phydro->nhydro;
  int &nscalars = pm->pmb_pack->phydro->nscalars;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb0;
  array_sum::GlobalSum sum_this_mb1;
  array_sum::GlobalSum sum_this_mb2;
  array_sum::GlobalSum sum_this_mb3;
  // set sum
  for (int n=0; n<NREDUCTION_VARIABLES; ++n) {
    sum_this_mb0.the_array[n] = 0.0;
    sum_this_mb1.the_array[n] = 0.0;
    sum_this_mb2.the_array[n] = 0.0;
    sum_this_mb3.the_array[n] = 0.0;
  }
  Kokkos::parallel_reduce("UserHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum0,
  array_sum::GlobalSum &mb_sum1, array_sum::GlobalSum &mb_sum2,
  array_sum::GlobalSum &mb_sum3) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));

    Real dens = w0(m,IDN,k,j,i);
    Real temp = 0.0;
    Real eint = 0.0;
    if (use_e) {
      temp = w0(m,IEN,k,j,i)/dens*gm1;
      eint = w0(m,IEN,k,j,i);
    } else {
      temp = w0(m,ITM,k,j,i);
      eint = w0(m,ITM,k,j,i)*dens/gm1;
    }
    Real etot = u0(m,IEN,k,j,i);
    Real vtot = sqrt(SQR(w0(m,IVX,k,j,i))+SQR(w0(m,IVY,k,j,i))+SQR(w0(m,IVZ,k,j,i)));
    Real ek = 0.5*dens*(SQR(w0(m,IVX,k,j,i))+SQR(w0(m,IVY,k,j,i))+SQR(w0(m,IVZ,k,j,i)));
    Real velr = (w0(m,IVX,k,j,i)*x1v+w0(m,IVY,k,j,i)*x2v+w0(m,IVZ,k,j,i)*x3v)/rad;

    Real dv_cnm  = (temp<t_cold)? vol : 0.0;
    Real dv_unm  = (temp>=t_cold && temp<t_warm)? vol : 0.0;
    Real dv_wnm  = (temp>=t_warm && temp<t_hot)? vol : 0.0;
    Real dv_hot  = (temp>=t_hot)? vol : 0.0;
    Real dv_sh   = (temp<t_hot && velr>v_shell)? vol : 0.0;
    Real dv_hv   = (vtot>v_bubble)? vol : 0.0;
    Real dm_cnm  = dv_cnm*dens;
    Real dm_unm  = dv_unm*dens;
    Real dm_wnm  = dv_wnm*dens;
    Real dm_hot  = dv_hot*dens;
    Real dm_sh   = dv_sh*dens;
    Real dm_hv   = dv_hv*dens;
    Real dei_cnm = dv_cnm*eint;
    Real dei_unm = dv_unm*eint;
    Real dei_wnm = dv_wnm*eint;
    Real dei_hot = dv_hot*eint;
    Real dei_sh  = dv_sh*eint;
    Real dei_hv  = dv_hv*eint;
    Real dek_cnm = dv_cnm*ek;
    Real dek_unm = dv_unm*ek;
    Real dek_wnm = dv_wnm*ek;
    Real dek_hot = dv_hot*ek;
    Real dek_sh  = dv_sh*ek;
    Real dek_hv  = dv_hv*ek;
    Real de_cnm  = dv_cnm*etot;
    Real de_unm  = dv_unm*etot;
    Real de_wnm  = dv_wnm*etot;
    Real de_hot  = dv_hot*etot;
    Real de_sh   = dv_sh*etot;
    Real de_hv   = dv_hv*etot;
    Real dpr_cnm = dm_cnm*velr;
    Real dpr_unm = dm_unm*velr;
    Real dpr_wnm = dm_wnm*velr;
    Real dpr_hot = dm_hot*velr;
    Real dpr_sh  = dm_sh*velr;
    Real dpr_hv  = dm_hv*velr;
    Real dmr_sh  = dm_sh*rad;

    Real vars[nuser] = {
      dv_cnm,  dv_unm,  dv_wnm,  dv_hot,  dv_sh,
      dm_cnm,  dm_unm,  dm_wnm,  dm_hot,  dm_sh,
      dei_cnm, dei_unm, dei_wnm, dei_hot, dei_sh,
      dek_cnm, dek_unm, dek_wnm, dek_hot, dek_sh,
      de_cnm,  de_unm,  de_wnm,  de_hot,  de_sh,
      dpr_cnm, dpr_unm, dpr_wnm, dpr_hot, dpr_sh,
      dmr_sh,  0.0,     0.0,     0.0,     0.0,
      0.0,     0.0,     0.0,     0.0,     0.0,
      0.0,     0.0,     0.0,     0.0,     0.0,
      0.0,     0.0,     0.0,     0.0,     0.0,
      0.0,     0.0,     0.0,     0.0,     0.0,
      dv_hv,   dm_hv,   dei_hv,  dek_hv,  de_hv,
      dpr_hv,  0.0,     0.0,     0.0,     0.0,
    };
    if (nscalars > 3) {
      for (int n=0; n<4; ++n) {
        vars[31+n] = vol*dens*w0(m,nhydro+n,k,j,i);
        for (int nm=0; nm<5; ++nm){
          vars[35+4*nm+n] = vars[5+nm]*w0(m,nhydro+n,k,j,i);
        }
        vars[61+n] = dm_hv*w0(m,nhydro+n,k,j,i);
      }
    }
    // Hydro conserved variables:
    array_sum::GlobalSum hvars0;
    array_sum::GlobalSum hvars1;
    array_sum::GlobalSum hvars2;
    array_sum::GlobalSum hvars3;
    for (int n=0; n<NREDUCTION_VARIABLES; ++n) {
      hvars0.the_array[n] = vars[n];
      hvars1.the_array[n] = vars[n+NREDUCTION_VARIABLES];
      hvars2.the_array[n] = vars[n+2*NREDUCTION_VARIABLES];
    }
    for (int n=0; n<nuser-3*NREDUCTION_VARIABLES; ++n) {
      hvars3.the_array[n] = vars[n+3*NREDUCTION_VARIABLES];
    }

    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nuser-3*NREDUCTION_VARIABLES; n<NREDUCTION_VARIABLES; ++n) {
      hvars3.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum0 += hvars0;
    mb_sum1 += hvars1;
    mb_sum2 += hvars2;
    mb_sum3 += hvars3;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb0),
     Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb1),
     Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb2),
     Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb3));

  // store data into hdata array
  for (int n=0; n<NREDUCTION_VARIABLES; ++n) {
    pdata->hdata[n0+n] = sum_this_mb0.the_array[n];
    pdata->hdata[NREDUCTION_VARIABLES+n0+n] = sum_this_mb1.the_array[n];
    pdata->hdata[2*NREDUCTION_VARIABLES+n0+n] = sum_this_mb2.the_array[n];
  }
  for (int n=0; n<nuser-3*NREDUCTION_VARIABLES; ++n) {
    pdata->hdata[3*NREDUCTION_VARIABLES+n0+n] = sum_this_mb3.the_array[n];
  }

  return;
}

} // namespace
