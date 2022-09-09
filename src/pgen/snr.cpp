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
#include "eos/ideal_c2p_hyd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/cell_locations.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/ismcooling.hpp"
#include "globals.hpp"
#include "units/units.hpp"

namespace {
struct pgen_snr {
  int ndiag;
  Real t_cold;
  Real t_warm;
  Real t_ion;
  Real t_hot;
  Real v_shell;
  Real v_bubble;
  bool cooling;
  Real hrate;
};
  pgen_snr psnr;

void AddUserSrcs(Mesh *pm, const Real bdt);
void UserHistOutput(HistoryData *pdata, Mesh *pm);
void Diagnostic(Mesh *pm, const Real bdt);
void AddISMCooling(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void LoadData(Mesh *pm, ParameterInput *pin);
void FineToCoarse(Mesh *pm, ParameterInput *pin);
void CoarseToFine(Mesh *pm, ParameterInput *pin);
} // namespace


//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Problem Generator for spherical blast problem

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_srcs_func = AddUserSrcs;
  user_hist_func = UserHistOutput;
  bool add_snr = pin->GetOrAddBoolean("problem", "add_snr", true);
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
  Real t_ion = psnr.t_ion = pin->GetOrAddReal("problem","t_ion",2.814e2);
  Real t_hot = psnr.t_hot = pin->GetOrAddReal("problem","t_hot",1.407e4);
  psnr.v_shell = pin->GetOrAddReal("problem","v_shell",1.0);
  psnr.v_bubble = pin->GetOrAddReal("problem","v_bubble",psnr.v_shell);
  psnr.cooling = pin->GetOrAddBoolean("problem","cooling",false);
  psnr.hrate = pin->GetOrAddReal("problem","hrate",2.0e-26);

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
    std::cout << "  user cooling = " << psnr.cooling << std::endl;
    std::cout << "  user hrate = " << psnr.hrate << std::endl;
    std::cout << "  t_cold = " << t_cold << std::endl;
    std::cout << "  t_warm = " << t_warm << std::endl;
    std::cout << "  t_ion = " << t_ion << std::endl;
    std::cout << "  t_hot = " << t_hot << std::endl;
    std::cout << "  v_shell = " << psnr.v_shell << std::endl;
    std::cout << "  v_bubble = " << psnr.v_bubble << std::endl;
    std::cout << "  user_hist = " << user_hist << std::endl;
  }
  // End print info

  bool rst_flag = pin->GetOrAddBoolean("problem","rst",false);
  //int rst_level = pin->GetOrAddInteger("problem", "rst_level", 0);
  if (rst_flag) {
    int rst_type = pin->GetOrAddInteger("problem","rst_type",0);
    if (rst_type==0) {
      LoadData(pmy_mesh_, pin);
    } else if (rst_type==1) {
      CoarseToFine(pmy_mesh_, pin);
    } else {
    }
  }

  // TODO(@mhguo): set snr if it is the fisrt restarting, else return
  // if (restart) return;
  if (!add_snr) return;

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
        if (add_snr) {
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
//! \fn void AddUserSrcs()
//! \brief Add User Source Terms
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void AddUserSrcs(Mesh *pm, const Real bdt) {
  DvceArray5D<Real> &u0 = pm->pmb_pack->phydro->u0;
  const DvceArray5D<Real> &w0 = pm->pmb_pack->phydro->w0;
  const EOS_Data &eos_data = pm->pmb_pack->phydro->peos->eos_data;
  if (psnr.cooling) {
    //std::cout << "AddISMCooling" << std::endl;
    AddISMCooling(pm,bdt,u0,w0,eos_data);
  }
  Diagnostic(pm,bdt);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::AddISMCooling()
//! \brief Add explict ISM cooling and heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void AddISMCooling(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  auto pmy_pack = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real beta = bdt/pm->dt;
  Real cfl_no = pm->cfl_no;
  auto &eos = eos_data;
  Real use_e = eos_data.use_e;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real heating_rate = psnr.hrate;
  Real temp_unit = pmy_pack->punit->temperature_cgs();
  Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                /pmy_pack->punit->atomic_mass_unit_cgs;
  Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()/n_unit;

  int nsubcycle=0, nsubcycle_count=0;
  Kokkos::parallel_reduce("cooling", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sum0, int &sum1) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real dens=1.0, temp = 1.0, eint = 1.0;
    dens = w0(m,IDN,k,j,i);
    if (use_e) {
      temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
      eint = w0(m,IEN,k,j,i);
    } else {
      temp = w0(m,ITM,k,j,i);
      eint = w0(m,ITM,k,j,i)*w0(m,IDN,k,j,i)/gm1;
    }

    Real gamma_heating = heating_rate/heating_unit;
    bool sub_cycling = true;
    bool sub_cycling_used = false;
    Real bdt_now = 0.0;
    while (sub_cycling) {
      Real lambda_cooling = ISMCoolFn(temp*temp_unit)/cooling_unit;
      Real cooling_heating =  dens * (dens * lambda_cooling - gamma_heating);
      Real dt_cool = (eint/(FLT_MIN + fabs(cooling_heating)));
      Real bdt_cool = beta*cfl_no*dt_cool;
      if (bdt_now+bdt_cool<bdt) {
        u0(m,IEN,k,j,i) -= bdt_cool * cooling_heating;

        // compute new temperature and internal energy
        
        // load single state conserved variables
        HydCons1D u;
        u.d  = u0(m,IDN,k,j,i);
        u.mx = u0(m,IM1,k,j,i);
        u.my = u0(m,IM2,k,j,i);
        u.mz = u0(m,IM3,k,j,i);
        u.e  = u0(m,IEN,k,j,i);

        // call c2p function
        // (inline function in ideal_c2p_hyd.hpp file)
        HydPrim1D w;
        bool dfloor_used=false, efloor_used=false, tfloor_used=false;
        SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used);
        dens = w.d;
        temp = gm1*w.e/w.d;
        eint = w.e;
        sub_cycling_used = true;
        sum1++;
      } else {
        u0(m,IEN,k,j,i) -= (bdt-bdt_now) * cooling_heating;
        sub_cycling = false;
      }
      bdt_now += bdt_cool;
    }
    if (sub_cycling_used) {
      sum0++;
    }
  }, Kokkos::Sum<int>(nsubcycle), Kokkos::Sum<int>(nsubcycle_count));
#if MPI_PARALLEL_ENABLED
  int* pnsubcycle = &(nsubcycle);
  int* pnsubcycle_count = &(nsubcycle_count);
  MPI_Allreduce(MPI_IN_PLACE, pnsubcycle, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pnsubcycle_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    if (psnr.ndiag>0 && pm->ncycle % psnr.ndiag == 0) {
      if (nsubcycle>0 || nsubcycle_count >0) {
        std::cout << " nsubcycle_cell=" << nsubcycle << std::endl
                  << " nsubcycle_count=" << nsubcycle_count << std::endl;
      }
    }
  }
  return;
}

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
  Real dt_hyd  = pmbp->phydro->dtnew;
  Real dt_cond = pmbp->phydro->pcond->dtnew;
  Real dt_src  = pmbp->phydro->psrc->dtnew;
#if MPI_PARALLEL_ENABLED
  Real m_min[8] = {dtnew,min_dens,min_vtot,min_temp,min_eint,dt_hyd,dt_cond,dt_src};
  Real m_max[4] = {max_dens,max_vtot,max_temp,max_eint};
  Real gm_min[8];
  Real gm_max[4];
  //MPI_Allreduce(MPI_IN_PLACE, &dtnew, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(m_min, gm_min, 8, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(m_max, gm_max, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  dtnew = gm_min[0];
  min_dens = gm_min[1];
  min_vtot = gm_min[2];
  min_temp = gm_min[3];
  min_eint = gm_min[4];
  dt_hyd   = gm_min[5];
  dt_cond  = gm_min[6];
  dt_src   = gm_min[7];
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
              << " dt_temp=" << dtnew   << " dt_hyd=" << dt_hyd << std::endl
              << " dt_cond=" << dt_cond << " dt_src=" << dt_src << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void UserHistOutput()
//! \brief Sets user-defined history output

void UserHistOutput(HistoryData *pdata, Mesh *pm) {
  int n0 = pdata->nhist;
  const int nuser = 75;
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
    "Vsb",   "Msb",   "eisb",  "eksb",  "esb",
    "prsb",  "Msbs0", "Msbs1", "Msbs2", "Msbs3",
    "Vig",   "Mig",   "eiig",  "ekig",  "eig",
    "prig",  "Migs0", "Migs1", "Migs2", "Migs3",
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
  Real t_ion = psnr.t_ion;
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
    Real dv_wnm  = (temp>=t_warm && temp<t_ion)? vol : 0.0;
    Real dv_ig   = (temp>=t_ion && temp<t_hot)? vol : 0.0;
    Real dv_hot  = (temp>=t_hot)? vol : 0.0;
    Real dv_sh   = (temp<t_hot && velr>v_shell)? vol : 0.0;
    Real dv_sb   = (temp>=t_hot || vtot>v_bubble)? vol : 0.0;
    Real dm_cnm  = dv_cnm*dens;
    Real dm_unm  = dv_unm*dens;
    Real dm_wnm  = dv_wnm*dens;
    Real dm_ig   = dv_ig*dens;
    Real dm_hot  = dv_hot*dens;
    Real dm_sh   = dv_sh*dens;
    Real dm_sb   = dv_sb*dens;
    Real dei_cnm = dv_cnm*eint;
    Real dei_unm = dv_unm*eint;
    Real dei_wnm = dv_wnm*eint;
    Real dei_ig  = dv_ig*eint;
    Real dei_hot = dv_hot*eint;
    Real dei_sh  = dv_sh*eint;
    Real dei_sb  = dv_sb*eint;
    Real dek_cnm = dv_cnm*ek;
    Real dek_unm = dv_unm*ek;
    Real dek_wnm = dv_wnm*ek;
    Real dek_ig  = dv_ig*ek;
    Real dek_hot = dv_hot*ek;
    Real dek_sh  = dv_sh*ek;
    Real dek_sb  = dv_sb*ek;
    Real de_cnm  = dv_cnm*etot;
    Real de_unm  = dv_unm*etot;
    Real de_wnm  = dv_wnm*etot;
    Real de_ig   = dv_ig*etot;
    Real de_hot  = dv_hot*etot;
    Real de_sh   = dv_sh*etot;
    Real de_sb   = dv_sb*etot;
    Real dpr_cnm = dm_cnm*velr;
    Real dpr_unm = dm_unm*velr;
    Real dpr_wnm = dm_wnm*velr;
    Real dpr_ig  = dm_ig*velr;
    Real dpr_hot = dm_hot*velr;
    Real dpr_sh  = dm_sh*velr;
    Real dpr_sb  = dm_sb*velr;
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
      dv_sb,   dm_sb,   dei_sb,  dek_sb,  de_sb,
      dpr_sb,  0.0,     0.0,     0.0,     0.0,
      dv_ig,   dm_ig,   dei_ig,  dek_ig,  de_ig,
      dpr_ig,  0.0,     0.0,     0.0,     0.0,
    };
    if (nscalars > 3) {
      for (int n=0; n<4; ++n) {
        vars[31+n] = vol*dens*w0(m,nhydro+n,k,j,i);
        for (int nm=0; nm<5; ++nm){
          vars[35+4*nm+n] = vars[5+nm]*w0(m,nhydro+n,k,j,i);
        }
        vars[61+n] = dm_sb*w0(m,nhydro+n,k,j,i);
        vars[71+n] = dm_ig*w0(m,nhydro+n,k,j,i);
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

void LoadData(Mesh *pm, ParameterInput *pin) {
  std::string rst_file = pin->GetOrAddString("problem", "rst_file", "none");
  MeshBlockPack *pmbp = pm->pmb_pack;
  ParameterInput* pinput = new ParameterInput;
  IOWrapper resfile;

  //--- STEP 1.  Root process reads header data (input file, critical variables)

  resfile.Open(rst_file.c_str(), IOWrapper::FileMode::read);
  pinput->LoadFromFile(resfile);

  // capture variables for kernel
  auto &indcs = pm->mb_indcs;
  // get spatial dimensions of arrays, including ghost zones
  int nmb = pmbp->nmb_thispack;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  //--- STEP 2.  Root process reads list of logical locations and cost of MeshBlocks
  // Similar to data read in Mesh::BuildTreeFromRestart()

  // At this point, the restartfile is already open and the ParameterInput (input file)
  // data has already been read. Thus the file pointer is set to after <par_end>
  IOWrapperSizeT headeroffset = resfile.GetPosition();

  // following must be identical to calculation of headeroffset (excluding size of
  // ParameterInput data) in restart.cpp
  IOWrapperSizeT headersize = 3*sizeof(int) + 2*sizeof(Real)
    + sizeof(RegionSize) + 2*sizeof(RegionIndcs);
  char *headerdata = new char[headersize];

  if (global_variable::my_rank == 0) { // the master process reads the header data
    if (resfile.Read_bytes(headerdata, 1, headersize) != headersize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Header size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

#if MPI_PARALLEL_ENABLED
  // then broadcast the header data
  MPI_Bcast(headerdata, headersize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // get old mesh data, time and cycle, actually useless here
  // Now copy mesh data read from restart file into Mesh variables. Order of variables
  // set by Write()'s in restart.cpp
  // Note this overwrites size and indices initialized in Mesh constructor.
  IOWrapperSizeT hdos = 0;
  int nmb_tot = 0; 
  std::memcpy(&nmb_tot, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  //std::memcpy(&root_level, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  //std::memcpy(&mesh_size, &(headerdata[hdos]), sizeof(RegionSize));
  hdos += sizeof(RegionSize);
  //std::memcpy(&mesh_indcs, &(headerdata[hdos]), sizeof(RegionIndcs));
  hdos += sizeof(RegionIndcs);
  //std::memcpy(&mb_indcs, &(headerdata[hdos]), sizeof(RegionIndcs));
  hdos += sizeof(RegionIndcs);
  //std::memcpy(&time, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  //std::memcpy(&dt, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  //std::memcpy(&ncycle, &(headerdata[hdos]), sizeof(int));
  delete [] headerdata;

  // allocate idlist buffer and read list of logical locations and cost
  IOWrapperSizeT listsize = sizeof(LogicalLocation) + sizeof(float);
  char *idlist = new char[listsize*nmb_tot];
  if (global_variable::my_rank == 0) { // only the master process reads the ID list
    if (resfile.Read_bytes(idlist,listsize,nmb_tot) !=
        static_cast<unsigned int>(nmb_tot)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Incorrect number of MeshBlocks in restart file; "
                << "restart file is broken." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  //--- STEP 3.  All ranks read data over all MeshBlocks (5D arrays) in parallel
  // Similar to data read in ProblemGenerator constructor for restarts
  // Only work for hydro

  // root process reads size of CC and FC data arrays from restart file
  IOWrapperSizeT variablesize = 2*sizeof(IOWrapperSizeT);
  char *variabledata = new char[variablesize];
  if (global_variable::my_rank == 0) { // the master process reads the variables data
    if (resfile.Read_bytes(variabledata, 1, variablesize) != variablesize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Variable data size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  // then broadcast the datasize information
  MPI_Bcast(variabledata, variablesize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // Read number of CC variables and FC fields per MeshBlock in restart file
  IOWrapperSizeT ccdata_cnt, fcdata_cnt;
  hdos = 0;
  std::memcpy(&ccdata_cnt, &(variabledata[hdos]), sizeof(IOWrapperSizeT));
  hdos += sizeof(IOWrapperSizeT);
  std::memcpy(&fcdata_cnt, &(variabledata[hdos]), sizeof(IOWrapperSizeT));
  
  // calculate total number of CC variables
  hydro::Hydro* phydro = pmbp->phydro;
  int nhydro_tot = 0, nmhd_tot = 0;
  if (phydro != nullptr) {
    nhydro_tot = phydro->nhydro + phydro->nscalars;
  }

  // master process gets file offset
  if (global_variable::my_rank == 0) {
    headeroffset = resfile.GetPosition();
  }
#if MPI_PARALLEL_ENABLED
  // then broadcasts it
  MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // allocate arrays for CC data
  HostArray5D<Real> ccin("pgen-ccin", nmb, (nhydro_tot + nmhd_tot), nout3, nout2, nout1);
  if (ccin.size() != (nmb*ccdata_cnt)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CC data size read from restart file not equal to size "
              << "of Hydro and MHD arrays, restart file is broken." << std::endl;
    exit(EXIT_FAILURE);
  }

  // calculate max/min number of MeshBlocks across all ranks
  int noutmbs_max = pm->nmblist[0];
  int noutmbs_min = pm->nmblist[0];
  for (int i=0; i<(global_variable::nranks); ++i) {
    noutmbs_max = std::max(noutmbs_max,pm->nmblist[i]);
    noutmbs_min = std::min(noutmbs_min,pm->nmblist[i]);
  }

  // read CC data into host array, one MeshBlock at a time to avoid exceeding 2^31 limit
  // on each read call for very large grids per MPI rank
  int mygids = pm->gidslist[global_variable::my_rank];
  IOWrapperSizeT myoffset = headeroffset + (ccdata_cnt+fcdata_cnt)*mygids*sizeof(Real);
  for (int m=0;  m<noutmbs_max; ++m) {
    // every rank has a MB to read, so read collectively
    if (m < noutmbs_min) {
      // get ptr to cell-centered MeshBlock data
      auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      int mbcnt = mbptr.size();
      if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset) != mbcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "CC data not read correctly from restart file, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
      myoffset += mbcnt*sizeof(Real);

    // some ranks are finished writing, so use non-collective write
    } else if (m < pm->nmb_thisrank) {
      // get ptr to MeshBlock data
      auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      int mbcnt = mbptr.size();
      if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset) != mbcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "CC data not read correctly from restart file, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
      myoffset += mbcnt*sizeof(Real);
    }
  }

  // copy CC Hydro data to device
  if (phydro != nullptr) {
    DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(phydro->u0);
    auto hst_slice = Kokkos::subview(ccin, Kokkos::ALL, std::make_pair(0,nhydro_tot),
                                      Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(host_u0, hst_slice);
    Kokkos::deep_copy(phydro->u0, host_u0);
  }

  resfile.Close();
  delete pinput;
  return;
}

void FineToCoarse(Mesh *pm, ParameterInput *pin) {
  std::string rst_file = pin->GetOrAddString("problem", "rst_file", "none");
  MeshBlockPack *pmbp = pm->pmb_pack;
  ParameterInput* pinput = new ParameterInput;
  IOWrapper resfile;

  //--- STEP 1.  Root process reads header data (input file, critical variables)

  resfile.Open(rst_file.c_str(), IOWrapper::FileMode::read);
  pinput->LoadFromFile(resfile);

  // capture variables for kernel
  auto &indcs = pm->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb1 = (pmbp->nmb_thispack-1);
  // get spatial dimensions of arrays, including ghost zones
  int nmb = pmbp->nmb_thispack;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  //--- STEP 2.  Root process reads list of logical locations and cost of MeshBlocks
  // Similar to data read in Mesh::BuildTreeFromRestart()

  // At this point, the restartfile is already open and the ParameterInput (input file)
  // data has already been read. Thus the file pointer is set to after <par_end>
  IOWrapperSizeT headeroffset = resfile.GetPosition();

  // following must be identical to calculation of headeroffset (excluding size of
  // ParameterInput data) in restart.cpp
  IOWrapperSizeT headersize = 3*sizeof(int) + 2*sizeof(Real)
    + sizeof(RegionSize) + 2*sizeof(RegionIndcs);
  char *headerdata = new char[headersize];

  if (global_variable::my_rank == 0) { // the master process reads the header data
    if (resfile.Read_bytes(headerdata, 1, headersize) != headersize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Header size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

#if MPI_PARALLEL_ENABLED
  // then broadcast the header data
  MPI_Bcast(headerdata, headersize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // get old mesh data, time and cycle, actually useless here
  // Now copy mesh data read from restart file into Mesh variables. Order of variables
  // set by Write()'s in restart.cpp
  // Note this overwrites size and indices initialized in Mesh constructor.
  IOWrapperSizeT hdos = 0;
  int nmb_tot = 0; 
  std::memcpy(&nmb_tot, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  //std::memcpy(&root_level, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  //std::memcpy(&mesh_size, &(headerdata[hdos]), sizeof(RegionSize));
  hdos += sizeof(RegionSize);
  //std::memcpy(&mesh_indcs, &(headerdata[hdos]), sizeof(RegionIndcs));
  hdos += sizeof(RegionIndcs);
  //std::memcpy(&mb_indcs, &(headerdata[hdos]), sizeof(RegionIndcs));
  // TODO(@mhguo): consider whether you want old time, dt, and ncycle?
  hdos += sizeof(RegionIndcs);
  //std::memcpy(&time, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  //std::memcpy(&dt, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  //std::memcpy(&ncycle, &(headerdata[hdos]), sizeof(int));
  delete [] headerdata;

  // allocate idlist buffer and read list of logical locations and cost
  IOWrapperSizeT listsize = sizeof(LogicalLocation) + sizeof(float);
  // TODO(@mhguo): set right number!
  // TODO(@mhguo): consider whether you can utilize the idlist here
  char *idlist = new char[listsize*nmb_tot];
  if (global_variable::my_rank == 0) { // only the master process reads the ID list
    if (resfile.Read_bytes(idlist,listsize,nmb_tot) !=
        static_cast<unsigned int>(nmb_tot)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Incorrect number of MeshBlocks in restart file; "
                << "restart file is broken." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  //--- STEP 3.  All ranks read data over all MeshBlocks (5D arrays) in parallel
  // Similar to data read in ProblemGenerator constructor for restarts

  // root process reads size of CC and FC data arrays from restart file
  IOWrapperSizeT variablesize = 2*sizeof(IOWrapperSizeT);
  char *variabledata = new char[variablesize];
  if (global_variable::my_rank == 0) { // the master process reads the variables data
    if (resfile.Read_bytes(variabledata, 1, variablesize) != variablesize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Variable data size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  // then broadcast the datasize information
  MPI_Bcast(variabledata, variablesize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // Read number of CC variables and FC fields per MeshBlock in restart file
  IOWrapperSizeT ccdata_cnt, fcdata_cnt;
  hdos = 0;
  std::memcpy(&ccdata_cnt, &(variabledata[hdos]), sizeof(IOWrapperSizeT));
  hdos += sizeof(IOWrapperSizeT);
  std::memcpy(&fcdata_cnt, &(variabledata[hdos]), sizeof(IOWrapperSizeT));
  
  // calculate total number of CC variables
  hydro::Hydro* phydro = pmbp->phydro;
  mhd::MHD* pmhd = pmbp->pmhd;
  int nhydro_tot = 0, nmhd_tot = 0;
  if (phydro != nullptr) {
    nhydro_tot = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd_tot = pmhd->nmhd + pmhd->nscalars;
  }

  // master process gets file offset
  if (global_variable::my_rank == 0) {
    headeroffset = resfile.GetPosition();
  }
#if MPI_PARALLEL_ENABLED
  // then broadcasts it
  MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // allocate arrays for CC data
  HostArray5D<Real> ccin("pgen-ccin", nmb, (nhydro_tot + nmhd_tot), nout3, nout2, nout1);
  if (ccin.size() != (nmb*ccdata_cnt)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CC data size read from restart file not equal to size "
              << "of Hydro and MHD arrays, restart file is broken." << std::endl;
    exit(EXIT_FAILURE);
  }

  // calculate max/min number of MeshBlocks across all ranks
  int noutmbs_max = pm->nmblist[0];
  int noutmbs_min = pm->nmblist[0];
  for (int i=0; i<(global_variable::nranks); ++i) {
    noutmbs_max = std::max(noutmbs_max,pm->nmblist[i]);
    noutmbs_min = std::min(noutmbs_min,pm->nmblist[i]);
  }

  // allocate arrays for finer data
  DvceArray5D<Real> fine_u0;
  Kokkos::realloc(fine_u0, nmb, (nhydro_tot), nout3, nout2, nout1);

  // initialize
  auto &u0 = pmbp->phydro->u0;
  int &nhydro = pmbp->phydro->nhydro;
  int &nscalars = pmbp->phydro->nscalars;
  par_for("f2c_init", DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = 0.0;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = 0.0;
    // add passive scalars
    for (int n=nhydro; n<(nhydro+nscalars); ++n) {
      u0(m,n,k,j,i) = 0.0;
    }
  });

  // TODO(@mhguo): only work for nmb=1 now!
  int nitr = 8, k0=0, j0=0, i0=0;
  for (int itr=0; itr<nitr; itr++) {
    i0 = indcs.nx1/2*(itr%2);
    j0 = indcs.nx2/2*((itr/2)%2);
    k0 = indcs.nx3/2*((itr/4)%2);
    // read CC data into host array, one MeshBlock at a time to avoid exceeding 2^31 limit
    // on each read call for very large grids per MPI rank
    int mygids = pm->gidslist[global_variable::my_rank];
    int itrgids = mygids*nitr+itr;
    IOWrapperSizeT myoffset = headeroffset + (ccdata_cnt+fcdata_cnt)*itrgids*sizeof(Real);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      // get ptr to cell-centered MeshBlock data
      auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                  Kokkos::ALL);
      int mbcnt = mbptr.size();
      //if (resfile.Read_Reals_at_all(ccin.data(), ccdata_size, 1, myoffset) != 1) {
      if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset) != mbcnt) {
        
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "CC data not read correctly from restart file, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
      myoffset += mbcnt*sizeof(Real);
    }

    // copy CC Hydro data to device
    
    if (phydro != nullptr) {
      DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(fine_u0);
      auto hst_slice = Kokkos::subview(ccin, Kokkos::ALL, std::make_pair(0,nhydro_tot),
                                        Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
      Kokkos::deep_copy(host_u0, hst_slice);
      Kokkos::deep_copy(fine_u0, host_u0);
    }
    // Set initial conditions
    par_for("f2c_set", DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      int cm = m;
      int ck = (k-ks)/2+ks+k0;
      int cj = (j-js)/2+js+j0;
      int ci = (i-is)/2+is+i0;
      u0(cm,IDN,ck,cj,ci) += fine_u0(m,IDN,k,j,i)/nitr;
      u0(cm,IM1,ck,cj,ci) += fine_u0(m,IM1,k,j,i)/nitr;
      u0(cm,IM2,ck,cj,ci) += fine_u0(m,IM2,k,j,i)/nitr;
      u0(cm,IM3,ck,cj,ci) += fine_u0(m,IM3,k,j,i)/nitr;
      u0(cm,IEN,ck,cj,ci) += fine_u0(m,IEN,k,j,i)/nitr;
      // add passive scalars
      for (int n=nhydro; n<(nhydro+nscalars); ++n) {
        u0(cm,n,ck,cj,ci) += fine_u0(m,n,k,j,i)/nitr;
      }
    });
  }

  resfile.Close();
  delete pinput;
  return;
}

void CoarseToFine(Mesh *pm, ParameterInput *pin) {
  std::string rst_file = pin->GetOrAddString("problem", "rst_file", "none");
  MeshBlockPack *pmbp = pm->pmb_pack;
  ParameterInput* pinput = new ParameterInput;
  IOWrapper resfile;

  //--- STEP 1.  Root process reads header data (input file, critical variables)

  resfile.Open(rst_file.c_str(), IOWrapper::FileMode::read);
  pinput->LoadFromFile(resfile);

  // capture variables for kernel
  auto &indcs = pm->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb1 = (pmbp->nmb_thispack-1);
  // get spatial dimensions of arrays, including ghost zones
  int nmb = pmbp->nmb_thispack;
  int rst_n = 8;
  int rst_nmb = pin->GetOrAddInteger("problem", "rst_nmb", (nmb+rst_n-1)/rst_n);
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  //--- STEP 2.  Root process reads list of logical locations and cost of MeshBlocks
  // Similar to data read in Mesh::BuildTreeFromRestart()

  // At this point, the restartfile is already open and the ParameterInput (input file)
  // data has already been read. Thus the file pointer is set to after <par_end>
  IOWrapperSizeT headeroffset = resfile.GetPosition();

  // following must be identical to calculation of headeroffset (excluding size of
  // ParameterInput data) in restart.cpp
  IOWrapperSizeT headersize = 3*sizeof(int) + 2*sizeof(Real)
    + sizeof(RegionSize) + 2*sizeof(RegionIndcs);
  char *headerdata = new char[headersize];

  if (global_variable::my_rank == 0) { // the master process reads the header data
    if (resfile.Read_bytes(headerdata, 1, headersize) != headersize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Header size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

#if MPI_PARALLEL_ENABLED
  // then broadcast the header data
  MPI_Bcast(headerdata, headersize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // get old mesh data, time and cycle, actually useless here
  // Now copy mesh data read from restart file into Mesh variables. Order of variables
  // set by Write()'s in restart.cpp
  // Note this overwrites size and indices initialized in Mesh constructor.
  IOWrapperSizeT hdos = 0;
  int nmb_tot = 0; 
  std::memcpy(&nmb_tot, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  //std::memcpy(&root_level, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  //std::memcpy(&mesh_size, &(headerdata[hdos]), sizeof(RegionSize));
  hdos += sizeof(RegionSize);
  //std::memcpy(&mesh_indcs, &(headerdata[hdos]), sizeof(RegionIndcs));
  hdos += sizeof(RegionIndcs);
  //std::memcpy(&mb_indcs, &(headerdata[hdos]), sizeof(RegionIndcs));
  // TODO(@mhguo): consider whether you want old time, dt, and ncycle?
  hdos += sizeof(RegionIndcs);
  //std::memcpy(&time, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  //std::memcpy(&dt, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  //std::memcpy(&ncycle, &(headerdata[hdos]), sizeof(int));
  delete [] headerdata;

  // allocate idlist buffer and read list of logical locations and cost
  IOWrapperSizeT listsize = sizeof(LogicalLocation) + sizeof(float);
  // TODO(@mhguo): set right number!
  // TODO(@mhguo): consider whether you can utilize the idlist here
  char *idlist = new char[listsize*nmb_tot];
  if (global_variable::my_rank == 0) { // only the master process reads the ID list
    if (resfile.Read_bytes(idlist,listsize,nmb_tot) !=
        static_cast<unsigned int>(nmb_tot)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Incorrect number of MeshBlocks in restart file; "
                << "restart file is broken." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  //--- STEP 3.  All ranks read data over all MeshBlocks (5D arrays) in parallel
  // Similar to data read in ProblemGenerator constructor for restarts

  // root process reads size of CC and FC data arrays from restart file
  IOWrapperSizeT variablesize = 2*sizeof(IOWrapperSizeT);
  char *variabledata = new char[variablesize];
  if (global_variable::my_rank == 0) { // the master process reads the variables data
    if (resfile.Read_bytes(variabledata, 1, variablesize) != variablesize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Variable data size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  // then broadcast the datasize information
  MPI_Bcast(variabledata, variablesize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // Read number of CC variables and FC fields per MeshBlock in restart file
  IOWrapperSizeT ccdata_cnt, fcdata_cnt;
  hdos = 0;
  std::memcpy(&ccdata_cnt, &(variabledata[hdos]), sizeof(IOWrapperSizeT));
  hdos += sizeof(IOWrapperSizeT);
  std::memcpy(&fcdata_cnt, &(variabledata[hdos]), sizeof(IOWrapperSizeT));
  
  // calculate total number of CC variables
  hydro::Hydro* phydro = pmbp->phydro;
  mhd::MHD* pmhd = pmbp->pmhd;
  int nhydro_tot = 0, nmhd_tot = 0;
  if (phydro != nullptr) {
    nhydro_tot = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd_tot = pmhd->nmhd + pmhd->nscalars;
  }

  // master process gets file offset
  if (global_variable::my_rank == 0) {
    headeroffset = resfile.GetPosition();
  }
#if MPI_PARALLEL_ENABLED
  // then broadcasts it
  MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  // allocate arrays for CC data
  HostArray5D<Real> ccin("pgen-ccin", rst_nmb, (nhydro_tot + nmhd_tot), nout3, nout2,
                         nout1);
  if (ccin.size() != (rst_nmb*ccdata_cnt)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CC data size read from restart file not equal to size "
              << "of Hydro and MHD arrays, restart file is broken." << std::endl;
    exit(EXIT_FAILURE);
  }

  // calculate max/min number of MeshBlocks across all ranks
  int noutmbs_max = pm->nmblist[0];
  int noutmbs_min = pm->nmblist[0];
  for (int i=0; i<(global_variable::nranks); ++i) {
    noutmbs_max = std::max(noutmbs_max,pm->nmblist[i]);
    noutmbs_min = std::min(noutmbs_min,pm->nmblist[i]);
  }

  // allocate arrays for coarser data
  DvceArray5D<Real> coarse_u0;
  Kokkos::realloc(coarse_u0, rst_nmb, (nhydro_tot), nout3, nout2, nout1);

  // initialize
  auto &u0 = pmbp->phydro->u0;
  int &nhydro = pmbp->phydro->nhydro;
  int &nscalars = pmbp->phydro->nscalars;
  par_for("c2f_init", DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = 0.0;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = 0.0;
    // add passive scalars
    for (int n=nhydro; n<(nhydro+nscalars); ++n) {
      u0(m,n,k,j,i) = 0.0;
    }
  });

  // TODO(@mhguo): only work when MeshBlock is evenly divided!
  // read CC data into host array, one MeshBlock at a time to avoid exceeding 2^31 limit
  // on each read call for very large grids per MPI rank
  int mygids = pm->gidslist[global_variable::my_rank];
  int rstgids = mygids/rst_n;
  IOWrapperSizeT myoffset = headeroffset + (ccdata_cnt+fcdata_cnt)*rstgids*sizeof(Real);
  for (int m=0;  m<rst_nmb; ++m) {
    // every rank has a MB to read, so read collectively
    // get ptr to cell-centered MeshBlock data
    auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                Kokkos::ALL);
    int mbcnt = mbptr.size();
    //if (resfile.Read_Reals_at_all(ccin.data(), ccdata_size, 1, myoffset) != 1) {
    if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset) != mbcnt) {
      
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "CC data not read correctly from restart file, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
    myoffset += mbcnt*sizeof(Real);
  }

  // copy CC Hydro data to device
  //std::cout << "### STEP 3: copy CC Hydro data to device" << std::endl;
  
  if (phydro != nullptr) {
    DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(coarse_u0);
    auto hst_slice = Kokkos::subview(ccin, Kokkos::ALL, std::make_pair(0,nhydro_tot),
                                      Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(host_u0, hst_slice);
    Kokkos::deep_copy(coarse_u0, host_u0);
  }
  
  // Set initial conditions
  int idmy = mygids%rst_n;
  par_for("c2f_set", DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    int idnow = idmy + m;
    int i0 = indcs.nx1/2*(idnow%2);
    int j0 = indcs.nx2/2*((idnow/2)%2);
    int k0 = indcs.nx3/2*((idnow/4)%2);
    int cm = m/rst_n;
    int ck = (k-ks)/2+ks+k0;
    int cj = (j-js)/2+js+j0;
    int ci = (i-is)/2+is+i0;
    u0(m,IDN,k,j,i) = coarse_u0(cm,IDN,ck,cj,ci);
    u0(m,IM1,k,j,i) = coarse_u0(cm,IM1,ck,cj,ci);
    u0(m,IM2,k,j,i) = coarse_u0(cm,IM2,ck,cj,ci);
    u0(m,IM3,k,j,i) = coarse_u0(cm,IM3,ck,cj,ci);
    u0(m,IEN,k,j,i) = coarse_u0(cm,IEN,ck,cj,ci);
    // add passive scalars
    for (int n=nhydro; n<(nhydro+nscalars); ++n) {
      u0(m,n,k,j,i) += coarse_u0(cm,n,ck,cj,ci);
    }
  });

  resfile.Close();
  delete pinput;
  return;
}

} // namespace
