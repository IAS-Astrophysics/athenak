//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ti.cpp
//! \brief Problem generator for nonlinear thermal instability

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/ismcooling.hpp"
#include "globals.hpp"
#include "units/units.hpp"

#include "turb_init.hpp"

#include <Kokkos_Random.hpp>

namespace {
struct pgen_ti {
  int ndiag;
  Real t_cold;
  Real t_warm;
  Real t_hot;
};
  pgen_ti pti;

void AddUserSrcs(Mesh *pm, const Real bdt);
void Diagnostic(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void UserHistOutput(HistoryData *pdata, Mesh *pm);
void LoadData(Mesh *pm, ParameterInput *pin);
void FineToCoarse(Mesh *pm, ParameterInput *pin);
void CoarseToFine(Mesh *pm, ParameterInput *pin);
} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Problem Generator for nonlinear thermal instability

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_srcs_func = AddUserSrcs;
  user_hist_func = UserHistOutput;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Thermal instability problem generator can only be run with Hydro and/or MHD, "
       << "but no <hydro> or <mhd> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb1 = (pmbp->nmb_thispack-1);

  // Get temperature in Kelvin
  Real temp_cgs = pin->GetOrAddReal("problem","temp",1.0);
  Real temp = temp_cgs/pmbp->punit->temperature_cgs();
  Real hrate = pin->GetOrAddReal("hydro","hrate",2.0e-26);
  pti.t_cold = pin->GetOrAddReal("problem","t_cold",2.58);
  pti.t_warm = pin->GetOrAddReal("problem","t_warm",71.0);
  pti.t_hot = pin->GetOrAddReal("problem","t_hot",2.81e2);

  pti.ndiag = pin->GetOrAddInteger("problem","ndiag",-1);

  bool turb = pin->GetOrAddBoolean("problem","turb",false);
  Real turb_amp = pin->GetOrAddReal("problem","turb_amp",0.0);

  // Find the equilibrium point of the cooling curve by n*Lambda-Gamma=0
  Real number_density=hrate/ISMCoolFn(temp_cgs);
  Real rho_0 = number_density*pmbp->punit->mu()*
               pmbp->punit->atomic_mass_unit_cgs/pmbp->punit->density_cgs();
  Real cs_iso = std::sqrt(temp);

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    Real gm1 = eos.gamma - 1.0;
    Real pgas_0 = rho_0*cs_iso*cs_iso;
    Real cs = std::sqrt(eos.gamma*pgas_0/rho_0);
    int &nhydro = pmbp->phydro->nhydro;
    int &nscalars = pmbp->phydro->nscalars;

    // Print info
    if (global_variable::my_rank == 0) {
      std::cout << "============== Check Initialization ===============" << std::endl;
      std::cout << "  rho_0 (code) = " << rho_0 << std::endl;
      std::cout << "  sound speed (code) = " << cs << std::endl;
      std::cout << "  temperature (code) = " << SQR(cs_iso) << std::endl;
      std::cout << "  mu = " << pmbp->punit->mu() << std::endl;
      std::cout << "  temperature (c.g.s) = " << temp_cgs << std::endl;
      std::cout << "  cooling function (c.g.s) = " << ISMCoolFn(temp_cgs) << std::endl;
      //std::cout << "  turb_init = " << turb_init << std::endl;
    }
    // End print info

    if (restart) return;

    // Set initial conditions
    par_for("pgen_thermal_instability", DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = rho_0;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      if (eos.is_ideal) {
        u0(m,IEN,k,j,i) = pgas_0/gm1;
      }
      // add passive scalars
      for (int n=nhydro; n<(nhydro+nscalars); ++n) {
        u0(m,n,k,j,i) = 0.0;
      }
    });
    // TODO(@mhguo): write a reasonable initial perturbation
    if (turb) {
      Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
      par_for("pgen_ti_turb", DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto rand_gen = rand_pool64.get_state();  // get random number state this thread
        Real dens = u0(m,IDN,k,j,i);
        u0(m,IM1,k,j,i) += dens*turb_amp*(rand_gen.frand() - 0.5);
        u0(m,IM2,k,j,i) += dens*turb_amp*(rand_gen.frand() - 0.5);
        u0(m,IM3,k,j,i) += dens*turb_amp*(rand_gen.frand() - 0.5);
        rand_pool64.free_state(rand_gen);  // free state for use by other threads
      });
    }
    //if (turb_init) {
    //  pturb->InitializeModes(1);
    //  pturb->AddForcing(1);
    //  delete pturb;
    //  turb_init = false;
    //}
    for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
      if (it->block_name.compare(0, 9, "turb_init") == 0) {
        TurbulenceInit *pturb;
        pturb = new TurbulenceInit(it->block_name,pmbp, pin);
        pturb->InitializeModes(1);
        pturb->AddForcing(1);
        delete pturb;
      }
    }
  }

  // TODO(@mhguo): read data from restart file and load from finer level to coarser level
  bool rst_flag = pin->GetOrAddBoolean("problem","rst",false);
  //int rst_level = pin->GetOrAddInteger("problem", "rst_level", 0);
  if (rst_flag) {
    int rst_type = pin->GetOrAddInteger("problem","rst_type",0);
    if (rst_type==0) {
      LoadData(pmy_mesh_, pin);
    }
    else if (rst_type==1) {
      CoarseToFine(pmy_mesh_, pin);
    } else {
      FineToCoarse(pmy_mesh_, pin);
    }
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::ISMCoolingNewTimeStep()
//! \brief Compute new time step for ISM cooling.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void Diagnostic(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  MeshBlockPack *pmbp = pm->pmb_pack;
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
//! \fn void AddUserSrcs()
//! \brief Add User Source Terms
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void AddUserSrcs(Mesh *pm, const Real bdt) {
  DvceArray5D<Real> &u0 = pm->pmb_pack->phydro->u0;
  const DvceArray5D<Real> &w0 = pm->pmb_pack->phydro->w0;
  const EOS_Data &eos_data = pm->pmb_pack->phydro->peos->eos_data;
  if (pti.ndiag>0 && pm->ncycle % pti.ndiag == 0) {
    Diagnostic(pm,bdt,u0,w0,eos_data);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn UserHistOutput
//! \brief Sets user-defined history output

void UserHistOutput(HistoryData *pdata, Mesh *pm) {
  int n0 = pdata->nhist;
  int nuser = 9;
  pdata->nhist += nuser;
  pdata->label[n0+0] = "E-int";
  pdata->label[n0+1] = "Vcnm ";
  pdata->label[n0+2] = "Vunm ";
  pdata->label[n0+3] = "Vwnm ";
  pdata->label[n0+4] = "Vhot ";
  pdata->label[n0+5] = "Mcnm ";
  pdata->label[n0+6] = "Munm ";
  pdata->label[n0+7] = "Vwnm ";
  pdata->label[n0+8] = "Mhot ";
  //pdata->label[n0+1] = "Mdot ";
  EOS_Data &eos = pm->pmb_pack->phydro->peos->eos_data;
  Real use_e = eos.use_e;
  Real gm1 = eos.gamma - 1.0;
  Real t_cold = pti.t_cold;
  Real t_warm = pti.t_warm;
  Real t_hot = pti.t_hot;
  // capture class variabels for kernel
  //auto &u0_ = pm->pmb_pack->phydro->u0;
  auto &w0 = pm->pmb_pack->phydro->w0;
  auto &size = pm->pmb_pack->pmb->mb_size;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;
  // store data into hdata array
  for (int n=0; n<NREDUCTION_VARIABLES; ++n) {
    sum_this_mb.the_array[n] = 0.0;
  }
  Kokkos::parallel_reduce("UserHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    Real dens = w0(m,IDN,k,j,i);
    //Real momsqr = SQR(u0_(m,IM1,k,j,i))+SQR(u0_(m,IM2,k,j,i))+SQR(u0_(m,IM3,k,j,i));
    //Real temp = gm1 * (u0_(m,IEN,k,j,i) - momsqr/u0_(m,IDN,k,j,i)) / u0_(m,IDN,k,j,i);
    Real temp = 0.0;
    Real eint = 0.0;
    if (use_e) {
      temp = w0(m,IEN,k,j,i)/dens*gm1;
      eint = w0(m,IEN,k,j,i);
    } else {
      temp = w0(m,ITM,k,j,i);
      eint = w0(m,ITM,k,j,i)*dens/gm1;
    }

    Real dv_cnm  = (temp<t_cold)? vol : 0.0;
    Real dv_unm  = (temp>=t_cold && temp<t_warm)? vol : 0.0;
    Real dv_wnm  = (temp>=t_warm && temp<t_hot)? vol : 0.0;
    Real dv_hot  = (temp>=t_hot)? vol : 0.0;
    Real dm_cnm = dv_cnm*dens;
    Real dm_unm = dv_unm*dens;
    Real dm_wnm = dv_wnm*dens;
    Real dm_hot  = dv_hot*dens;

    // Hydro conserved variables:
    array_sum::GlobalSum hvars;
    hvars.the_array[0] = eint*vol;
    hvars.the_array[1] = dv_cnm;
    hvars.the_array[2] = dv_unm;
    hvars.the_array[3] = dv_wnm;
    hvars.the_array[4] = dv_hot;
    hvars.the_array[5] = dm_cnm;
    hvars.the_array[6] = dm_unm;
    hvars.the_array[7] = dm_wnm;
    hvars.the_array[8] = dm_hot;

    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nuser; n<NREDUCTION_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));

  // store data into hdata array
  for (int n=0; n<nuser; ++n) {
    pdata->hdata[n0+n] = sum_this_mb.the_array[n];
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
    int nhydro = phydro->nhydro;
    int nscalars = pinput->GetOrAddInteger("hydro","nscalars",0);
    nhydro_tot = nhydro + nscalars;
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
    auto u0_slice = Kokkos::subview(host_u0, Kokkos::ALL, std::make_pair(0,nhydro_tot),
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    auto hst_slice = Kokkos::subview(ccin, Kokkos::ALL, std::make_pair(0,nhydro_tot),
                                     Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(u0_slice, hst_slice);
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
  int rst_c2f = pin->GetOrAddInteger("problem", "rst_c2f", 0);
  if (rst_c2f==0) {
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
      for (int n=0; n<(nhydro+nscalars); ++n) {
        u0(m,n,k,j,i) = coarse_u0(cm,n,ck,cj,ci);
      }
    });
  } else {
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
      Real sk = ((k-ks)%2>0) ? 1.0:-1.0;
      Real sj = ((j-js)%2>0) ? 1.0:-1.0;
      Real si = ((i-is)%2>0) ? 1.0:-1.0;
      for (int n=0; n<(nhydro+nscalars); ++n) {
        int cn = n;
        // calculate gradient using the min-mod limiter
        Real dl = coarse_u0(cm,cn,ck,cj,ci  ) - coarse_u0(cm,cn,ck,cj,ci-1);
        Real dr = coarse_u0(cm,cn,ck,cj,ci+1) - coarse_u0(cm,cn,ck,cj,ci  );
        Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

        dl = coarse_u0(cm,cn,ck,cj  ,ci) - coarse_u0(cm,cn,ck,cj-1,ci);
        dr = coarse_u0(cm,cn,ck,cj+1,ci) - coarse_u0(cm,cn,ck,cj  ,ci);
        Real dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

        dl = coarse_u0(cm,cn,ck  ,cj,ci) - coarse_u0(cm,cn,ck-1,cj,ci);
        dr = coarse_u0(cm,cn,ck+1,cj,ci) - coarse_u0(cm,cn,ck  ,cj,ci);
        Real dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

        u0(m,n,k,j,i) = coarse_u0(cm,n,ck,cj,ci) + si*dvar1 + sj*dvar2 + sk*dvar3;
      }
    });
  }
  
  resfile.Close();
  delete pinput;
  return;
}

} // namespace
