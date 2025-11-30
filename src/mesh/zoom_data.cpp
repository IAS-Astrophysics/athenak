//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_data.cpp
//  \brief implementation of constructor and functions in CyclicZoom class

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "mesh/cyclic_zoom.hpp"

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

ZoomData::ZoomData(CyclicZoom *pz) :
    pzoom(pz),
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    efld("efld",1,1,1,1),
    emf0("emf0",1,1,1,1),
    delta_efld("delta_efld",1,1,1,1),
    max_emf0("max_emf0",1,1)
  {
  // allocate memory for primitive variables
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &mzoom = pzoom->mzoom;
  int &nvars = pzoom->nvars;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(u0, mzoom, nvars, ncells3, ncells2, ncells1);
  Kokkos::realloc(w0, mzoom, nvars, ncells3, ncells2, ncells1);
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(coarse_u0, mzoom, nvars, n_ccells3, n_ccells2, n_ccells1);
  Kokkos::realloc(coarse_w0, mzoom, nvars, n_ccells3, n_ccells2, n_ccells1);

  // allocate electric fields
  Kokkos::realloc(efld.x1e, mzoom, n_ccells3+1, n_ccells2+1, n_ccells1);
  Kokkos::realloc(efld.x2e, mzoom, n_ccells3+1, n_ccells2, n_ccells1+1);
  Kokkos::realloc(efld.x3e, mzoom, n_ccells3, n_ccells2+1, n_ccells1+1);

  // allocate electric fields just after zoom
  Kokkos::realloc(emf0.x1e, mzoom, n_ccells3+1, n_ccells2+1, n_ccells1);
  Kokkos::realloc(emf0.x2e, mzoom, n_ccells3+1, n_ccells2, n_ccells1+1);
  Kokkos::realloc(emf0.x3e, mzoom, n_ccells3, n_ccells2+1, n_ccells1+1);

  // allocate delta electric fields
  Kokkos::realloc(delta_efld.x1e, mzoom, n_ccells3+1, n_ccells2+1, n_ccells1);
  Kokkos::realloc(delta_efld.x2e, mzoom, n_ccells3+1, n_ccells2, n_ccells1+1);
  Kokkos::realloc(delta_efld.x3e, mzoom, n_ccells3, n_ccells2+1, n_ccells1+1);

  Kokkos::realloc(max_emf0, mzoom, 3);
  for (int i = 0; i < mzoom; i++) {
    for (int j = 0; j < 3; j++) {
      max_emf0(i,j) = 0.0;
    }
  }

  Initialize();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::Initialize()
//! \brief Initialize ZoomData variables

void ZoomData::Initialize()
{
  auto &indcs = pzoom->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 0;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 0;
  int nc1 = indcs.cnx1 + 2*ng;
  int nc2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*ng) : 0;
  int nc3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*ng) : 0;
  int &mzoom = pzoom->mzoom;

  auto &u0_ = u0;
  auto &w0_ = w0;
  auto &cu0 = coarse_u0;
  auto &cw0 = coarse_w0;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto e01 = emf0.x1e;
  auto e02 = emf0.x2e;
  auto e03 = emf0.x3e;
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  bool is_mhd = (pzoom->pmesh->pmb_pack->pmhd != nullptr);
  auto peos = (is_mhd)? pzoom->pmesh->pmb_pack->pmhd->peos : pzoom->pmesh->pmb_pack->phydro->peos;
  Real gm1 = peos->eos_data.gamma - 1.0;

  Real d0 = pzoom->d_zoom;
  Real p0 = pzoom->p_zoom;

  par_for("zoom_init", DevExeSpace(),0,mzoom-1,0,n3-1,0,n2-1,0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0_(m,IDN,k,j,i) = d0;
    w0_(m,IM1,k,j,i) = 0.0;
    w0_(m,IM2,k,j,i) = 0.0;
    w0_(m,IM3,k,j,i) = 0.0;
    w0_(m,IEN,k,j,i) = p0/gm1;
  });

  par_for("zoom_init_c",DevExeSpace(),0,mzoom-1,0,nc3-1,0,nc2-1,0,nc1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    cw0(m,IDN,k,j,i) = d0;
    cw0(m,IM1,k,j,i) = 0.0;
    cw0(m,IM2,k,j,i) = 0.0;
    cw0(m,IM3,k,j,i) = 0.0;
    cw0(m,IEN,k,j,i) = p0/gm1;
  });

  // In MHD, we don't use conserved variables so no need to convert
  // if (!is_mhd) {
  //   peos->PrimToCons(w0_,u0_,0,n3-1,0,n2-1,0,n1-1);
  //   peos->PrimToCons(cw0,cu0,0,nc3-1,0,nc2-1,0,nc1-1);
  // }

  par_for("zoom_init_e1",DevExeSpace(),0,mzoom-1,0,nc3,0,nc2,0,nc1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e1(m,k,j,i) = 0.0;
    e01(m,k,j,i) = 0.0;
    de1(m,k,j,i) = 0.0;
  });
  par_for("zoom_init_e2",DevExeSpace(),0,mzoom-1,0,nc3,0,nc2-1,0,nc1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e2(m,k,j,i) = 0.0;
    e02(m,k,j,i) = 0.0;
    de2(m,k,j,i) = 0.0;
  });
  par_for("zoom_init_e3",DevExeSpace(),0,mzoom-1,0,nc3-1,0,nc2,0,nc1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e3(m,k,j,i) = 0.0;
    e03(m,k,j,i) = 0.0;
    de3(m,k,j,i) = 0.0;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ZoomData::DumpData()
//! \brief dump zoom data to file

// TODO: dumping on a single rank now, should consider parallel dumping
void ZoomData::DumpData() {
  if (global_variable::my_rank == 0) {
    std::cout << "CyclicZoom: Dumping data" << std::endl;
    auto pm = pzoom->pmesh;
    auto &indcs = pm->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    int &mzoom = pzoom->mzoom;
    int &nvars = pzoom->nvars;

    std::string fname;
    fname.assign("CyclicZoom");
    // add pmesh ncycles
    fname.append(".");
    fname.append(std::to_string(pm->ncycle));
    fname.append(".dat");
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "wb")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Error output file could not be opened" <<std::endl;
      std::exit(EXIT_FAILURE);
    }
    int datasize = sizeof(Real);
    // xyz? bcc?
    IOWrapperSizeT cnt = mzoom*nvars*(n_ccells3)*(n_ccells2)*(n_ccells1);
    std::fwrite(coarse_w0.data(),datasize,cnt,pfile);
    auto mbptr = efld.x1e;
    cnt = mzoom*(n_ccells3+1)*(n_ccells2+1)*(n_ccells1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = efld.x2e;
    cnt = mzoom*(n_ccells3+1)*(n_ccells2)*(n_ccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = efld.x3e;
    cnt = mzoom*(n_ccells3)*(n_ccells2+1)*(n_ccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = emf0.x1e;
    cnt = mzoom*(n_ccells3+1)*(n_ccells2+1)*(n_ccells1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = emf0.x2e;
    cnt = mzoom*(n_ccells3+1)*(n_ccells2)*(n_ccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = emf0.x3e;
    cnt = mzoom*(n_ccells3)*(n_ccells2+1)*(n_ccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    std::fclose(pfile);
  }
  return;
}
