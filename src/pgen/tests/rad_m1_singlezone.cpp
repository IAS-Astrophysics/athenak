//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_m1_latticetest.cpp
//  \brief Radiation M1 single zone equilibratioon test for grey M1 + hydro

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "eos/eos.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "driver/driver.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for radiation M1 single zone equilibratioon test

void ProblemGenerator::RadiationM1SingleZoneTest(ParameterInput *pin,
                                              const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Check required modules are called
  if (pmbp->pmhd == nullptr) {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
            << "DyHydro is required for the single zone equilibration test" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1 == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The single zone equilibration test problem generator requires "
                 "radiation-m1, but no "
              << "<radiation_m1> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pradm1->params.opacity_type != radiationm1::BnsNurates) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "The single zone equilibration test problem generator requires "
                 "bns-nurates" << std::endl;
    exit(EXIT_FAILURE);
  }  

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  int nmb1 = (pmbp->nmb_thispack-1);
  // get problem parameters
  Real rho = pin->GetReal("problem", "rho");
  Real temp = pin->GetReal("problem", "temp");
  Real vx = pin->GetReal("problem", "vx");
  Real vy = pin->GetReal("problem", "vy");
  Real vz = pin->GetReal("problem", "vz");
  Real ye = pin->GetReal("problem", "Y_e");

  // set primitive variables
  auto &w0_ = pmbp->pmhd->w0;
  par_for("pgen_singlezone",DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IVX,k,j,i) = vx;
    w0_(m,IVY,k,j,i) = vy;
    w0_(m,IVZ,k,j,i) = vz;
    w0_(m,IEN,k,j,i) = temp;
    w0_(m,IYF,k,j,i) = ye;
  });

  // Convert primitives to conserved
  // auto &u0 = pmbp->pmhd->u0;
  pmbp->pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), 0, (n3-1));

  // initialize ADM variables
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }
}
