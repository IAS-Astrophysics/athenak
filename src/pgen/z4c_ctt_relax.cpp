//========================================================================================
//! \file z4c_ctt_relax.cpp
//! \brief Flat starter pgen for hyperbolic CTT relaxation initial data.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"

void ProblemGenerator::Z4cCTTRelax(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->padm == nullptr || pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in Z4cCTTRelax" << std::endl
              << "pgen_name=z4c_ctt_relax requires <adm> and <z4c> blocks."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is - indcs.ng, ie = indcs.ie + indcs.ng;
  int js = indcs.js - indcs.ng, je = indcs.je + indcs.ng;
  int ks = indcs.ks - indcs.ng, ke = indcs.ke + indcs.ng;
  int nmb = pmbp->nmb_thispack;
  auto &admvars = pmbp->padm->adm;

  par_for("pgen_z4c_ctt_relax_flat", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    admvars.psi4(m,k,j,i) = 1.0;
    admvars.alpha(m,k,j,i) = 1.0;
    for (int a = 0; a < 3; ++a) admvars.beta_u(m,a,k,j,i) = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        admvars.g_dd(m,a,b,k,j,i) = (a == b) ? 1.0 : 0.0;
        admvars.vK_dd(m,a,b,k,j,i) = 0.0;
      }
    }
  });

  int fd = pmbp->pz4c->opt.fd_stencil;
  if (fd == 2) pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
  else if (fd == 3) pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
  else pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
  pmbp->pz4c->Z4cToADM(pmbp);
}
