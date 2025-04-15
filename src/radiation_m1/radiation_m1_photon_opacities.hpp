#ifndef RADIATION_M1_PHOTON_OPACITIES_HPP
#define RADIATION_M1_PHOTON_OPACITIES_HPP

#include "athena.hpp"
#include "radiation_m1_macro.hpp"

namespace radiationm1 {

KOKKOS_INLINE_FUNCTION
Real PhotonBlackBodyImpl(const Real temp) {
  Real rad_const{}; //@TODO: fix
  return rad_const * POW4(temp);
}

KOKKOS_INLINE_FUNCTION
Real PhotonOpacityImpl(
        Real rho,
        Real temp,
        Real Y_e,
        Real &abs_1,
        Real &scat_1) {

  Real kappa_abs{}; //@TODO: fix
  Real kappa_scat{}; //@TODO: fix

  abs_1 = kappa_abs * rho;
  scat_1 = kappa_scat * rho;

  return 0;
}

}
#endif //RADIATION_M1_PHOTON_OPACITIES_HPP
