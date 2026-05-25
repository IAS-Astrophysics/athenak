#include "gravity.hpp"

#include <cstdlib>
#include <iostream>

#include "../parameter_input.hpp"

namespace gravity {

Gravity::Gravity(MeshBlockPack *pmbp, ParameterInput *pin)
    : pmy_pack(pmbp), phi("phi",1,1,1,1,1), four_pi_G(0.0) {
  std::cout << "### FATAL ERROR in Gravity::Gravity" << std::endl
            << "The shared multigrid Poisson gravity solver was removed on "
            << "project/collapse_relaxation. Do not use a <gravity> block on "
            << "this branch." << std::endl;
  std::exit(EXIT_FAILURE);
}

Gravity::~Gravity() {}

} // namespace gravity
