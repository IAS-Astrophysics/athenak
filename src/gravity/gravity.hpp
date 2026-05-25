#ifndef GRAVITY_GRAVITY_HPP_
#define GRAVITY_GRAVITY_HPP_

#include "../athena.hpp"

class MeshBlockPack;
class ParameterInput;

namespace gravity {

class Gravity {
 public:
  Gravity(MeshBlockPack *pmbp, ParameterInput *pin);
  ~Gravity();

  MeshBlockPack *pmy_pack;
  DvceArray5D<Real> phi;
  Real four_pi_G;
};

} // namespace gravity

#endif // GRAVITY_GRAVITY_HPP_
