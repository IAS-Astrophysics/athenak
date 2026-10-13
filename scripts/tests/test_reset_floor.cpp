// Regression: non-aggressive density flooring must also enforce temperature.
#include <cstdio>
#include "eos/primitive-solver/reset_floor.hpp"

class FloorProbe : public Primitive::ResetFloor {
 public:
  FloorProbe(bool aggressive) {
    n_atm=1e-10; n_threshold=1.0; T_atm=1e-12;
    aggressive_floor=aggressive;
    for (int i=0; i<MAX_SPECIES; ++i) Y_atm[i]=0.0;
  }
  using Primitive::ResetFloor::PrimitiveFloor;
};
int main() {
  for (bool aggressive : {false,true}) {
    FloorProbe policy(aggressive);
    for (Real density : {1e-14,1.0}) {
      for (Real temperature : {-1e-6,1.0}) {
        Real n=density, t=temperature, v[3]={1.0,2.0,3.0}, y[MAX_SPECIES]={0.0};
        bool changed=policy.PrimitiveFloor(n,v,t,y,0);
        if (n<1e-10 || t<1e-12) return 1;
        if (changed != (density<1e-10 || temperature<1e-12)) return 2;
        if (!aggressive && v[0]!=1.0) return 3;
        if (aggressive && density<1e-10 && (v[0]!=0.0 || t!=1e-12)) return 4;
      }
    }
  }
  std::puts("Simultaneous density/temperature floor regression passed.");
}
