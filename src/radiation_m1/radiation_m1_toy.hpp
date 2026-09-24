#ifndef RADIATION_M1_TOY_HPP
#define RADIATION_M1_TOY_HPP

#include "athena.hpp"

namespace radiationm1 {

enum class ToyOpacityModel { Lattice, Sphere, DiffusionExplicit, DiffusionImplicit };

struct LatticeOpacities {
  KOKKOS_INLINE_FUNCTION
  void operator()(Real x1, Real x2, Real x3, Real dx, Real dy, Real dz, Real nuidx,
                  Real &eta_0, Real &abs_0, Real &eta_1, Real &abs_1,
                  Real &scat_1) const {
    eta_0 = 0;
    abs_0 = 0;
    eta_1 = 0;
    abs_1 = 0;
    scat_1 = 0;

    if (x1 >= 3 && x1 <= 4 && x2 >= 3 && x2 <= 4) {
      eta_1 = 1.;
    }

    if ((x1 >= 1 && x1 <= 2 && x2 >= 1 && x2 <= 2) ||
        (x1 >= 3 && x1 <= 4 && x2 >= 1 && x2 <= 2) ||
        (x1 >= 5 && x1 <= 6 && x2 >= 1 && x2 <= 2) ||
        (x1 >= 2 && x1 <= 3 && x2 >= 2 && x2 <= 3) ||
        (x1 >= 4 && x1 <= 5 && x2 >= 2 && x2 <= 3) ||
        (x1 >= 1 && x1 <= 2 && x2 >= 3 && x2 <= 4) ||
        (x1 >= 5 && x1 <= 6 && x2 >= 3 && x2 <= 4) ||
        (x1 >= 2 && x1 <= 3 && x2 >= 4 && x2 <= 5) ||
        (x1 >= 4 && x1 <= 5 && x2 >= 4 && x2 <= 5) ||
        (x1 >= 1 && x1 <= 2 && x2 >= 5 && x2 <= 6) ||
        (x1 >= 5 && x1 <= 6 && x2 >= 5 && x2 <= 6)) {
      abs_1 = 10.;
    } else {
      abs_1 = 1.;
    }
  }
};

struct SphereOpacities {
  KOKKOS_INLINE_FUNCTION
  void operator()(Real x1, Real x2, Real x3, Real dx, Real dy, Real dz, Real nuidx,
                  Real &eta_0, Real &abs_0, Real &eta_1, Real &abs_1, Real &scat_1) {
    eta_0 = 0;
    abs_0 = 0;
    eta_1 = 0;
    abs_1 = 0;
    scat_1 = 0;

    const Real R = 1.;
    const int npoints = 10;
    int inside = 0;
    int count = 0;
    for (int ii = 0; ii < npoints; ++ii) {
      Real const myx = (x1 - dx / 2.) + (ii + 0.5) * (dx / npoints);
      for (int jj = 0; jj < npoints; ++jj) {
        Real const myy = (x2 - dy / 2.) + (jj + 0.5) * (dy / npoints);
        for (int kk = 0; kk < npoints; ++kk) {
          Real const myz = (x3 - dz / 2.) + (kk + 0.5) * (dz / npoints);
          count++;
          if (myx * myx + myy * myy + myz * myz <= R * R) {
            inside++;
          }
        }
      }
    }
    Real fraction_inside_sphere = static_cast<Real>(inside) / static_cast<Real>(count);
    abs_1 = fraction_inside_sphere * 10.0;
    eta_1 = fraction_inside_sphere * 10.0;
  }
};

struct DiffusionOpacitiesExplicit {
  KOKKOS_INLINE_FUNCTION
  void operator()(Real x1, Real x2, Real x3, Real dx, Real dy, Real dz, Real nuidx,
                  Real &eta_0, Real &abs_0, Real &eta_1, Real &abs_1, Real &scat_1) {
    eta_0 = 0;
    abs_0 = 0;
    eta_1 = 0;
    abs_1 = 0;
    scat_1 = 100;
  }
};

struct DiffusionOpacitiesImplicit {
  KOKKOS_INLINE_FUNCTION
  void operator()(Real x1, Real x2, Real x3, Real dx, Real dy, Real dz, Real nuidx,
                  Real &eta_0, Real &abs_0, Real &eta_1, Real &abs_1, Real &scat_1) {
    eta_0 = 0;
    abs_0 = 0;
    eta_1 = 0;
    abs_1 = 0;
    scat_1 = 1000;
  }
};

struct ToyOpacity {
  ToyOpacityModel model;

  KOKKOS_INLINE_FUNCTION
  void operator()(Real x1, Real x2, Real x3, Real dx, Real dy, Real dz, Real nuidx,
                  Real &eta_0, Real &abs_0, Real &eta_1, Real &abs_1,
                  Real &scat_1) const {
    if (model == ToyOpacityModel::Lattice) {
      LatticeOpacities{}(x1, x2, x3, dx, dy, dz, nuidx, eta_0, abs_0, eta_1, abs_1,
                         scat_1);
    } else if (model == ToyOpacityModel::Sphere) {
      SphereOpacities{}(x1, x2, x3, dx, dy, dz, nuidx, eta_0, abs_0, eta_1, abs_1,
                        scat_1);
    } else if (model == ToyOpacityModel::DiffusionExplicit) {
      DiffusionOpacitiesExplicit{}(x1, x2, x3, dx, dy, dz, nuidx, eta_0, abs_0, eta_1,
                                   abs_1, scat_1);
    } else if (model == ToyOpacityModel::DiffusionImplicit) {
      DiffusionOpacitiesImplicit{}(x1, x2, x3, dx, dy, dz, nuidx, eta_0, abs_0, eta_1,
                                   abs_1, scat_1);
    } else {
      // default behavior
    }
  }
};

}  // namespace radiationm1
#endif  // RADIATION_M1_TOY_HPP
