//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_shadowtest.cpp
//! \brief initializes the 2d shadow test problem with FEM_N/FP_N/M1

// C++ headers
#include <iostream>
#include <sstream>
#include <cmath>        // exp
#include <algorithm>    // max

// AthenaK headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "radiation_femn/radiation_femn.hpp"
#include "radiation_femn/radiation_femn_geodesic_grid.hpp"
#include "adm/adm.hpp"

void ProblemGenerator::RadiationFEMNSpharms(ParameterInput *pin, const bool restart) {
  if (restart) return;

  Real error = -42;

  Real phi = -3. * M_PI / 2.;
  Real theta = 1.9889;

  std::cout << "Test: Check fpn basis implementation works and legendre matches with the gsl implementation" << std::endl;
  for (int l = 0; l < 25; l++) {
    for (int m = -l; m <= l; m++) {
      Real fpn_new_basis = radiationfemn::fpn_basis_lm(l, m, phi, theta);
      Real fpn_basis_gsl = radiationfemn::fpn_basis_lm_alt(l, m, phi, theta);

      if (fabs(fpn_new_basis - fpn_basis_gsl) > error) {
        error = fabs(fpn_new_basis - fpn_basis_gsl);
      }

      std::cout << "l = " << l << ", m = " << m << ": " << fpn_new_basis << " " << fpn_basis_gsl << std::endl;
    }
  }
  std::cout << "Error: " << error << std::endl;
  std::cout << std::endl;

  std::cout << "Test: check legendre implementation for l >= 0, |m| <= l" << std::endl;
  Real error2 = -42;
  Real x = 0.8987;
  int N = 30;
  Real l_arr[] = {11, 4, 4, 26, 0, 4, 22, 13, 8, 23, 23, 10, 29, 20, 27, 19, 20, 11, 2, 18, 28, 11, 27, 24, 26, 14, 9, 26, 3, 26, 23, 27, 23, 14, 15, 27, 21, 2, 14, 1, 0, 26, 23,
                  2, 26, 6, 20, 20, 0, 3};
  Real m_arr[] = {7, 0, -1, -12, 0, 0, -22, -7, 7, 0, 11, 10, 20, 9, 18, 4, -15, 8, 1, 0, -8, 10, -17, -5, 8, -8, -6, 12, -3, 15, -14, -20, -13, -10, 10, 1, -4, 0, 6, 0, 0, -4,
                  -13, 1, 21, -2, -1, 12, 0, -1};
  Real answers[] = {-767068.0698190533, 0.20015731153474592, 0.13073694140932204, 2.1636171267985502e-18, 1.0, 0.20015731153474592, 2.8268141479831695e-36, 2.0019898541051555e-09,
                    -5684.655251828969, -0.22388166933330705, -113469147045446.45, 172341.7926871465, 1.4055020950276696e+25, -102558690792.88368, 2.132776678486919e+22,
                    -35413.397549732734, 5.412508224579258e-23, 1873706.56176514, -1.1824121787926132, 0.07035318127627838, -2.661385439814381e-13, 3252554.95084671,
                    4.406176734001765e-27, -2.7919927986855215e-08, 16705573721.962795, 5.648473516213318e-11, 1.0636075512287724e-07, 1.2980533080851954e+16,
                    0.0017573467245310269, -1.650902887446956e+19, 1.918964221063777e-21, 1.1369855638015587e-32, 9.846651817720011e-20, 4.3222553164519686e-14, 4795489134.8200865,
                    4.7053765235150085, -1.2419559894218114e-06, 0.7114925350000003, 2281588.7055624607, 0.8987, 1.0, 3.7804300687725965e-07, 9.846651817720011e-20,
                    -1.1824121787926132, -3.3908185064157403e+23, 0.012003960090949603, 0.009912456468071734, 54434511824144.58, 1.0, 0.16656155716630072};

  for (int i = 0; i < N; i++) {
    Real legval = radiationfemn::legendre(l_arr[i], m_arr[i], x);

    if (fabs((legval - answers[i]) / answers[i]) > error2) {
      error2 = fabs((legval - answers[i]) / answers[i]);
    }
    std::cout << "l = " << l_arr[i] << ", m = " << m_arr[i] << ": " << legval << " " << answers[i] << std::endl;
  }
  std::cout << "Relative Error: " << error2 << std::endl;
  std::cout << std::endl;

  std::cout << "Test: check legendre implementation for |m| > l (should be zero)" << std::endl;
  Real x1 = 0.23;
  Real error3 = -42;
  for (int m = 20; m < 35; m++) {
    for (int l = -(m); l <= (m - 1); l++) {
      Real legval = radiationfemn::legendre(l, m, x);
      if (fabs(legval) > error3) {
        error3 = fabs(legval);
      }
      std::cout << "l = " << l << ", m = " << m << ": " << legval << " " << "0" << std::endl;
    }
  }
  std::cout << "Error: " << error3 << std::endl;
  std::cout << std::endl;

  std::cout << "Test: Recurrence relation for associated Legendre functions" << std::endl;
  Real x2 = 0.9999;
  Real lwhere = -42;
  Real mwhere = -42;
  Real error4 = -42;
  for (int l = 0; l < 20; l++) {
    for (int m = -l; m <= l; m++) {
      Real legval = m * radiationfemn::legendre(l, m, x2) / sqrt(1 - x2 * x2);
      Real recur_legval1 = radiationfemn::recurrence_legendre(l, m, x2);
      Real recur_legval2 = radiationfemn::recurrence_legendre_alt(l, m, x2);
      Real rel_error = fabs(legval) > 1e-14 ? fabs(legval - recur_legval1) / fabs(legval) : fabs(legval - recur_legval1);
      if (rel_error > error4) {
        error4 = rel_error;
        lwhere = l;
        mwhere = m;
      }
      std::cout << "l = " << l << ", m = " << m << ": " << legval << " " << recur_legval1 << " " << recur_legval2 << std::endl;
    }
  }
  std::cout << "Error: (" << lwhere << ", " << mwhere << ") " << error4 << std::endl;
  std::cout << std::endl;
}
