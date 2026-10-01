#include "BHaH_defines.h"
/**
 * Compute Cartesian coordinates {x, y, z} = {xCart[0], xCart[1], xCart[2]} in terms of
 * local grid coordinates {xx[0][i0], xx[1][i1], xx[2][i2]} = {xx0, xx1, xx2},
 * taking into account the possibility that the origin of this grid is off-center.
 */
void bah_xx_to_Cart(const params_struct *restrict params, REAL xx[3], REAL xCart[3]) {

  const REAL xx0 = xx[0];
  const REAL xx1 = xx[1];
  const REAL xx2 = xx[2];
  /*
   *  Original SymPy expressions:
   *  "[xCart[0] = params->Cart_originx + xx0*sin(xx1)*cos(xx2)]"
   *  "[xCart[1] = params->Cart_originy + xx0*sin(xx1)*sin(xx2)]"
   *  "[xCart[2] = params->Cart_originz + xx0*cos(xx1)]"
   */
  {
    const REAL tmp0 = xx0 * sin(xx1);
    xCart[0] = params->Cart_originx + tmp0 * cos(xx2);
    xCart[1] = params->Cart_originy + tmp0 * sin(xx2);
    xCart[2] = params->Cart_originz + xx0 * cos(xx1);
  }
} // END FUNCTION bah_xx_to_Cart
