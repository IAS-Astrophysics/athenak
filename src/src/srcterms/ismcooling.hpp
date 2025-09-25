#ifndef SRCTERMS_ISMCOOLING_HPP_
#define SRCTERMS_ISMCOOLING_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ismcooling.hpp
//! \brief function to implement ISM cooling

// Athena++ headers
#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real ISMCoolFn()
//! \brief SPEX cooling curve, taken from Table 2 of Schure et al, A&A 508, 751 (2009)

KOKKOS_INLINE_FUNCTION
Real ISMCoolFn(Real temp) {
  // original data from Shure et al. paper, covers 4.12 < logt < 8.16
  const float lhd[102] = {
      -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
      -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
      -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
      -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
      -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
      -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
      -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
      -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
      -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
      -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
      -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
      -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
      -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928};

  Real logt = log10(temp);

  // for temperatures less than 10^4 K, use Koyama & Inutsuka (2002)
  if (logt <= 4.2) {
    return (2.0e-19*exp(-1.184e5/(temp + 1.0e3)) + 2.8e-28*sqrt(temp)*exp(-92.0/temp));
  }

  // for temperatures above 10^8.15 use CGOLS fit
  if (logt > 8.15) {
    return pow(10.0, (0.45*logt - 26.065));
  }

  // in between values of 4.2 < log(T) < 8.15
  // linear interpolation of tabulated SPEX cooling rate
  int ipps  = static_cast<int>(25.0*logt) - 103;
  ipps = (ipps < 100)? ipps : 100;
  ipps = (ipps > 0 )? ipps : 0;
  Real x0 = 4.12 + 0.04*static_cast<Real>(ipps);
  Real dx = logt - x0;
  Real logcool = (lhd[ipps+1]*dx - lhd[ipps]*(dx - 0.04))*25.0;
  return pow(10.0,logcool);
}
#endif // SRCTERMS_ISMCOOLING_HPP_
