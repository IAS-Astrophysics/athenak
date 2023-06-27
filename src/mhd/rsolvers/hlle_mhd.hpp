#ifndef MHD_RSOLVERS_HLLE_MHD_HPP_
#define MHD_RSOLVERS_HLLE_MHD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlle_mhd.hpp
//! \brief HLLE Riemann solver for MHD. See the hydro version for details.

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void HLLE
//! \brief The HLLE Riemann solver for hydrodynamics (both ideal gas and isothermal)

KOKKOS_INLINE_FUNCTION
void HLLE(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;
  Real gm1 = eos.gamma - 1.0;
  Real igm1 = 1.0/gm1;
  Real iso_cs = eos.iso_cs;

  par_for_inner(member, il, iu, [&](const int i) {
    //--- Step 1.  Create local references for L/R states (helps compiler vectorize)

    Real &wl_idn = wl(IDN,i);
    Real &wl_ivx = wl(ivx,i);
    Real &wl_ivy = wl(ivy,i);
    Real &wl_ivz = wl(ivz,i);
    Real &wl_iby = bl(iby,i);
    Real &wl_ibz = bl(ibz,i);

    Real &wr_idn = wr(IDN,i);
    Real &wr_ivx = wr(ivx,i);
    Real &wr_ivy = wr(ivy,i);
    Real &wr_ivz = wr(ivz,i);
    Real &wr_iby = br(iby,i);
    Real &wr_ibz = br(ibz,i);

    Real wl_ipr, wr_ipr;
    if (eos.is_ideal) {
      wl_ipr = eos.IdealGasPressure(wl(IEN,i));
      wr_ipr = eos.IdealGasPressure(wr(IEN,i));
    }

    Real bxi = bx(m,k,j,i);

    //--- Step 2. Compute Roe-averaged state

    Real sqrtdl = sqrt(wl_idn);
    Real sqrtdr = sqrt(wr_idn);
    Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

    Real wroe_idn = sqrtdl*sqrtdr;
    Real wroe_ivx = (sqrtdl*wl_ivx + sqrtdr*wr_ivx)*isdlpdr;
    Real wroe_ivy = (sqrtdl*wl_ivy + sqrtdr*wr_ivy)*isdlpdr;
    Real wroe_ivz = (sqrtdl*wl_ivz + sqrtdr*wr_ivz)*isdlpdr;
    // Note Roe average of magnetic field is different
    Real wroe_iby = (sqrtdr*wl_iby + sqrtdl*wr_iby)*isdlpdr;
    Real wroe_ibz = (sqrtdr*wl_ibz + sqrtdl*wr_ibz)*isdlpdr;
    Real x = 0.5*(SQR(wl_iby-wr_iby) + SQR(wl_ibz-wr_ibz))/(SQR(sqrtdl+sqrtdr));
    Real y = 0.5*(wl_idn + wr_idn)/wroe_idn;

    // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for ideal gas EOS,
    // rather than E or P directly. sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
    Real pbl = 0.5*(bxi*bxi + SQR(wl_iby) + SQR(wl_ibz));
    Real pbr = 0.5*(bxi*bxi + SQR(wr_iby) + SQR(wr_ibz));
    Real el,er,hroe,cl,cr;
    if (eos.is_ideal) {
      el = wl_ipr*igm1 + 0.5*wl_idn*(SQR(wl_ivx)+SQR(wl_ivy)+SQR(wl_ivz)) + pbl;
      er = wr_ipr*igm1 + 0.5*wr_idn*(SQR(wr_ivx)+SQR(wr_ivy)+SQR(wr_ivz)) + pbr;
      hroe = ((el + wl_ipr + pbl)/sqrtdl + (er + wr_ipr + pbr)/sqrtdr)*isdlpdr;
      cl = eos.IdealMHDFastSpeed(wl_idn, wl_ipr, bxi, wl_iby, wl_ibz);
      cr = eos.IdealMHDFastSpeed(wr_idn, wr_ipr, bxi, wr_iby, wr_ibz);
    } else {
      cl = eos.IdealMHDFastSpeed(wl_idn, bxi, wl_iby, wl_ibz);
      cr = eos.IdealMHDFastSpeed(wr_idn, bxi, wr_iby, wr_ibz);
    }

    //--- Step 3. Compute fast magnetosonic speed in L,R, and Roe-averaged states


    // Compute Roe-averaged Cf using eq. B18 (ideal gas) or B39 (isothermal)
    Real btsq = SQR(wroe_iby) + SQR(wroe_ibz);
    Real vaxsq = bxi*bxi/wroe_idn;
    Real bt_starsq, twid_asq;
    if (eos.is_ideal) {
      bt_starsq = (gm1 - (gm1 - 1.0)*y)*btsq;
      Real hp = hroe - (vaxsq + btsq/wroe_idn);
      Real vsq = SQR(wroe_ivx) + SQR(wroe_ivy) + SQR(wroe_ivz);
      twid_asq = fmax((gm1*(hp-0.5*vsq)-(gm1-1.0)*x), 0.0);
    } else {
      bt_starsq = btsq*y;
      twid_asq = iso_cs*iso_cs + x;
    }
    Real ct2 = bt_starsq/wroe_idn;
    Real tsum = vaxsq + ct2 + twid_asq;
    Real tdif = vaxsq + ct2 - twid_asq;
    Real cf2_cs2 = sqrt(tdif*tdif + 4.0*twid_asq*ct2);

    Real cfsq = 0.5*(tsum + cf2_cs2);
    Real a = sqrt(cfsq);

    //--- Step 4. Compute the max/min wave speeds based on L/R and Roe-averaged values

    Real al = fmin((wroe_ivx - a),(wl_ivx - cl));
    Real ar = fmax((wroe_ivx + a),(wr_ivx + cr));

    // following min/max set to TINY_NUMBER to fix bug found in converging supersonic flow
    Real bp = ar > 0.0 ? ar : 1.0e-20;
    Real bm = al < 0.0 ? al : -1.0e-20;

    //--- Step 5. Compute L/R fluxes along the lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R

    Real vxl = wl_ivx - bm;
    Real vxr = wr_ivx - bp;

    MHDCons1D fl,fr;
    fl.d  = wl_idn*vxl;
    fr.d  = wr_idn*vxr;

    fl.mx = wl_idn*wl_ivx*vxl + pbl - SQR(bxi);
    fr.mx = wr_idn*wr_ivx*vxr + pbr - SQR(bxi);

    fl.my = wl_idn*wl_ivy*vxl - bxi*wl_iby;
    fr.my = wr_idn*wr_ivy*vxr - bxi*wr_iby;

    fl.mz = wl_idn*wl_ivz*vxl - bxi*wl_ibz;
    fr.mz = wr_idn*wr_ivz*vxr - bxi*wr_ibz;

    if (eos.is_ideal) {
      fl.mx += wl_ipr;
      fr.mx += wr_ipr;
      fl.e   = el*vxl + wl_ivx*(wl_ipr + pbl - bxi*bxi);
      fr.e   = er*vxr + wr_ivx*(wr_ipr + pbr - bxi*bxi);
      fl.e  -= bxi*(wl_iby*wl_ivy + wl_ibz*wl_ivz);
      fr.e  -= bxi*(wr_iby*wr_ivy + wr_ibz*wr_ivz);
    } else {
      fl.mx += (iso_cs*iso_cs)*wl_idn;
      fr.mx += (iso_cs*iso_cs)*wr_idn;
    }

    fl.by = wl_iby*vxl - bxi*wl_ivy;
    fr.by = wr_iby*vxr - bxi*wr_ivy;

    fl.bz = wl_ibz*vxl - bxi*wl_ivz;
    fr.bz = wr_ibz*vxr - bxi*wr_ivz;

    //--- Step 6. Compute the HLLE flux at interface.

    Real tmp=0.0;
    if (bp != bm) tmp = 0.5*(bp + bm)/(bp - bm);

    flx(m,IDN,k,j,i) = 0.5*(fl.d  + fr.d ) + (fl.d  - fr.d )*tmp;
    flx(m,ivx,k,j,i) = 0.5*(fl.mx + fr.mx) + (fl.mx - fr.mx)*tmp;
    flx(m,ivy,k,j,i) = 0.5*(fl.my + fr.my) + (fl.my - fr.my)*tmp;
    flx(m,ivz,k,j,i) = 0.5*(fl.mz + fr.mz) + (fl.mz - fr.mz)*tmp;
    if (eos.is_ideal) flx(m,IEN,k,j,i) = 0.5*(fl.e + fr.e ) + (fl.e - fr.e)*tmp;
    ey(m,k,j,i) = -0.5*(fl.by + fr.by) - (fl.by - fr.by)*tmp;
    ez(m,k,j,i) =  0.5*(fl.bz + fr.bz) + (fl.bz - fr.bz)*tmp;
  });

  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_HLLE_MHD_HPP_
