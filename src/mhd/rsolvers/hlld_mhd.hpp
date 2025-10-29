#ifndef MHD_RSOLVERS_HLLD_MHD_HPP_
#define MHD_RSOLVERS_HLLD_MHD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlld_mhd.hpp
//! \brief HLLD Riemann solver for ideal gas EOS in MHD.
//!
//! REFERENCES:
//! - T. Miyoshi & K. Kusano, "A multi-state HLL approximate Riemann solver for ideal
//!   MHD", JCP, 208, 315 (2005)

namespace mhd {

#define HLLD_SMALL_NUMBER 1.0e-4
#define vbstr ((urst.mx*bxi+(urst.my*urst.by+urst.mz*urst.bz))*urst_d_inv)
#define vbstl ((ulst.mx*bxi+(ulst.my*ulst.by+ulst.mz*ulst.bz))*ulst_d_inv)

#define _ivy IVX + ((ivx-IVX)+1)%3
#define _ivz IVX + ((ivx-IVX)+2)%3
#define _iby ((ivx-IVX) + 1)%3
#define _ibz ((ivx-IVX) + 2)%3

//----------------------------------------------------------------------------------------
//! \fn

KOKKOS_INLINE_FUNCTION
void HLLD(const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int i, const int ivx,
     const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
     const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br, const DvceArray4D<Real> &bx,
     const DvceArray5D<Real> &flx, const DvceArray4D<Real> &ey, const DvceArray4D<Real> &ez) {
  //int ivy = IVX + ((ivx-IVX)+1)%3;
  //int ivz = IVX + ((ivx-IVX)+2)%3;
  //int iby = ((ivx-IVX) + 1)%3;
  //int ibz = ((ivx-IVX) + 2)%3;
  Real spd[5];         // signal speeds, left to right

  //------------------------ ADIABATIC HLLD solver ---------------------------------------
  //if (eos.is_ideal) {
    Real gm1 = eos.gamma - 1.0;
    Real igm1 = 1.0/gm1;
    //--- Step 1.  Create local references for L/R states (helps compiler vectorize)

    Real wl_idn=wl(m, IDN,0,0,0);
    Real wl_ivx=wl(m, ivx,0,0,0);
    Real wl_ivy=wl(m,_ivy,0,0,0);
    Real wl_ivz=wl(m,_ivz,0,0,0);
    Real wl_iby=bl(m,_iby,0,0,0);
    Real wl_ibz=bl(m,_ibz,0,0,0);  

    Real wr_idn=wr(m, IDN,0,0,0);
    Real wr_ivx=wr(m, ivx,0,0,0);
    Real wr_ivy=wr(m,_ivy,0,0,0);
    Real wr_ivz=wr(m,_ivz,0,0,0);
    Real wr_iby=br(m,_iby,0,0,0);
    Real wr_ibz=br(m,_ibz,0,0,0);

    Real wl_ipr, wr_ipr;
    wl_ipr = eos.IdealGasPressure(wl(m,IEN,0,0,0));
    wr_ipr = eos.IdealGasPressure(wr(m,IEN,0,0,0));

    Real bxi = bx(m,k,j,i);

      // Compute L/R states for selected conserved variables
      {
        Real bxsq = bxi*bxi;
        // (KGF): group transverse components for floating-point associativity symmetry
        Real pbl = 0.5*(bxsq + (SQR(wl_iby) + SQR(wl_ibz)));  // magnetic pressure (l/r)
        Real pbr = 0.5*(bxsq + (SQR(wr_iby) + SQR(wr_ibz)));
        Real kel = 0.5*wl_idn*(SQR(wl_ivx) + (SQR(wl_ivy) + SQR(wl_ivz)));
        Real ker = 0.5*wr_idn*(SQR(wr_ivx) + (SQR(wr_ivy) + SQR(wr_ivz)));

        MHDCons1D ul,ur;  // L/R states, conserved variables (computed)
        ul.d  = wl_idn;
        ul.mx = wl_ivx*ul.d;
        ul.my = wl_ivy*ul.d;
        ul.mz = wl_ivz*ul.d;
        ul.e  = wl_ipr*igm1 + kel + pbl;
        ul.by = wl_iby;
        ul.bz = wl_ibz;

        ur.d  = wr_idn;
        ur.mx = wr_ivx*ur.d;
        ur.my = wr_ivy*ur.d;
        ur.mz = wr_ivz*ur.d;
        ur.e  = wr_ipr*igm1 + ker + pbr;
        ur.by = wr_iby;
        ur.bz = wr_ibz;

        //--- Step 2.  Compute L & R wave speeds according to Miyoshi & Kusano, eqn. (67)

        Real cfl = eos.IdealMHDFastSpeed(wl_idn, wl_ipr, bxi, wl_iby, wl_ibz);
        Real cfr = eos.IdealMHDFastSpeed(wr_idn, wr_ipr, bxi, wr_iby, wr_ibz);

        spd[0] = fmin( wl_ivx-cfl, wr_ivx-cfr );
        spd[4] = fmax( wl_ivx+cfl, wr_ivx+cfr );

      // Real cfmax = std::max(cfl,cfr);
      // if (wl_ivx <= wr_ivx) {
      //   spd[0] = wl_ivx - cfmax;
      //   spd[4] = wr_ivx + cfmax;
      // } else {
      //   spd[0] = wr_ivx - cfmax;
      //   spd[4] = wl_ivx + cfmax;
      // }

        //--- Step 3.  Compute L/R fluxes

        Real ptl = wl_ipr + pbl; // total pressures L,R
        Real ptr = wr_ipr + pbr;

        MHDCons1D fl,fr;           // Fluxes for left & right states
        fl.d  = ul.mx;
        fl.mx = ul.mx*wl_ivx + ptl - bxsq;
        fl.my = ul.my*wl_ivx - bxi*ul.by;
        fl.mz = ul.mz*wl_ivx - bxi*ul.bz;
        fl.e  = wl_ivx*(ul.e + ptl - bxsq) - bxi*(wl_ivy*ul.by + wl_ivz*ul.bz);
        fl.by = ul.by*wl_ivx - bxi*wl_ivy;
        fl.bz = ul.bz*wl_ivx - bxi*wl_ivz;

        fr.d  = ur.mx;
        fr.mx = ur.mx*wr_ivx + ptr - bxsq;
        fr.my = ur.my*wr_ivx - bxi*ur.by;
        fr.mz = ur.mz*wr_ivx - bxi*ur.bz;
        fr.e  = wr_ivx*(ur.e + ptr - bxsq) - bxi*(wr_ivy*ur.by + wr_ivz*ur.bz);
        fr.by = ur.by*wr_ivx - bxi*wr_ivy;
        fr.bz = ur.bz*wr_ivx - bxi*wr_ivz;

        //--- Step 4.  Compute middle and Alfven wave speeds

        Real sdl = spd[0] - wl_ivx;  // S_i-u_i (i=L or R)
        Real sdr = spd[4] - wr_ivx;

        // S_M: eqn (38) of Miyoshi & Kusano
        // (KGF): group ptl, ptr terms for floating-point associativity symmetry
        spd[2] = (sdr*ur.mx - sdl*ul.mx + (ptl - ptr))/(sdr*ur.d - sdl*ul.d);

        Real sdml   = spd[0] - spd[2];  // S_i-S_M (i=L or R)
        Real sdmr   = spd[4] - spd[2];

        MHDCons1D ulst,uldst,urdst,urst;   // intermadiate states for conserved variables
        // eqn (43) of Miyoshi & Kusano
        ulst.d = ul.d * sdl/sdml;
        urst.d = ur.d * sdr/sdmr;
        Real ulst_d_inv = 1.0/ulst.d;
        Real urst_d_inv = 1.0/urst.d;
        Real sqrtdl = sqrt(ulst.d);
        Real sqrtdr = sqrt(urst.d);

        // eqn (51) of Miyoshi & Kusano
        spd[1] = spd[2] - fabs(bxi)/sqrtdl;
        spd[3] = spd[2] + fabs(bxi)/sqrtdr;

        //--- Step 5.  Compute intermediate states
        // eqn (23) explicitly becomes eq (41) of Miyoshi & Kusano
        // TODO(felker): place an assertion that ptstl==ptstr
        //Real ptstl = ptl + ul.d*sdl*(spd[2]-wl_ivx);
        //Real ptstr = ptr + ur.d*sdr*(spd[2]-wr_ivx);
        // Real ptstl = ptl + ul.d*sdl*(sdl-sdml); // these eqns had issues when averaged
        // Real ptstr = ptr + ur.d*sdr*(sdr-sdmr);
        Real ptst = 0.5*(ptl + ul.d*sdl*(spd[2]-wl_ivx) + ptr + ur.d*sdr*(spd[2]-wr_ivx));  // total pressure (star state)

        // ul* - eqn (39) of M&K
        ulst.mx = ulst.d * spd[2];
        if (fabs(ul.d*sdl*sdml-bxsq) < (HLLD_SMALL_NUMBER)*ptst) {
          // Degenerate case
          ulst.my = ulst.d * wl_ivy;
          ulst.mz = ulst.d * wl_ivz;

          ulst.by = ul.by;
          ulst.bz = ul.bz;
        } else {
          // eqns (44) and (46) of M&K
          Real tmp = bxi*(sdl - sdml)/(ul.d*sdl*sdml - bxsq);
          ulst.my = ulst.d * (wl_ivy - ul.by*tmp);
          ulst.mz = ulst.d * (wl_ivz - ul.bz*tmp);

          // eqns (45) and (47) of M&K
          tmp = (ul.d*SQR(sdl) - bxsq)/(ul.d*sdl*sdml - bxsq);
          ulst.by = ul.by * tmp;
          ulst.bz = ul.bz * tmp;
        }
        // v_i* dot B_i*
        // (KGF): group transverse momenta terms for floating-point associativity symmetry
        //Real vbstl = (ulst.mx*bxi+(ulst.my*ulst.by+ulst.mz*ulst.bz))*ulst_d_inv;
        // eqn (48) of M&K
        // (KGF): group transverse by, bz terms for floating-point associativity symmetry
        ulst.e = (sdl*ul.e - ptl*wl_ivx + ptst*spd[2] +
                  bxi*(wl_ivx*bxi + (wl_ivy*ul.by + wl_ivz*ul.bz) - vbstl))/sdml;

        // ur* - eqn (39) of M&K
        urst.mx = urst.d * spd[2];
        if (fabs(ur.d*sdr*sdmr - bxsq) < (HLLD_SMALL_NUMBER)*ptst) {
          // Degenerate case
          urst.my = urst.d * wr_ivy;
          urst.mz = urst.d * wr_ivz;

          urst.by = ur.by;
          urst.bz = ur.bz;
        } else {
          // eqns (44) and (46) of M&K
          Real tmp = bxi*(sdr - sdmr)/(ur.d*sdr*sdmr - bxsq);
          urst.my = urst.d * (wr_ivy - ur.by*tmp);
          urst.mz = urst.d * (wr_ivz - ur.bz*tmp);

          // eqns (45) and (47) of M&K
          tmp = (ur.d*SQR(sdr) - bxsq)/(ur.d*sdr*sdmr - bxsq);
          urst.by = ur.by * tmp;
          urst.bz = ur.bz * tmp;
        }
        // v_i* dot B_i*
        // (KGF): group transverse momenta terms for floating-point associativity symmetry
        //Real vbstr = (urst.mx*bxi+(urst.my*urst.by+urst.mz*urst.bz))*urst_d_inv;
        // eqn (48) of M&K
        // (KGF): group transverse by, bz terms for floating-point associativity symmetry
        urst.e = (sdr*ur.e - ptr*wr_ivx + ptst*spd[2] +
                  bxi*(wr_ivx*bxi + (wr_ivy*ur.by + wr_ivz*ur.bz) - vbstr))/sdmr;
        // ul** and ur** - if Bx is near zero, same as *-states
        if (0.5*bxsq < (HLLD_SMALL_NUMBER)*ptst) {
          uldst = ulst;
          urdst = urst;
        } else {
          Real invsumd = 1.0/(sqrtdl + sqrtdr);
          Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);

          uldst.d = ulst.d;
          urdst.d = urst.d;

          uldst.mx = ulst.mx;
          urdst.mx = urst.mx;

          // eqn (59) of M&K
          Real tmp = invsumd*(sqrtdl*(ulst.my*ulst_d_inv) + sqrtdr*(urst.my*urst_d_inv) +
                              bxsig*(urst.by - ulst.by));
          uldst.my = uldst.d * tmp;
          urdst.my = urdst.d * tmp;

          // eqn (60) of M&K
          tmp = invsumd*(sqrtdl*(ulst.mz*ulst_d_inv) + sqrtdr*(urst.mz*urst_d_inv) +
                         bxsig*(urst.bz - ulst.bz));
          uldst.mz = uldst.d * tmp;
          urdst.mz = urdst.d * tmp;

          // eqn (61) of M&K
          tmp = invsumd*(sqrtdl*urst.by + sqrtdr*ulst.by +
                         bxsig*sqrtdl*sqrtdr*((urst.my*urst_d_inv) - (ulst.my*ulst_d_inv)));
          uldst.by = urdst.by = tmp;

          // eqn (62) of M&K
          tmp = invsumd*(sqrtdl*urst.bz + sqrtdr*ulst.bz +
                         bxsig*sqrtdl*sqrtdr*((urst.mz*urst_d_inv) - (ulst.mz*ulst_d_inv)));
          uldst.bz = urdst.bz = tmp;

          // eqn (63) of M&K
          tmp = spd[2]*bxi + (uldst.my*uldst.by + uldst.mz*uldst.bz)/uldst.d;
          uldst.e = ulst.e - sqrtdl*bxsig*(vbstl - tmp);
          urdst.e = urst.e + sqrtdr*bxsig*(vbstr - tmp);
        }

        //--- Step 6.  Compute flux
        uldst.d = spd[1] * (uldst.d - ulst.d);
        uldst.mx = spd[1] * (uldst.mx - ulst.mx);
        uldst.my = spd[1] * (uldst.my - ulst.my);
        uldst.mz = spd[1] * (uldst.mz - ulst.mz);
        uldst.e = spd[1] * (uldst.e - ulst.e);
        uldst.by = spd[1] * (uldst.by - ulst.by);
        uldst.bz = spd[1] * (uldst.bz - ulst.bz);

        ulst.d = spd[0] * (ulst.d - ul.d);
        ulst.mx = spd[0] * (ulst.mx - ul.mx);
        ulst.my = spd[0] * (ulst.my - ul.my);
        ulst.mz = spd[0] * (ulst.mz - ul.mz);
        ulst.e = spd[0] * (ulst.e - ul.e);
        ulst.by = spd[0] * (ulst.by - ul.by);
        ulst.bz = spd[0] * (ulst.bz - ul.bz);

        urdst.d = spd[3] * (urdst.d - urst.d);
        urdst.mx = spd[3] * (urdst.mx - urst.mx);
        urdst.my = spd[3] * (urdst.my - urst.my);
        urdst.mz = spd[3] * (urdst.mz - urst.mz);
        urdst.e = spd[3] * (urdst.e - urst.e);
        urdst.by = spd[3] * (urdst.by - urst.by);
        urdst.bz = spd[3] * (urdst.bz - urst.bz);

        urst.d = spd[4] * (urst.d  - ur.d);
        urst.mx = spd[4] * (urst.mx - ur.mx);
        urst.my = spd[4] * (urst.my - ur.my);
        urst.mz = spd[4] * (urst.mz - ur.mz);
        urst.e = spd[4] * (urst.e - ur.e);
        urst.by = spd[4] * (urst.by - ur.by);
        urst.bz = spd[4] * (urst.bz - ur.bz);

        if (spd[0] >= 0.0) {
          // return Fl if flow is supersonic
          flx(m,IDN,k,j,i) = fl.d;
          flx(m,ivx,k,j,i) = fl.mx;
          flx(m,_ivy,k,j,i) = fl.my;
          flx(m,_ivz,k,j,i) = fl.mz;
          flx(m,IEN,k,j,i) = fl.e;
          ey(m,k,j,i) = -fl.by;
          ez(m,k,j,i) =  fl.bz;
        } else if (spd[4] <= 0.0) {
          // return Fr if flow is supersonic
          flx(m,IDN,k,j,i) = fr.d;
          flx(m,ivx,k,j,i) = fr.mx;
          flx(m,_ivy,k,j,i) = fr.my;
          flx(m,_ivz,k,j,i) = fr.mz;
          flx(m,IEN,k,j,i) = fr.e;
          ey(m,k,j,i) = -fr.by;
          ez(m,k,j,i) =  fr.bz;
        } else if (spd[1] >= 0.0) {
          // return Fl*
          flx(m,IDN,k,j,i) = fl.d  + ulst.d;
          flx(m,ivx,k,j,i) = fl.mx + ulst.mx;
          flx(m,_ivy,k,j,i) = fl.my + ulst.my;
          flx(m,_ivz,k,j,i) = fl.mz + ulst.mz;
          flx(m,IEN,k,j,i) = fl.e  + ulst.e;
          ey(m,k,j,i) = -(fl.by + ulst.by);
          ez(m,k,j,i) =  (fl.bz + ulst.bz);
        } else if (spd[2] >= 0.0) {
          // return Fl**
          flx(m,IDN,k,j,i) = fl.d  + ulst.d + uldst.d;
          flx(m,ivx,k,j,i) = fl.mx + ulst.mx + uldst.mx;
          flx(m,_ivy,k,j,i) = fl.my + ulst.my + uldst.my;
          flx(m,_ivz,k,j,i) = fl.mz + ulst.mz + uldst.mz;
          flx(m,IEN,k,j,i) = fl.e  + ulst.e + uldst.e;
          ey(m,k,j,i) = -(fl.by + ulst.by + uldst.by);
          ez(m,k,j,i) =  (fl.bz + ulst.bz + uldst.bz);
        } else if (spd[3] > 0.0) {
          // return Fr**
          flx(m,IDN,k,j,i) = fr.d + urst.d + urdst.d;
          flx(m,ivx,k,j,i) = fr.mx + urst.mx + urdst.mx;
          flx(m,_ivy,k,j,i) = fr.my + urst.my + urdst.my;
          flx(m,_ivz,k,j,i) = fr.mz + urst.mz + urdst.mz;
          flx(m,IEN,k,j,i) = fr.e + urst.e + urdst.e;
          ey(m,k,j,i) = -(fr.by + urst.by + urdst.by);
          ez(m,k,j,i) =  (fr.bz + urst.bz + urdst.bz);
        } else {
          // return Fr*
          flx(m,IDN,k,j,i) = fr.d  + urst.d;
          flx(m,ivx,k,j,i) = fr.mx + urst.mx;
          flx(m,_ivy,k,j,i) = fr.my + urst.my;
          flx(m,_ivz,k,j,i) = fr.mz + urst.mz;
          flx(m,IEN,k,j,i) = fr.e  + urst.e;
          ey(m,k,j,i) = -(fr.by + urst.by);
          ez(m,k,j,i) =  (fr.bz + urst.bz);
        }
      }
  //}
  //------------------------- ISOTHERMAL HLLD solver -------------------------------------
 

  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_HLLD_MHD_HPP_
