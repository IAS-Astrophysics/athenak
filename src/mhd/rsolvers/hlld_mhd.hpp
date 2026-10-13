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

//----------------------------------------------------------------------------------------
//! \fn

KOKKOS_INLINE_FUNCTION
void HLLD(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;
  Real spd[5];         // signal speeds, left to right

  //------------------------ ADIABATIC HLLD solver ---------------------------------------
  if (eos.is_ideal) {
    Real gm1 = eos.gamma - 1.0;
    Real igm1 = 1.0/gm1;
    par_for_inner(member, il, iu, [&](const int i) {
      //--- Step 1.  Create local references for L/R states (helps compiler vectorize)

      Real &wl_idn=wl(IDN,i);
      Real &wl_ivx=wl(ivx,i);
      Real &wl_ivy=wl(ivy,i);
      Real &wl_ivz=wl(ivz,i);
      Real &wl_iby=bl(iby,i);
      Real &wl_ibz=bl(ibz,i);

      Real &wr_idn=wr(IDN,i);
      Real &wr_ivx=wr(ivx,i);
      Real &wr_ivy=wr(ivy,i);
      Real &wr_ivz=wr(ivz,i);
      Real &wr_iby=br(iby,i);
      Real &wr_ibz=br(ibz,i);

      Real wl_ipr, wr_ipr;
      wl_ipr = eos.IdealGasPressure(wl(IEN,i));
      wr_ipr = eos.IdealGasPressure(wr(IEN,i));

      Real &bxi = bx(m,k,j,i);

      // Compute L/R states for selected conserved variables
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

      MHDCons1D fl,fr,flxi;           // Fluxes for left & right states
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
      Real sdml_inv = 1.0/sdml;
      Real sdmr_inv = 1.0/sdmr;

      MHDCons1D ulst,uldst,urdst,urst;   // intermadiate states for conserved variables
      // eqn (43) of Miyoshi & Kusano
      ulst.d = ul.d * sdl * sdml_inv;
      urst.d = ur.d * sdr * sdmr_inv;
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
      Real ptstl = ptl + ul.d*sdl*(spd[2]-wl_ivx);
      Real ptstr = ptr + ur.d*sdr*(spd[2]-wr_ivx);
      // Real ptstl = ptl + ul.d*sdl*(sdl-sdml); // these eqns had issues when averaged
      // Real ptstr = ptr + ur.d*sdr*(sdr-sdmr);
      Real ptst = 0.5*(ptstr + ptstl);  // total pressure (star state)

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
      Real vbstl = (ulst.mx*bxi+(ulst.my*ulst.by+ulst.mz*ulst.bz))*ulst_d_inv;
      // eqn (48) of M&K
      // (KGF): group transverse by, bz terms for floating-point associativity symmetry
      ulst.e = (sdl*ul.e - ptl*wl_ivx + ptst*spd[2] +
                bxi*(wl_ivx*bxi + (wl_ivy*ul.by + wl_ivz*ul.bz) - vbstl))*sdml_inv;

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
      Real vbstr = (urst.mx*bxi+(urst.my*urst.by+urst.mz*urst.bz))*urst_d_inv;
      // eqn (48) of M&K
      // (KGF): group transverse by, bz terms for floating-point associativity symmetry
      urst.e = (sdr*ur.e - ptr*wr_ivx + ptst*spd[2] +
                bxi*(wr_ivx*bxi + (wr_ivy*ur.by + wr_ivz*ur.bz) - vbstr))*sdmr_inv;
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
        flxi.d = fl.d;
        flxi.mx = fl.mx;
        flxi.my = fl.my;
        flxi.mz = fl.mz;
        flxi.e  = fl.e;
        flxi.by = fl.by;
        flxi.bz = fl.bz;
      } else if (spd[4] <= 0.0) {
        // return Fr if flow is supersonic
        flxi.d = fr.d;
        flxi.mx = fr.mx;
        flxi.my = fr.my;
        flxi.mz = fr.mz;
        flxi.e  = fr.e;
        flxi.by = fr.by;
        flxi.bz = fr.bz;
      } else if (spd[1] >= 0.0) {
        // return Fl*
        flxi.d = fl.d  + ulst.d;
        flxi.mx = fl.mx + ulst.mx;
        flxi.my = fl.my + ulst.my;
        flxi.mz = fl.mz + ulst.mz;
        flxi.e  = fl.e  + ulst.e;
        flxi.by = fl.by + ulst.by;
        flxi.bz = fl.bz + ulst.bz;
      } else if (spd[2] >= 0.0) {
        // return Fl**
        flxi.d = fl.d  + ulst.d + uldst.d;
        flxi.mx = fl.mx + ulst.mx + uldst.mx;
        flxi.my = fl.my + ulst.my + uldst.my;
        flxi.mz = fl.mz + ulst.mz + uldst.mz;
        flxi.e  = fl.e  + ulst.e + uldst.e;
        flxi.by = fl.by + ulst.by + uldst.by;
        flxi.bz = fl.bz + ulst.bz + uldst.bz;
      } else if (spd[3] > 0.0) {
        // return Fr**
        flxi.d = fr.d + urst.d + urdst.d;
        flxi.mx = fr.mx + urst.mx + urdst.mx;
        flxi.my = fr.my + urst.my + urdst.my;
        flxi.mz = fr.mz + urst.mz + urdst.mz;
        flxi.e  = fr.e + urst.e + urdst.e;
        flxi.by = fr.by + urst.by + urdst.by;
        flxi.bz = fr.bz + urst.bz + urdst.bz;
      } else {
        // return Fr*
        flxi.d = fr.d  + urst.d;
        flxi.mx = fr.mx + urst.mx;
        flxi.my = fr.my + urst.my;
        flxi.mz = fr.mz + urst.mz;
        flxi.e  = fr.e  + urst.e;
        flxi.by = fr.by + urst.by;
        flxi.bz = fr.bz + urst.bz;
      }

      flx(m,IDN,k,j,i) = flxi.d;
      flx(m,ivx,k,j,i) = flxi.mx;
      flx(m,ivy,k,j,i) = flxi.my;
      flx(m,ivz,k,j,i) = flxi.mz;
      flx(m,IEN,k,j,i) = flxi.e;
      ey(m,k,j,i) = -flxi.by;
      ez(m,k,j,i) =  flxi.bz;
    });

  //------------------------- ISOTHERMAL HLLD solver -------------------------------------
  } else {
    auto &dfloor_ = eos.dfloor;
    Real iso_cs = eos.iso_cs;
    par_for_inner(member, il, iu, [&](const int i) {
      //--- Step 1.  Load L/R states into local variables

      Real &wl_idn=wl(IDN,i);
      Real &wl_ivx=wl(ivx,i);
      Real &wl_ivy=wl(ivy,i);
      Real &wl_ivz=wl(ivz,i);
      Real &wl_iby=bl(iby,i);
      Real &wl_ibz=bl(ibz,i);

      Real &wr_idn=wr(IDN,i);
      Real &wr_ivx=wr(ivx,i);
      Real &wr_ivy=wr(ivy,i);
      Real &wr_ivz=wr(ivz,i);
      Real &wr_iby=br(iby,i);
      Real &wr_ibz=br(ibz,i);

      Real &bxi = bx(m,k,j,i);

      // Compute L/R states for selected conserved variables
      MHDCons1D ul,ur;
      ul.d  = wl_idn;
      ul.mx = wl_ivx*ul.d;
      ul.my = wl_ivy*ul.d;
      ul.mz = wl_ivz*ul.d;
      ul.by = wl_iby;
      ul.bz = wl_ibz;

      ur.d  = wr_idn;
      ur.mx = wr_ivx*ur.d;
      ur.my = wr_ivy*ur.d;
      ur.mz = wr_ivz*ur.d;
      ur.by = wr_iby;
      ur.bz = wr_ibz;

      //--- Step 2.  Compute L & R wave speeds according to Miyoshi & Kusano, eqn. (67)

      Real cfl = eos.IdealMHDFastSpeed(wl_idn, bxi, wl_iby, wl_ibz);
      Real cfr = eos.IdealMHDFastSpeed(wr_idn, bxi, wr_iby, wr_ibz);

      spd[0] = fmin( wl_ivx-cfl, wr_ivx-cfr );
      spd[4] = fmax( wl_ivx+cfl, wr_ivx+cfr );

      //--- Step 3.  Compute L/R fluxes

      // total pressures L,R
      Real bxsq = bxi*bxi;
      Real ptl = SQR(iso_cs)*wl_idn + 0.5*(bxsq + SQR(wl_iby) + SQR(wl_ibz));
      Real ptr = SQR(iso_cs)*wr_idn + 0.5*(bxsq + SQR(wr_iby) + SQR(wr_ibz));

      MHDCons1D fl,fr,flxi;  // Fluxes for left & right states, and interface
      fl.d  = ul.mx;
      fl.mx = ul.mx*wl_ivx + ptl - bxsq;
      fl.my = ul.my*wl_ivx - bxi*ul.by;
      fl.mz = ul.mz*wl_ivx - bxi*ul.bz;
      fl.by = ul.by*wl_ivx - bxi*wl_ivy;
      fl.bz = ul.bz*wl_ivx - bxi*wl_ivz;

      fr.d  = ur.mx;
      fr.mx = ur.mx*wr_ivx + ptr - bxsq;
      fr.my = ur.my*wr_ivx - bxi*ur.by;
      fr.mz = ur.mz*wr_ivx - bxi*ur.bz;
      fr.by = ur.by*wr_ivx - bxi*wr_ivy;
      fr.bz = ur.bz*wr_ivx - bxi*wr_ivz;

      //--- Step 4.  Compute hll averages and Alfven wave speed

      // inverse of difference between right and left signal speeds
      Real idspd = 1.0/(spd[4]-spd[0]);

      // rho component of U^{hll} from Mignone eqn. (15); uses F_L and F_R from eqn. (6)
      Real dhll = (spd[4]*ur.d - spd[0]*ul.d - fr.d + fl.d)*idspd;
      dhll = fmax(dhll, dfloor_);
      Real sqrtdhll = sqrt(dhll);

      // rho and mx components of F^{hll} from Mignone eqn. (17)
      Real fdhll  = (spd[4]*fl.d  - spd[0]*fr.d  + spd[4]*spd[0]*(ur.d -ul.d ))*idspd;
      Real fmxhll = (spd[4]*fl.mx - spd[0]*fr.mx + spd[4]*spd[0]*(ur.mx-ul.mx))*idspd;

      // ustar from paragraph between eqns. (23) and (24)
      Real ustar = fdhll/dhll;

      // mx component of U^{hll} from Mignone eqn. (15); paragraph referenced
      // above states that mxhll should NOT be used to compute ustar
      Real mxhll = (spd[4]*ur.mx - spd[0]*ul.mx - fr.mx + fl.mx)*idspd;

      // S*_L and S*_R from Mignone eqn. (29)
      spd[1] = ustar - fabs(bxi)/sqrtdhll;
      spd[3] = ustar + fabs(bxi)/sqrtdhll;

      //--- Step 5. Compute intermediate states

      MHDCons1D ulst,urst,ucst;  // Conserved variable for all states
      // Ul* - eqn. (20) of Mignone
      ulst.d  = dhll;
      ulst.mx = mxhll; // eqn. (24) of Mignone

      Real tmp = (spd[0]-spd[1])*(spd[0]-spd[3]);
      if (fabs(spd[0]-spd[1]) < (HLLD_SMALL_NUMBER)*iso_cs) {
        // degenerate case described below eqn. (39)
        ulst.my = ul.my;
        ulst.mz = ul.mz;
        ulst.by = ul.by;
        ulst.bz = ul.bz;
      } else {
        Real mfact = bxi*(ustar-wl_ivx)/tmp;
        Real bfact = (ul.d*SQR(spd[0]-wl_ivx) - bxsq)/(dhll*tmp);

        ulst.my = dhll*wl_ivy - ul.by*mfact; // eqn. (30) of Mignone
        ulst.mz = dhll*wl_ivz - ul.bz*mfact; // eqn. (31) of Mignone
        ulst.by = ul.by*bfact; // eqn. (32) of Mignone
        ulst.bz = ul.bz*bfact; // eqn. (33) of Mignone
      }

      // Ur* - eqn. (20) of Mignone */
      urst.d  = dhll;
      urst.mx = mxhll; // eqn. (24) of Mignone

      tmp = (spd[4]-spd[1])*(spd[4]-spd[3]);
      if (fabs(spd[4]-spd[3]) < (HLLD_SMALL_NUMBER)*iso_cs) {
        // degenerate case described below eqn. (39)
        urst.my = ur.my;
        urst.mz = ur.mz;
        urst.by = ur.by;
        urst.bz = ur.bz;
      } else {
        Real mfact = bxi*(ustar-wr_ivx)/tmp;
        Real bfact = (ur.d*SQR(spd[4]-wr_ivx) - bxsq)/(dhll*tmp);

        urst.my = dhll*wr_ivy - ur.by*mfact; // eqn. (30) of Mignone
        urst.mz = dhll*wr_ivz - ur.bz*mfact; // eqn. (31) of Mignone
        urst.by = ur.by*bfact; // eqn. (32) of Mignone
        urst.bz = ur.bz*bfact; // eqn. (33) of Mignone
      }

      // Uc*
      Real x = sqrtdhll*(bxi > 0.0 ? 1.0 : -1.0); // from below eqn. (37) of Mignone
      ucst.d  = dhll;  // eqn. (20) of Mignone
      ucst.mx = mxhll; // eqn. (24) of Mignone
      ucst.my = 0.5*(ulst.my + urst.my + (urst.by-ulst.by)*x); // eqn. (34) of Mignone
      ucst.mz = 0.5*(ulst.mz + urst.mz + (urst.bz-ulst.bz)*x); // eqn. (35) of Mignone
      ucst.by = 0.5*(ulst.by + urst.by + (urst.my-ulst.my)/x); // eqn. (36) of Mignone
      ucst.bz = 0.5*(ulst.bz + urst.bz + (urst.mz-ulst.mz)/x); // eqn. (37) of Mignone

      //--- Step 6.  Compute flux

      if (spd[0] >= 0.0) {
        // return Fl if flow is supersonic, eqn. (38a) of Mignone
        flxi.d  = fl.d;
        flxi.mx = fl.mx;
        flxi.my = fl.my;
        flxi.mz = fl.mz;
        flxi.by = fl.by;
        flxi.bz = fl.bz;
      } else if (spd[4] <= 0.0) {
        // return Fr if flow is supersonic, eqn. (38e) of Mignone
        flxi.d  = fr.d;
        flxi.mx = fr.mx;
        flxi.my = fr.my;
        flxi.mz = fr.mz;
        flxi.by = fr.by;
        flxi.bz = fr.bz;
      } else if (spd[1] >= 0.0) {
        // return (Fl+Sl*(Ulst-Ul)), eqn. (38b) of Mignone
        flxi.d  = fl.d  + spd[0]*(ulst.d  - ul.d);
        flxi.mx = fl.mx + spd[0]*(ulst.mx - ul.mx);
        flxi.my = fl.my + spd[0]*(ulst.my - ul.my);
        flxi.mz = fl.mz + spd[0]*(ulst.mz - ul.mz);
        flxi.by = fl.by + spd[0]*(ulst.by - ul.by);
        flxi.bz = fl.bz + spd[0]*(ulst.bz - ul.bz);
      } else if (spd[3] <= 0.0) {
        // return (Fr+Sr*(Urst-Ur)), eqn. (38d) of Mignone
        flxi.d  = fr.d  + spd[4]*(urst.d  - ur.d);
        flxi.mx = fr.mx + spd[4]*(urst.mx - ur.mx);
        flxi.my = fr.my + spd[4]*(urst.my - ur.my);
        flxi.mz = fr.mz + spd[4]*(urst.mz - ur.mz);
        flxi.by = fr.by + spd[4]*(urst.by - ur.by);
        flxi.bz = fr.bz + spd[4]*(urst.bz - ur.bz);
      } else {
        // return Fcst, eqn. (38c) of Mignone, using eqn. (24)
        flxi.d  = dhll*ustar;
        flxi.mx = fmxhll;
        flxi.my = ucst.my*ustar - bxi*ucst.by;
        flxi.mz = ucst.mz*ustar - bxi*ucst.bz;
        flxi.by = ucst.by*ustar - bxi*ucst.my/ucst.d;
        flxi.bz = ucst.bz*ustar - bxi*ucst.mz/ucst.d;
      }

      flx(m,IDN,k,j,i) = flxi.d;
      flx(m,ivx,k,j,i) = flxi.mx;
      flx(m,ivy,k,j,i) = flxi.my;
      flx(m,ivz,k,j,i) = flxi.mz;
      ey(m,k,j,i) = -flxi.by;
      ez(m,k,j,i) =  flxi.bz;
    });
  } // end ideal gas/isothermal solvers

  return;
}
} // namespace mhd
#endif // MHD_RSOLVERS_HLLD_MHD_HPP_
