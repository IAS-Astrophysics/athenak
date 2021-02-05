//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_corner_e.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::CornerE
//  \brief calculate the corner EMFs

TaskStatus MHD::CornerE(Driver *pdriver, int stage)
{
  int is = pmy_pack->mb_cells.is; int ie = pmy_pack->mb_cells.ie;
  int js = pmy_pack->mb_cells.js; int je = pmy_pack->mb_cells.je;
  int ks = pmy_pack->mb_cells.ks; int ke = pmy_pack->mb_cells.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // capture class variables for the kernels
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto e2x1 = e2x1_;
  auto e3x1 = e3x1_;
  auto e1x2 = e1x2_;
  auto e3x2 = e3x2_;
  auto e1x3 = e1x3_;
  auto e2x3 = e2x3_;
  auto w0_ = w0;
  auto b0_ = bcc0;


  //---- 1-D problem:
  //  copy face-centered E-fields to edges and return.
  //  Note e2[is:ie+1,js:je,  ks:ke+1]
  //       e3[is:ie+1,js:je+1,ks:ke  ]

  if (!(pmy_pack->pmesh->nx2gt1)) {
    par_for("emf1", DevExeSpace(), 0, nmb1, is, ie+1,
      KOKKOS_LAMBDA(int m, int i)
      {
        e2(m,ks  ,js  ,i) = e2x1(m,ks,js,i);
        e2(m,ke+1,js  ,i) = e2x1(m,ks,js,i);
        e3(m,ks  ,js  ,i) = e3x1(m,ks,js,i);
        e3(m,ks  ,je+1,i) = e3x1(m,ks,js,i);
      }
    );
    return TaskStatus::complete;
  }

  //---- 2-D problem:
  // Copy face-centered E1 and E2 to edges, use GS07 algorithm to compute E3

  if (!(pmy_pack->pmesh->nx3gt1)) {

    // Compute cell-centered E3 = -(v X B) = VyBx-VxBy
    par_for("e_cc_2d", DevExeSpace(), 0, nmb1, js-1, je+1, is-1, ie+1,
      KOKKOS_LAMBDA(int m, int j, int i)
      {
        e3_cc_(m,ks,j,i) = w0_(m,IVY,ks,j,i)*b0_(m,IBX,ks,j,i) -
                           w0_(m,IVX,ks,j,i)*b0_(m,IBY,ks,j,i);
      }
    );

    // integrate E3 to corner using SG07
    //  Note e1[is:ie,  js:je+1,ks:ke+1]
    //       e2[is:ie+1,js:je,  ks:ke+1]
    //       e3[is:ie+1,js:je+1,ks:ke  ]
    auto flx1 = uflx.x1f;
    auto flx2 = uflx.x2f;
    par_for("emf2", DevExeSpace(), 0, nmb1, js, je+1, is, ie+1,
      KOKKOS_LAMBDA(const int m, const int j, const int i)
      {
        e2(m,ks  ,j,i) = e2x1(m,ks,j,i);
        e2(m,ke+1,j,i) = e2x1(m,ks,j,i);
        e1(m,ks  ,j,i) = e1x2(m,ks,j,i);
        e1(m,ke+1,j,i) = e1x2(m,ks,j,i);

        Real e3_l1, e3_l2, e3_r1, e3_r2;
        if (flx1(m,IDN,ks,j-1,i) >= 0.0) {
          e3_l2 = e3x2(m,ks,j,i-1) - e3_cc_(m,ks,j-1,i-1);
        } else {
          e3_l2 = e3x2(m,ks,j,i  ) - e3_cc_(m,ks,j-1,i  );
        }
        if (flx1(m,IDN,ks,j,i) >= 0.0) {
          e3_r2 = e3x2(m,ks,j,i-1) - e3_cc_(m,ks,j  ,i-1);
        } else {
          e3_r2 = e3x2(m,ks,j,i  ) - e3_cc_(m,ks,j  ,i  );
        }
        if (flx2(m,IDN,ks,j,i-1) >= 0.0) {
          e3_l1 = e3x1(m,ks,j-1,i) - e3_cc_(m,ks,j-1,i-1);
        } else {
          e3_l1 = e3x1(m,ks,j  ,i) - e3_cc_(m,ks,j  ,i-1);
        }
        if (flx2(m,IDN,ks,j,i) >= 0.0) {
          e3_r1 = e3x1(m,ks,j-1,i) - e3_cc_(m,ks,j-1,i  );
        } else {
          e3_r1 = e3x1(m,ks,j  ,i) - e3_cc_(m,ks,j  ,i  );
        }
        e3(m,ks,j,i) = 0.25*(e3_l1 + e3_r1 + e3_l2 + e3_r2 +
               e3x2(m,ks,j,i-1) + e3x2(m,ks,j,i) + e3x1(m,ks,j-1,i) + e3x1(m,ks,j,i));
      }
    );
    return TaskStatus::complete;
  }

  //---- 3-D problem:
  // Use GS07 algorithm to compute all three of E1, E2, and E3

  // Compute cell-centered electric fields
  // E1=-(v X B)=VzBy-VyBz
  // E2=-(v X B)=VxBz-VzBx
  // E3=-(v X B)=VyBx-VxBy
  par_for("e_cc_2d", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1, is-1, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i)
    {
      e1_cc_(m,k,j,i)=w0_(m,IVZ,k,j,i)*b0_(m,IBY,k,j,i)-w0_(m,IVY,k,j,i)*b0_(m,IBZ,k,j,i);
      e2_cc_(m,k,j,i)=w0_(m,IVX,k,j,i)*b0_(m,IBZ,k,j,i)-w0_(m,IVZ,k,j,i)*b0_(m,IBX,k,j,i);
      e3_cc_(m,k,j,i)=w0_(m,IVY,k,j,i)*b0_(m,IBX,k,j,i)-w0_(m,IVX,k,j,i)*b0_(m,IBY,k,j,i);
    }
  );

  // Integrate E1, E2, E3 to corners
  //  Note e1[is:ie,  js:je+1,ks:ke+1]
  //       e2[is:ie+1,js:je,  ks:ke+1]
  //       e3[is:ie+1,js:je+1,ks:ke  ]
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  par_for("emf3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
    {
      // integrate E1 to corner using SG07
      Real e1_l3, e1_r3, e1_l2, e1_r2;
      if (flx2(m,IDN,k-1,j,i) >= 0.0) {
        e1_l3 = e1x3(m,k,j  ,i) - e1_cc_(m,k-1,j  ,i);
      } else {
        e1_l3 = e1x3(m,k,j-1,i) - e1_cc_(m,k-1,j-1,i);
      }
      if (flx2(m,IDN,k,j,i) >= 0.0) {
        e1_r3 = e1x3(m,k,j  ,i) - e1_cc_(m,k  ,j  ,i);
      } else {
        e1_r3 = e1x3(m,k,j-1,i) - e1_cc_(m,k  ,j-1,i);
      }
      if (flx3(m,IDN,k,j-1,i) >= 0.0) {
        e1_l2 = e1x2(m,k  ,j,i) - e1_cc_(m,k  ,j-1,i);
      } else {
        e1_l2 = e1x2(m,k-1,j,i) - e1_cc_(m,k-1,j-1,i);
      }
      if (flx3(m,IDN,k,j,i) >= 0.0) {
        e1_r2 = e1x2(m,k  ,j,i) - e1_cc_(m,k  ,j  ,i);
      } else {
        e1_r2 = e1x2(m,k-1,j,i) - e1_cc_(m,k-1,j  ,i);
      }
      e1(m,k,j,i) = 0.25*(e1_l3 + e1_r3 + e1_l2 + e1_r2 +
                    e1x2(m,k-1,j,i) + e1x2(m,k,j,i) + e1x3(m,k,j-1,i) + e1x3(m,k,j,i));

      // integrate E2 to corner using SG07
      Real e2_l3, e2_r3, e2_l1, e2_r1;
      if (flx1(m,IDN,k-1,j,i) >= 0.0) {
        e2_l3 = e2x3(m,k,j,i  ) - e2_cc_(m,k-1,j,i  );
      } else {
        e2_l3 = e2x3(m,k,j,i-1) - e2_cc_(m,k-1,j,i-1);
      }
      if (flx1(m,IDN,k,j,i) >= 0.0) {
        e2_r3 = e2x3(m,k,j,i  ) - e2_cc_(m,k  ,j,i  );
      } else {
        e2_r3 = e2x3(m,k,j,i-1) - e2_cc_(m,k  ,j,i-1);
      }
      if (flx3(m,IDN,k,j,i-1) >= 0.0) {
        e2_l1 = e2x1(m,k  ,j,i) - e2_cc_(m,k  ,j,i-1);
      } else {
        e2_l1 = e2x1(m,k-1,j,i) - e2_cc_(m,k-1,j,i-1);
      }
      if (flx3(m,IDN,k,j,i) >= 0.0) {
        e2_r1 = e2x1(m,k  ,j,i) - e2_cc_(m,k  ,j,i  );
      } else {
        e2_r1 = e2x1(m,k-1,j,i) - e2_cc_(m,k-1,j,i  );
      }
      e2(m,k,j,i) = 0.25*(e2_l3 + e2_r3 + e2_l1 + e2_r1 +
                    e2x3(m,k,j,i-1) + e2x3(m,k,j,i) + e2x1(m,k-1,j,i) + e2x1(m,k,j,i));

      // integrate E3 to corner using SG07
      Real e3_l2, e3_r2, e3_l1, e3_r1;
      if (flx1(m,IDN,k,j-1,i) >= 0.0) {
        e3_l2 = e3x2(m,k,j,i  ) - e3_cc_(m,k,j-1,i  );
      } else {
        e3_l2 = e3x2(m,k,j,i-1) - e3_cc_(m,k,j-1,i-1);
      }
      if (flx1(m,IDN,k,j,i) >= 0.0) {
        e3_r2 = e3x2(m,k,j,i  ) - e3_cc_(m,k,j  ,i  );
      } else {
        e3_r2 = e3x2(m,k,j,i-1) - e3_cc_(m,k,j  ,i-1);
      }
      if (flx2(m,IDN,k,j,i-1) >= 0.0) {
        e3_l1 = e3x1(m,k,j  ,i) - e3_cc_(m,k,j  ,i-1);
      } else {
        e3_l1 = e3x1(m,k,j-1,i) - e3_cc_(m,k,j-1,i-1);
      }
      if (flx2(m,IDN,k,j,i) >= 0.0) {
        e3_r1 = e3x1(m,k,j  ,i) - e3_cc_(m,k,j  ,i  );
      } else {
        e3_r1 = e3x1(m,k,j-1,i) - e3_cc_(m,k,j-1,i  );
      }
      e3(m,k,j,i) = 0.25*(e3_l1 + e3_r1 + e3_l2 + e3_r2 +
                    e3x2(m,k,j,i-1) + e3x2(m,k,j,i) + e3x1(m,k,j-1,i) + e3x1(m,k,j,i));

    }
  );

  return TaskStatus::complete;
}
} // namespace mhd
