//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement.cpp
//! \brief Implements constructor and functions in MeshRefinement class.
//! Note prolongation is part of BVals classes.

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// Mesh constructor: initializes some mesh variables at start of calculation using
// parameters in input file.  Most objects in Mesh are constructed in the BuildTree()
// function, so that they can store a pointer to the Mesh which can be reliably referenced
// only after the Mesh constructor has finished

MeshRefinement::MeshRefinement(Mesh *pm, ParameterInput *pin) :
  pmy_mesh(pm) {
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictCC
//!  \brief Restricts cell-centered variables to coarse mesh

void MeshRefinement::RestrictCC(DvceArray5D<Real> &u, DvceArray5D<Real> &cu) {
  int nmb  = u.extent_int(0);  // TODO(@user): 1st index from L of in array must be NMB
  int nvar = u.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  auto &cis = pmy_mesh->mb_indcs.cis;
  auto &cie = pmy_mesh->mb_indcs.cie;
  auto &cjs = pmy_mesh->mb_indcs.cjs;
  auto &cje = pmy_mesh->mb_indcs.cje;
  auto &cks = pmy_mesh->mb_indcs.cks;
  auto &cke = pmy_mesh->mb_indcs.cke;

  // restrict in 1D
  if (pmy_mesh->one_d) {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, 0, nvar-1, cis, cie,
    KOKKOS_LAMBDA(const int m, const int n, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      cu(m,n,cks,cjs,i) = 0.5*(u(m,n,cks,cjs,finei) + u(m,n,cks,cjs,finei+1));
    });
  // restrict in 2D
  } else if (pmy_mesh->two_d) {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, 0, nvar-1, cjs, cje, cis, cie,
    KOKKOS_LAMBDA(const int m, const int n, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      cu(m,n,cks,j,i) = 0.25*(u(m,n,cks,finej  ,finei) + u(m,n,cks,finej  ,finei+1)
                            + u(m,n,cks,finej+1,finei) + u(m,n,cks,finej+1,finei+1));
    });

  // restrict in 3D
  } else {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, 0, nvar-1, cks, cke, cjs, cje, cis, cie,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      int finek = 2*k - cks;  // correct when cks=ks
      cu(m,n,k,j,i) =
          0.125*(u(m,n,finek  ,finej  ,finei) + u(m,n,finek  ,finej  ,finei+1)
               + u(m,n,finek  ,finej+1,finei) + u(m,n,finek  ,finej+1,finei+1)
               + u(m,n,finek+1,finej,  finei) + u(m,n,finek+1,finej,  finei+1)
               + u(m,n,finek+1,finej+1,finei) + u(m,n,finek+1,finej+1,finei+1));
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::RestrictFC
//! \brief Restricts face-centered variables to coarse mesh

void MeshRefinement::RestrictFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb) {
  int nmb  = b.x1f.extent_int(0);  // TODO(@user): 1st idx from L of in array must be NMB

  auto &cis = pmy_mesh->mb_indcs.cis;
  auto &cie = pmy_mesh->mb_indcs.cie;
  auto &cjs = pmy_mesh->mb_indcs.cjs;
  auto &cje = pmy_mesh->mb_indcs.cje;
  auto &cks = pmy_mesh->mb_indcs.cks;
  auto &cke = pmy_mesh->mb_indcs.cke;

  // restrict in 1D
  if (pmy_mesh->one_d) {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, cis, cie,
    KOKKOS_LAMBDA(const int m, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      // restrict B1
      cb.x1f(m,cks,cjs,i) = b.x1f(m,cks,cjs,finei);
      if (i==cie) {
        cb.x1f(m,cks,cjs,i+1) = b.x1f(m,cks,cjs,finei+2);
      }
      // restrict B2
      Real b2coarse = 0.5*(b.x2f(m,cks,cjs,finei) + b.x2f(m,cks,cjs,finei+1));
      cb.x2f(m,cks,cjs  ,i) = b2coarse;
      cb.x2f(m,cks,cjs+1,i) = b2coarse;
      // restrict B3
      Real b3coarse = 0.5*(b.x3f(m,cks,cjs,finei) + b.x3f(m,cks,cjs,finei+1));
      cb.x3f(m,cks  ,cjs,i) = b3coarse;
      cb.x3f(m,cks+1,cjs,i) = b3coarse;
    });

  // restrict in 2D
  } else if (pmy_mesh->two_d) {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, cjs, cje, cis, cie,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      // restrict B1
      cb.x1f(m,cks,j,i) = 0.5*(b.x1f(m,cks,finej,finei) + b.x1f(m,cks,finej+1,finei));
      if (i==cie) {
        cb.x1f(m,cks,j,i+1) =
          0.5*(b.x1f(m,cks,finej,finei+2) + b.x1f(m,cks,finej+1,finei+2));
      }
      // restrict B2
      cb.x2f(m,cks,j,i) = 0.5*(b.x2f(m,cks,finej,finei) + b.x2f(m,cks,finej,finei+1));
      if (j==cje) {
        cb.x2f(m,cks,j+1,i) =
          0.5*(b.x2f(m,cks,finej+2,finei) + b.x2f(m,cks,finej+2,finei+1));
      }
      // restrict B3
      Real b3coarse = 0.25*(b.x3f(m,cks,finej  ,finei) + b.x3f(m,cks,finej  ,finei+1)
                          + b.x3f(m,cks,finej+1,finei) + b.x3f(m,cks,finej+1,finei+1));
      cb.x3f(m,cks  ,j,i) = b3coarse;
      cb.x3f(m,cks+1,j,i) = b3coarse;
    });

  // restrict in 3D
  } else {
    par_for("restrict3D",DevExeSpace(),0, nmb-1, cks, cke, cjs, cje, cis, cie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      int finek = 2*k - cks;  // correct when cks=ks
      // restrict B1
      cb.x1f(m,k,j,i) =
        0.25*(b.x1f(m,finek  ,finej,finei) + b.x1f(m,finek  ,finej+1,finei)
            + b.x1f(m,finek+1,finej,finei) + b.x1f(m,finek+1,finej+1,finei));
      if (i==cie) {
        cb.x1f(m,k,j,i+1) =
          0.25*(b.x1f(m,finek  ,finej,finei+2) + b.x1f(m,finek  ,finej+1,finei+2)
              + b.x1f(m,finek+1,finej,finei+2) + b.x1f(m,finek+1,finej+1,finei+2));
      }
      // restrict B2
      cb.x2f(m,k,j,i) =
        0.25*(b.x2f(m,finek  ,finej,finei) + b.x2f(m,finek  ,finej,finei+1)
            + b.x2f(m,finek+1,finej,finei) + b.x2f(m,finek+1,finej,finei+1));
      if (j==cje) {
        cb.x2f(m,k,j+1,i) =
          0.25*(b.x2f(m,finek  ,finej+2,finei) + b.x2f(m,finek  ,finej+2,finei+1)
              + b.x2f(m,finek+1,finej+2,finei) + b.x2f(m,finek+1,finej+2,finei+1));
      }
      // restrict B3
      cb.x3f(m,k,j,i) =
        0.25*(b.x3f(m,finek,finej  ,finei) + b.x3f(m,finek,finej  ,finei+1)
            + b.x3f(m,finek,finej+1,finei) + b.x3f(m,finek,finej+1,finei+1));
      if (k==cke) {
        cb.x3f(m,k+1,j,i) =
          0.25*(b.x3f(m,finek+2,finej  ,finei) + b.x3f(m,finek+2,finej  ,finei+1)
              + b.x3f(m,finek+2,finej+1,finei) + b.x3f(m,finek+2,finej+1,finei+1));
      }
    });
  }
  return;
}
