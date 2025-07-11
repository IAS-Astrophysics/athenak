//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grmhd_zoomin.cpp
//  \brief pgen for a zoom-in GRMHD simulation

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "utils/tov/tov_tabulated.hpp"
#include "utils/tov/tov_utils.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"

//#define DEBUG_PGEN

#ifndef NCHEBY
#define NCHEBY 63
#endif

#ifndef NSTENCIL
#define NSTENCIL 2
#endif

// Lagrange interpolation on the Chebyshev grid
template <int NB, int NS>
class ChebyshevInterpolation {
 public:
  static KOKKOS_INLINE_FUNCTION Real MapToCollocation(Real xmin, Real xmax,
                                                      Real x) {
    Real xi = 2.0 * (x - xmin) / (xmax - xmin) - 1.0;
    assert(fabs(xi) <= 1.0);
    return xi;
  }

 public:
  KOKKOS_INLINE_FUNCTION
  ChebyshevInterpolation() {
    static_assert(NS < NB/2);
    for (int j = 0; j <= NB; ++j) {
      xp[j] = cos((M_PI * j) / NB);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void ComputeInterpolationWeights(Real *w, int & xs, int & xe, Real x) {
    int j0 = 0;
    while (xp[j0 + 1] > x && j0 < NB) {
      j0++;
    }

    xs = j0 - NS;
    xe = j0 + NS + 1;
    while (xs < 0) {
      xs++; xe++;
    }
    while (xe > NB) {
      xe--; xs--;
    }

    for (int j = xs; j <= xe; ++j) {
      Real num = 1.0;
      Real den = 1.0;
      for (int k = xs; k <= xe; ++k) {
        if (k != j) {
          num = num*(x - xp[k]);
          den = den*(xp[j] - xp[k]);
        }
      }
      w[j-xs] = num/den;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void SetInterpolationPoint(Real x, Real y, Real z) {
    ComputeInterpolationWeights(&wx[0], is, ie, x);
    ComputeInterpolationWeights(&wy[0], js, je, y);
    ComputeInterpolationWeights(&wz[0], ks, ke, z);
  }

  KOKKOS_INLINE_FUNCTION
  Real Eval(DvceArray4D<Real> const &var, int vidx) {
    Real out = 0.0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          out += wx[i-is] * wy[j-js] * wz[k-ks] * var(vidx, k, j, i);
        }
      }
    }
    return out;
  }

 public:
  // Quadrature points
  Real xp[NB + 1];
  // Interpolation weights
  Real wx[2*NS + 2];
  Real wy[2*NS + 2];
  Real wz[2*NS + 2];
  // Interpolation stencil
  int is, ie;
  int js, je;
  int ks, ke;
};

class CartesianDumpReader {
 public:
  struct MetaData {
    int cycle;
    float time;
    float center[3];
    float extent[3];
    int numpoints[3];
    bool is_cheb;
    int noutvars;
  };
  CartesianDumpReader(char const *fname) {
    // Open output file
    FILE *fp = fopen(fname, "rb");
    if (fp == NULL) {
      fprintf(stderr, "I/O Error: could not open \"%s\"", fname);
      fflush(stderr);
      abort();
    }

    // Read metadata
    size_t sread = fread(&mdata, sizeof(MetaData), 1, fp);
    assert(mdata.numpoints[0] == mdata.numpoints[1]);
    assert(mdata.numpoints[0] == mdata.numpoints[2]);
    int const N = mdata.numpoints[0] - 1;

    std::vector<float> var((N + 1) * (N + 1) * (N + 1));

    // Read list of variables
    int len;
    sread = fread(&len, sizeof(int), 1, fp);
    assert(len < BUFSIZ);
    char msg[BUFSIZ];
    sread = fread(&msg[0], 1, len, fp);
    msg[len] = '\0';

    // Now read all variables
    char *token = strtok(msg, " ");
    while (NULL != token) {
      for (int i = 0; i < (N + 1) * (N + 1) * (N + 1); ++i) {
        sread = fread(&var[i], sizeof(float), 1, fp);
        assert(sread == 1);
      }
      data[std::string(token)] = var;
      token = strtok(NULL, " ");
    }

    fclose(fp);
  }

 public:
  MetaData mdata;
  std::map<std::string, std::vector<float>> data;
};

template <int N>
class BackgroundData {
 public:
  enum VariableIndex {
    IDN = 0,
    IVX = 1,
    IVY = 2,
    IVZ = 3,
    IPR = 4,
    IYF = 5,  // fluid variables
    IBX = 6,
    IBY = 7,
    IBZ = 8,  // magnetic field (densitized)
    IAVECX = 9,
    IAVECY = 10,
    IAVECZ = 11,  // magnetic vector potential
    IALP = 12,
    IBETAX = 13,
    IBETAY = 14,
    IBETAZ = 15,  // gauge
    IGXX = 16,
    IGXY = 17,
    IGXZ = 18,
    IGYY = 19,
    IGYZ = 20,
    IGZZ = 21,  // metric
    IKXX = 22,
    IKXY = 23,
    IKXZ = 24,
    IKYY = 25,
    IKYZ = 26,
    IKZZ = 27,  // vector potential
    NVARS = 28
  };
  constexpr static char const *const vnames[] = {
      "dens",      "velx",      "vely",      "velz",      "press",   "s_00",
      "bcc1",      "bcc2",      "bcc3",      "avec1",     "avec2",   "avec3",
      "z4c_alpha", "z4c_betax", "z4c_betay", "z4c_betaz", "adm_gxx", "adm_gxy",
      "adm_gxz",   "adm_gyy",   "adm_gyz",   "adm_gzz",   "adm_Kxx", "adm_Kxy",
      "adm_Kxz",   "adm_Kyy",   "adm_Kyz",   "adm_Kzz"};

  BackgroundData(ParameterInput *pin) {
    file_basename = pin->GetString("problem", "file_basename");
    fnmin = pin->GetInteger("problem", "fnmin");
    fnmax = pin->GetInteger("problem", "fnmax");
    ncache = pin->GetOrAddInteger("problem", "ncache", 128);
    if (ncache < 2) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "problem/ncache needs to be 2 or larger"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    Kokkos::realloc(cheb_data, NVARS, N + 1, N + 1, N + 1);
#ifdef DEBUG_PGEN
    printf("Reading cart data...");
    fflush(stdout);
#endif
    ReadMetaData();
    ReadData(fnmin);
#ifdef DEBUG_PGEN
    printf("done!\n");
#endif
  }

  void ReadMetaData() {
    Real center[3], extent[3];
    int ndumps = fnmax - fnmin + 1;
    mtimes.resize(ndumps);
    if (0 == global_variable::my_rank) {
      for (int n = fnmin; n <= fnmax; ++n) {
        char fname[BUFSIZ];
        {
          snprintf(fname, BUFSIZ, "%s%s.%05d.bin", file_basename.c_str(),
                   "mhd_w_bcc", n);
          CartesianDumpReader cart(fname);
          mtimes[n - fnmin] = cart.mdata.time;
          // Read grid extent (assumed to be the same for all files)
          if (n == fnmin) {
            for (int d = 0; d < 3; ++d) {
              center[d] = cart.mdata.center[d];
              extent[d] = cart.mdata.extent[d];
            }
          }
        }
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Bcast(&center[0], 3, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&extent[0], 3, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(mtimes.data(), mtimes.size(), MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif
    xmin = center[0] - extent[0];
    xmax = center[0] + extent[0];
    ymin = center[1] - extent[1];
    ymax = center[1] + extent[1];
    zmin = center[2] - extent[2];
    zmax = center[2] + extent[2];
  }

  void ReadData(int f0) {
    m_fn_start = f0;
    int nread = std::min(ncache, fnmax - f0 + 1);
    Kokkos::realloc(m_raw_data, nread, NVARS, N + 1, N + 1, N + 1);
    if (0 == global_variable::my_rank) {
      for (int n = f0; n < f0 + nread; ++n) {
        char fname[BUFSIZ];
        {
          snprintf(fname, BUFSIZ, "%s%s.%05d.bin", file_basename.c_str(),
                   "mhd_w_bcc", n);
          CartesianDumpReader cart(fname);
          for (int vi = IDN; vi <= IBZ; ++vi) {
#define BACKGROUND_DATA_READ_VARIABLE_FROM_FILE(XX)                      \
  std::vector<float> const &var = cart.data.at(std::string(vnames[XX])); \
  int ijk = 0;                                                           \
  for (int k = 0; k <= N; ++k) {                                         \
    for (int j = 0; j <= N; ++j) {                                       \
      for (int i = 0; i <= N; ++i) {                                     \
        m_raw_data(n - f0, XX, k, j, i) = var[ijk++];                    \
      }                                                                  \
    }                                                                    \
  }
            BACKGROUND_DATA_READ_VARIABLE_FROM_FILE(vi);
          }
        }
        {
          snprintf(fname, BUFSIZ, "%s%s.%05d.bin", file_basename.c_str(),
                   "avec", n);
          CartesianDumpReader cart(fname);
          for (int vi = IAVECX; vi <= IAVECZ; ++vi) {
            BACKGROUND_DATA_READ_VARIABLE_FROM_FILE(vi);
          }
        }
        {
          snprintf(fname, BUFSIZ, "%s%s.%05d.bin", file_basename.c_str(),
                   "z4c_alpha", n);
          CartesianDumpReader cart(fname);
          for (int vi = IALP; vi <= IALP; ++vi) {
            BACKGROUND_DATA_READ_VARIABLE_FROM_FILE(vi);
          }
        }
        {
          snprintf(fname, BUFSIZ, "%s%s.%05d.bin", file_basename.c_str(),
                   "z4c_betax", n);
          CartesianDumpReader cart(fname);
          BACKGROUND_DATA_READ_VARIABLE_FROM_FILE(IBETAX);
        }
        {
          snprintf(fname, BUFSIZ, "%s%s.%05d.bin", file_basename.c_str(),
                   "z4c_betay", n);
          CartesianDumpReader cart(fname);
          BACKGROUND_DATA_READ_VARIABLE_FROM_FILE(IBETAY);
        }
        {
          snprintf(fname, BUFSIZ, "%s%s.%05d.bin", file_basename.c_str(),
                   "z4c_betaz", n);
          CartesianDumpReader cart(fname);
          BACKGROUND_DATA_READ_VARIABLE_FROM_FILE(IBETAZ);
        }
        {
          snprintf(fname, BUFSIZ, "%s%s.%05d.bin", file_basename.c_str(),
                   "adm", n);
          CartesianDumpReader cart(fname);
          for (int vi = IGXX; vi <= IKZZ; ++vi) {
            BACKGROUND_DATA_READ_VARIABLE_FROM_FILE(vi);
          }
        }
#undef BACKGROUND_DATA_READ_VARIABLE_FROM_FILE
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Bcast(m_raw_data.data(), m_raw_data.size(), MPI_FLOAT, 0,
              MPI_COMM_WORLD);
#endif
  }

  void ExtractSlice(int idx) {
    if (idx < m_fn_start) {
      ReadData(idx);
    }
    if (idx >= m_fn_start + ncache) {
      ReadData(idx);
    }

    for (int vi = 0; vi < NVARS; ++vi) {
      for (int k = 0; k <= N; ++k) {
        for (int j = 0; j <= N; ++j) {
          for (int i = 0; i <= N; ++i) {
            cheb_data.h_view(vi, k, j, i) = m_raw_data(idx - m_fn_start, vi, k, j, i);
          }
        }
      }
    }
    // copy data to GPU
    cheb_data.template modify<HostMemSpace>();
    cheb_data.template sync<DevExeSpace>();
  }

  void InterpToTime(Real time) {
    time += mtimes.front();
    if (time <= mtimes.front()) {
      ExtractSlice(0);
    } else if (time > mtimes.back()) {
      ExtractSlice(fnmax);
    } else {
      auto tp = std::lower_bound(mtimes.begin(), mtimes.end(), time);
      int idx = std::distance(mtimes.begin(), tp) + fnmin - 1;
      assert(idx >= fnmin && idx <= fnmax);

      if (idx < m_fn_start || idx >= m_fn_start + ncache - 1) {
        ReadData(idx);
      }

      Real w = (time - mtimes[idx - fnmin]) /
               (mtimes[idx + 1 - fnmin] - mtimes[idx - fnmin]);

      for (int vi = 0; vi < NVARS; ++vi) {
        for (int k = 0; k <= N; ++k) {
          for (int j = 0; j <= N; ++j) {
            for (int i = 0; i <= N; ++i) {
              cheb_data.h_view(vi, k, j, i) =
                  (1.0 - w) * m_raw_data(idx - m_fn_start, vi, k, j, i) +
                  w * m_raw_data(idx + 1 - m_fn_start, vi, k, j, i);
            }
          }
        }
      }
      // copy data to GPU
      cheb_data.template modify<HostMemSpace>();
      cheb_data.template sync<DevExeSpace>();
    }
  }

 public:
  DualArray4D<Real> cheb_data;  // time interpolated data on the Chebyshev grid
  Real xmin, xmax, ymin, ymax, zmin, zmax;  // grid extent

  std::string file_basename;
  int fnmin, fnmax;  // minimum/maximum file numbers to read
  int ncache;  // number of snapshots to keep in memory

 private:
  std::vector<Real> mtimes;      // time of all the frames
  int m_fn_start;                // index of first file actually read
  HostArray5D<float> m_raw_data; // data as a function of time for all fields
                                 // on the Chebyshev grid
};

static BackgroundData<NCHEBY> *pmy_data;

// User defined BC
void TurbulenceBC(Mesh *pm);
void SetADMVariables(MeshBlockPack *pmbp);

// Prototypes for user-defined BCs and history
void ZoomHistory(HistoryData *pdata, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for BNS with LORENE
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "GRMHD zoom-in problem can only be run when <adm> block is present"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  user_bcs_func = TurbulenceBC;
  pmbp->padm->SetADMVariables = SetADMVariables;
  user_hist_func = &ZoomHistory;

  pmy_data = new BackgroundData<NCHEBY>(pin);
  if (restart) return;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  int &nmb = pmbp->nmb_thispack;
  Real box_x1min = pmy_data->xmin;
  Real box_x1max = pmy_data->xmax;
  Real box_x2min = pmy_data->ymin;
  Real box_x2max = pmy_data->ymax;
  Real box_x3min = pmy_data->zmin;
  Real box_x3max = pmy_data->zmax;

  auto w0_ = pmbp->pmhd->w0;
  auto adm = pmbp->padm->adm;

  pmy_data->ExtractSlice(0);
  auto &cheb_data = pmy_data->cheb_data;

#ifdef DEBUG_PGEN
  printf("Interpolating grid data...");
  fflush(stdout);
#endif
  // initialize all fields, apart from magnetic field -------------------------
  par_for("pgen_data", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real xi1 = interp.MapToCollocation(box_x1min, box_x1max, x1v);
    Real xi2 = interp.MapToCollocation(box_x2min, box_x2max, x2v);
    Real xi3 = interp.MapToCollocation(box_x3min, box_x3max, x3v);
    interp.SetInterpolationPoint(xi1, xi2, xi3);

    w0_(m,IDN,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IDN);
    w0_(m,IPR,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IPR);
    w0_(m,IVX,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IVX);
    w0_(m,IVY,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IVY);
    w0_(m,IVZ,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IVZ);
    w0_(m,IYF,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IYF);

    adm.alpha(m,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IALP);
    adm.beta_u(m,0,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBETAX);
    adm.beta_u(m,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBETAY);
    adm.beta_u(m,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBETAZ);

    adm.g_dd(m,0,0,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGXX);
    adm.g_dd(m,0,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGXY);
    adm.g_dd(m,0,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGXZ);
    adm.g_dd(m,1,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGYY);
    adm.g_dd(m,1,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGYZ);
    adm.g_dd(m,2,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGZZ);

    adm.vK_dd(m,0,0,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKXX);
    adm.vK_dd(m,0,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKXY);
    adm.vK_dd(m,0,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKXZ);
    adm.vK_dd(m,1,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKYY);
    adm.vK_dd(m,1,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKYZ);
    adm.vK_dd(m,2,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKZZ);
  });
#ifdef DEBUG_PGEN
  printf("done!\n");
#endif

  // initialize magnetic field -----------------------------------------------
  // compute vector potential over all faces
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  DvceArray4D<Real> a1, a2, a3;
  Kokkos::realloc(a1, nmb, ncells3+1, ncells2+1, ncells1+1);
  Kokkos::realloc(a2, nmb, ncells3+1, ncells2+1, ncells1+1);
  Kokkos::realloc(a3, nmb, ncells3+1, ncells2+1, ncells1+1);
#ifdef DEBUG_PGEN
  printf("Interpolating vector potential...");
  fflush(stdout);
#endif
  par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+2,js,je+2,is,ie+2,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL+1> interp;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2f = LeftEdgeX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f = LeftEdgeX(k-ks, nx3, x3min, x3max);

    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    Real xi1 = interp.MapToCollocation(box_x1min, box_x1max, x1v);
    Real xi2 = interp.MapToCollocation(box_x2min, box_x2max, x2f);
    Real xi3 = interp.MapToCollocation(box_x3min, box_x3max, x3f);
    interp.SetInterpolationPoint(xi1, xi2, xi3);
    a1(m,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IAVECX);

    xi1 = interp.MapToCollocation(box_x1min, box_x1max, x1f);
    xi2 = interp.MapToCollocation(box_x2min, box_x2max, x2v);
    xi3 = interp.MapToCollocation(box_x3min, box_x3max, x3f);
    interp.SetInterpolationPoint(xi1, xi2, xi3);
    a2(m,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IAVECY);

    xi1 = interp.MapToCollocation(box_x1min, box_x1max, x1f);
    xi2 = interp.MapToCollocation(box_x2min, box_x2max, x2f);
    xi3 = interp.MapToCollocation(box_x3min, box_x3max, x3v);
    interp.SetInterpolationPoint(xi1, xi2, xi3);
    a3(m,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IAVECZ);

    // TODO(dur566): add averaging of A for AMR runs
  });
#ifdef DEBUG_PGEN
  printf("done!\n");
#endif

  // Compute face-centered fields from curl(A).
  auto &b0 = pmbp->pmhd->b0;
#ifdef DEBUG_PGEN
  printf("Computing face centered field...");
  fflush(stdout);
#endif
  par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;

    b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                       (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
    b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                       (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
    b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                       (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

    // Include extra face-component at edge of block in each direction
    if (i==ie) {
      b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                           (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                           (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                           (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
    }
  });
#ifdef DEBUG_PGEN
  printf("done!\n");
#endif

  // Compute cell-centered fields
  auto &bcc_ = pmbp->pmhd->bcc0;
#ifdef DEBUG_PGEN
  printf("Computing cell centered field...");
  fflush(stdout);
#endif
  par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // cell-centered fields are simple linear average of face-centered fields
    bcc_(m,IBX,k,j,i) = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    bcc_(m,IBY,k,j,i) = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    bcc_(m,IBZ,k,j,i) = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
  });
#ifdef DEBUG_PGEN
  printf("done!\n");
#endif
  
#ifdef DEBUG_PGEN
  printf("Computing conservative variables...");
  fflush(stdout);
#endif
  pmbp->pdyngr->PrimToConInit(0, (ncells1-1), 0, (ncells2-1), 0, (ncells3-1));
#ifdef DEBUG_PGEN
  printf("done!\n");
#endif

  return;
}

// Boundary function
void TurbulenceBC(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto &size = pm->pmb_pack->pmb->mb_size;

  auto &w0 = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &b0 = pm->pmb_pack->pmhd->b0;
  auto &bcc = pm->pmb_pack->pmhd->bcc0;
  int nmb = pm->pmb_pack->nmb_thispack;

  Real box_x1min = pmy_data->xmin;
  Real box_x1max = pmy_data->xmax;
  Real box_x2min = pmy_data->ymin;
  Real box_x2max = pmy_data->ymax;
  Real box_x3min = pmy_data->zmin;
  Real box_x3max = pmy_data->zmax;

  auto &pdyngr = pm->pmb_pack->pdyngr;

#ifdef DEBUG_PGEN
  printf("Apply boundary conditions at time %f...", pm->time);
  fflush(stdout);
#endif
  
  pmy_data->InterpToTime(pm->time);
  auto &cheb_data = pmy_data->cheb_data;

  // Compute primitives in the boundary region -------------------------------
  pdyngr->ConToPrimBC(is-ng, is-1, 0, (n2-1), 0, (n3-1));
  pdyngr->ConToPrimBC(ie+1, ie+ng, 0, (n2-1), 0, (n3-1));
  pdyngr->ConToPrimBC(0, (n1-1), js-ng, js-1, 0, (n3-1));
  pdyngr->ConToPrimBC(0, (n1-1), je+1, je+ng, 0, (n3-1));
  pdyngr->ConToPrimBC(0, (n1-1), 0, (n2-1), ks-ng, ks-1);
  pdyngr->ConToPrimBC(0, (n1-1), 0, (n2-1), ke+1, ke+ng);

  // X1-inner boundary -------------------------------------------------------
  par_for("zoomin_inner_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=is-ng; i<is; ++i) {
#define TURBULENCE_BC_INTERPOLATE_CC_DATA                                   \
  Real &x1min = size.d_view(m).x1min;                                       \
  Real &x1max = size.d_view(m).x1max;                                       \
  int nx1 = indcs.nx1;                                                      \
  Real x1v = CellCenterX(i - is, nx1, x1min, x1max);                        \
                                                                            \
  Real &x2min = size.d_view(m).x2min;                                       \
  Real &x2max = size.d_view(m).x2max;                                       \
  int nx2 = indcs.nx2;                                                      \
  Real x2v = CellCenterX(j - js, nx2, x2min, x2max);                        \
                                                                            \
  Real &x3min = size.d_view(m).x3min;                                       \
  Real &x3max = size.d_view(m).x3max;                                       \
  int nx3 = indcs.nx3;                                                      \
  Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);                        \
                                                                            \
  Real xi1 = interp.MapToCollocation(box_x1min, box_x1max, x1v);            \
  Real xi2 = interp.MapToCollocation(box_x2min, box_x2max, x2v);            \
  Real xi3 = interp.MapToCollocation(box_x3min, box_x3max, x3v);            \
  interp.SetInterpolationPoint(xi1, xi2, xi3);                              \
                                                                            \
  w0(m, IDN, k, j, i) =                                                     \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IDN);    \
  w0(m, IPR, k, j, i) =                                                     \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IPR);    \
  w0(m, IVX, k, j, i) =                                                     \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IVX);    \
  w0(m, IVY, k, j, i) =                                                     \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IVY);    \
  w0(m, IVZ, k, j, i) =                                                     \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IVZ);    \
  w0(m, IYF, k, j, i) =                                                     \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IYF);    \
                                                                            \
  adm.alpha(m, k, j, i) =                                                   \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IALP);   \
  adm.beta_u(m, 0, k, j, i) =                                               \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBETAX); \
  adm.beta_u(m, 1, k, j, i) =                                               \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBETAY); \
  adm.beta_u(m, 2, k, j, i) =                                               \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBETAZ); \
                                                                            \
  adm.g_dd(m, 0, 0, k, j, i) =                                              \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGXX);   \
  adm.g_dd(m, 0, 1, k, j, i) =                                              \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGXY);   \
  adm.g_dd(m, 0, 2, k, j, i) =                                              \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGXZ);   \
  adm.g_dd(m, 1, 1, k, j, i) =                                              \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGYY);   \
  adm.g_dd(m, 1, 2, k, j, i) =                                              \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGYZ);   \
  adm.g_dd(m, 2, 2, k, j, i) =                                              \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGZZ);   \
                                                                            \
  adm.vK_dd(m, 0, 0, k, j, i) =                                             \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKXX);   \
  adm.vK_dd(m, 0, 1, k, j, i) =                                             \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKXY);   \
  adm.vK_dd(m, 0, 2, k, j, i) =                                             \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKXZ);   \
  adm.vK_dd(m, 1, 1, k, j, i) =                                             \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKYY);   \
  adm.vK_dd(m, 1, 2, k, j, i) =                                             \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKYZ);   \
  adm.vK_dd(m, 2, 2, k, j, i) =                                             \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKZZ);   \
                                                                            \
  bcc(m, IBX, k, j, i) =                                                    \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBX);    \
  bcc(m, IBY, k, j, i) =                                                    \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBY);    \
  bcc(m, IBZ, k, j, i) =                                                    \
      interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBZ)

        TURBULENCE_BC_INTERPOLATE_CC_DATA;
      }
    }
  });
  par_for("zoomin_inner_bfc_x1", DevExeSpace(),0,(nmb-1),0,(n3),0,(n2),
  KOKKOS_LAMBDA(int m, int k, int j) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i=is-ng; i<is; ++i) {
#define TURBULENCE_BC_INTERPOLATE_FC_DATA                                  \
  Real &x1min = size.d_view(m).x1min;                                      \
  Real &x1max = size.d_view(m).x1max;                                      \
  int nx1 = indcs.nx1;                                                     \
  Real x1v = CellCenterX(i - is, nx1, x1min, x1max);                       \
  Real x1f = LeftEdgeX(i - is, nx1, x1min, x1max);                         \
                                                                           \
  Real &x2min = size.d_view(m).x2min;                                      \
  Real &x2max = size.d_view(m).x2max;                                      \
  int nx2 = indcs.nx2;                                                     \
  Real x2v = CellCenterX(j - js, nx2, x2min, x2max);                       \
  Real x2f = LeftEdgeX(j - js, nx2, x2min, x2max);                         \
                                                                           \
  Real &x3min = size.d_view(m).x3min;                                      \
  Real &x3max = size.d_view(m).x3max;                                      \
  int nx3 = indcs.nx3;                                                     \
  Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);                       \
  Real x3f = LeftEdgeX(k - ks, nx3, x3min, x3max);                         \
                                                                           \
  Real xi1, xi2, xi3;                                                      \
                                                                           \
  if (j < n2 && k < n3) {                                                  \
    xi1 = interp.MapToCollocation(box_x1min, box_x1max, x1f);              \
    xi2 = interp.MapToCollocation(box_x2min, box_x2max, x2v);              \
    xi3 = interp.MapToCollocation(box_x3min, box_x3max, x3v);              \
    interp.SetInterpolationPoint(xi1, xi2, xi3);                           \
    b0.x1f(m, k, j, i) =                                                   \
        interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBX); \
  }                                                                        \
                                                                           \
  if (i < n1 && k < n3) {                                                  \
    xi1 = interp.MapToCollocation(box_x1min, box_x1max, x1v);              \
    xi2 = interp.MapToCollocation(box_x2min, box_x2max, x2f);              \
    xi3 = interp.MapToCollocation(box_x3min, box_x3max, x3v);              \
    interp.SetInterpolationPoint(xi1, xi2, xi3);                           \
    b0.x2f(m, k, j, i) =                                                   \
        interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBY); \
  }                                                                        \
                                                                           \
  if (i < n1 && j < n2) {                                                  \
    xi1 = interp.MapToCollocation(box_x1min, box_x1max, x1v);              \
    xi2 = interp.MapToCollocation(box_x2min, box_x2max, x2v);              \
    xi3 = interp.MapToCollocation(box_x3min, box_x3max, x3f);              \
    interp.SetInterpolationPoint(xi1, xi2, xi3);                           \
    b0.x3f(m, k, j, i) =                                                   \
        interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBZ); \
  }

        TURBULENCE_BC_INTERPOLATE_FC_DATA;
      }
    }
  });
  par_for("zoomin_inner_bcc_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      for (int i = is-ng; i < is; ++i) {
#define TURBULENCE_BC_COMPUTE_BCC_FROM_BFC                                    \
  bcc(m, IBX, k, j, i) = 0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i + 1)); \
  bcc(m, IBY, k, j, i) = 0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j + 1, i)); \
  bcc(m, IBZ, k, j, i) = 0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k + 1, j, i))

        TURBULENCE_BC_COMPUTE_BCC_FROM_BFC;
      }
    }
  });

  // X1-outer boundary -------------------------------------------------------
  par_for("zoomin_outer_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i = ie+1; i <= ie+ng; ++i) {
        TURBULENCE_BC_INTERPOLATE_CC_DATA;
      }
    }
  });
  par_for("zoomin_outer_bfc_x1", DevExeSpace(),0,(nmb-1),0,(n3),0,(n2),
  KOKKOS_LAMBDA(int m, int k, int j) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i = ie+2; i <= ie+ng+1; ++i) {
        TURBULENCE_BC_INTERPOLATE_FC_DATA;
      }
    }
  });
  par_for("zoomin_outer_bcc_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      for (int i = ie+1; i <= ie+ng; ++i) {
        TURBULENCE_BC_COMPUTE_BCC_FROM_BFC;
      }
    }
  });

  // X2-inner boundary -------------------------------------------------------
  par_for("zoomin_inner_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j = js-ng; j < js; ++j) {
        TURBULENCE_BC_INTERPOLATE_CC_DATA;
      }
    }
  });
  par_for("zoomin_inner_bfc_x2", DevExeSpace(),0,(nmb-1),0,(n3),0,(n1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j = js-ng; j < js; ++j) {
        TURBULENCE_BC_INTERPOLATE_FC_DATA;
      }
    }
  });
  par_for("zoomin_inner_bcc_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      for (int j = js-ng; j < js; ++j) {
        TURBULENCE_BC_COMPUTE_BCC_FROM_BFC;
      }
    }
  });

  // X2-outer boundary -------------------------------------------------------
  par_for("zoomin_outer_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j = je+1; j <= je+ng; ++j) {
        TURBULENCE_BC_INTERPOLATE_CC_DATA;
      }
    }
  });
  par_for("zoomin_outer_bfc_x2", DevExeSpace(),0,(nmb-1),0,(n3),0,(n1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j = je+2; j <= je+ng+1; ++j) {
        TURBULENCE_BC_INTERPOLATE_FC_DATA;
      }
    }
  });
  par_for("zoomin_outer_bcc_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      for (int j = je+1; j <= je+ng; ++j) {
        TURBULENCE_BC_COMPUTE_BCC_FROM_BFC;
      }
    }
  });

  // X3-inner boundary -------------------------------------------------------
  par_for("zoomin_inner_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k = ks-ng; k < ks; ++k) {
        TURBULENCE_BC_INTERPOLATE_CC_DATA;
      }
    }
  });
  par_for("zoomin_inner_bfc_x3", DevExeSpace(),0,(nmb-1),0,(n2),0,(n1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k = ks-ng; k < ks; ++k) {
        TURBULENCE_BC_INTERPOLATE_FC_DATA;
      }
    }
  });
  par_for("zoomin_inner_bcc_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      for (int k = ks-ng; k < ks; ++k) {
        TURBULENCE_BC_COMPUTE_BCC_FROM_BFC;
      }
    }
  });

  // X3-outer boundary -------------------------------------------------------
  par_for("zoomin_outer_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k = ke+1; k <= ke+ng; ++k) {
        TURBULENCE_BC_INTERPOLATE_CC_DATA;
      }
    }
  });
  par_for("zoomin_outer_bfc_x3", DevExeSpace(),0,(nmb-1),0,(n2),0,(n1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k = ke+2; k <= ke+ng+1; ++k) {
        TURBULENCE_BC_INTERPOLATE_FC_DATA;
      }
    }
  });
  par_for("zoomin_outer_bcc_x3", DevExeSpace(),0,(nmb-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      for (int k = ke+1; k <= ke+ng; ++k) {
        TURBULENCE_BC_COMPUTE_BCC_FROM_BFC;
      }
    }
  });

  // Convert primitives to conservative --------------------------------------
  pdyngr->PrimToConInit(is-ng, is-1, 0, (n2-1), 0, (n3-1));
  pdyngr->PrimToConInit(ie+1, ie+ng, 0, (n2-1), 0, (n3-1));
  pdyngr->PrimToConInit(0, (n1-1), js-ng, js-1, 0, (n3-1));
  pdyngr->PrimToConInit(0, (n1-1), je+1, je+ng, 0, (n3-1));
  pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), ks-ng, ks-1);
  pdyngr->PrimToConInit(0, (n1-1), 0, (n2-1), ke+1, ke+ng);

#ifdef DEBUG_PGEN
  printf("done!\n");
#endif

#undef TURBULENCE_BC_INTERPOLATE_CC_DATA
#undef TURBULENCE_BC_INTERPOLATE_FC_DATA
#undef TURBULENCE_BC_COMPUTE_BCC_FROM_BFC
}

void SetADMVariables(MeshBlockPack *pmbp) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  auto &size = pmbp->pmb->mb_size;
  int &nmb = pmbp->nmb_thispack;
  Real box_x1min = pmy_data->xmin;
  Real box_x1max = pmy_data->xmax;
  Real box_x2min = pmy_data->ymin;
  Real box_x2max = pmy_data->ymax;
  Real box_x3min = pmy_data->zmin;
  Real box_x3max = pmy_data->zmax;

  auto &adm = pmbp->padm->adm;

#ifdef DEBUG_PGEN
  printf("Update metric fields at time %f...", pmbp->pmesh->time);
  fflush(stdout);
#endif

  pmy_data->InterpToTime(pmbp->pmesh->time);
  auto &cheb_data = pmy_data->cheb_data;

  par_for("update_adm_vars", DevExeSpace(), 0, (nmb-1), 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    ChebyshevInterpolation<NCHEBY, NSTENCIL> interp;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real xi1 = interp.MapToCollocation(box_x1min, box_x1max, x1v);
    Real xi2 = interp.MapToCollocation(box_x2min, box_x2max, x2v);
    Real xi3 = interp.MapToCollocation(box_x3min, box_x3max, x3v);
    interp.SetInterpolationPoint(xi1, xi2, xi3);

    adm.alpha(m,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IALP);
    adm.beta_u(m,0,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBETAX);
    adm.beta_u(m,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBETAY);
    adm.beta_u(m,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IBETAZ);

    adm.g_dd(m,0,0,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGXX);
    adm.g_dd(m,0,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGXY);
    adm.g_dd(m,0,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGXZ);
    adm.g_dd(m,1,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGYY);
    adm.g_dd(m,1,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGYZ);
    adm.g_dd(m,2,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IGZZ);

    adm.vK_dd(m,0,0,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKXX);
    adm.vK_dd(m,0,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKXY);
    adm.vK_dd(m,0,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKXZ);
    adm.vK_dd(m,1,1,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKYY);
    adm.vK_dd(m,1,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKYZ);
    adm.vK_dd(m,2,2,k,j,i) = interp.Eval(cheb_data.view_device(), BackgroundData<NCHEBY>::IKZZ);
  });

#ifdef DEBUG_PGEN
  printf("done!\n");
#endif
}

// History function
void ZoomHistory(HistoryData *pdata, Mesh *pm) {
  // Select the number of outputs and create labels for them.
  pdata->nhist = 2;
  pdata->label[0] = "btor-max";
  pdata->label[1] = "bpol-max";

  // capture class variables for kernel
  auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &size = pm->pmb_pack->pmb->mb_size;
  Real btor_max = 0.0;
  Real bpol_max = 0.0;
  Kokkos::parallel_reduce("ZoomHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_btor_max, Real &mb_bpol_max) {
    // coompute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    // compute coordinate position
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rcyl = sqrt(x1v*x1v + x2v*x2v);

    Real phivec[3] = {-x2v/rcyl, x1v/rcyl, 0.0};

    Real gamma = sqrt(
        adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                   adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                   adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i)));
    Real bvec[3] = {bcc0_(m,IBX,k,j,i)/gamma, bcc0_(m,IBY,k,j,i)/gamma, bcc0_(m,IBZ,k,j,i)/gamma};

    Real btor = 0.0;
    Real bsqr = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        btor += adm.g_dd(m, a, b, k, j, i) * phivec[a] * bvec[b];
        bsqr += adm.g_dd(m, a, b, k, j, i) * bvec[a] * bvec[b];
      }
    }
    btor = fabs(btor);
    Real bpol = sqrt(bsqr - btor*btor);

    mb_btor_max = fmax(btor, mb_btor_max);
    mb_bpol_max = fmax(bpol, mb_bpol_max);
  }, Kokkos::Max<Real>(btor_max), Kokkos::Max<Real>(bpol_max));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we need MPI_MAX
  // and MPI_MIN operations instead. This is a cheap hack to make it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &btor_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &bpol_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&btor_max, &btor_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bpol_max, &bpol_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    btor_max = 0.0;
    bpol_max = 0.0;
  }
#endif

  // store data in hdata array
  pdata->hdata[0] = btor_max;
  pdata->hdata[1] = bpol_max;
}
