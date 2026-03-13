//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ahf.hpp
//! \brief Basic functionality for the AHF class.

#ifndef AHF_HPP
#define AHF_HPP

#include <string>
#include <vector>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "geodesic-grid/gauss_legendre.hpp"
#include "z4c_macros.hpp"

// Forward declaration
class Mesh;
class MeshBlock;
class MeshBlockPack;
class ParameterInput;

//! \class AHF
//! \brief Apparent Horizon Finder class
class AHF {
public:
  // Constructor for AHF object
  AHF(MeshBlockPack *pmbp, ParameterInput *pin, int n);

  // Default Destructor for AHF object (closes output file)
  ~AHF();

  void Find(int iter, Real time); // main functionality for finding AH
  void Write(int iter, Real time); // function for result writing
  template <int NGHOST>
  bool MetricDerivatives(Real time); // compute the metric derivatives

  Real GetHorizonRadius() const { return ah_prop[hmeanradius]; }

  // Some of the main parameters in the fast-flow algorithm
  bool ah_found; // Horizon found
  Real time_first_found; // Time, when horizon first found
  Real initial_radius; // Initial guess for the radius of the horizon
  Real rr_min; // Minimum radius
  Real expand_guess; // Expand the initial guess by this factor
  Real center[3]; // Center around which the horizon is searched

  // Fast-Flow parameters
  Real hmean_tol; // for convergence 
  Real mass_tol; // for convergence
  int flow_iterations; // number of flow iterations
  Real flow_alpha_beta_const; // alpha & beta constants in the iteration formula
                              // Eqs. (43) & (44) of https://arxiv.org/pdf/gr-qc/9707050
  bool verbose;
  bool output_ylm;
  bool output_grid;

  // Spherical harmonics & Legendre polynomials
  int lmax; // Multipoles
  int nangles; // Number of angles on Gauss-Legendre grid
  int ntheta; // Number of theta points

  // Compact Object Tracker variables
  int use_puncture; // n surface follows the puncture tracker if use_puncture[n] > 0
  Real merger_distance; // Distance in M at which BHs are considered as merged
  bool use_puncture_massweighted_center;

  // Start and Stop times for each surface
  Real start_time;
  Real stop_time;
  
private:
  int npunct; // Number of punctures
  int lmax1; // lmax + 1
  int lmpoints; // lmax * lmax
  int nh; // Counter variable
  bool wait_until_punc_are_close;
  bool use_stored_metric_drvts;
  int nhorizon; // Number of horizons
  int fastflow_iter = 0;

  // Pointer to Gauss-Legendre object
  GaussLegendreGrid *gl_grid;
 
  // Arrays of spherical harmonics and derivatives
  HostArray2D<Real> Y0, Yc, Ys; 
  HostArray2D<Real> dY0dth, dYcdth, dYsdth, dYcdph, dYsdph; 
  HostArray2D<Real> dY0dth2, dYcdth2, dYcdthdph, dYsdth2, dYsdthdph, dYcdph2, dYsdph2; 

  // Arrays for spectral coefficients
  HostArray1D<Real> a0; 
  HostArray1D<Real> ac; 
  HostArray1D<Real> as; 
  Real last_a0; // last coefficient a_00

  // Arrays used for the fields on the sphere
  HostArray1D<Real> rr, rr_dth, rr_dph; 

  // Array computed in Surface Integrals
  HostArray1D<Real> rho;

  // Indexes of surface integrals
  enum {
    iarea,
    icoarea,
    ihrms,
    ihmean,
    iSx, iSy, iSz,
    invar
  };
  Real integrals[invar]; // Array of surface integrals

  // Indexes of horizon quantities
  enum{
    harea,
    hcoarea,
    hhrms,
    hhmean,
    hSx, hSy, hSz, hS,
    hmass,
    hmeanradius,
    hminradius,
    hnvar
  };
  Real ah_prop[hnvar]; // Array of horizon quantities

  // Indices to interpolate over
  /* int g_idx[6] = {
    pmbp->padm->I_ADM_GXX,
    pmbp->padm->I_ADM_GXY,
    pmbp->padm->I_ADM_GXZ,
    pmbp->padm->I_ADM_GYY,
    pmbp->padm->I_ADM_GYZ,
    pmbp->padm->I_ADM_GZZ
  };
  
  int K_idx[6] = {
    pmbp->padm->I_ADM_KXX,
    pmbp->padm->I_ADM_KXY,
    pmbp->padm->I_ADM_KXZ,
    pmbp->padm->I_ADM_KYY,
    pmbp->padm->I_ADM_KYZ,
    pmbp->padm->I_ADM_KZZ
  }; */

  // Enumerators for readability when calling interpolated arrays
  enum {
    DX_GXX, DX_GXY, DX_GXZ, DX_GYY, DX_GYZ, DX_GZZ,
    DY_GXX, DY_GXY, DY_GXZ, DY_GYY, DY_GYZ, DY_GZZ,
    DZ_GXX, DZ_GXY, DZ_GXZ, DZ_GYY, DZ_GYZ, DZ_GZZ
  };

  enum {GXX, GXY, GXZ, GYY, GYZ, GZZ};
  enum {KXX, KXY, KXZ, KYY, KYZ, KZZ};

  // 5D Device array for the metric derivatives
  // (OS): this is a bad workaround, since we cannot use rank-3 tensors
  //       and the InterpolateToSphere function in GaussLegendreGrid expects
  //       this type
  DvceArray5D<Real> dg;

  // Vectors to hold the DualArray1D interpolated values of GaussLegendreGrid
  /* std::vector<DualArray1D<Real>> g_interp; 
  std::vector<DualArray1D<Real>> K_interp; 
  std::vector<DualArray1D<Real>> dg_interp; */
  HostArray2D<Real> g_interp, K_interp, dg_interp;

  // Flag points
  HostArray1D<int> havepoint;

  // Functions used in the fast-flow algorithm
  void MetricInterp();
  void SurfaceIntegrals();
  void FastFlowLoop();
  void UpdateFlowSpectralComponents();
  void RadiiFromSphericalHarmonics();
  void InitialGuess();
  void ComputeSphericalHarmonics();
  int lmindex(const int l, const int m);

  // Pointers to MeshBlockPack and ParameterInput
  MeshBlockPack *pmbp;
  ParameterInput *pin;

  // Control parameters
  int root;
  int ioproc;
  std::string ofname_summary;
  std::string ofname_shape;
  std::string ofname_verbose;
  std::string ofname_ylm;
  std::string ofname_grid;
  FILE *pofile_summary;
  FILE *pofile_shape;
  FILE *pofile_verbose;
  FILE *pofile_ylm;
  FILE *pofile_grid;

  // Functions to interface with puncture tracker 
  Real PuncMaxDistance();
  Real PuncMaxDistance(const int pix);
  Real PuncSumMasses();
  void PuncWeightedMassCentralPoint(Real *xc, Real *yc, Real *zc);
  bool PuncAreClose();
};

#endif