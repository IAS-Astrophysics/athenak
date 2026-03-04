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

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "geodesic-grid/gauss_legendre.hpp"
#include "z4c_macros.hpp"

// Forward declaration
class Mesh;
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

  bool CalculateMetricDerivatives(int iter, Real time);
  bool DeleteMetricDerivatives(int iter, Real time);

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

  // Start and Stop times for each surface
  Real start_time;
  Real stop_time;

  // Compute every n iterations
  int compute_every_iter;

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
  DualArray2D<Real> Y0, Yc, Ys; 
  DualArray2D<Real> dY0dth, dYcdth, dYsdth, dYcdph, dYsdph; 
  DualArray2D<Real> dY0dth2, dYcdth2, dYcdthdph, dYsdth2, dYsdthdph, dYcdph2, dYsdph2; 

  // Arrays for spectral coefficients
  DualArray1D<Real> a0; 
  DualArray1D<Real> ac; 
  DualArray1D<Real> as; 
  Real last_a0; // last coefficient a_00

  // Arrays used for the fields on the sphere
  AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> g; 
  AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 2> K; 
  AthenaPointTensor<Real, TensorSymm::SYM2, NDIM, 3> dg; 
  DualArray1D<Real> rr, rr_dth, rr_dph; 

  // Array computed in Surface Integrals
  DualArray1D<Real> rho;

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

  // Flag points
  DualArray1D<int> havepoint;

  // Functions used in the fast-flow algorithm
  void MetricDerivatives(MeshBlockPack *pmbp); // (OS): MeshBlockPack?
  void MetricInterp(MeshBlockPack *pmbp); // (OS): MeshBlockPack?
  void SurfaceIntegrals();
  void FastFlowLoop();
  void UpdateFlowSpectralComponents();
  void RadiiFromSphericalHarmonics();
  void InitialGuess();
  void ComputeSphericalHarmonics();
  int lmindex(const int l, const int m);
  int tpindex(const int i, const int j);

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