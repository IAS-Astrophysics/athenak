#ifndef GRAVITY_MG_GRAVITY_HPP_
#define GRAVITY_MG_GRAVITY_HPP_
//========================================================================================
// Athenak astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mg_gravity.hpp
//! \brief defines MGGravity class

// Athenak headers
#include "../athena.hpp"
#include "../multigrid/multigrid.hpp"

class MeshBlockPack;
class ParameterInput;
class Coordinates;
class Multigrid;
class MultigridDriver;

struct PoissonCompositeMaskCounts {
  long long solve = 0;
  long long resid = 0;
  long long reset = 0;
  long long stencil = 0;
  long long covered = 0;
};

struct PoissonBoundaryClosureStats {
  long long closure_writes = 0;
  long long solve_overlap_writes = 0;
  long long reset_writes = 0;
  long long stencil_writes = 0;
  long long covered_writes = 0;
  long long face_only_count = 0;
  long long edge_corner_skipped = 0;
  Real delta_sum2 = 0.0;
  Real max_delta = 0.0;
  long long interface_residual_count = 0;
  Real interface_residual_before_sum2 = 0.0;
  Real interface_residual_after_sum2 = 0.0;
};

// 7-point Laplacian stencil for Poisson gravity
struct GravityStencil {
  Real omega_over_diag;

  template <typename ViewType>
  KOKKOS_INLINE_FUNCTION
  Real Apply(const ViewType &u, const ViewType &coeff,
             int m, int v, int k, int j, int i) const {
    return 6.0*u(m,v,k,j,i) - u(m,v,k+1,j,i) - u(m,v,k,j+1,i)
           - u(m,v,k,j,i+1) - u(m,v,k-1,j,i) - u(m,v,k,j-1,i)
           - u(m,v,k,j,i-1);
  }
};

//! \class MGGravity
//! \brief Multigrid gravity solver for each block

class MGGravity : public Multigrid {
 public:
  MGGravity(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost,
            bool on_host = false);
  ~MGGravity();

  void SmoothPack(int color) final;
  void CalculateDefectPack() final;
  void CalculateFASRHSPack() final;

 private:
  void ApplyScalarBoundaryContract(const char *phase);
};


//! \class MGGravityDriver
//! \brief Multigrid gravity solver

class MGGravityDriver : public MultigridDriver {
  public:
    MGGravityDriver(MeshBlockPack *pmbp, ParameterInput *pin);
    ~MGGravityDriver();

    // Allocate a pack-aware multigrid instance for meshblock levels (attach to pack->pgrav->pmg)
    void CreateMeshblockMultigrids(MeshBlockPack *pmbp);
    void Solve(Driver *pdriver, int stage, Real dt = 0.0) final;
    void SetFourPiG(Real four_pi_G);

    // octet physics (host-side)
    void SmoothOctet(MGOctet &oct, int rlev, int color) final;
    void CalculateDefectOctet(MGOctet &oct, int rlev) final;
    void CalculateFASRHSOctet(MGOctet &oct, int rlev) final;
    void ProlongateOctetBoundariesFluxCons(MGOctet &oct,
         std::vector<Real> &cbuf, const std::vector<bool> &ncoarse) final;

    friend class MGGravity;
  private:
    Real four_pi_G_, omega_;
    bool poisson_test_enabled_;
    bool poisson_test_composite_fas_;
    bool poisson_test_debug_composite_masks_;
    bool poisson_test_debug_residual_split_;
    bool poisson_test_debug_boundary_contract_;
    int poisson_test_scalar_boundary_contract_;
    PoissonBoundaryClosureStats poisson_last_boundary_closure_stats_;
    void BuildPoissonCompositeMasks();
    void PrintPoissonCompositeMaskDiagnostics() const;
    void PrintPoissonResidualSplit(const char *label);
    void PrintPoissonBoundaryContractDiagnostics(const char *label);
    void PrintPoissonBoundaryClosureDiagnostics(const char *label) const;
    bool PoissonCellCoveredByFiner(int level, int ix, int iy, int iz) const;
    bool PoissonCellCoveredAtOrAbove(int level, int ix, int iy, int iz) const;
    bool PoissonCellNeedsReset(int level, int ix, int iy, int iz) const;
    PoissonCompositeMaskCounts CountPoissonMeshBlockMasks(int level,
                                                          bool active_only) const;
    PoissonCompositeMaskCounts CountPoissonRootMasks(int level,
                                                     bool active_only) const;
    PoissonCompositeMaskCounts CountPoissonOctetMasks(int level,
                                                      bool active_only) const;
};

#endif // GRAVITY_MG_GRAVITY_HPP_