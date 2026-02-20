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

//! \class MGGravity
//! \brief Multigrid gravity solver for each block

class MGGravity : public Multigrid {
 public:
  MGGravity(MultigridDriver *pmd, MeshBlockPack *pmbp, int nghost);
  ~MGGravity();

  void Smooth(DvceArray5D<Real> &dst, const DvceArray5D<Real> &src,
              const DvceArray5D<Real> &coeff, const DvceArray5D<Real> &matrix, int rlev,
              int il, int iu, int jl, int ju, int kl, int ku, int color, bool th) final;
  void CalculateDefect(DvceArray5D<Real> &def, const DvceArray5D<Real> &u,
                const DvceArray5D<Real> &src, const DvceArray5D<Real> &coeff,
                const DvceArray5D<Real> &matrix, int rlev, int il, int iu, int jl, int ju,
                int kl, int ku, bool th) final;
  void CalculateFASRHS(DvceArray5D<Real> &def, const DvceArray5D<Real> &src,
                const DvceArray5D<Real> &coeff, const DvceArray5D<Real> &matrix,
                int rlev, int il, int iu, int jl, int ju, int kl, int ku, bool th) final;

  void SmoothFAS();
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

    friend class MGGravity;
  private:
    Real four_pi_G_, omega_;
};

#endif // GRAVITY_MG_GRAVITY_HPP_