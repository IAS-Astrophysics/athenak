//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file reconstruct.cpp
//  \brief implements constructors for all Reconstruction base and derived classes

#include "athena.hpp"
#include "parameter_input.hpp"
#include "reconstruct.hpp"

//----------------------------------------------------------------------------------------
// Reconstruction constructor

Reconstruction::Reconstruction(ParameterInput *pin, int nghost)
{
  // select reconstruction method (default PLM)
  std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");

  if (xorder.compare("dc") == 0) {
    recon_method_ = ReconstructionMethod::donor_cell;
  } else if (xorder.compare("plm") == 0) {
    recon_method_ = ReconstructionMethod::piecewise_linear;
  } else if (xorder.compare("ppm") == 0) {
    // check that nghost > 2
    if (nghost < 3) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "PPM reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << nghost << std::endl;
      std::exit(EXIT_FAILURE);
    }
    recon_method_ = ReconstructionMethod::piecewise_parabolic;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro> recon = '" << xorder << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}


//----------------------------------------------------------------------------------------
// ReconstructX1()

KOKKOS_FUNCTION
void Reconstruction::ReconstructX1(TeamMember_t const &member, const int k, const int j,
     const int il, const int iu, const AthenaArray4D<Real> &q,
     AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr)
{
  switch (recon_method_) {
    case ReconstructionMethod::donor_cell:
      DonorCellX1(member, k, j, il, iu, q, ql, qr);
      break;
    case ReconstructionMethod::piecewise_linear:
      PLMX1(member, k, j, il, iu, q, ql, qr);
      break;
    case ReconstructionMethod::piecewise_parabolic:
      PPMX1(member, k, j, il, iu, q, ql, qr);
      break;
    default:
      break;
  }
  return;
}

//----------------------------------------------------------------------------------------
// ReconstructX2()
 
KOKKOS_FUNCTION
void Reconstruction::ReconstructX2(TeamMember_t const &member, const int k, const int j,
     const int il, const int iu, const AthenaArray4D<Real> &q,
     AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr)
{
  switch (recon_method_) {
    case ReconstructionMethod::donor_cell:
      DonorCellX2(member, k, j, il, iu, q, ql, qr);
      break;
    case ReconstructionMethod::piecewise_linear:
      PLMX2(member, k, j, il, iu, q, ql, qr);
      break;
    case ReconstructionMethod::piecewise_parabolic:
      PPMX2(member, k, j, il, iu, q, ql, qr);
      break;
    default:
      break;
  }
  return;
}
//----------------------------------------------------------------------------------------
// ReconstructX3()

KOKKOS_FUNCTION
void Reconstruction::ReconstructX3(TeamMember_t const &member, const int k, const int j,
     const int il, const int iu, const AthenaArray4D<Real> &q,
     AthenaScratch2D<Real> &ql, AthenaScratch2D<Real> &qr)
{
  switch (recon_method_) {
    case ReconstructionMethod::donor_cell:
      DonorCellX3(member, k, j, il, iu, q, ql, qr);
      break;
    case ReconstructionMethod::piecewise_linear:
      PLMX3(member, k, j, il, iu, q, ql, qr);
      break;
    case ReconstructionMethod::piecewise_parabolic:
      PPMX3(member, k, j, il, iu, q, ql, qr);
      break;
    default:
      break;
  }
  return;
}
