#ifndef RADIATION_M1_HPP
#define RADIATION_M1_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1.hpp
//  \brief definitions for Grey M1 radiation class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "bvals/bvals.hpp"
#include "parameter_input.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
#include "radiation_m1/radiation_m1_photon_opacities.hpp"
#include "radiation_m1/radiation_m1_roots_fns.hpp"
#include "radiation_m1/radiation_m1_toy.hpp"
#include "tasklist/task_list.hpp"

#if ENABLE_NURATES
#include "bns_nurates/include/bns_nurates.hpp"
#include "radiation_m1/radiation_m1_nurates.hpp"
#endif

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \struct RadiationTaskIDs
//  \brief container to hold TaskIDs of all radiation M1 tasks
struct RadiationM1TaskIDs {
  TaskID M1_irecv;
  TaskID M1_copyu;
  TaskID M1_setmask;
  TaskID M1_closure;
  TaskID M1_flux;
  TaskID M1_sendf;
  TaskID M1_recvf;
  TaskID M1_rkupdt;
  TaskID M1_mattersrc;
  TaskID M1_restu;
  TaskID M1_sendu;
  TaskID M1_recvu;
  TaskID M1_bcs;
  TaskID M1_prol;
  TaskID M1_newdt;
  TaskID M1_csend;
  TaskID M1_crecv;
};

//----------------------------------------------------------------------------------------
//! \class RadiationM1
//  \brief class for grey M1
class RadiationM1 {
 public:
  RadiationM1(MeshBlockPack* ppack, ParameterInput* pin);
  ~RadiationM1();

  BrentFunctor BrentFunc;        // function to minimize for closure
  HybridsjFunctor HybridsjFunc;  // function to minimize for multiroots solver

  ToyOpacity toy_opacity_fn{};  // use only if toy opacities enabled

  int nvars;                               // no. of evolved variables per species
  int nspecies;                            // no. of species
  int nvarstot;                            // total no. of evolved variables
  bool ismhd;                              // flag to check if <mhd> present
  bool ishydro;                            // flag to check if <hydro> present
  bool isunits;                            // flag to check if <units> present
  bool isadm;                              // flag to check if <adm> present

  bool is_chi_updated;                     // true if the closure is already up to date
  
  RadiationM1Params params{};              // user parameters for grey M1
  PhotonOpacityParams photon_op_params{};  // params for photon opacities
#if ENABLE_NURATES
  NuratesParams nurates_params{};  // params for nurates (reactions, quadratures, etc)
#endif

  DvceArray5D<Real> u0;              // evolved variables
  DvceArray5D<Real> coarse_u0;       // evolved variables on 2x coarser grid
  DvceArray5D<Real> chi;             // Eddington factor
  DvceArray4D<bool> radiation_mask;  // radiation mask
  DvceArray5D<Real> u1;              // evolved variables at intermediate step
  DvceFaceFld5D<Real> uflx;          // fluxes of evo. quantities on cell faces
  DvceArray5D<Real> eta_0;           // number emissivity coefficient
  DvceArray5D<Real> abs_0;           // number absorptivity coefficient
  DvceArray5D<Real> eta_1;           // energy emissivity coefficient
  DvceArray5D<Real> abs_1;           // energy absorptivity coefficient
  DvceArray5D<Real> scat_1;          // energy scattering coefficient

  //! Diagnostics of the partially-equilibrated (T*, Ye*) predictor, written by
  //! CalcOpacityNurates and output as "rad_m1_peq". Sized zero unless
  //! nurates_params.use_partial_equilibrium is set, so it costs nothing when off.
  //! These are diagnostics of the path the kernel actually took, not of a shadow
  //! calculation, so they can be checked against an independent implementation.
  enum PeqDiagnostic {
    PEQ_W1E    = 0,  //! weight of the nu_e pair energy channel
    PEQ_W1X    = 1,  //! weight of the heavy pair energy channel
    PEQ_W0E    = 2,  //! weight of the lepton number channel
    PEQ_DTAU   = 3,  //! alpha*dt/W, the proper time the weights were built from
    PEQ_TSTAR  = 4,  //! predicted temperature (= T when the cell is gated out)
    PEQ_YESTAR = 5,  //! predicted electron fraction (= Y_e when gated out)
    PEQ_RUNG   = 6,  //! which rung of the fallback ladder answered; see PeqRung
    PEQ_STATUS = 7,  //! EOSCompOSE::NeutrinoEquilibriumStatus, or -1 if never called
    //! max_x |J^eq_x(T*,Ye*)/J^eq_x(T,Ye) - 1|, i.e. eta^corr/eta^n - 1 on the THERMAL
    //! channel; the stored eta_1 also carries eta_1_non_th, so the two coincide only
    //! where the NEPS emissivity vanishes
    PEQ_RATIO  = 8,
    PEQ_NDIAG  = 9
  };
  //! How a cell's answer was arrived at, stored in PEQ_RUNG.
  enum PeqRung {
    PEQ_GATE_W    = 0,  //! tier-0: every weight below peq_w_floor
    PEQ_GATE_DTOL = 1,  //! tier-1: predicted excursion below both tolerances
    PEQ_SOLVED    = 2,  //! solved at the cell's own weights, inside the trust region
    PEQ_SOFTENED  = 3,  //! solved after halving the weights one or more times
    PEQ_UNUSABLE  = 4,  //! no accepted solution; the state is left at (T, Y_e)
    //! a weight, c_v, or the predicted blackbody was non-finite or unusable, so the
    //! local blackbody stands. Unlike PEQ_UNUSABLE, where only the root was rejected,
    //! this says an input to the scheme was not a number.
    PEQ_NONFINITE = 5
  };
  DvceArray5D<Real> peq_diag;        // partial-equilibrium predictor diagnostics

  MeshBoundaryValuesCC* pbval_u;  // Communication buffers and functions for u
  RadiationM1TaskIDs id;          // container to hold names of TaskIDs
  Real dtnew{};

  // conditional quantities
  DvceArray5D<Real> w0;        // fluid quantities, when MHD is off, unused otherwise 
  // TODO: get rid of this:
  RadiationM1Beam rad_m1_beam;  // beam ID values (only needed when beams on)

  // functions...
  void AssembleRadiationM1Tasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver* d, int stage);
  // ...in "stagen_tl" list
  TaskStatus CopyCons(Driver* d, int stage);
  TaskStatus SetMask(Driver* d, int stage);
  TaskStatus FloorAndCalcClosure(Driver* d, int stage);
  TaskStatus CalculateFluxes(Driver* d, int stage);
  TaskStatus SendFlux(Driver* d, int stage);
  TaskStatus RecvFlux(Driver* d, int stage);
  TaskStatus TimeUpdate(Driver* d, int stage);
  TaskStatus CalcOpacityNurates(Driver* pdrive, int stage);
  TaskStatus CalcOpacityPhotons(Driver* pdrive, int stage);
  TaskStatus CalcOpacityToy(Driver* pdrive, int stage);
  TaskStatus RestrictU(Driver* d, int stage);
  TaskStatus SendU(Driver* d, int stage);
  TaskStatus RecvU(Driver* d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus NewTimeStep(Driver* d, int stage);
  // ...in "after_stagen_tl" list
  TaskStatus ClearSend(Driver* d, int stage);
  TaskStatus ClearRecv(Driver* d, int stage);  // also in Driver::Initialize
  TaskStatus SetTmunu(Driver* d, int stage);

  // eos related quantites (only when mhd is on)
  template <class EOSPolicy, class ErrorPolicy>
  TaskStatus CalcOpacityNurates_(Driver* pdrive, int stage);
  template <class EOSPolicy, class ErrorPolicy>
  TaskStatus CalcOpacityPhotons_(Driver* pdrive, int stage);
  template <class EOSPolicy, class ErrorPolicy, int M1_NGHOST>
  TaskStatus TimeUpdate_(Driver* d, int stage);

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack
};

// beam boundary conditions
void ApplyBeamSources1D(Mesh* pmesh);
void ApplyBeamSources2D(Mesh* pmesh);
void ApplyBeamSourcesBlackHole(Mesh* pmesh);

}  // namespace radiationm1

#endif  // RADIATION_M1_HPP
