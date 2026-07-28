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

#ifdef ENABLE_NURATES
#include "bns_nurates/include/bns_nurates.hpp"
#include "radiation_m1/radiation_m1_nurates.hpp"
#endif

#if ENABLE_TORCH
#include "radiation_m1/radiation_m1_rhea.hpp"
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
  TaskID M1_flvmix;
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

#if ENABLE_TORCH
  // Rhea ML flavor-mixing state. Null unless params.flavor_mix_type == FlavMixRhea;
  // constructed once in the RadiationM1 constructor (the direct analog of THC's
  // THC_M1_InitRhea).
  std::unique_ptr<RheaModel> prhea;
  // Preallocated once at construction, shape (n_batch, 2, NF=3, 4), n_batch =
  // nmb_thispack * nx1*nx2*nx3. Deliberately float32 regardless of Real.
  Kokkos::View<float****, LayoutWrapper, DevMemSpace> rhea_f4_in_scratch;
#endif

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
  TaskStatus FlavorMix(Driver* d, int stage);
#if ENABLE_TORCH
  // Rhea ML flavor-mixing pipeline, called sequentially from the FlavMixRhea branch of
  // FlavorMix -- no new TaskIDs (precedented by CalculateFluxes's sequential par_for
  // calls in one task body).
  TaskStatus PackRheaInputs(Driver* pdrive, int stage);
  TaskStatus ApplyRheaMixing(Driver* pdrive, int stage,
                              const DvceArray4D<const float>& rhea_f4_out,
                              const DvceArray1D<const float>& rhea_growthrate,
                              const DvceArray1D<const float>& rhea_stability,
                              Real unit_num_dens);
#endif
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
#if ENABLE_TORCH
  // Needs eos.GetEOSUnitSystem(), hence EOS-templated like CalcOpacityNurates_;
  // dispatched to from the non-templated PackRheaInputs above via the same
  // dynamic_cast(pmy_pack->pdyngr) pattern as RadiationM1::CalcOpacityNurates.
  template <class EOSPolicy, class ErrorPolicy>
  TaskStatus PackRheaInputs_(Driver* pdrive, int stage);
#endif

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack
};

// beam boundary conditions
void ApplyBeamSources1D(Mesh* pmesh);
void ApplyBeamSources2D(Mesh* pmesh);
void ApplyBeamSourcesBlackHole(Mesh* pmesh);

}  // namespace radiationm1

#endif  // RADIATION_M1_HPP
