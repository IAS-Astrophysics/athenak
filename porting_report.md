# src/mesh
## mesh.hpp
- Add `Real idx1, idx2, idx3;` to `struct RegionSize`. These are inverse dx needed for derivatives and are defined in MeshBlock constructor
## mesh.cpp
- Add `#include "z4c/z4c.hpp"`
- Add
```
if (pmb_pack->pz4c != nullptr) {
  dt = std::min(dt, (cfl_no)*(pmb_pack->pz4c->dtnew) );
}
```
in function `Mesh::NewTimeStep`
## meshblock.cpp
- Add 
```
mb_size.h_view(m).idx1 = 1./mb_size.h_view(m).dx1;
mb_size.h_view(m).idx2 = 1./mb_size.h_view(m).dx2;
mb_size.h_view(m).idx3 = 1./mb_size.h_view(m).dx3;
```
in MeshBlock constructor
## meshblock_pack.hpp
- Add `namespace z4c {class Z4c;}`
- Add `z4c::Z4c *pz4c=nullptr;`
## meshblock_pack.cpp
- Add `#include "z4c/z4c.hpp"`
- Add `if (pz4c   != nullptr) {delete pz4c;}`
in `MeshBlockPack` destructor
- Add Z4c physics module in `MeshBlockPack::AddPhysics`, line 138
# src/driver
## driver.cpp
- Add `#include "z4c/z4c.hpp"`
- Add `Initialize Z4c` in `Driver::Initialize`
- Add
```
if (pz4c != nullptr) {
  (void) pmesh->pmb_pack->pz4c->NewTimeStep(this, nexp_stages);
}
```
in `Driver::Initialize`

# src/output
## outputs.hpp 
- Change `#define NOUTPUT_CHOICES` to 44
- Add `"adm",       "con",       "z4c"` to `var_choice`
##
- Add `#include "z4c/z4c.hpp"`
- Add `void LoadZ4cHistoryData(HistoryData *pdata, Mesh *pm)`

## basetype_output.cpp
- Add
```
if ((ivar>=41) && (ivar<=43) && (pm->pmb_pack->pz4c == nullptr)) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
     << "Output of Z4c variable requested in <output> block '"
     << out_params.block_name << "' but no Z4c object has been constructed."
     << std::endl << "Input file is likely missing a <z4c> block" << std::endl;
  exit(EXIT_FAILURE);
}
```
to `BaseTypeOutput` constructor
- Add all output variables, lines 316-367
- Fix slicing in `LoadOutputData` adding `+indcs.ng` at lines 425,435,445 

## restart.cpp
- Add `#include "z4c/z4c.hpp"`
- Add 
```
z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
int nhydro=0, nmhd=0, nz4c=0;
if (pz4c != nullptr) {
  nz4c = pz4c->N_Z4c;
}
Kokkos::realloc(outarray, nmb, (nhydro+nmhd+nz4c), nout3, nout2, nout1);
```
```
// load z4c (CC) data (copy to host)
if (pz4c != nullptr) {
  DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(pz4c->u0);
  Kokkos::deep_copy(host_u0,pz4c->u0);
  auto hst_slice = Kokkos::subview(outarray, Kokkos::ALL, std::make_pair(nhydro+nmhd,nz4c),
                                   Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  Kokkos::deep_copy(hst_slice,host_u0);
}
```
in `LoadOutputData`

## history
- Add `#include "z4c/z4c.hpp"`
- Add
```

if (pm->pmb_pack->pz4c != nullptr) {
  hist_data.emplace_back(PhysicsModule::SpaceTimeDynamics);
}
```
in HystoryOutput
- Add 
```
} else if (data.physics == PhysicsModule::SpaceTimeDynamics) {
  LoadZ4cHistoryData(&data, pm);
```
in LoadOutputData
- Add
```
case PhysicsModule::SpaceTimeDynamics:
  fname.append(".z4c");
  break;
```
in WriteOutputFile
- Add `void HistoryOutput::LoadZ4cHistoryData(HistoryData *pdata, Mesh *pm)` module

# src/athena.hpp
- Add `SpaceTimeDynamics` to `PhysicsModule`

# src/pgen
## z4c_one_puncture.cpp
## z4c_two_puncture.cpp
## pgen.cpp
- Add `#include "z4c/z4c.hpp"`
- Add
```
z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
int nhydro_tot = 0, nmhd_tot = 0, nz4c_tot = 0;
if (pz4c != nullptr) {
  nz4c_tot = pz4c->N_Z4c;
}
```
in `ProblemGenerator` constructor (resfile)
- Add 
```
// allocate arrays for CC data
HostArray5D<Real> ccin("pgen-ccin", nmb, (nhydro_tot + nmhd_tot + nz4c_tot), nout3, nout2, nout1);
```
in `ProblemGenerator` constructor (resfile)
- Add
```
// copy CC MHD data to device
if (pz4c != nullptr) {
  DvceArray5D<Real>::HostMirror host_u0 = Kokkos::create_mirror(pz4c->u0);
  auto hst_slice = Kokkos::subview(ccin,Kokkos::ALL,std::make_pair(nhydro_tot+nmhd_tot,nz4c_tot),
                                   Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  Kokkos::deep_copy(host_u0, hst_slice);
  Kokkos::deep_copy(pz4c->u0, host_u0);
}
```
in `ProblemGenerator` constructor (resfile)

# CMakeLists/configuration
- Add files to compile in `src/CMakeLists.txt`
- Modify `CMakelists.txt` in order to compile twopuncture problem. Add dependence on libraries (`gsl`, `twopuncturesc`)
- Add `#define TWO_PUNCTURES` in `config.hpp.in` to compile for two punctures

# Compile twopunctures
- Download and install `gsl` in the base dir
- Download and install `twopuncturesc` in the base dir
- Create build dir, e.g. `mkdir build_twopuncture && cd build_twopuncture`
- Configure with cmake `cmake ../ -DPROBLEM=z4c_two_puncture`
- Make


# TODO
- RK4: to implement
- AMR: to implement
- Puncture Tracker: to implement
- Wave Extraction: to implement
- SBC: Sommerfeld boundary condition need a way ad hoc to be implemented in this code base
- Derivatives-ghosts: how it is done now (switch/case over nghost in each derivative call) is stupid and expensive. Best thing would be to have nghost compile-time. If not possible, do only one if clause at the task-list level, for example.  
- Test hist output (e.g. compare with Athena++)
- Test restart (e.g. compare two runs: one from 0-20 timestep, the other with 0-10, 10-20)
- Test source terms (todo once they are actually used)
- Check if in generation of initial data it is correct to have 2nd order derivative only to calculate Gamma (lines 25-30 of `src/z4c/z4c_adm.cpp`)
- Discuss addition of inverse dx with AthenaK team
- Discuss slicing in `LoadOutputData` function: is it wrong? 
- Discuss modification of `CMakeLists.txt` and `config.hpp.in` for twopuncture compilation
