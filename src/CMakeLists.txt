#=========================================================================================
# AthenaXXX astrophysical plasma code
# Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
# Licensed under the 3-clause BSD License (the "LICENSE")
#=========================================================================================

# list of files to be compiled into executable name
add_executable(
    athena
        main.cpp
        globals.cpp
        parameter_input.cpp

        bvals/buffs_cc.cpp
        bvals/bvals_cc.cpp
        bvals/bvals_fc.cpp

        coordinates/coordinates.cpp

        diffusion/resistivity.cpp
        diffusion/viscosity.cpp

        driver/driver.cpp

        eos/eos.cpp
        eos/ideal_hyd.cpp
        eos/ideal_mhd.cpp
        eos/isothermal_hyd.cpp
        eos/isothermal_mhd.cpp
        eos/ideal_srhyd.cpp
        eos/ideal_grhyd.cpp
        eos/ideal_srmhd.cpp
        eos/ideal_grmhd.cpp

        hydro/hydro.cpp
        hydro/hydro_fluxes.cpp
        hydro/hydro_newdt.cpp
        hydro/hydro_tasks.cpp
        hydro/hydro_update.cpp
        hydro/bvals/apply_physical_bcs.cpp
        hydro/bvals/outflow_hydro.cpp
        hydro/bvals/reflect_hydro.cpp
        hydro/bvals/shear_hydro.cpp

	ion-neutral/ion_neutral.cpp

        mesh/load_balance.cpp
        mesh/mesh.cpp
        mesh/meshblock.cpp
        mesh/meshblock_pack.cpp
        mesh/meshblock_tree.cpp

        mhd/mhd.cpp
        mhd/mhd_corner_e.cpp
        mhd/mhd_ct.cpp
        mhd/mhd_fluxes.cpp
        mhd/mhd_newdt.cpp
        mhd/mhd_tasks.cpp
        mhd/mhd_update.cpp
        mhd/bvals/apply_physical_bcs.cpp
        mhd/bvals/outflow_mhd.cpp
        mhd/bvals/reflect_mhd.cpp
        mhd/bvals/shear_mhd.cpp

        outputs/io_wrapper.cpp
        outputs/outputs.cpp
        outputs/output_type.cpp
        outputs/formatted_table.cpp
        outputs/history.cpp
        outputs/vtk.cpp
        outputs/binary.cpp

        pgen/pgen.cpp
        pgen/advection.cpp
        pgen/linear_wave.cpp
        pgen/lw_implode.cpp
        pgen/orszag_tang.cpp
        pgen/shock_tube.cpp

        srcterms/srcterms.cpp
        srcterms/turb_driver.cpp

        utils/change_rundir.cpp
        utils/show_config.cpp
)

# custom problem generator to be included in compile
# specify on command line using '-D PROBLEM=file' where 'file' is name of file in
# pgen/ directory (not including .cpp extension)

if (NOT ${PROBLEM} STREQUAL "built_in_pgens")
  target_sources(athena
      PRIVATE
          pgen/${PROBLEM}.cpp
  )
endif()

# enable include of header files with /src/ as root of path
target_include_directories(athena PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
