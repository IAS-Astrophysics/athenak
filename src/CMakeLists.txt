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

        bvals/bvals_cc.cpp
        bvals/bvals_fc.cpp

        eos/eos.cpp
        eos/adiabatic_hydro.cpp
        eos/adiabatic_mhd.cpp
        eos/isothermal_hydro.cpp
        eos/isothermal_mhd.cpp

        driver/driver.cpp

        hydro/hydro.cpp
        hydro/hydro_divflux.cpp
        hydro/hydro_newdt.cpp
        hydro/hydro_update.cpp
        hydro/bvals/apply_physical_bcs.cpp
        hydro/bvals/outflow_hydro.cpp
        hydro/bvals/reflect_hydro.cpp

        mesh/mesh_physics.cpp
        mesh/load_balance.cpp
        mesh/mesh.cpp
        mesh/meshblock.cpp
        mesh/meshblock_pack.cpp
        mesh/meshblock_tree.cpp

        mhd/mhd.cpp
        mhd/mhd_calc_fluxes.cpp
        mhd/bvals/apply_physical_bcs.cpp
        mhd/bvals/outflow_mhd.cpp
        mhd/bvals/reflect_mhd.cpp

        outputs/io_wrapper.cpp
        outputs/outputs.cpp
        outputs/output_type.cpp
        outputs/formatted_table.cpp
        outputs/history.cpp
        outputs/vtk.cpp

        pgen/pgen.cpp
        pgen/advection.cpp
        pgen/linear_wave.cpp
        pgen/lw_implode.cpp
        pgen/shock_tube.cpp

        utils/change_rundir.cpp
        utils/show_config.cpp
)

# enable include of header files with /src/ as root of path
target_include_directories(athena PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
