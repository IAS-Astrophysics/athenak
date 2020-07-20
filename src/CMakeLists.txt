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

        bvals/boundary_flag.cpp

        driver/driver.cpp

        hydro/hydro.cpp
        hydro/hydro_divflux.cpp
        hydro/hydro_newdt.cpp
        hydro/hydro_update.cpp
        hydro/eos/eos.cpp
        hydro/eos/adiabatic_hydro.cpp
        hydro/eos/isothermal_hydro.cpp
        hydro/rsolver/rsolver.cpp
        hydro/rsolver/advection.cpp
        hydro/rsolver/llf.cpp

        mesh/interface_physics.cpp
        mesh/load_balance.cpp
        mesh/mesh.cpp
        mesh/meshblock.cpp
        mesh/meshblock_tree.cpp

        outputs/io_wrapper.cpp
        outputs/outputs.cpp
        outputs/output_type.cpp
        outputs/formatted_table.cpp

        pgen/pgen.cpp
        pgen/advection.cpp
        pgen/shock_tube.cpp

        reconstruct/reconstruct.cpp
        reconstruct/dc.cpp

        utils/change_rundir.cpp
        utils/show_config.cpp
)

# enable include of header files with /src/ as root of path
target_include_directories(athena PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
