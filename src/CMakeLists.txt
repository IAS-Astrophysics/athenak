#===================================================================================================
# AthenaXXX astrophysical plasma code
# Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
# Licensed under the 3-clause BSD License (the "LICENSE")
#===================================================================================================

# list of files to be compiled into executable name
add_executable(
    athena
        main.cpp
        globals.cpp
        parameter_input.cpp

        mesh/mesh.cpp

        outputs/io_wrapper.cpp

        utils/show_config.cpp
)

# enable include of header files with /src/ as root of path
target_include_directories(athena PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
