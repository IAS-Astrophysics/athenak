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

        adm/adm.cpp

        bvals/bvals.cpp
        bvals/buffs_cc.cpp
        bvals/buffs_fc.cpp
        bvals/bvals_cc.cpp
        bvals/bvals_fc.cpp
        bvals/flux_correct_cc.cpp
        bvals/flux_correct_fc.cpp
        bvals/prolongation.cpp
        bvals/physics/hydro_bcs.cpp
        bvals/physics/bfield_bcs.cpp
        bvals/physics/radiation_bcs.cpp

        coordinates/coordinates.cpp
        coordinates/excision.cpp

        diffusion/conduction.cpp
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

        geodesic-grid/geodesic_grid.cpp
        geodesic-grid/spherical_grid.cpp

        hydro/hydro.cpp
        hydro/hydro_fluxes.cpp
        hydro/hydro_fofc.cpp
        hydro/hydro_newdt.cpp
        hydro/hydro_tasks.cpp
        hydro/hydro_update.cpp

        ion-neutral/ion_neutral.cpp

        mesh/build_tree.cpp
        mesh/load_balance.cpp
        mesh/mesh.cpp
        mesh/meshblock.cpp
        mesh/meshblock_pack.cpp
        mesh/meshblock_tree.cpp
        mesh/mesh_refinement.cpp

        mhd/mhd.cpp
        mhd/mhd_corner_e.cpp
        mhd/mhd_ct.cpp
        mhd/mhd_fluxes.cpp
        mhd/mhd_fofc.cpp
        mhd/mhd_newdt.cpp
        mhd/mhd_tasks.cpp
        mhd/mhd_update.cpp

	z4c/z4c.cpp
	z4c/z4c_adm.cpp
	z4c/z4c_calcrhs.cpp
	z4c/z4c_newdt.cpp
        z4c/z4c_tasks.cpp
        z4c/z4c_update.cpp
        z4c/z4c_gauge.cpp
        z4c/z4c_Sbc.cpp
        
	outputs/io_wrapper.cpp
        outputs/outputs.cpp
        outputs/basetype_output.cpp
        outputs/derived_variables.cpp
        outputs/binary.cpp
        outputs/eventlog.cpp
        outputs/formatted_table.cpp
        outputs/history.cpp
        outputs/restart.cpp
        outputs/vtk.cpp

        pgen/pgen.cpp
        pgen/tests/advection.cpp
        pgen/tests/cpaw.cpp
        pgen/tests/gr_bondi.cpp
        pgen/tests/gr_monopole.cpp
        pgen/tests/linear_wave.cpp
        pgen/tests/lw_implode.cpp
        pgen/tests/orszag_tang.cpp
        pgen/tests/shock_tube.cpp
        pgen/tests/rad_check_tetrad.cpp
        pgen/tests/rad_hohlraum.cpp
        pgen/tests/rad_linear_wave.cpp
        pgen/tests/z4c_linear_wave.cpp

        radiation/radiation.cpp
        radiation/radiation_fluxes.cpp
        radiation/radiation_newdt.cpp
        radiation/radiation_source.cpp
        radiation/radiation_tasks.cpp
        radiation/radiation_tetrad.cpp
        radiation/radiation_update.cpp

        srcterms/srcterms.cpp
        srcterms/srcterms_newdt.cpp
        srcterms/turb_driver.cpp

        units/units.cpp
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
