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

        bvals/bvals.cpp
        bvals/buffs_cc.cpp
        bvals/buffs_fc.cpp
        bvals/bvals_cc.cpp
        bvals/bvals_fc.cpp
        bvals/bvals_part.cpp
        bvals/bvals_tasks.cpp
        bvals/flux_correct_cc.cpp
        bvals/flux_correct_fc.cpp
        bvals/prolongation.cpp
        bvals/prolong_prims.cpp
        bvals/physics/hydro_bcs.cpp
        bvals/physics/bfield_bcs.cpp
        bvals/physics/radiation_bcs.cpp
        bvals/physics/z4c_bcs.cpp

        coordinates/adm.cpp
        coordinates/coordinates.cpp
        coordinates/excision.cpp

        diffusion/conduction.cpp
        diffusion/resistivity.cpp
        diffusion/viscosity.cpp

        driver/driver.cpp

        dyn_grmhd/dyn_grmhd.cpp
        dyn_grmhd/dyn_grmhd_fluxes.cpp
        dyn_grmhd/dyn_grmhd_fofc.cpp

        eos/eos.cpp
        eos/ideal_hyd.cpp
        eos/ideal_mhd.cpp
        eos/isothermal_hyd.cpp
        eos/isothermal_mhd.cpp
        eos/ideal_srhyd.cpp
        eos/ideal_grhyd.cpp
        eos/ideal_srmhd.cpp
        eos/ideal_grmhd.cpp
        eos/primitive-solver/eos_compose.cpp
        eos/primitive-solver/eos_hybrid.cpp
        eos/primitive-solver/piecewise_polytrope.cpp
        eos/primitive-solver/unit_system.cpp

        geodesic-grid/geodesic_grid.cpp
        geodesic-grid/spherical_grid.cpp
	geodesic-grid/gauss_legendre.cpp

        hydro/hydro.cpp
        hydro/hydro_fluxes.cpp
        hydro/hydro_fofc.cpp
        hydro/hydro_newdt.cpp
        hydro/hydro_tasks.cpp
        hydro/hydro_update.cpp

        ion-neutral/ion-neutral.cpp
        ion-neutral/ion-neutral_tasks.cpp

        mesh/build_tree.cpp
        mesh/load_balance.cpp
        mesh/mesh.cpp
        mesh/meshblock.cpp
        mesh/meshblock_pack.cpp
        mesh/meshblock_tree.cpp
        mesh/mesh_refinement.cpp
        mesh/refinement_criteria.cpp

        mhd/mhd.cpp
        mhd/mhd_corner_e.cpp
        mhd/mhd_ct.cpp
        mhd/mhd_fluxes.cpp
        mhd/mhd_fofc.cpp
        mhd/mhd_newdt.cpp
        mhd/mhd_tasks.cpp
        mhd/mhd_update.cpp

        outputs/io_wrapper.cpp
        outputs/outputs.cpp
        outputs/basetype_output.cpp
        outputs/cartgrid.cpp
        outputs/derived_variables.cpp
        outputs/binary.cpp
        outputs/eventlog.cpp
        outputs/formatted_table.cpp
        outputs/history.cpp
        outputs/restart.cpp
        outputs/spherical_surface.cpp
        outputs/coarsened_binary.cpp
        outputs/track_prtcl.cpp
        outputs/vtk_mesh.cpp
        outputs/vtk_prtcl.cpp

        particles/particles.cpp
        particles/particles_pushers.cpp
        particles/particles_tasks.cpp
        outputs/pdf.cpp

        pgen/pgen.cpp
        pgen/tests/advection.cpp
        pgen/tests/collapse.cpp
        pgen/tests/cpaw.cpp
        pgen/tests/cshock.cpp
        pgen/tests/diffusion.cpp
        pgen/tests/gr_bondi.cpp
        pgen/tests/gr_monopole.cpp
        pgen/tests/linear_wave.cpp
        pgen/tests/lw_implode.cpp
        pgen/tests/orszag_tang.cpp
        pgen/tests/mri3d.cpp
        pgen/tests/shock_tube.cpp
        pgen/tests/shwave.cpp
        pgen/tests/rad_beam.cpp
        pgen/tests/rad_linear_wave.cpp
        pgen/tests/z4c_boosted_puncture.cpp
        pgen/tests/z4c_linear_wave.cpp

        radiation/radiation.cpp
        radiation/radiation_fluxes.cpp
        radiation/radiation_newdt.cpp
        radiation/radiation_source.cpp
        radiation/radiation_tasks.cpp
        radiation/radiation_tetrad.cpp
        radiation/radiation_update.cpp

        shearing_box/orbital_advection.cpp
        shearing_box/orbital_advection_cc.cpp
        shearing_box/orbital_advection_fc.cpp
        shearing_box/orbital_advection_tasks.cpp
        shearing_box/shearing_box.cpp
        shearing_box/shearing_box_cc.cpp
        shearing_box/shearing_box_fc.cpp
        shearing_box/shearing_box_srcterms.cpp
        shearing_box/shearing_box_tasks.cpp

        srcterms/srcterms.cpp
        srcterms/srcterms_newdt.cpp
        srcterms/turb_driver.cpp

        tasklist/numerical_relativity.cpp

        units/units.cpp
        utils/change_rundir.cpp
        utils/derived_vars.cpp
        utils/show_config.cpp
        utils/lagrange_interpolator.cpp
        utils/tov/tov.cpp
        utils/tr_table.cpp
        utils/cart_grid.cpp
        utils/spherical_surface.cpp

        z4c/compact_object_tracker.cpp
        z4c/horizon_dump.cpp
        z4c/tmunu.cpp
        z4c/z4c.cpp
        z4c/z4c_adm.cpp
        z4c/z4c_calcrhs.cpp
        z4c/z4c_newdt.cpp
        z4c/z4c_tasks.cpp
        z4c/z4c_update.cpp
        z4c/z4c_gauge.cpp
        z4c/z4c_Sbc.cpp
        z4c/z4c_calculate_weyl_scalars.cpp
        z4c/z4c_wave_extr.cpp
        z4c/z4c_amr.cpp
        z4c/cce/cce.cpp
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
