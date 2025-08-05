CC ?= gcc  # assigns the value CC to gcc only if environment variable CC is not already set

CFLAGS = -std=gnu99 -O2 -march=native -g -Wall
VALGRIND_CFLAGS = -std=gnu99 -O2 -g -Wall -Wno-unknown-pragmas
INCLUDEDIRS =
LDFLAGS = -lm

# Check for OpenMP support
OPENMP_FLAG = -fopenmp
COMPILER_SUPPORTS_OPENMP := $(shell echo | $(CC) $(OPENMP_FLAG) -E - >/dev/null 2>&1 && echo YES || echo NO)

ifeq ($(COMPILER_SUPPORTS_OPENMP), YES)
    CFLAGS += $(OPENMP_FLAG)
    LDFLAGS += $(OPENMP_FLAG)
endif

OBJ_FILES = bah_apply_bcs_inner_only.o bah_apply_bcs_r_maxmin_partial_r_hDD_upwinding.o bah_bcstruct_set_up.o bah_cfl_limited_timestep_based_on_h_equals_r.o bah_commondata_struct_set_to_default.o bah_diagnostics.o bah_diagnostics_area_centroid_and_Theta_norms.o bah_diagnostics_file_output.o bah_diagnostics_integration_weights.o bah_diagnostics_min_max_mean_radii_wrt_centroid.o bah_diagnostics_proper_circumferences.o bah_error_message.o bah_find_horizon.o bah_hDD_dD_and_W_dD_in_interp_src_grid_interior.o bah_initial_data.o bah_interpolation_1d_radial_spokes_on_3d_src_grid.o bah_interpolation_2d_external_input_to_interp_src_grid.o bah_interpolation_2d_general__uniform_src_grid.o bah_KO_apply.o bah_MoL_free_memory_non_y_n_gfs.o bah_MoL_free_memory_y_n_gfs.o bah_MoL_malloc_non_y_n_gfs.o bah_MoL_malloc_y_n_gfs.o bah_MoL_step_forward_in_time.o bah_numgrid__evol_set_up.o bah_numgrid__external_input_set_up.o bah_numgrid__interp_src_set_up.o bah_over_relaxation.o bah_params_struct_set_to_default.o bah_poisoning_check_inputs.o bah_poisoning_set_inputs.o bah_quadratic_extrapolation.o bah_radial_grid_cell_centered_set_up.o bah_rfm_precompute_defines.o bah_rfm_precompute_free.o bah_rfm_precompute_malloc.o bah_rhs_eval.o bah_xx_to_Cart.o bah_xyz_center_r_minmax.o

all: libBHaHAHA.a

%.o: %.c $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c $< -o $@

libBHaHAHA.a: $(OBJ_FILES)
	ar rcs $@ $^


valgrind: clean
	$(MAKE) CFLAGS="$(VALGRIND_CFLAGS)" all
# Use $(RM) to be cross-platform compatible.
clean:
	$(RM) *.o */*.o *~ */*~ ./#* *.txt *.gp *.dat *.avi *.png libBHaHAHA.a
