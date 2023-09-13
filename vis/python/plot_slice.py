#! /usr/bin/env python3

"""
Script for plotting a 2D slice from a 2D or 3D AthenaK data dump.

Usage:
[python3] plot_slice.py <input_file> <quantity_to_plot> <output_file> [options]

Example:
~/athenak/vis/python/plot_slice.py basename.prim.00100.bin dens image.png

<input_file> can be any standard AthenaK .bin data dump. <output_file> can have
any extension recognized by Matplotlib (e.g., .png). If <output_file> is simply
"show", the script will open a live Matplotlib viewing window.

Available quantities include anything found in the input file (e.g., dens, velx,
eint, bcc1, r00_ff). If an invalid quantity is requested (e.g.,
"plot_slice.py <input_file> ? show"), the error message will list all available
quantities in the file.

Additional derived quantities can be computed as well. These are identified by
prefixing with "derived:". An invalid request (e.g.,
"plot_slice.py <input_file> derived:? show") will list available options.
Currently, these include the following:
  - Gas pressure and related quantities:
    - pgas: gas pressure
    - pgas_rho: pgas / rho
    - T: temperature in K
  - Non-relativistic velocity:
    - vr_nr, vth_nr, vph_nr: orthonormal spherical components v^i
  - Relativistic velocity:
    - uut: normal-frame Lorentz factor u^{t'} = tilde{u}^t
    - ut, ux, uy, uz: contravariant CKS 4-velocity components u^mu
    - ur, uth, uph: contravariant SKS 4-velocity components u^i
    - u_t, u_x, u_y, u_z: covariant CKS 4-velocity components u_mu
    - u_r, u_th, u_ph: covariant SKS 4-velocity components u_i
    - vx, vy, vz: CKS 3-velocity components v^i = u^i / u^t
    - vr_rel, vth_rel, vph_rel: SKS 3-velocity components v^i = u^i / u^t
  - Non-relativistic magnetic field and related quantities:
    - Br_nr, Bth_nr, Bph_nr: orthonormal spherical components B^i
    - pmag_nr: magnetic pressure, pmag = B^2 / 2
    - beta_inv_nr: reciprocal of plasma beta, beta^{-1} = pmag / pgas
    - sigma_nr: plasma sigma, sigma = B^2 / rho
  - Relativistic magnetic field and related quantities:
    - bt, bx, by, bz: contravariant CKS 4-field components b^mu
    - br, bth, bph: contravariant SKS 4-field components b^i
    - b_t, b_x, b_y, b_z: covariant CKS 4-field components b_mu
    - b_r, b_th, b_ph: covariant SKS 4-field components b_i
    - Br_rel, Bth_rel, Bph_rel: SKS 3-field components B^i = *F^{it}
    - pmag_rel: magnetic pressure, pmag = (B^2 - E^2) / 2 = b_mu b^mu / 2
    - beta_inv_rel: reciprocal of plasma beta, beta^{-1} = pmag / pgas
    - sigma_rel: cold plasma sigma, sigma = 2 pmag / rho
    - sigmah_rel: hot plasma sigma, sigma_hot = 2 pmag / (rho + ugas + pgas + 2 * pmag)
  - Relativistic radiation quantities:
    - prad: (radiation pressure) = (fluid-frame radiation energy density) / 3
    - prad_pgas: (radiation pressure) / (gas pressure)
    - pmag_prad: (magnetic pressure) / (radiation pressure)
  - Relativistic enthalpy densities and Bernoulli parameters:
    - wgas: hydrodynamic enthalpy rho + ugas + pgas
    - wmhd: magnetohydrodynamic enthalpy rho + ugas + pgas + 2 * pmag
    - wgasrad: radiation-hydrodynamic enthalpy rho + ugas + pgas + 4 * prad
    - wmhdrad:
        radiation-magnetohydrodynamic enthalpy rho + ugas + pgas + 2 * pmag + 4 * prad
    - Begas: -u_t * wgas / rho - 1
    - Bemhd: -u_t * wmhd / rho - 1
    - Begasrad: -u_t * wgasrad / rho - 1
    - Bemhdrad: -u_t * wmhdrad / rho - 1
  - Non-relativistic conserved quantities
    - cons_hydro_nr_t: pure hydrodynamical energy density
    - cons_hydro_nr_x, cons_hydro_nr_y, cons_hydro_nr_z: momentum density
    - cons_em_nr_t: pure electromagnetic energy density
    - cons_mhd_nr_t: MHD energy density
    - cons_mhd_nr_x, cons_mhd_nr_y, cons_mhd_nr_z: MHD momentum density
  - Relativistic conserved quantities
    - cons_hydro_rel_t, : (T_hydro)^t_t
    - cons_hydro_rel_x, cons_hydro_rel_y, cons_hydro_rel_z: (T_hydro)^t_i
    - cons_em_rel_t, : (T_EM)^t_t
    - cons_em_rel_x, cons_em_rel_y, cons_em_rel_z: (T_EM)^t_i
    - cons_mhd_rel_t, : (T_MHD)^t_t
    - cons_mhd_rel_x, cons_mhd_rel_y, cons_mhd_rel_z: (T_MHD)^t_i

Only temperature T is in physical units (K); all others are in code units.

Optional inputs include:
  -d: direction orthogonal to slice of 3D data
  -l: location of slice of 3D data if not 0
  --r_max: half-width of plot in both coordinates, centered at the origin
  --x1_min, --x1_max, --x2_min, --x2_max: horizontal and vertical limits of plot
  -c: colormap recognized by Matplotlib
  -n: colormap normalization (e.g., "-n log") if not linear
  --vmin, --vmax: limits of colorbar if not the full range of data
  --notex: flag to disable Latex typesetting of labels
  --horizon: flag for outlining outer event horizon of GR simulation
  --horizon_mask: flag for covering black hole of GR simulation
  --ergosphere: flag for outlining boundary of ergosphere in GR simulation
  --horizon_color, --horizon_mask_color, --ergosphere_color: color choices

Run "plot_slice.py -h" to see a full description of inputs.
"""

# Python standard modules
import argparse
import struct
import warnings

# Numerical modules
import numpy as np

# Load plotting modules
import matplotlib


# Main function
def main(**kwargs):

    # Load additional numerical modules
    if kwargs['ergosphere']:
        from scipy.optimize import brentq

    # Load additional plotting modules
    if kwargs['output_file'] != 'show':
        matplotlib.use('agg')
    if not kwargs['notex']:
        matplotlib.rc('text', usetex=True)
    import matplotlib.colors as colors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    # Plotting parameters
    horizon_line_style = '-'
    horizon_line_width = 1.0
    ergosphere_num_points = 129
    ergosphere_line_style = '-'
    ergosphere_line_width = 1.0
    x1_labelpad = 2.0
    x2_labelpad = 2.0
    dpi = 300

    # Adjust user inputs
    if kwargs['dimension'] == '1':
        kwargs['dimension'] = 'x'
    if kwargs['dimension'] == '2':
        kwargs['dimension'] = 'y'
    if kwargs['dimension'] == '3':
        kwargs['dimension'] = 'z'

    # Set physical units
    c_cgs = 2.99792458e10
    kb_cgs = 1.380649e-16
    mp_cgs = 1.67262192369e-24
    gg_msun_cgs = 1.32712440018e26

    # Set derived dependencies
    derived_dependencies = {}
    derived_dependencies['pgas'] = ('eint',)
    names = ('pgas_rho', 'T', 'wgas')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint')
    names = ('vr_nr', 'vth_nr', 'vph_nr', 'uut', 'ut', 'ux', 'uy', 'uz', 'ur', 'uth',
             'uph', 'u_t', 'u_x', 'u_y', 'u_z', 'u_r', 'u_th', 'u_ph', 'vx', 'vy', 'vz',
             'vr_rel', 'vth_rel', 'vph_rel')
    for name in names:
        derived_dependencies[name] = ('velx', 'vely', 'velz')
    names = ('Br_nr', 'Bth_nr', 'Bph_nr', 'pmag_nr', 'cons_em_nr_t')
    for name in names:
        derived_dependencies[name] = ('bcc1', 'bcc2', 'bcc3')
    derived_dependencies['beta_inv_nr'] = ('eint', 'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['sigma_nr'] = ('dens', 'bcc1', 'bcc2', 'bcc3')
    names = ('bt', 'bx', 'by', 'bz', 'br', 'bth', 'bph', 'b_t', 'b_x', 'b_y', 'b_z',
             'b_r', 'b_th', 'b_ph', 'Br_rel', 'Bth_rel', 'Bph_rel', 'pmag_rel',
             'cons_em_rel_t', 'cons_em_rel_x', 'cons_em_rel_y', 'cons_em_rel_z')
    for name in names:
        derived_dependencies[name] = ('velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['beta_inv_rel'] = ('eint', 'velx', 'vely', 'velz', 'bcc1',
                                            'bcc2', 'bcc3')
    derived_dependencies['sigma_rel'] = ('dens', 'velx', 'vely', 'velz', 'bcc1', 'bcc2',
                                         'bcc3')
    names = ('sigmah_rel', 'wmhd', 'Bemhd', 'cons_mhd_nr_t', 'cons_mhd_rel_t',
             'cons_mhd_rel_x', 'cons_mhd_rel_y', 'cons_mhd_rel_z')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint', 'velx', 'vely', 'velz', 'bcc1',
                                      'bcc2', 'bcc3')
    derived_dependencies['prad'] = ('r00_ff',)
    derived_dependencies['prad_pgas'] = ('eint', 'r00_ff')
    derived_dependencies['pmag_prad'] = ('velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3',
                                         'r00_ff')
    derived_dependencies['wgasrad'] = ('dens', 'eint', 'r00_ff')
    names = ('wmhdrad', 'Bemhdrad')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint', 'velx', 'vely', 'velz', 'bcc1',
                                      'bcc2', 'bcc3', 'r00_ff')
    names = ('Begas', 'cons_hydro_nr_t', 'cons_hydro_rel_t', 'cons_hydro_rel_x',
             'cons_hydro_rel_y', 'cons_hydro_rel_z')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['Begasrad'] = ('dens', 'eint', 'velx', 'vely', 'velz', 'r00_ff')
    names = ('cons_hydro_nr_x', 'cons_mhd_nr_x')
    for name in names:
        derived_dependencies[name] = ('dens', 'velx')
    names = ('cons_hydro_nr_y', 'cons_mhd_nr_y')
    for name in names:
        derived_dependencies[name] = ('dens', 'vely')
    names = ('cons_hydro_nr_z', 'cons_mhd_nr_z')
    for name in names:
        derived_dependencies[name] = ('dens', 'velz')

    # Read data
    with open(kwargs['data_file'], 'rb') as f:

        # Get file size
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)

        # Read header metadata
        line = f.readline().decode('ascii')
        if line != 'Athena binary output version=1.1\n':
            raise RuntimeError('Unrecognized data file format.')
        next(f)
        next(f)
        next(f)
        line = f.readline().decode('ascii')
        if line[:19] != '  size of location=':
            raise RuntimeError('Could not read location size.')
        location_size = int(line[19:])
        line = f.readline().decode('ascii')
        if line[:19] != '  size of variable=':
            raise RuntimeError('Could not read variable size.')
        variable_size = int(line[19:])
        next(f)
        line = f.readline().decode('ascii')
        if line[:12] != '  variables:':
            raise RuntimeError('Could not read variable names.')
        variable_names_base = line[12:].split()
        line = f.readline().decode('ascii')
        if line[:16] != '  header offset=':
            raise RuntimeError('Could not read header offset.')
        header_offset = int(line[16:])

        # Process header metadata
        if location_size not in (4, 8):
            raise RuntimeError('Only 4- and 8-byte integer types supported for '
                               'location data.')
        location_format = 'f' if location_size == 4 else 'd'
        if variable_size not in (4, 8):
            raise RuntimeError('Only 4- and 8-byte integer types supported for cell '
                               'data.')
        variable_format = 'f' if variable_size == 4 else 'd'
        num_variables_base = len(variable_names_base)
        if kwargs['variable'][:8] == 'derived:':
            variable_name = kwargs['variable'][8:]
            if variable_name not in derived_dependencies:
                raise RuntimeError('Derived variable "{0}" not valid; options are " \
                        "{{{1}}}.'.format(variable_name,
                                          ', '.join(derived_dependencies.keys())))
            variable_names = []
            variable_inds = []
            for dependency in derived_dependencies[variable_name]:
                if dependency not in variable_names_base:
                    raise RuntimeError('Requirement "{0}" for "{1}" not found.'
                                       .format(dependency, variable_name))
                variable_names.append(dependency)
                variable_ind = 0
                while variable_names_base[variable_ind] != dependency:
                    variable_ind += 1
                variable_inds.append(variable_ind)
        elif kwargs['variable'] == 'level':
            variable_name = kwargs['variable']
            variable_names = [variable_name]
            variable_inds = [-1]
        else:
            variable_name = kwargs['variable']
            if variable_name not in variable_names_base:
                raise RuntimeError('Variable "{0}" not found; options are {{{1}}}.'
                                   .format(variable_name,
                                           ', '.join(variable_names_base)))
            variable_names = [variable_name]
            variable_ind = 0
            while variable_names_base[variable_ind] != variable_name:
                variable_ind += 1
            variable_inds = [variable_ind]
        variable_names_sorted = \
            [name for _, name in sorted(zip(variable_inds, variable_names))]
        variable_inds_sorted = \
            [ind for ind, _ in sorted(zip(variable_inds, variable_names))]

        # Read input file metadata
        input_data = {}
        start_of_data = f.tell() + header_offset
        while f.tell() < start_of_data:
            line = f.readline().decode('ascii')
            if line[0] == '#':
                continue
            if line[0] == '<':
                section_name = line[1:-2]
                input_data[section_name] = {}
                continue
            key, val = line.split('=', 1)
            input_data[section_name][key.strip()] = val.split('#', 1)[0].strip()

        # Extract number of ghost cells from input file metadata
        try:
            num_ghost = int(input_data['mesh']['nghost'])
        except:  # noqa: E722
            raise RuntimeError('Unable to find number of ghost cells in input file.')

        # Extract adiabatic index from input file metadata
        names = ('pgas', 'pgas_rho', 'T', 'prad_pgas', 'sigmah_rel', 'wgas', 'wgasrad',
                 'Begas', 'Begasrad', 'cons_hydro_rel_t', 'cons_hydro_rel_x',
                 'cons_hydro_rel_y', 'cons_hydro_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            try:
                gamma_adi = float(input_data['hydro']['gamma'])
            except:  # noqa: E722
                try:
                    gamma_adi = float(input_data['mhd']['gamma'])
                except:  # noqa: E722
                    raise RuntimeError('Unable to find adiabatic index in input file.')
        names = ('beta_inv_nr', 'beta_inv_rel', 'wmhd', 'wmhdrad', 'Bemhd', 'Bemhdrad',
                 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y', 'cons_mhd_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            try:
                gamma_adi = float(input_data['mhd']['gamma'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find adiabatic index in input file.')

        # Extract units from input file metadata
        if kwargs['variable'] == 'derived:T':
            if input_data['coord']['general_rel'] == 'true':
                try:
                    length_cgs = float(input_data['units']['bhmass_msun']) * gg_msun_cgs \
                            / c_cgs ** 2
                except:  # noqa: E722
                    raise RuntimeError('Unable to find black hole mass in input file.')
                time_cgs = length_cgs / c_cgs
            else:
                try:
                    length_cgs = float(input_data['units']['length_cgs'])
                except:  # noqa: E722
                    raise RuntimeError('Unable to find length unit in input file.')
                try:
                    time_cgs = float(input_data['units']['time_cgs'])
                except:  # noqa: E722
                    raise RuntimeError('Unable to find time unit in input file.')
            try:
                mu = float(input_data['units']['mu'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find molecular weight in input file.')

        # Check input file metadata for relativity
        names = ('vr_nr', 'vth_nr', 'vph_nr', 'Br_nr', 'Bth_nr', 'Bph_nr', 'pmag_nr',
                 'beta_inv_nr', 'sigma_nr', 't', 'x', 'y', 'z', 'cons_em_nr_t',
                 'cons_mhd_nr_t', 'cons_mhd_nr_x', 'cons_mhd_nr_y', 'cons_mhd_nr_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            assert input_data['coord']['general_rel'] == 'false', \
                    '"{0}" is only defined for non-GR data.'.format(variable_name)
        names = ('uut', 'ut', 'ux', 'uy', 'uz', 'ur', 'uth', 'uph', 'u_t', 'u_x', 'u_y',
                 'u_z', 'u_r', 'u_th', 'u_ph', 'vx', 'vy', 'vz', 'vr_rel', 'vth_rel',
                 'vph_rel', 'bt', 'bx', 'by', 'bz', 'br', 'bth', 'bph', 'b_t', 'b_x',
                 'b_y', 'b_z', 'b_r', 'b_th', 'b_ph', 'Br_rel', 'Bth_rel', 'Bph_rel',
                 'pmag_rel', 'beta_inv_rel', 'sigma_rel', 'sigmah_rel', 'pmag_prad',
                 'wgas', 'wmhd', 'wgasrad', 'wmhdrad', 'Begas', 'Bemhd', 'Begasrad',
                 'Bemhdrad', 'cons_hydro_rel_t', 'cons_hydro_rel_x', 'cons_hydro_rel_y',
                 'cons_hydro_rel_z', 'cons_em_rel_t', 'cons_em_rel_x', 'cons_em_rel_y',
                 'cons_em_rel_z', 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y',
                 'cons_mhd_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            assert input_data['coord']['general_rel'] == 'true', \
                    '"{0}" is only defined for GR data.'.format(variable_name)
        if kwargs['horizon'] or kwargs['horizon_mask'] or kwargs['ergosphere']:
            assert input_data['coord']['general_rel'] == 'true', '"horizon", ' \
                    '"horizon_mask", and "ergosphere" options only pertain to GR data.'

        # Extract black hole spin from input file metadata
        names = ('uut', 'ut', 'ux', 'uy', 'uz', 'ur', 'uth', 'uph', 'u_t', 'u_x', 'u_y',
                 'u_z', 'u_r', 'u_th', 'u_ph', 'vx', 'vy', 'vz', 'vr_rel', 'vth_rel',
                 'vph_rel', 'bt', 'bx', 'by', 'bz', 'br', 'bth', 'bph', 'b_t', 'b_x',
                 'b_y', 'b_z', 'b_r', 'b_th', 'b_ph', 'Br_rel', 'Bth_rel', 'Bph_rel',
                 'pmag_rel', 'beta_inv_rel', 'sigma_rel', 'sigmah_rel', 'pmag_prad',
                 'wmhd', 'wmhdrad', 'Begas', 'Bemhd', 'Begasrad', 'Bemhdrad',
                 'cons_hydro_rel_t', 'cons_hydro_rel_x', 'cons_hydro_rel_y',
                 'cons_hydro_rel_z', 'cons_em_rel_t', 'cons_em_rel_x', 'cons_em_rel_y',
                 'cons_em_rel_z', 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y',
                 'cons_mhd_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            try:
                bh_a = float(input_data['coord']['a'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find black hole spin in input file.')
        if kwargs['horizon'] or kwargs['horizon_mask'] or kwargs['ergosphere']:
            try:
                bh_a = float(input_data['coord']['a'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find black hole spin in input file.')

        # Prepare lists to hold results
        max_level_calculated = -1
        block_loc_for_level = []
        block_ind_for_level = []
        num_blocks_used = 0
        extents = []
        quantities = {}
        for name in variable_names_sorted:
            quantities[name] = []

        # Go through blocks
        first_time = True
        while f.tell() < file_size:

            # Read grid structure data
            block_indices = np.array(struct.unpack('@6i', f.read(24))) - num_ghost
            block_i, block_j, block_k, block_level = struct.unpack('@4i', f.read(16))

            # Process grid structure data
            if first_time:
                block_nx = block_indices[1] - block_indices[0] + 1
                block_ny = block_indices[3] - block_indices[2] + 1
                block_nz = block_indices[5] - block_indices[4] + 1
                cells_per_block = block_nz * block_ny * block_nx
                block_cell_format = '=' + str(cells_per_block) + variable_format
                variable_data_size = cells_per_block * variable_size
                if kwargs['dimension'] is None:
                    if block_nx > 1 and block_ny > 1 and block_nz > 1:
                        kwargs['dimension'] = 'z'
                    elif block_nx > 1 and block_ny > 1:
                        kwargs['dimension'] = 'z'
                    elif block_nx > 1 and block_nz > 1:
                        kwargs['dimension'] = 'y'
                    elif block_ny > 1 and block_nz > 1:
                        kwargs['dimension'] = 'x'
                    else:
                        raise RuntimeError('Input file only contains 1D data.')
                if kwargs['dimension'] == 'x':
                    if block_ny == 1:
                        raise RuntimeError('Data in file has no extent in y-direction.')
                    if block_nz == 1:
                        raise RuntimeError('Data in file has no extent in z-direction.')
                    block_nx1 = block_ny
                    block_nx2 = block_nz
                    slice_block_n = block_nx
                    slice_location_min = float(input_data['mesh']['x1min'])
                    slice_location_max = float(input_data['mesh']['x1max'])
                    slice_root_blocks = (int(input_data['mesh']['nx1'])
                                         // int(input_data['meshblock']['nx1']))
                if kwargs['dimension'] == 'y':
                    if block_nx == 1:
                        raise RuntimeError('Data in file has no extent in x-direction.')
                    if block_nz == 1:
                        raise RuntimeError('Data in file has no extent in z-direction.')
                    block_nx1 = block_nx
                    block_nx2 = block_nz
                    slice_block_n = block_ny
                    slice_location_min = float(input_data['mesh']['x2min'])
                    slice_location_max = float(input_data['mesh']['x2max'])
                    slice_root_blocks = (int(input_data['mesh']['nx2'])
                                         // int(input_data['meshblock']['nx2']))
                if kwargs['dimension'] == 'z':
                    if block_nx == 1:
                        raise RuntimeError('Data in file has no extent in x-direction.')
                    if block_ny == 1:
                        raise RuntimeError('Data in file has no extent in y-direction.')
                    block_nx1 = block_nx
                    block_nx2 = block_ny
                    slice_block_n = block_nz
                    slice_location_min = float(input_data['mesh']['x3min'])
                    slice_location_max = float(input_data['mesh']['x3max'])
                    slice_root_blocks = (int(input_data['mesh']['nx3'])
                                         // int(input_data['meshblock']['nx3']))
                slice_normalized_coord = (kwargs['location'] - slice_location_min) \
                    / (slice_location_max - slice_location_min)
                first_time = False

            # Determine if block is needed
            if block_level > max_level_calculated:
                for level in range(max_level_calculated + 1, block_level + 1):
                    if kwargs['location'] <= slice_location_min:
                        block_loc_for_level.append(0)
                        block_ind_for_level.append(0)
                    elif kwargs['location'] >= slice_location_max:
                        block_loc_for_level.append(slice_root_blocks - 1)
                        block_ind_for_level.append(slice_block_n - 1)
                    else:
                        slice_mesh_n = slice_block_n * slice_root_blocks * 2 ** level
                        mesh_ind = int(slice_normalized_coord * slice_mesh_n)
                        block_loc_for_level.append(mesh_ind // slice_block_n)
                        block_ind_for_level.append(mesh_ind - slice_block_n
                                                   * block_loc_for_level[-1])
                max_level_calculated = block_level
            if kwargs['dimension'] == 'x' and block_i != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables_base * variable_data_size, 1)
                continue
            if kwargs['dimension'] == 'y' and block_j != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables_base * variable_data_size, 1)
                continue
            if kwargs['dimension'] == 'z' and block_k != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables_base * variable_data_size, 1)
                continue
            num_blocks_used += 1

            # Read coordinate data
            block_lims = struct.unpack('=6' + location_format, f.read(6 * location_size))
            if kwargs['dimension'] == 'x':
                extents.append((block_lims[2], block_lims[3], block_lims[4],
                                block_lims[5]))
            if kwargs['dimension'] == 'y':
                extents.append((block_lims[0], block_lims[1], block_lims[4],
                                block_lims[5]))
            if kwargs['dimension'] == 'z':
                extents.append((block_lims[0], block_lims[1], block_lims[2],
                                block_lims[3]))

            # Read cell data
            cell_data_start = f.tell()
            for ind, name in zip(variable_inds_sorted, variable_names_sorted):
                if ind == -1:
                    if kwargs['dimension'] == 'x':
                        quantities[name].append(np.full((block_nz, block_ny),
                                                        block_level))
                    if kwargs['dimension'] == 'y':
                        quantities[name].append(np.full((block_nz, block_nx),
                                                        block_level))
                    if kwargs['dimension'] == 'z':
                        quantities[name].append(np.full((block_ny, block_nx),
                                                        block_level))
                else:
                    f.seek(cell_data_start + ind * variable_data_size, 0)
                    cell_data = (np.array(struct.unpack(block_cell_format,
                                                        f.read(variable_data_size)))
                                 .reshape(block_nz, block_ny, block_nx))
                    block_ind = block_ind_for_level[block_level]
                    if kwargs['dimension'] == 'x':
                        quantities[name].append(cell_data[:, :, block_ind])
                    if kwargs['dimension'] == 'y':
                        quantities[name].append(cell_data[:, block_ind, :])
                    if kwargs['dimension'] == 'z':
                        quantities[name].append(cell_data[block_ind, :, :])
            f.seek((num_variables_base - ind - 1) * variable_data_size, 1)

    # Prepare to calculate derived quantity
    for name in variable_names_sorted:
        quantities[name] = np.array(quantities[name])

    # Calculate gas pressure or related quantity
    names = ('pgas', 'pgas_rho', 'T', 'prad_pgas')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        pgas = (gamma_adi - 1.0) * quantities['eint']
        if kwargs['variable'] == 'derived:pgas':
            quantity = pgas
        elif kwargs['variable'] == 'derived:pgas_rho':
            quantity = pgas / quantities['dens']
        elif kwargs['variable'] == 'derived:T':
            quantity = (mu * mp_cgs / kb_cgs * (length_cgs / time_cgs) ** 2 * pgas
                        / quantities['dens'])
        else:
            prad = quantities['r00_ff'] / 3.0
            quantity = prad / pgas

    # Calculate non-relativistic velocity
    names = ('vr_nr', 'vth_nr', 'vph_nr')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                      kwargs['location'])
        vx = quantities['velx']
        vy = quantities['vely']
        vz = quantities['velz']
        vr, vth, vph = cart_to_sph(vx, vy, vz, x, y, z)
        if kwargs['variable'] == 'derived:vr':
            quantity = vr
        elif kwargs['variable'] == 'derived:vth':
            quantity = vth
        else:
            quantity = vph

    # Calculate relativistic velocity
    names = ('uut', 'ut', 'ux', 'uy', 'uz', 'ur', 'uth', 'uph', 'u_t', 'u_x', 'u_y',
             'u_z', 'u_r', 'u_th', 'u_ph', 'vx', 'vy', 'vz', 'vr_rel', 'vth_rel',
             'vph_rel')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                      kwargs['location'])
        alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
            g_yz, g_zz = cks_geometry(bh_a, x, y, z)
        uux = quantities['velx']
        uuy = quantities['vely']
        uuz = quantities['velz']
        uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
        if kwargs['variable'] == 'derived:uut':
            quantity = uut
        names = ('ut', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            if kwargs['variable'] == 'derived:ut':
                quantity = ut
            elif kwargs['variable'] == 'derived:ux':
                quantity = ux
            elif kwargs['variable'] == 'derived:uy':
                quantity = uy
            elif kwargs['variable'] == 'derived:uz':
                quantity = uz
            elif kwargs['variable'] == 'derived:vx':
                quantity = ux / ut
            elif kwargs['variable'] == 'derived:vy':
                quantity = uy / ut
            else:
                quantity = uz / ut
        names = ('ur', 'uth', 'uph', 'vr_rel', 'vth_rel', 'vph_rel')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            ur, uth, uph = cks_to_sks_con(ux, uy, uz, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:ur':
                quantity = ur
            elif kwargs['variable'] == 'derived:uth':
                quantity = uth
            elif kwargs['variable'] == 'derived:uph':
                quantity = uph
            elif kwargs['variable'] == 'derived:vr_rel':
                quantity = ur / ut
            elif kwargs['variable'] == 'derived:vth_rel':
                quantity = uth / ut
            else:
                quantity = uph / ut
        names = ('u_t', 'u_x', 'u_y', 'u_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            if kwargs['variable'] == 'derived:u_t':
                quantity = u_t
            elif kwargs['variable'] == 'derived:u_x':
                quantity = u_x
            elif kwargs['variable'] == 'derived:u_y':
                quantity = u_y
            else:
                quantity = u_z
        names = ('u_r', 'u_th', 'u_ph')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            u_r, u_th, u_ph = cks_to_sks_cov(u_x, u_y, u_z, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:u_r':
                quantity = u_r
            elif kwargs['variable'] == 'derived:u_th':
                quantity = u_th
            else:
                quantity = u_ph

    # Calculate non-relativistic magnetic field or related quantity
    names = ('Br_nr', 'Bth_nr', 'Bph_nr', 'pmag_nr', 'beta_inv_nr', 'sigma_nr')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        bbx = quantities['bcc1']
        bby = quantities['bcc2']
        bbz = quantities['bcc3']
        names = ('Br_nr', 'Bth_nr', 'Bph_nr')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                          kwargs['dimension'], kwargs['location'])
            bbr, bbth, bbph = cart_to_sph(bbx, bby, bbz, x, y, z)
            if kwargs['variable'] == 'derived:Br_nr':
                quantity = bbr
            elif kwargs['variable'] == 'derived:Bth_nr':
                quantity = bbth
            else:
                quantity = bbph
        names = ('pmag_nr', 'beta_inv_nr', 'sigma_nr')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            pmag = 0.5 * (bbx ** 2 + bby ** 2 + bbz ** 2)
            if kwargs['variable'] == 'derived:pmag_nr':
                quantity = pmag
            elif kwargs['variable'] == 'derived:beta_inv_nr':
                pgas = (gamma_adi - 1.0) * quantities['eint']
                quantity = pmag / pgas
            else:
                quantity = 2.0 * pmag / quantities['dens']

    # Calculate relativistic magnetic field or related quantity
    names = ('bt', 'bx', 'by', 'bz', 'br', 'bth', 'bph', 'b_t', 'b_x', 'b_y', 'b_z',
             'b_r', 'b_th', 'b_ph', 'Br_rel', 'Bth_rel', 'Bph_rel', 'pmag_rel',
             'beta_inv_rel', 'sigma_rel', 'sigmah_rel', 'pmag_prad')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                      kwargs['location'])
        alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
            g_yz, g_zz = cks_geometry(bh_a, x, y, z)
        uux = quantities['velx']
        uuy = quantities['vely']
        uuz = quantities['velz']
        uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
        ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
        u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz, g_xx,
                                          g_xy, g_xz, g_yy, g_yz, g_zz)
        bbx = quantities['bcc1']
        bby = quantities['bcc2']
        bbz = quantities['bcc3']
        bt, bx, by, bz = three_field_to_four_field(bbx, bby, bbz, ut, ux, uy, uz, u_x,
                                                   u_y, u_z)
        names = ('bt', 'bx', 'by', 'bz')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            if kwargs['variable'] == 'derived:bt':
                quantity = bt
            elif kwargs['variable'] == 'derived:bx':
                quantity = bx
            elif kwargs['variable'] == 'derived:by':
                quantity = by
            else:
                quantity = bz
        names = ('br', 'bth', 'bph')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            br, bth, bph = cks_to_sks_con(bx, by, bz, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:br':
                quantity = br
            elif kwargs['variable'] == 'derived:bth':
                quantity = bth
            else:
                quantity = bph
        names = ('b_t', 'b_x', 'b_y', 'b_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            if kwargs['variable'] == 'derived:b_t':
                quantity = b_t
            elif kwargs['variable'] == 'derived:b_x':
                quantity = b_x
            elif kwargs['variable'] == 'derived:b_y':
                quantity = b_y
            else:
                quantity = b_z
        names = ('b_r', 'b_th', 'b_ph')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            b_r, b_th, b_ph = cks_to_sks_cov(b_x, b_y, b_z, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:b_r':
                quantity = b_r
            elif kwargs['variable'] == 'derived:b_th':
                quantity = b_th
            else:
                quantity = b_ph
        names = ('Br_rel', 'Bth_rel', 'Bph_rel')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ur, uth, uph = cks_to_sks_con(ux, uy, uz, bh_a, x, y, z)
            br, bth, bph = cks_to_sks_con(bx, by, bz, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:Br_rel':
                quantity = br * ut - bt * ur
            elif kwargs['variable'] == 'derived:Bth_rel':
                quantity = bth * ut - bt * uth
            else:
                quantity = bph * ut - bt * uph
        names = ('pmag_rel', 'beta_inv_rel', 'sigma_rel', 'sigmah_rel', 'pmag_prad')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            pmag = 0.5 * (b_t * bt + b_x * bx + b_y * by + b_z * bz)
            if kwargs['variable'] == 'derived:pmag_rel':
                quantity = pmag
            elif kwargs['variable'] == 'derived:beta_inv_rel':
                pgas = (gamma_adi - 1.0) * quantities['eint']
                quantity = pmag / pgas
            elif kwargs['variable'] == 'derived:sigma_rel':
                quantity = 2.0 * pmag / quantities['dens']
            elif kwargs['variable'] == 'derived:sigmah_rel':
                w = quantities['dens'] + gamma_adi * quantities['eint'] + 2.0 * pmag
                quantity = 2.0 * pmag / w
            else:
                prad = quantities['r00_ff'] / 3.0
                with warnings.catch_warnings():
                    message = 'divide by zero encountered in true_divide'
                    warnings.filterwarnings('ignore', message=message,
                                            category=RuntimeWarning)
                    quantity = pmag / prad

    # Calculate relativistic radiation quantity
    if kwargs['variable'] == 'derived:prad':
        quantity = quantities['r00_ff'] / 3.0

    # Calculate relativistic enthalpy density or Bernoulli parameter
    names = ('wgas', 'wmhd', 'wgasrad', 'wmhdrad', 'Begas', 'Bemhd', 'Begasrad',
             'Bemhdrad')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        rho = quantities['dens']
        ugas = quantities['eint']
        w = rho + gamma_adi * ugas
        names = ('wgasrad', 'wmhdrad', 'Begasrad', 'Bemhdrad')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            urad = quantities['r00_ff']
            w += 4.0/3.0 * urad
        names = ('wgas', 'wgasrad')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            quantity = w
        names = ('wmhd', 'wmhdrad', 'Begas', 'Bemhd', 'Begasrad', 'Bemhdrad')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                          kwargs['dimension'], kwargs['location'])
            alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
                g_yz, g_zz = cks_geometry(bh_a, x, y, z)
            uux = quantities['velx']
            uuy = quantities['vely']
            uuz = quantities['velz']
            uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            names = ('wmhd', 'wmhdrad', 'Bemhd', 'Bemhdrad')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                bbx = quantities['bcc1']
                bby = quantities['bcc2']
                bbz = quantities['bcc3']
                bt, bx, by, bz = three_field_to_four_field(bbx, bby, bbz, ut, ux, uy, uz,
                                                           u_x, u_y, u_z)
                b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                                  g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
                w += b_t * bt + b_x * bx + b_y * by + b_z * bz
            names = ('wmhd', 'wmhdrad')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = w
            names = ('Begas', 'Bemhd', 'Begasrad', 'Bemhdrad')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = -u_t * w / rho - 1.0

    # Calculate non-relativistic conserved quantity
    names = ('cons_hydro_nr_t', 'cons_hydro_nr_x', 'cons_hydro_nr_y', 'cons_hydro_nr_z',
             'cons_em_nr_t', 'cons_mhd_nr_t', 'cons_mhd_nr_x', 'cons_mhd_nr_y',
             'cons_mhd_nr_z')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        quantity = 0.0
        names = ('cons_hydro_nr_t', 'cons_mhd_nr_t')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            ugas = quantities['eint']
            vx = quantities['velx']
            vy = quantities['vely']
            vz = quantities['velz']
            quantity = 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2) + ugas
        names = ('cons_hydro_nr_x', 'cons_mhd_nr_x')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            vx = quantities['velx']
            quantity = rho * vx
        names = ('cons_hydro_nr_y', 'cons_mhd_nr_y')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            vy = quantities['vely']
            quantity = rho * vy
        names = ('cons_hydro_nr_z', 'cons_mhd_nr_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            vz = quantities['velz']
            quantity = rho * vz
        names = ('cons_em_nr_t', 'cons_mhd_nr_t')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            bbx = quantities['bcc1']
            bby = quantities['bcc2']
            bbz = quantities['bcc3']
            quantity += 0.5 * (bbx ** 2 + bby ** 2 + bbz ** 2)

    # Calculate relativistic conserved quantity
    names = ('cons_hydro_rel_t', 'cons_hydro_rel_x', 'cons_hydro_rel_y',
             'cons_hydro_rel_z', 'cons_em_rel_t', 'cons_em_rel_x', 'cons_em_rel_y',
             'cons_em_rel_z', 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y',
             'cons_mhd_rel_z')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                      kwargs['location'])
        alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
            g_yz, g_zz = cks_geometry(bh_a, x, y, z)
        uux = quantities['velx']
        uuy = quantities['vely']
        uuz = quantities['velz']
        uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
        ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
        u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz, g_xx,
                                          g_xy, g_xz, g_yy, g_yz, g_zz)
        quantity = 0.0
        names = ('cons_hydro_rel_t', 'cons_hydro_rel_x', 'cons_hydro_rel_y',
                 'cons_hydro_rel_z', 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y',
                 'cons_mhd_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            ugas = quantities['eint']
            pgas = (gamma_adi - 1.0) * ugas
            wgas = rho + ugas + pgas
            names = ('cond_hydro_rel_t', 'cons_mhd_rel_t')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = wgas * ut * u_t + pgas
            names = ('cond_hydro_rel_x', 'cons_mhd_rel_x')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = wgas * ut * u_x
            names = ('cond_hydro_rel_y', 'cons_mhd_rel_y')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = wgas * ut * u_y
            names = ('cond_hydro_rel_z', 'cons_mhd_rel_z')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = wgas * ut * u_z
        names = ('cons_em_rel_t', 'cons_em_rel_x', 'cons_em_rel_y', 'cons_em_rel_z',
                 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y', 'cons_mhd_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            bbx = quantities['bcc1']
            bby = quantities['bcc2']
            bbz = quantities['bcc3']
            bt, bx, by, bz = three_field_to_four_field(bbx, bby, bbz, ut, ux, uy, uz, u_x,
                                                       u_y, u_z)
            b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            umag = 0.5 * (b_t * bt + b_x * bx + b_y * by + b_z * bz)
            pmag = umag
            names = ('cond_em_rel_t', 'cons_mhd_rel_t')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity += (umag + pmag) * ut * u_t + pmag - bt * b_t
            names = ('cond_em_rel_x', 'cons_mhd_rel_x')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity += (umag + pmag) * ut * u_x - bt * b_x
            names = ('cond_em_rel_y', 'cons_mhd_rel_y')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity += (umag + pmag) * ut * u_y - bt * b_y
            names = ('cond_em_rel_z', 'cons_mhd_rel_z')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity += (umag + pmag) * ut * u_z - bt * b_z

    # Extract quantity without derivation
    if kwargs['variable'][:8] != 'derived:':
        quantity = quantities[variable_name]

    # Calculate colors
    if kwargs['vmin'] is None:
        vmin = np.nanmin(quantity)
    else:
        vmin = kwargs['vmin']
    if kwargs['vmax'] is None:
        vmax = np.nanmax(quantity)
    else:
        vmax = kwargs['vmax']

    # Choose colormap norm
    if kwargs['norm'] == 'linear':
        norm = colors.Normalize(vmin, vmax)
        vmin = None
        vmax = None
    elif kwargs['norm'] == 'log':
        norm = colors.LogNorm(vmin, vmax)
        vmin = None
        vmax = None
    else:
        norm = kwargs['norm']

    # Prepare figure
    plt.figure()

    # Plot data
    for block_num in range(num_blocks_used):
        plt.imshow(quantity[block_num], cmap=kwargs['cmap'], norm=norm, vmin=vmin,
                   vmax=vmax, interpolation='none', origin='lower',
                   extent=extents[block_num])

    # Make colorbar
    plt.colorbar()

    # Mark and/or mask horizon
    if kwargs['horizon'] or kwargs['horizon_mask']:
        r_hor = 1.0 + (1.0 - bh_a ** 2) ** 0.5
        if kwargs['dimension'] in ('x', 'y') \
                and kwargs['location'] ** 2 < r_hor ** 2 + bh_a ** 2:
            full_width = 2.0 * (r_hor ** 2 + bh_a ** 2 - kwargs['location'] ** 2) ** 0.5
            full_height = 2.0 * ((r_hor ** 2 + bh_a ** 2 - kwargs['location'] ** 2)
                                 / (1.0 + bh_a ** 2 / r_hor ** 2)) ** 0.5
            if kwargs['horizon_mask']:
                horizon_mask = patches.Ellipse((0.0, 0.0), full_width, full_height,
                                               facecolor=kwargs['horizon_mask_color'],
                                               edgecolor='none')
                plt.gca().add_artist(horizon_mask)
            if kwargs['horizon']:
                horizon = patches.Ellipse((0.0, 0.0), full_width, full_height,
                                          linestyle=horizon_line_style,
                                          linewidth=horizon_line_width,
                                          facecolor='none',
                                          edgecolor=kwargs['horizon_color'])
                plt.gca().add_artist(horizon)
        if kwargs['dimension'] == 'z' and abs(kwargs['location']) < r_hor:
            radius = ((r_hor ** 2 + bh_a ** 2)
                      * (1.0 - kwargs['location'] ** 2 / r_hor ** 2)) ** 0.5
            if kwargs['horizon_mask']:
                horizon_mask = patches.Circle((0.0, 0.0), radius=radius,
                                              facecolor=kwargs['horizon_mask_color'],
                                              edgecolor='none')
                plt.gca().add_artist(horizon_mask)
            if kwargs['horizon']:
                horizon = patches.Circle((0.0, 0.0), radius=radius,
                                         linestyle=horizon_line_style,
                                         linewidth=horizon_line_width,
                                         facecolor='none',
                                         edgecolor=kwargs['horizon_color'])
                plt.gca().add_artist(horizon)

    # Mark ergosphere
    if kwargs['ergosphere']:
        r_hor = 1.0 + (1.0 - bh_a ** 2) ** 0.5
        if kwargs['dimension'] in ('x', 'y') \
                and kwargs['location'] ** 2 < 4.0 + bh_a ** 2:
            w = np.linspace(abs(kwargs['location']), (4.0 + bh_a ** 2) ** 0.5,
                            ergosphere_num_points)
            z = np.empty_like(w)
            for ind, w_val in enumerate(w):
                def residual_hor(z_val):
                    rr2 = w_val ** 2 + z_val ** 2
                    r2 = 0.5 * (rr2 - bh_a ** 2 + ((rr2 - bh_a ** 2) ** 2
                                + 4.0 * bh_a ** 2 * z_val ** 2) ** 0.5)
                    return r2 - r_hor ** 2
                if residual_hor(0.0) < 0.0:
                    z_min = brentq(residual_hor, 0.0, 2.0)
                else:
                    z_min = 0.0

                def residual_ergo(z_val):
                    rr2 = w_val ** 2 + z_val ** 2
                    r2 = 0.5 * (rr2 - bh_a ** 2 + ((rr2 - bh_a ** 2) ** 2
                                + 4.0 * bh_a ** 2 * z_val ** 2) ** 0.5)
                    return r2 ** 2 - 2.0 * r2 ** 1.5 + bh_a ** 2 * z_val ** 2
                if residual_ergo(z_min) <= 0.0:
                    z[ind] = brentq(residual_ergo, z_min, 2.0)
                else:
                    z[ind] = 0.0
            xy_plot = np.sqrt(w ** 2 - kwargs['location'] ** 2)
            xy_plot = np.concatenate((-xy_plot[::-1], xy_plot))
            xy_plot = np.concatenate((xy_plot, xy_plot[::-1]))
            z_plot = np.concatenate((z[::-1], z))
            z_plot = np.concatenate((z_plot, -z_plot[::-1]))
            plt.plot(xy_plot, z_plot, linestyle=ergosphere_line_style,
                     linewidth=ergosphere_line_width, color=kwargs['ergosphere_color'],
                     zorder=0)
        if kwargs['dimension'] == 'z' and abs(kwargs['location']) < r_hor:
            def residual_ergo(r):
                return r ** 4 - 2.0 * r ** 3 + bh_a ** 2 * kwargs['location'] ** 2
            r_ergo = brentq(residual_ergo, r_hor, 2.0)
            radius = ((r_ergo ** 2 + bh_a ** 2)
                      * (1.0 - kwargs['location'] ** 2 / r_ergo ** 2)) ** 0.5
            ergosphere = patches.Circle((0.0, 0.0), radius=radius,
                                        linestyle=ergosphere_line_style,
                                        linewidth=ergosphere_line_width,
                                        facecolor='none',
                                        edgecolor=kwargs['ergosphere_color'])
            plt.gca().add_artist(ergosphere)

    # Adjust axes
    if kwargs['dimension'] == 'x':
        x1_min = float(input_data['mesh']['x2min'])
        x1_max = float(input_data['mesh']['x2max'])
        x2_min = float(input_data['mesh']['x3min'])
        x2_max = float(input_data['mesh']['x3max'])
    if kwargs['dimension'] == 'y':
        x1_min = float(input_data['mesh']['x1min'])
        x1_max = float(input_data['mesh']['x1max'])
        x2_min = float(input_data['mesh']['x3min'])
        x2_max = float(input_data['mesh']['x3max'])
    if kwargs['dimension'] == 'z':
        x1_min = float(input_data['mesh']['x1min'])
        x1_max = float(input_data['mesh']['x1max'])
        x2_min = float(input_data['mesh']['x2min'])
        x2_max = float(input_data['mesh']['x2max'])
    if kwargs['x1_min'] is not None:
        x1_min = kwargs['x1_min']
    if kwargs['x1_max'] is not None:
        x1_max = kwargs['x1_max']
    if kwargs['x2_min'] is not None:
        x2_min = kwargs['x2_min']
    if kwargs['x2_max'] is not None:
        x2_max = kwargs['x2_max']
    if kwargs['r_max'] is not None:
        x1_min = -kwargs['r_max']
        x1_max = kwargs['r_max']
        x2_min = -kwargs['r_max']
        x2_max = kwargs['r_max']
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    if kwargs['dimension'] == 'x':
        plt.xlabel('$y$', labelpad=x1_labelpad)
        plt.ylabel('$z$', labelpad=x2_labelpad)
    if kwargs['dimension'] == 'y':
        plt.xlabel('$x$', labelpad=x1_labelpad)
        plt.ylabel('$z$', labelpad=x2_labelpad)
    if kwargs['dimension'] == 'z':
        plt.xlabel('$x$', labelpad=x1_labelpad)
        plt.ylabel('$y$', labelpad=x2_labelpad)

    # Adjust layout
    plt.tight_layout()

    # Save or display figure
    if kwargs['output_file'] != 'show':
        plt.savefig(kwargs['output_file'], dpi=dpi)
    else:
        plt.show()


# Function for calculating cell coordinates
def xyz(num_blocks_used, block_nx1, block_nx2, extents, dimension, location):
    x1 = np.empty((num_blocks_used, block_nx2, block_nx1))
    x2 = np.empty((num_blocks_used, block_nx2, block_nx1))
    for block_ind in range(len(extents)):
        x1f = np.linspace(extents[block_ind][0], extents[block_ind][1], block_nx1 + 1)
        x1v = 0.5 * (x1f[:-1] + x1f[1:])
        x1[block_ind, :, :] = x1v[None, :]
        x2f = np.linspace(extents[block_ind][2], extents[block_ind][3], block_nx2 + 1)
        x2v = 0.5 * (x2f[:-1] + x2f[1:])
        x2[block_ind, :, :] = x2v[:, None]
    if dimension == 'x':
        x = np.full((num_blocks_used, block_nx2, block_nx1), location)
        y = x1
        z = x2
    if dimension == 'y':
        y = np.full((num_blocks_used, block_nx2, block_nx1), location)
        x = x1
        z = x2
    if dimension == 'z':
        z = np.full((num_blocks_used, block_nx2, block_nx1), location)
        x = x1
        y = x2
    return x, y, z


# Function for converting Cartesian coordinates to spherical
def cart_to_sph(ax, ay, az, x, y, z):
    rr = np.sqrt(x ** 2 + y ** 2)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dr_dx = x / r
    dr_dy = y / r
    dr_dz = z / r
    dth_dx = x * z / (rr * r)
    dth_dy = y * z / (rr * r)
    dth_dz = -rr / r
    dph_dx = -y / rr
    dph_dy = x / rr
    dph_dz = 0.0
    ar = dr_dx * ax + dr_dy * ay + dr_dz * az
    ath = dth_dx * ax + dth_dy * ay + dth_dz * az
    aph = dph_dx * ax + dph_dy * ay + dph_dz * az
    return ar, ath, aph


# Function for calculating quantities related to CKS metric
def cks_geometry(a, x, y, z):
    a2 = a ** 2
    z2 = z ** 2
    rr2 = x ** 2 + y ** 2 + z2
    r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z2))
    r = np.sqrt(r2)
    with warnings.catch_warnings():
        message = 'invalid value encountered in true_divide'
        warnings.filterwarnings('ignore', message=message, category=RuntimeWarning)
        f = 2.0 * r2 * r / (r2 ** 2 + a2 * z2)
        lx = (r * x + a * y) / (r2 + a2)
        ly = (r * y - a * x) / (r2 + a2)
        lz = z / r
    gtt = -1.0 - f
    alpha2 = -1.0 / gtt
    alpha = np.sqrt(alpha2)
    betax = alpha2 * f * lx
    betay = alpha2 * f * ly
    betaz = alpha2 * f * lz
    g_tt = -1.0 + f
    g_tx = f * lx
    g_ty = f * ly
    g_tz = f * lz
    g_xx = 1.0 + f * lx ** 2
    g_xy = f * lx * ly
    g_xz = f * lx * lz
    g_yy = 1.0 + f * ly ** 2
    g_yz = f * ly * lz
    g_zz = 1.0 + f * lz ** 2
    return alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
        g_yz, g_zz


# Function for converting contravariant CKS components to SKS
def cks_to_sks_con(ax, ay, az, a, x, y, z):
    a2 = a ** 2
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    rr2 = x2 + y2 + z2
    r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z2))
    r = np.sqrt(r2)
    dr_dx = r * x / (2.0 * r2 - rr2 + a2)
    dr_dy = r * y / (2.0 * r2 - rr2 + a2)
    dr_dz = r * z * (1.0 + a2 / r2) / (2.0 * r2 - rr2 + a2)
    dth_dx = z / r * dr_dx / np.sqrt(r2 - z2)
    dth_dy = z / r * dr_dy / np.sqrt(r2 - z2)
    dth_dz = (z / r * dr_dz - 1.0) / np.sqrt(r2 - z2)
    dph_dx = -y / (x2 + y2) + a / (r2 + a2) * dr_dx
    dph_dy = x / (x2 + y2) + a / (r2 + a2) * dr_dy
    dph_dz = a / (r2 + a2) * dr_dz
    ar = dr_dx * ax + dr_dy * ay + dr_dz * az
    ath = dth_dx * ax + dth_dy * ay + dth_dz * az
    aph = dph_dx * ax + dph_dy * ay + dph_dz * az
    return ar, ath, aph


# Function for calculating normal-frame Lorentz factor
def normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz):
    uut = np.sqrt(1.0 + g_xx * uux ** 2 + 2.0 * g_xy * uux * uuy
                  + 2.0 * g_xz * uux * uuz + g_yy * uuy ** 2 + 2.0 * g_yz * uuy * uuz
                  + g_zz * uuz ** 2)
    return uut


# Function for transforming velocity from normal frame to coordinate frame
def norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz):
    ut = uut / alpha
    ux = uux - betax * ut
    uy = uuy - betay * ut
    uz = uuz - betaz * ut
    return ut, ux, uy, uz


# Function for transforming vector from contravariant to covariant components
def lower_vector(at, ax, ay, az,
                 g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz):
    a_t = g_tt * at + g_tx * ax + g_ty * ay + g_tz * az
    a_x = g_tx * at + g_xx * ax + g_xy * ay + g_xz * az
    a_y = g_ty * at + g_xy * ax + g_yy * ay + g_yz * az
    a_z = g_tz * at + g_xz * ax + g_yz * ay + g_zz * az
    return a_t, a_x, a_y, a_z


# Function for converting covariant CKS components to SKS
def cks_to_sks_cov(a_x, a_y, a_z, a, x, y, z):
    a2 = a ** 2
    z2 = z ** 2
    rr2 = x ** 2 + y ** 2 + z2
    r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z2))
    r = np.sqrt(r2)
    th = np.arccos(z / r)
    sth = np.sin(th)
    cth = np.cos(th)
    ph = np.arctan2(y, x) - np.arctan2(a, r)
    sph = np.sin(ph)
    cph = np.cos(ph)
    dx_dr = sth * cph
    dy_dr = sth * sph
    dz_dr = cth
    dx_dth = cth * (r * cph - a * sph)
    dy_dth = cth * (r * sph + a * cph)
    dz_dth = -r * sth
    dx_dph = sth * (-r * sph - a * cph)
    dy_dph = sth * (r * cph - a * sph)
    dz_dph = 0.0
    a_r = dx_dr * a_x + dy_dr * a_y + dz_dr * a_z
    a_th = dx_dth * a_x + dy_dth * a_y + dz_dth * a_z
    a_ph = dx_dph * a_x + dy_dph * a_y + dz_dph * a_z
    return a_r, a_th, a_ph


# Function for converting 3-magnetic field to 4-magnetic field
def three_field_to_four_field(bbx, bby, bbz, ut, ux, uy, uz, u_x, u_y, u_z):
    bt = u_x * bbx + u_y * bby + u_z * bbz
    bx = (bbx + bt * ux) / ut
    by = (bby + bt * uy) / ut
    bz = (bbz + bt * uz) / ut
    return bt, bx, by, bz


# Parse inputs and execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='name of input file, possibly including path')
    parser.add_argument('variable', help='name of variable to be plotted, any valid'
                                         'derived quantity prefaced by "derived:"')
    parser.add_argument('output_file', help='name of output to be (over)written; use '
                        '"show" to show interactive plot instead')
    parser.add_argument('-d', '--dimension', choices=('x', 'y', 'z', '1', '2', '3'),
                        help='dimension orthogonal to slice for 3D data')
    parser.add_argument('-l', '--location', type=float, default=0.0,
                        help='coordinate value along which slice is to be taken '
                             '(default: 0)')
    parser.add_argument('--r_max', type=float,
                        help='half-width of plot in both coordinates, centered at the '
                             'origin')
    parser.add_argument('--x1_min', type=float,
                        help='horizontal coordinate of left edge of plot')
    parser.add_argument('--x1_max', type=float,
                        help='horizontal coordinate of right edge of plot')
    parser.add_argument('--x2_min', type=float,
                        help='vertical coordinate of bottom edge of plot')
    parser.add_argument('--x2_max', type=float,
                        help='vertical coordinate of top edge of plot')
    parser.add_argument('-c', '--cmap', help='name of Matplotlib colormap to use')
    parser.add_argument('-n', '--norm', help='name of Matplotlib norm to use')
    parser.add_argument('--vmin', type=float, help='colormap minimum')
    parser.add_argument('--vmax', type=float, help='colormap maximum')
    parser.add_argument('--notex', action='store_true',
                        help='flag indicating LaTeX integration is not to be used')
    parser.add_argument('--horizon', action='store_true',
                        help='flag indicating black hole event horizon should be marked')
    parser.add_argument('--horizon_color', default='k',
                        help='color string for event horizon marker')
    parser.add_argument('--horizon_mask', action='store_true',
                        help='flag indicating black hole event horizon should be masked')
    parser.add_argument('--horizon_mask_color', default='k',
                        help='color string for event horizon mask')
    parser.add_argument('--ergosphere', action='store_true',
                        help='flag indicating black hole ergosphere should be marked')
    parser.add_argument('--ergosphere_color', default='gray',
                        help='color string for ergosphere marker')
    args = parser.parse_args()
    main(**vars(args))
