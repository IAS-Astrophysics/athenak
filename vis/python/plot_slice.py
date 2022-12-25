#! /usr/bin/env python3

"""
Script for plotting a 2D slice from a 2D or 3D AthenaK data dump.

Run "plot_slice.py -h" to see a description of inputs.
"""

# Python standard modules
import argparse
import struct

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
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Plotting parameters
    ergosphere_num_points = 129
    ergosphere_line_width = 1.0
    x1_labelpad = 2.0
    x2_labelpad = 2.0

    # Adjust user inputs
    if kwargs['dimension'] == '1':
        kwargs['dimension'] = 'x'
    if kwargs['dimension'] == '2':
        kwargs['dimension'] = 'y'
    if kwargs['dimension'] == '3':
        kwargs['dimension'] = 'z'

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
        variable_names = line[12:].split()
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
            raise RuntimeError('Only 4- and 8-byte integer types supported for '
                               'cell data.')
        variable_format = 'f' if variable_size == 4 else 'd'
        num_variables = len(variable_names)
        if kwargs['variable'] != 'level' and kwargs['variable'] not in variable_names:
            raise RuntimeError('Variable "{0}" not found; options are {{{1}}}.'.format(
                kwargs['variable'], ', '.join(variable_names)))
        if kwargs['variable'] == 'level':
            variable_ind = -1
        else:
            variable_ind = 0
            while variable_names[variable_ind] != kwargs['variable']:
                variable_ind += 1

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

        # Extract black hole metadata
        if kwargs['horizon'] or kwargs['ergosphere']:
            try:
                bh_m = float(input_data['coord']['m'])
                bh_a = float(input_data['coord']['a'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find black hole mass and spin in '
                                   'input file.')

        # Prepare lists to hold results
        max_level_calculated = -1
        block_loc_for_level = []
        block_ind_for_level = []
        num_blocks_used = 0
        extents = []
        quantity = []

        # Go through blocks
        first_time = True
        while f.tell() < file_size:
            # Read grid structure data
            block_indices = struct.unpack('@6i', f.read(24))
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
                        block_ind_for_level.append(mesh_ind -
                                                   (slice_block_n
                                                    * block_loc_for_level[-1]))
                max_level_calculated = block_level
            if kwargs['dimension'] == 'x' and block_i != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables * variable_data_size, 1)
                continue
            if kwargs['dimension'] == 'y' and block_j != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables * variable_data_size, 1)
                continue
            if kwargs['dimension'] == 'z' and block_k != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables * variable_data_size, 1)
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
            if variable_ind == -1:
                if kwargs['dimension'] == 'x':
                    quantity.append(np.full((block_nz, block_ny), block_level))
                if kwargs['dimension'] == 'y':
                    quantity.append(np.full((block_nz, block_nx), block_level))
                if kwargs['dimension'] == 'z':
                    quantity.append(np.full((block_ny, block_nx), block_level))
                f.seek(num_variables * variable_data_size, 1)
            else:
                f.seek(variable_ind * variable_data_size, 1)
                cell_data = (np.array(struct.unpack(block_cell_format,
                                                    f.read(variable_data_size)))
                             .reshape(block_nz, block_ny, block_nx))
                block_ind = block_ind_for_level[block_level]
                if kwargs['dimension'] == 'x':
                    quantity.append(cell_data[:, :, block_ind])
                if kwargs['dimension'] == 'y':
                    quantity.append(cell_data[:, block_ind, :])
                if kwargs['dimension'] == 'z':
                    quantity.append(cell_data[block_ind, :, :])
                f.seek((num_variables - variable_ind - 1) * variable_data_size, 1)

    # Calculate colors
    quantity = np.array(quantity)
    if kwargs['vmin'] is None:
        vmin = np.nanmin(quantity)
    else:
        vmin = kwargs['vmin']
    if kwargs['vmax'] is None:
        vmax = np.nanmax(quantity)
    else:
        vmax = kwargs['vmax']

    # Prepare figure
    plt.figure()

    # Plot data
    for block_num in range(num_blocks_used):
        plt.imshow(quantity[block_num], cmap=kwargs['cmap'], norm=kwargs['norm'],
                   vmin=vmin, vmax=vmax, interpolation='none', origin='lower',
                   extent=extents[block_num])

    # Make colorbar
    plt.colorbar()

    # Mask horizon
    if kwargs['horizon']:
        r_hor = bh_m + (bh_m ** 2 - bh_a ** 2) ** 0.5
        if kwargs['dimension'] in ('x', 'y') and \
           kwargs['location'] ** 2 < r_hor ** 2 + bh_a ** 2:
            full_width = 2.0 * (r_hor ** 2 + bh_a ** 2 - kwargs['location'] ** 2) ** 0.5
            full_height = 2.0 * ((r_hor ** 2 + bh_a ** 2 - kwargs['location'] ** 2)
                                 / (1.0 + bh_a ** 2 / r_hor ** 2)) ** 0.5
            horizon = patches.Ellipse((0.0, 0.0), full_width, full_height,
                                      facecolor=kwargs['horizon_color'], edgecolor='none')
            plt.gca().add_artist(horizon)
        if kwargs['dimension'] == 'z' and abs(kwargs['location']) < r_hor:
            radius = ((r_hor ** 2 + bh_a ** 2)
                      * (1.0 - kwargs['location'] ** 2 / r_hor ** 2)) ** 0.5
            horizon = patches.Circle((0.0, 0.0), radius=radius,
                                     facecolor=kwargs['horizon_color'], edgecolor='none')
            plt.gca().add_artist(horizon)

    # Mark ergosphere
    if kwargs['ergosphere']:
        r_hor = bh_m + (bh_m ** 2 - bh_a ** 2) ** 0.5
        if kwargs['dimension'] in ('x', 'y') and \
           kwargs['location'] ** 2 < 4.0 * bh_m ** 2 + bh_a ** 2:
            xy = np.linspace(abs(kwargs['location']),
                             (4.0 * bh_m ** 2 + bh_a ** 2) ** 0.5,
                             ergosphere_num_points)
            z = np.empty_like(xy)
            for ind, xy_val in enumerate(xy):
                def residual(z_val):
                    rr2 = xy_val ** 2 + z_val ** 2
                    r2 = 0.5 * (rr2 - bh_a ** 2
                                + ((rr2 - bh_a ** 2) ** 2
                                   + 4.0 * bh_a ** 2 * z_val ** 2) ** 0.5)
                    return r2 ** 2 - 2.0 * bh_m * r2 ** 1.5 + bh_a ** 2 * z_val ** 2
                if abs(xy_val) < (r_hor ** 2 + bh_a ** 2) ** 0.5:
                    z[ind] = brentq(residual, 1.0, 2.0 * bh_m)
                else:
                    z[ind] = brentq(residual, 0.0, 2.0 * bh_m)
            xy_plot = np.sqrt(xy ** 2 - kwargs['location'] ** 2)
            xy_plot = np.concatenate((-xy_plot[::-1], xy_plot))
            xy_plot = np.concatenate((xy_plot, xy_plot[::-1]))
            z_plot = np.concatenate((z[::-1], z))
            z_plot = np.concatenate((z_plot, -z_plot[::-1]))
            plt.plot(xy_plot, z_plot, linewidth=ergosphere_line_width,
                     color=kwargs['ergosphere_color'])
        if kwargs['dimension'] == 'z' and abs(kwargs['location']) < r_hor:
            def residual(r):
                return r ** 4 - 2.0 * bh_m * r ** 3 + bh_a ** 2 * kwargs['location'] ** 2
            r_ergo = brentq(residual, r_hor, 2.0 * bh_m)
            radius = ((r_ergo ** 2 + bh_a ** 2)
                      * (1.0 - kwargs['location'] ** 2 / r_ergo ** 2)) ** 0.5
            ergosphere = patches.Circle((0.0, 0.0), radius=radius,
                                        linewidth=ergosphere_line_width, facecolor='none',
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
        plt.savefig(kwargs['output_file'])
    else:
        plt.show()


# Parse inputs and execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='name of input file, possibly including path')
    parser.add_argument('variable', help='name of variable to be plotted, or "level"')
    parser.add_argument('output_file', help='name of output to be (over)written; use'
                        '"show" to show interactive plot instead')
    parser.add_argument('-d', '--dimension', choices=('x', 'y', 'z', '1', '2', '3'),
                        help='dimension orthogonal to slice for 3D data')
    parser.add_argument('-l', '--location', type=float, default=0.0,
                        help=('coordinate value along which slice is to be taken '
                              '(default: 0)'))
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
                        help='flag indicating black hole event horizon should be masked')
    parser.add_argument('--horizon_color', default='k',
                        help='color string for event horizon mask')
    parser.add_argument('--ergosphere', action='store_true',
                        help='flag indicating black hole ergosphere should be marked')
    parser.add_argument('--ergosphere_color', default='gray',
                        help='color string for ergosphere marker')
    args = parser.parse_args()
    main(**vars(args))
