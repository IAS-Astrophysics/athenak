#! /usr/bin/env python3

"""
Script for calculating average magnetization in an AthenaK GRMHD data dump.

Usage:
[python3] calculate_tori_magnetization.py <input_file> [options]

Example:
~/athenak/vis/python/calculate_tori_magnetization.py basename.prim.00000.bin

<input_file> can be any standard AthenaK .bin data dump that uses GR (Cartesian
Kerr-Schild coordinates) and MHD.

Options include:
  --r_max: maximum radial coordinate to consider in the analysis
  --rho_min: minimum code density to consider in the analysis

Run "calculate_tori_magnetization.py -h" to see a full description of inputs.

The results will be printed to screen. The include volume- and mass-weighted
averages of plasma sigma and beta^{-1} over the region of interest.

The domain extends from the outer horizon to r <= r_max (default: infinity), and
counts cells with rho >= rho_min (default: 0). Volume weighting weights cells by
dV = sqrt(-g)*dx*dy*dv = dx*dy*dv. Mass weighting weights cells by dm = rho*dV.

Plasma sigma is defined as sigma = b_mu b^mu / rho. Plasma beta^{-1} is defined
as beta^{-1} = b_mu b^mu / (2 p_gas). Radiation is not considered in this
calculation.
"""

# Python standard modules
import argparse
import struct

# Numerical modules
import numpy as np


# Main function
def main(**kwargs):

    # Parameters
    variable_names = ('dens', 'eint', 'velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3')

    # Prepare summed values
    vol_sum = 0.0
    mass_sum = 0.0
    sigma_vol_sum = 0.0
    sigma_mass_sum = 0.0
    beta_inv_vol_sum = 0.0
    beta_inv_mass_sum = 0.0

    # Read data
    with open(kwargs['filename'], 'rb') as f:

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
            raise RuntimeError('Only 4- and 8-byte integer types supported '
                               'for location data.')
        location_format = 'f' if location_size == 4 else 'd'
        if variable_size not in (4, 8):
            raise RuntimeError('Only 4- and 8-byte integer types supported '
                               'for cell data.')
        variable_format = 'f' if variable_size == 4 else 'd'
        num_variables_base = len(variable_names_base)
        variable_inds = []
        for variable_name in variable_names:
            if variable_name not in variable_names_base:
                raise RuntimeError('{0} not found.'.format(variable_name))
            variable_ind = 0
            while variable_names_base[variable_ind] != variable_name:
                variable_ind += 1
            variable_inds.append(variable_ind)
        variable_names_sorted = [name for _, name
                                 in sorted(zip(variable_inds, variable_names))]
        variable_inds_sorted = [ind for ind, _
                                in sorted(zip(variable_inds, variable_names))]

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
        try:
            gamma_adi = float(input_data['hydro']['gamma'])
        except:  # noqa: E722
            try:
                gamma_adi = float(input_data['mhd']['gamma'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find adiabatic index in input file.')

        # Extract black hole spin from input file metadata
        try:
            a = float(input_data['coord']['a'])
            a2 = a ** 2
        except:  # noqa: E722
            raise RuntimeError('Unable to find black hole spin in input file.')

        # Prepare lists to hold results
        quantities = {}
        for name in variable_names_sorted:
            quantities[name] = []

        # Go through blocks
        first_time = True
        while f.tell() < file_size:

            # Read and process grid structure data
            if first_time:
                block_indices = [block_index - num_ghost
                                 for block_index in struct.unpack('@6i', f.read(24))]
                block_nx = block_indices[1] - block_indices[0] + 1
                block_ny = block_indices[3] - block_indices[2] + 1
                block_nz = block_indices[5] - block_indices[4] + 1
                cells_per_block = block_nz * block_ny * block_nx
                block_cell_format = '=' + str(cells_per_block) + variable_format
                variable_data_size = cells_per_block * variable_size
                first_time = False
            else:
                f.seek(24, 1)
            f.seek(16, 1)

            # Read and process coordinate data
            block_lims = struct.unpack('=6' + location_format, f.read(6 * location_size))
            xf, dx = np.linspace(block_lims[0], block_lims[1], block_nx + 1, retstep=True)
            yf, dy = np.linspace(block_lims[2], block_lims[3], block_ny + 1, retstep=True)
            zf, dz = np.linspace(block_lims[4], block_lims[5], block_nz + 1, retstep=True)
            x = 0.5 * (xf[:-1] + xf[1:])
            y = 0.5 * (yf[:-1] + yf[1:])
            z = 0.5 * (zf[:-1] + zf[1:])

            # Read cell data
            quantities = {}
            cell_data_start = f.tell()
            for ind, name in zip(variable_inds_sorted, variable_names_sorted):
                f.seek(cell_data_start + ind * variable_data_size, 0)
                quantities[name] = np.array(struct.unpack(block_cell_format,
                                                          f.read(variable_data_size))). \
                    reshape(block_nz, block_ny, block_nx)
            f.seek((num_variables_base - ind - 1) * variable_data_size, 1)

            # Calculate radial coordinate
            rr2 = np.maximum(x[None, None, :] ** 2 + y[None, :, None] ** 2
                             + z[:, None, None] ** 2, 1.0)
            r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2
                        + 4.0 * a2 * z[:, None, None] ** 2))
            r = np.sqrt(r2)

            # Calculate volume and mass
            rho = quantities['dens']
            vol = np.full_like(r, dx * dy * dz)
            vol = np.where(r < 1.0 + (1.0 - a2) ** 0.5, np.nan, vol)
            vol = np.where(r > kwargs['r_max'], np.nan, vol)
            vol = np.where(rho < kwargs['rho_min'], np.nan, vol)
            mass = rho * vol

            # Calculate metric
            factor = 2.0 * r2 * r / (r2 ** 2 + a2 * z[:, None, None] ** 2)
            l1 = (r * x[None, None, :] + a * y[None, :, None]) / (r2 + a2)
            l2 = (r * y[None, :, None] - a * x[None, None, :]) / (r2 + a2)
            l3 = z[:, None, None] / r
            g_00 = factor - 1.0
            g_01 = factor * l1
            g_02 = factor * l2
            g_03 = factor * l3
            g_11 = factor * l1 ** 2 + 1.0
            g_12 = factor * l1 * l2
            g_13 = factor * l1 * l3
            g_22 = factor * l2 ** 2 + 1.0
            g_23 = factor * l2 * l3
            g_33 = factor * l3 ** 2 + 1.0
            g00 = -factor - 1.0
            g01 = factor * l1
            g02 = factor * l2
            g03 = factor * l3
            alpha = 1.0 / np.sqrt(-g00)
            beta1 = -g01 / g00
            beta2 = -g02 / g00
            beta3 = -g03 / g00

            # Calculate gas pressure
            pgas = quantities['eint'] * (gamma_adi - 1.0)

            # Calculate velocity
            uu1 = quantities['velx']
            uu2 = quantities['vely']
            uu3 = quantities['velz']
            uu0 = np.sqrt(1.0 + g_11 * uu1 ** 2 + 2.0 * g_12 * uu1 * uu2
                          + 2.0 * g_13 * uu1 * uu3 + g_22 * uu2 ** 2
                          + 2.0 * g_23 * uu2 * uu3 + g_33 * uu3 ** 2)
            u0 = uu0 / alpha
            u1 = uu1 - beta1 * u0
            u2 = uu2 - beta2 * u0
            u3 = uu3 - beta3 * u0
            u_0 = g_00 * u0 + g_01 * u1 + g_02 * u2 + g_03 * u3  # noqa: F841
            u_1 = g_01 * u0 + g_11 * u1 + g_12 * u2 + g_13 * u3
            u_2 = g_02 * u0 + g_12 * u1 + g_22 * u2 + g_23 * u3
            u_3 = g_03 * u0 + g_13 * u1 + g_23 * u2 + g_33 * u3

            # Calculate magnetic field
            bb1 = quantities['bcc1']
            bb2 = quantities['bcc2']
            bb3 = quantities['bcc3']
            b0 = u_1 * bb1 + u_2 * bb2 + u_3 * bb3
            b1 = (bb1 + b0 * u1) / u0
            b2 = (bb2 + b0 * u2) / u0
            b3 = (bb3 + b0 * u3) / u0
            b_0 = g_00 * b0 + g_01 * b1 + g_02 * b2 + g_03 * b3
            b_1 = g_01 * b0 + g_11 * b1 + g_12 * b2 + g_13 * b3
            b_2 = g_02 * b0 + g_12 * b1 + g_22 * b2 + g_23 * b3
            b_3 = g_03 * b0 + g_13 * b1 + g_23 * b2 + g_33 * b3
            pmag = (b_0 * b0 + b_1 * b1 + b_2 * b2 + b_3 * b3) / 2.0

            # Add to summed values
            vol_sum += np.nansum(vol)
            mass_sum += np.nansum(mass)
            sigma_vol_sum += np.nansum(2.0 * pmag / rho * vol)
            sigma_mass_sum += np.nansum(2.0 * pmag / rho * mass)
            beta_inv_vol_sum += np.nansum(pmag / pgas * vol)
            beta_inv_mass_sum += np.nansum(pmag / pgas * mass)

    # Report results
    print('')
    print('<sigma>_vol = ' + repr(sigma_vol_sum / vol_sum))
    print('<sigma>_mass = ' + repr(sigma_mass_sum / mass_sum))
    print('<beta_inv>_vol = ' + repr(beta_inv_vol_sum / vol_sum))
    print('<beta_inv>_mass = ' + repr(beta_inv_mass_sum / mass_sum))
    print('')


# Process inputs and execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='name of primitive file to analyze')
    parser.add_argument('--r_max', type=float,
                        default=np.inf, help='maximum radius to analyze')
    parser.add_argument('--rho_min', type=float, default=0.0,
                        help='minimum density to analyze')
    args = parser.parse_args()
    main(**vars(args))
