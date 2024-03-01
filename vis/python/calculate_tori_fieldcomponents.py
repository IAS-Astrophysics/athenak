#! /usr/bin/env python3

"""
Script for calculating average Bpol and Btor in an AthenaK GRMHD data dump.

Usage:
[python3] calculate_tori_fieldcomponents.py <input_file> [options]

Example:
~/athenak/vis/python/calculate_tori_fieldcomponents.py basename.prim.00000.bin

<input_file> can be any standard AthenaK .bin data dump that uses GR (Cartesian
Kerr-Schild coordinates) and MHD.

Options include:
  --r_max: maximum radial coordinate to consider in the analysis
  --rho_min: minimum code density to consider in the analysis

Run "calculate_tori_fieldcomponents.py -h" to see a full description of inputs.

The results will be printed to screen. They include volume- and mass-weighted
averages of the toroidal and poloidal field components over the region of interest.

The domain extends from the outer horizon to r <= r_max (default: infinity), and
counts cells with rho >= rho_min (default: 0). Volume weighting weights cells by
dV = sqrt(-g)*dx*dy*dv = dx*dy*dv. Mass weighting weights cells by dm = rho*dV.

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
    pmag_vol_sum = 0.0
    pmag_mass_sum = 0.0
    beta_inv_vol_sum = 0.0
    beta_inv_mass_sum = 0.0
    bph_vol_sum = 0.0
    bph_mass_sum = 0.0
    br_vol_sum = 0.0
    br_mass_sum = 0.0
    bth_vol_sum = 0.0
    bth_mass_sum = 0.0

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
            bh_a = float(input_data['coord']['a'])
            a2 = bh_a ** 2
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
            xx = x[None, None, :]
            yy = y[None, :, None]
            zz = z[:, None, None]

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
            l1 = (r * x[None, None, :] + bh_a * y[None, :, None]) / (r2 + a2)
            l2 = (r * y[None, :, None] - bh_a * x[None, None, :]) / (r2 + a2)
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
            
            br, bth, bph = cks_to_sks_vec_con(b1, b2, b3, bh_a, xx, yy, zz)
            b_r, b_th, b_ph = cks_to_sks_vec_cov(b_1, b_2, b_3, bh_a, xx, yy, zz)
            ur, uth, uph = cks_to_sks_vec_con(u1, u2, u3, bh_a, xx, yy, zz)
            Br_rel = br * u0 - b0 * ur
            Bth_rel = bth * u0 - b0 * uth
            Bph_rel = bph * u0 - b0 * uph
            
            Eph = bph*b_ph
            Er = br*b_r
            Eth = bth*b_th
            
            bph = np.sqrt(abs(Eph))
            br = np.sqrt(abs(Er))
            bth = np.sqrt(abs(Eth))

            # Add to summed values
            vol_sum += np.nansum(vol)
            mass_sum += np.nansum(mass)
            pmag_vol_sum += np.nansum(pmag * vol)
            pmag_mass_sum += np.nansum(pmag * mass)
            bph_vol_sum += np.nansum(bph * vol)
            bph_mass_sum += np.nansum(bph * mass)
            br_vol_sum += np.nansum(br * vol)
            br_mass_sum += np.nansum(br * mass)
            bth_vol_sum += np.nansum(bth * vol)
            bth_mass_sum += np.nansum(bth * mass)
            beta_inv_vol_sum += np.nansum(pmag / pgas * vol)
            beta_inv_mass_sum += np.nansum(pmag / pgas * mass)

    # Report results
    print('')
    print('bph_vol = ' + repr(bph_vol_sum / vol_sum))
    print('bph_mass = ' + repr(bph_mass_sum / mass_sum))
    print('br_vol = ' + repr(br_vol_sum / vol_sum))
    print('br_mass = ' + repr(br_mass_sum / mass_sum))
    print('bth_vol = ' + repr(bth_vol_sum / vol_sum))
    print('bth_mass = ' + repr(bth_mass_sum / mass_sum))
    print('pmag_vol = ' + repr(pmag_vol_sum / vol_sum))
    print('pmag_mass = ' + repr(pmag_mass_sum / mass_sum))
    print('beta_inv_vol = ' + repr(beta_inv_vol_sum / vol_sum))
    print('beta_inv_mass = ' + repr(beta_inv_mass_sum / mass_sum))
    print('')

# Function for calculating cell coordinates
def xyz(num_blocks_used, block_nx1, block_nx2, extents, dimension, location):
  x1 = np.empty((num_blocks_used, block_nx2, block_nx1))
  x2 = np.empty((num_blocks_used, block_nx2, block_nx1))
  for block_ind in range(len(extents)):
    x1f = np.linspace(extents[block_ind][0], extents[block_ind][1], block_nx1 + 1)
    x1v = 0.5 * (x1f[:-1] + x1f[1:])
    x1[block_ind,:,:] = x1v[None,:]
    x2f = np.linspace(extents[block_ind][2], extents[block_ind][3], block_nx2 + 1)
    x2v = 0.5 * (x2f[:-1] + x2f[1:])
    x2[block_ind,:,:] = x2v[:,None]
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
    warnings.filterwarnings('ignore', message='invalid value encountered in divide', \
        category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='invalid value encountered in true_divide', \
        category=RuntimeWarning)
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
  return alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz

# Function for calculating normal-frame Lorentz factor
def normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz):
  uut = np.sqrt(1.0 + g_xx * uux ** 2 + 2.0 * g_xy * uux * uuy + 2.0 * g_xz * uux * uuz \
      + g_yy * uuy ** 2 + 2.0 * g_yz * uuy * uuz + g_zz * uuz ** 2)
  return uut

# Function for transforming velocity from normal frame to coordinate frame
def norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz):
  ut = uut / alpha
  ux = uux - betax * ut
  uy = uuy - betay * ut
  uz = uuz - betaz * ut
  return ut, ux, uy, uz

# Function for converting contravariant vector CKS components to SKS
def cks_to_sks_vec_con(ax, ay, az, a, x, y, z):
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

# Function for transforming vector from contravariant to covariant components
def lower_vector(at, ax, ay, az, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz):
  a_t = g_tt * at + g_tx * ax + g_ty * ay + g_tz * az
  a_x = g_tx * at + g_xx * ax + g_xy * ay + g_xz * az
  a_y = g_ty * at + g_xy * ax + g_yy * ay + g_yz * az
  a_z = g_tz * at + g_xz * ax + g_yz * ay + g_zz * az
  return a_t, a_x, a_y, a_z

# Function for converting covariant covector CKS components to SKS
def cks_to_sks_vec_cov(a_x, a_y, a_z, a, x, y, z):
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

# Function for converting contravariant rank-2 tensor CKS components to SKS
def cks_to_sks_tens_con(axx, axy, axz, ayx, ayy, ayz, azx, azy, azz, a, x, y, z):
  axr, axth, axph = cks_to_sks_vec_con(axx, axy, axz, a, x, y, z)
  ayr, ayth, ayph = cks_to_sks_vec_con(ayx, ayy, ayz, a, x, y, z)
  azr, azth, azph = cks_to_sks_vec_con(azx, azy, azz, a, x, y, z)
  arr, athr, aphr = cks_to_sks_vec_con(axr, ayr, azr, a, x, y, z)
  arth, athth, aphth = cks_to_sks_vec_con(axth, ayth, azth, a, x, y, z)
  arph, athph, aphph = cks_to_sks_vec_con(axph, ayph, azph, a, x, y, z)
  return arr, arth, arph, athr, athth, athph, aphr, aphth, aphph

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

