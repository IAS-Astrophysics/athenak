"""bin_convert_new.py
Utility helpers for working with Athena++ binary output files.

This module offers a high-performance pure-Python reader for Athena++ *.bin
snapshots, supporting:
  • Mesh-refined outputs (multiple AMR levels).
  • Coarsened (on-the-fly averaged) outputs.
  • Multi-rank dumps written with one file per MPI rank.
  • On-the-fly translation to an athdf-like in-memory representation.
  • Generation of companion XDMF files for ParaView/VisIt visualisation.

The original reader was developed by Lev Arzamasskiy (2021-11-15) and later
extended by George Wong (2022-01-27) and Drummond Fielding (2024-09-09).
This docstring and accompanying clean-up were added in 2025-06-19 to improve
clarity, bring the file closer to PEP-8 compliance, and document the public
API.  Down-stream code should rely only on the symbols exposed via ``__all__``
(defined at the bottom of the file).
"""
import numpy as np
import os
import glob
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Module-level constants ----------------------------------------------------------------
# --------------------------------------------------------------------------------------

CODE_HEADER_EXPECTED: bytes = b"Athena"
SUPPORTED_VERSION: bytes = b"1.1"
COORD_CARTESIAN: str = "cartesian"

# --------------------------------------------------------------------------------------
# Private utility helpers ---------------------------------------------------------------
# --------------------------------------------------------------------------------------

def _get_from_header(header: List[str], blockname: str, keyname: str) -> str:
    """Return the value for *keyname* inside *blockname* from a parsed Athena header.

    Parameters
    ----------
    header
        List of header lines with XML-like ``<section> key = value`` structure.
    blockname
        Name of the block (e.g. ``"<mesh>"``).  A leading ``<`` and trailing
        ``>`` are optional.
    keyname
        Parameter to extract (e.g. ``"nx1"``).
    """
    blockname = blockname.strip()
    keyname = keyname.strip()
    if not blockname.startswith("<"):
        blockname = "<" + blockname
    if not blockname.endswith(">"):
        blockname += ">"

    current_block = "<none>"
    for line in header:
        line = line.strip()
        # Skip empty lines or comment lines (though comments should have been removed)
        if not line or line.startswith("#"):
            continue
        if line.startswith("<"):
            current_block = line
            continue
        # Safely skip lines that don't contain an assignment
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if current_block == blockname and key.strip() == keyname:
            return value
    raise KeyError(f"no parameter called {blockname}/{keyname}")

def read_binary(filename: str) -> Dict[str, Any]:
    """Parse an Athena++ ``*.bin`` file produced **without** on-the-fly
    coarsening and return a dictionary identical to what `athdf` would
    contain on disk.
    """
    # New fast-path implementation
    return _read_binary_impl(filename, coarsened=False)

def read_coarsened_binary(filename: str) -> Dict[str, Any]:
    """Parse a *coarsened* Athena++ ``*.bin`` snapshot produced with
    ``output_coarsening`` enabled."""
    return _read_binary_impl(filename, coarsened=True)

def read_all_ranks_binary(rank0_filename: str) -> Dict[str, Any]:
    """
    Reads binary files from all ranks and combines them into a single dictionary.

    args:
      rank0_filename - string
          filename of the rank 0 binary file

    returns:
      combined_filedata - dict
          dictionary of combined fluid file data from all ranks
    """
    # Determine the directory and base filename pattern
    # rank0_dir = os.path.dirname(rank0_filename)
    # rank0_base = os.path.basename(rank0_filename).replace("rank_00000000", "rank_*")

    # Find all rank files
    rank_files = sorted(glob.glob(os.path.dirname(rank0_filename).replace("rank_00000000", "rank_*") + "/" + os.path.basename(rank0_filename)))
    # print(rank_files)

    file_sizes = np.array([os.path.getsize(file) for file in rank_files])
    if len(np.unique(file_sizes)) > 1:
        # print("Files are not the same size! you are probably trying to read a slice written with single_file_per_rank=True")
        unique_file_sizes = np.unique(file_sizes)
        larger_file_size = max(unique_file_sizes)
        rank_files = [file for file, size in zip(rank_files, file_sizes) if size == larger_file_size]

    # Read the rank 0 file to get the metadata
    rank0_filedata = read_binary(rank_files[0])

    # Initialize combined filedata with rank 0 data
    combined_filedata = rank0_filedata.copy()

    # Initialize lists to hold combined data
    combined_filedata["mb_index"] = []
    combined_filedata["mb_logical"] = []
    combined_filedata["mb_geometry"] = []
    combined_filedata["mb_data"] = {var: [] for var in rank0_filedata["var_names"]}

    # Read data from all ranks
    for rank_filename in rank_files:
        rank_filedata = read_binary(rank_filename)

        combined_filedata["mb_index"].extend(rank_filedata["mb_index"])
        combined_filedata["mb_logical"].extend(rank_filedata["mb_logical"])
        combined_filedata["mb_geometry"].extend(rank_filedata["mb_geometry"])
        for var in rank0_filedata["var_names"]:
            combined_filedata["mb_data"][var].extend(rank_filedata["mb_data"][var])

    # Convert lists to numpy arrays
    combined_filedata["mb_index"] = np.array(combined_filedata["mb_index"])
    combined_filedata["mb_logical"] = np.array(combined_filedata["mb_logical"])
    combined_filedata["mb_geometry"] = np.array(combined_filedata["mb_geometry"])
    for var in rank0_filedata["var_names"]:
        combined_filedata["mb_data"][var] = np.array(combined_filedata["mb_data"][var])

    # Ensure all relevant fields are stored
    combined_filedata["header"] = rank0_filedata["header"]
    combined_filedata["time"] = rank0_filedata["time"]
    combined_filedata["cycle"] = rank0_filedata["cycle"]
    combined_filedata["var_names"] = rank0_filedata["var_names"]
    combined_filedata["Nx1"] = rank0_filedata["Nx1"]
    combined_filedata["Nx2"] = rank0_filedata["Nx2"]
    combined_filedata["Nx3"] = rank0_filedata["Nx3"]
    combined_filedata["nvars"] = rank0_filedata["nvars"]
    combined_filedata["x1min"] = rank0_filedata["x1min"]
    combined_filedata["x1max"] = rank0_filedata["x1max"]
    combined_filedata["x2min"] = rank0_filedata["x2min"]
    combined_filedata["x2max"] = rank0_filedata["x2max"]
    combined_filedata["x3min"] = rank0_filedata["x3min"]
    combined_filedata["x3max"] = rank0_filedata["x3max"]
    combined_filedata["n_mbs"] = len(combined_filedata["mb_index"])
    combined_filedata["nx1_mb"] = rank0_filedata["nx1_mb"]
    combined_filedata["nx2_mb"] = rank0_filedata["nx2_mb"]
    combined_filedata["nx3_mb"] = rank0_filedata["nx3_mb"]
    combined_filedata["nx1_out_mb"] = rank0_filedata["nx1_out_mb"]
    combined_filedata["nx2_out_mb"] = rank0_filedata["nx2_out_mb"]
    combined_filedata["nx3_out_mb"] = rank0_filedata["nx3_out_mb"]

    return combined_filedata

def read_all_ranks_coarsened_binary(rank0_filename: str) -> Dict[str, Any]:
    """
    Reads binary files from all ranks and combines them into a single dictionary.

    args:
      rank0_filename - string
          filename of the rank 0 binary file

    returns:
      combined_filedata - dict
          dictionary of combined fluid file data from all ranks
    """
    # Determine the directory and base filename pattern
    # rank0_dir = os.path.dirname(rank0_filename)
    # rank0_base = os.path.basename(rank0_filename).replace("rank_00000000", "rank_*")

    # Find all rank files
    rank_files = sorted(glob.glob(os.path.dirname(rank0_filename).replace("rank_00000000", "rank_*") + "/" + os.path.basename(rank0_filename)))
    # print(rank_files)

    # Read the rank 0 file to get the metadata
    rank0_filedata = read_coarsened_binary(rank0_filename)

    # Initialize combined filedata with rank 0 data
    combined_filedata = rank0_filedata.copy()

    # Initialize lists to hold combined data
    combined_filedata["mb_index"] = []
    combined_filedata["mb_logical"] = []
    combined_filedata["mb_geometry"] = []
    combined_filedata["mb_data"] = {var: [] for var in rank0_filedata["var_names"]}

    # Read data from all ranks
    for rank_filename in rank_files:
        rank_filedata = read_coarsened_binary(rank_filename)

        combined_filedata["mb_index"].extend(rank_filedata["mb_index"])
        combined_filedata["mb_logical"].extend(rank_filedata["mb_logical"])
        combined_filedata["mb_geometry"].extend(rank_filedata["mb_geometry"])
        for var in rank0_filedata["var_names"]:
            combined_filedata["mb_data"][var].extend(rank_filedata["mb_data"][var])

    # Convert lists to numpy arrays
    combined_filedata["mb_index"] = np.array(combined_filedata["mb_index"])
    combined_filedata["mb_logical"] = np.array(combined_filedata["mb_logical"])
    combined_filedata["mb_geometry"] = np.array(combined_filedata["mb_geometry"])
    for var in rank0_filedata["var_names"]:
        combined_filedata["mb_data"][var] = np.array(combined_filedata["mb_data"][var])

    # Ensure all relevant fields are stored
    combined_filedata["header"] = rank0_filedata["header"]
    combined_filedata["time"] = rank0_filedata["time"]
    combined_filedata["cycle"] = rank0_filedata["cycle"]
    combined_filedata["var_names"] = rank0_filedata["var_names"]
    combined_filedata["Nx1"] = rank0_filedata["Nx1"]
    combined_filedata["Nx2"] = rank0_filedata["Nx2"]
    combined_filedata["Nx3"] = rank0_filedata["Nx3"]
    combined_filedata["nvars"] = rank0_filedata["nvars"]
    combined_filedata["x1min"] = rank0_filedata["x1min"]
    combined_filedata["x1max"] = rank0_filedata["x1max"]
    combined_filedata["x2min"] = rank0_filedata["x2min"]
    combined_filedata["x2max"] = rank0_filedata["x2max"]
    combined_filedata["x3min"] = rank0_filedata["x3min"]
    combined_filedata["x3max"] = rank0_filedata["x3max"]
    combined_filedata["n_mbs"] = len(combined_filedata["mb_index"])
    combined_filedata["nx1_mb"] = rank0_filedata["nx1_mb"]
    combined_filedata["nx2_mb"] = rank0_filedata["nx2_mb"]
    combined_filedata["nx3_mb"] = rank0_filedata["nx3_mb"]
    combined_filedata["nx1_out_mb"] = rank0_filedata["nx1_out_mb"]
    combined_filedata["nx2_out_mb"] = rank0_filedata["nx2_out_mb"]
    combined_filedata["nx3_out_mb"] = rank0_filedata["nx3_out_mb"]

    return combined_filedata

def read_binary_as_athdf(filename, raw=False, data=None, quantities=None, dtype=None, level=None,
                         return_levels=False, subsample=False, fast_restrict=False, x1_min=None,
                         x1_max=None, x2_min=None, x2_max=None, x3_min=None, x3_max=None,
                         vol_func=None, vol_params=None, face_func_1=None, face_func_2=None,
                         face_func_3=None, center_func_1=None, center_func_2=None,
                         center_func_3=None, num_ghost=0):
    """
    Reads a bin file and organizes data similar to athdf format without writing to file.
    """
    # Step 1: Read binary data
    filedata = read_binary(filename)

    # Step 2: Organize data similar to athdf
    if raw:
        return filedata

    # Prepare dictionary for results
    if data is None:
        data = {}
        new_data = True
    else:
        new_data = False

    # Extract size information
    max_level = max(filedata['mb_logical'][:, 3])
    if level is None:
        level = max_level
    block_size = [filedata['nx1_out_mb'], filedata['nx2_out_mb'], filedata['nx3_out_mb']]
    root_grid_size = [filedata['Nx1'], filedata['Nx2'], filedata['Nx3']]
    levels = filedata['mb_logical'][:, 3]
    logical_locations = filedata['mb_logical'][:, :3]
    if dtype is None:
        dtype = np.float32

    # Calculate nx_vals
    nx_vals = []
    for d in range(3):
        if block_size[d] == 1 and root_grid_size[d] > 1:  # sum or slice
            other_locations = [location
                                for location in zip(levels,
                                                    logical_locations[:, (d+1) % 3],
                                                    logical_locations[:, (d+2) % 3])]
            if len(set(other_locations)) == len(other_locations):  # effective slice
                nx_vals.append(1)
            else:  # nontrivial sum
                num_blocks_this_dim = 0
                for level_this_dim, loc_this_dim in zip(levels,
                                                        logical_locations[:, d]):
                    if level_this_dim <= level:
                        possible_max = (loc_this_dim+1) * 2**(level-level_this_dim)
                        num_blocks_this_dim = max(num_blocks_this_dim, possible_max)
                    else:
                        possible_max = (loc_this_dim+1) // 2**(level_this_dim-level)
                        num_blocks_this_dim = max(num_blocks_this_dim, possible_max)
                nx_vals.append(num_blocks_this_dim)
        elif block_size[d] == 1:  # singleton dimension
            nx_vals.append(1)
        else:  # normal case
            nx_vals.append(root_grid_size[d] * 2**level + 2 * num_ghost)
    nx1, nx2, nx3 = nx_vals
    lx1, lx2, lx3 = [nx // bs for nx, bs in zip(nx_vals, block_size)]

    # Set coordinate system and related functions
    coord = 'cartesian'  # Adjust based on your data
    if vol_func is None:
        def vol_func(xm, xp, ym, yp, zm, zp):
            return (xp-xm) * (yp-ym) * (zp-zm)
    # Define center functions if not provided
    if center_func_1 is None:
        def center_func_1(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_2 is None:
        def center_func_2(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_3 is None:
        def center_func_3(xm, xp):
            return 0.5 * (xm + xp)

    # Populate coordinate arrays
    center_funcs = [center_func_1, center_func_2, center_func_3]
    for d in range(1, 4):
        xf = f'x{d}f'
        xv = f'x{d}v'
        nx = nx_vals[d-1]
        if nx == 1:
            xmin = filedata[f'x{d}min']
            xmax = filedata[f'x{d}max']
            data[xf] = np.array([xmin, xmax], dtype=dtype)
        else:
            xmin = filedata[f'x{d}min']
            xmax = filedata[f'x{d}max']
            data[xf] = np.linspace(xmin, xmax, nx + 1, dtype=dtype)
        data[xv] = np.empty(nx, dtype=dtype)
        for i in range(nx):
            data[xv][i] = center_funcs[d-1](data[xf][i], data[xf][i+1])

    # Create list of quantities
    if quantities is None:
        quantities = filedata['var_names']

    # Account for selection
    i_min, i_max = 0, nx1
    j_min, j_max = 0, nx2
    k_min, k_max = 0, nx3
    if x1_min is not None:
        i_min = max(i_min, np.searchsorted(data['x1f'], x1_min))
    if x1_max is not None:
        i_max = min(i_max, np.searchsorted(data['x1f'], x1_max))
    if x2_min is not None:
        j_min = max(j_min, np.searchsorted(data['x2f'], x2_min))
    if x2_max is not None:
        j_max = min(j_max, np.searchsorted(data['x2f'], x2_max))
    if x3_min is not None:
        k_min = max(k_min, np.searchsorted(data['x3f'], x3_min))
    if x3_max is not None:
        k_max = min(k_max, np.searchsorted(data['x3f'], x3_max))

    # Prepare arrays for data and bookkeeping
    if new_data:
        for q in quantities:
            data[q] = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min), dtype=dtype)
        if return_levels:
            data['Levels'] = np.empty((k_max-k_min, j_max-j_min, i_max-i_min), dtype=np.int32)
    else:
        for q in quantities:
            data[q].fill(0.0)
    if not subsample and not fast_restrict and max_level > level:
        restricted_data = np.zeros((lx3, lx2, lx1), dtype=bool)

    # Step 3: Process each block
    for block_num in range(filedata['n_mbs']):
        block_level = levels[block_num]
        block_location = logical_locations[block_num]

        # Implement logic for prolongation, restriction, subsampling
        if block_level <= level:
            s = 2 ** (level - block_level)
            il_d = block_location[0] * block_size[0] * s if nx1 > 1 else 0
            jl_d = block_location[1] * block_size[1] * s if nx2 > 1 else 0
            kl_d = block_location[2] * block_size[2] * s if nx3 > 1 else 0
            iu_d = il_d + block_size[0] * s if nx1 > 1 else 1
            ju_d = jl_d + block_size[1] * s if nx2 > 1 else 1
            ku_d = kl_d + block_size[2] * s if nx3 > 1 else 1

            il_s = max(il_d, i_min) - il_d
            jl_s = max(jl_d, j_min) - jl_d
            kl_s = max(kl_d, k_min) - kl_d
            iu_s = min(iu_d, i_max) - il_d
            ju_s = min(ju_d, j_max) - jl_d
            ku_s = min(ku_d, k_max) - kl_d

            if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                continue

            il_d = max(il_d, i_min) - i_min
            jl_d = max(jl_d, j_min) - j_min
            kl_d = max(kl_d, k_min) - k_min
            iu_d = min(iu_d, i_max) - i_min
            ju_d = min(ju_d, j_max) - j_min
            ku_d = min(ku_d, k_max) - k_min

            for q in quantities:
                block_data = filedata['mb_data'][q][block_num]
                if s > 1:
                    block_data = np.repeat(np.repeat(np.repeat(block_data, s, axis=2), s, axis=1), s, axis=0)
                data[q][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_data[kl_s:ku_s, jl_s:ju_s, il_s:iu_s]
        else:
            # Implement restriction logic here (similar to athdf function)
            pass

        if return_levels:
            data['Levels'][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_level

    # Step 4: Finalize data
    if level < max_level and not subsample and not fast_restrict:
        # Remove volume factors from restricted data
        for loc3 in range(lx3):
            for loc2 in range(lx2):
                for loc1 in range(lx1):
                    if restricted_data[loc3, loc2, loc1]:
                        il = loc1 * block_size[0]
                        jl = loc2 * block_size[1]
                        kl = loc3 * block_size[2]
                        iu = il + block_size[0]
                        ju = jl + block_size[1]
                        ku = kl + block_size[2]
                        il = max(il, i_min) - i_min
                        jl = max(jl, j_min) - j_min
                        kl = max(kl, k_min) - k_min
                        iu = min(iu, i_max) - i_min
                        ju = min(ju, j_max) - j_min
                        ku = min(ku, k_max) - k_min
                        for k in range(kl, ku):
                            for j in range(jl, ju):
                                for i in range(il, iu):
                                    x1m, x1p = data['x1f'][i], data['x1f'][i+1]
                                    x2m, x2p = data['x2f'][j], data['x2f'][j+1]
                                    x3m, x3p = data['x3f'][k], data['x3f'][k+1]
                                    vol = vol_func(x1m, x1p, x2m, x2p, x3m, x3p)
                                    for q in quantities:
                                        data[q][k, j, i] /= vol

    # Add metadata
    data['Time'] = filedata['time']
    data['NumCycles'] = filedata['cycle']
    data['MaxLevel'] = max_level

    return data

def read_all_ranks_binary_as_athdf(rank0_filename, raw=False, data=None, quantities=None, dtype=None, level=None,
                         return_levels=False, subsample=False, fast_restrict=False, x1_min=None,
                         x1_max=None, x2_min=None, x2_max=None, x3_min=None, x3_max=None,
                         vol_func=None, vol_params=None, face_func_1=None, face_func_2=None,
                         face_func_3=None, center_func_1=None, center_func_2=None,
                         center_func_3=None, num_ghost=0):
    """
    Reads a bin file and organizes data similar to athdf format without writing to file.
    """
    # Step 1: Read binary data
    filedata = read_all_ranks_binary(rank0_filename)

    # Step 2: Organize data similar to athdf
    if raw:
        return filedata

    # Prepare dictionary for results
    if data is None:
        data = {}
        new_data = True
    else:
        new_data = False

    # Extract size information
    max_level = max(filedata['mb_logical'][:, 3])
    if level is None:
        level = max_level
    block_size = [filedata['nx1_out_mb'], filedata['nx2_out_mb'], filedata['nx3_out_mb']]
    root_grid_size = [filedata['Nx1'], filedata['Nx2'], filedata['Nx3']]
    levels = filedata['mb_logical'][:, 3]
    logical_locations = filedata['mb_logical'][:, :3]
    if dtype is None:
        dtype = np.float32

    # Calculate nx_vals
    nx_vals = []
    for d in range(3):
        if block_size[d] == 1 and root_grid_size[d] > 1:  # sum or slice
            other_locations = [location
                                for location in zip(levels,
                                                    logical_locations[:, (d+1) % 3],
                                                    logical_locations[:, (d+2) % 3])]
            if len(set(other_locations)) == len(other_locations):  # effective slice
                nx_vals.append(1)
            else:  # nontrivial sum
                num_blocks_this_dim = 0
                for level_this_dim, loc_this_dim in zip(levels,
                                                        logical_locations[:, d]):
                    if level_this_dim <= level:
                        possible_max = (loc_this_dim+1) * 2**(level-level_this_dim)
                        num_blocks_this_dim = max(num_blocks_this_dim, possible_max)
                    else:
                        possible_max = (loc_this_dim+1) // 2**(level_this_dim-level)
                        num_blocks_this_dim = max(num_blocks_this_dim, possible_max)
                nx_vals.append(num_blocks_this_dim)
        elif block_size[d] == 1:  # singleton dimension
            nx_vals.append(1)
        else:  # normal case
            nx_vals.append(root_grid_size[d] * 2**level + 2 * num_ghost)
    nx1, nx2, nx3 = nx_vals
    lx1, lx2, lx3 = [nx // bs for nx, bs in zip(nx_vals, block_size)]
    # Set coordinate system and related functions
    coord = 'cartesian'  # Adjust based on your data
    if vol_func is None:
        def vol_func(xm, xp, ym, yp, zm, zp):
            return (xp-xm) * (yp-ym) * (zp-zm)
    # Define center functions if not provided
    if center_func_1 is None:
        def center_func_1(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_2 is None:
        def center_func_2(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_3 is None:
        def center_func_3(xm, xp):
            return 0.5 * (xm + xp)

    # Populate coordinate arrays
    center_funcs = [center_func_1, center_func_2, center_func_3]
    for d in range(1, 4):
        xf = f'x{d}f'
        xv = f'x{d}v'
        nx = nx_vals[d-1]
        if nx == 1:
            xmin = filedata[f'x{d}min']
            xmax = filedata[f'x{d}max']
            data[xf] = np.array([xmin, xmax], dtype=dtype)
        else:
            xmin = filedata[f'x{d}min']
            xmax = filedata[f'x{d}max']
            data[xf] = np.linspace(xmin, xmax, nx + 1, dtype=dtype)
        data[xv] = np.empty(nx, dtype=dtype)
        for i in range(nx):
            data[xv][i] = center_funcs[d-1](data[xf][i], data[xf][i+1])

    # Create list of quantities
    if quantities is None:
        quantities = filedata['var_names']

    # Account for selection
    i_min, i_max = 0, nx1
    j_min, j_max = 0, nx2
    k_min, k_max = 0, nx3
    if x1_min is not None:
        i_min = max(i_min, np.searchsorted(data['x1f'], x1_min))
    if x1_max is not None:
        i_max = min(i_max, np.searchsorted(data['x1f'], x1_max))
    if x2_min is not None:
        j_min = max(j_min, np.searchsorted(data['x2f'], x2_min))
    if x2_max is not None:
        j_max = min(j_max, np.searchsorted(data['x2f'], x2_max))
    if x3_min is not None:
        k_min = max(k_min, np.searchsorted(data['x3f'], x3_min))
    if x3_max is not None:
        k_max = min(k_max, np.searchsorted(data['x3f'], x3_max))

    # Prepare arrays for data and bookkeeping
    if new_data:
        for q in quantities:
            data[q] = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min), dtype=dtype)
        if return_levels:
            data['Levels'] = np.empty((k_max-k_min, j_max-j_min, i_max-i_min), dtype=np.int32)
    else:
        for q in quantities:
            data[q].fill(0.0)
    if not subsample and not fast_restrict and max_level > level:
        restricted_data = np.zeros((lx3, lx2, lx1), dtype=bool)

    # Step 3: Process each block
    for block_num in range(filedata['n_mbs']):
        block_level = levels[block_num]
        block_location = logical_locations[block_num]

        # Implement logic for prolongation, restriction, subsampling
        if block_level <= level:
            s = 2 ** (level - block_level)
            il_d = block_location[0] * block_size[0] * s if nx1 > 1 else 0
            jl_d = block_location[1] * block_size[1] * s if nx2 > 1 else 0
            kl_d = block_location[2] * block_size[2] * s if nx3 > 1 else 0
            iu_d = il_d + block_size[0] * s if nx1 > 1 else 1
            ju_d = jl_d + block_size[1] * s if nx2 > 1 else 1
            ku_d = kl_d + block_size[2] * s if nx3 > 1 else 1

            il_s = max(il_d, i_min) - il_d
            jl_s = max(jl_d, j_min) - jl_d
            kl_s = max(kl_d, k_min) - kl_d
            iu_s = min(iu_d, i_max) - il_d
            ju_s = min(ju_d, j_max) - jl_d
            ku_s = min(ku_d, k_max) - kl_d

            if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                continue

            il_d = max(il_d, i_min) - i_min
            jl_d = max(jl_d, j_min) - j_min
            kl_d = max(kl_d, k_min) - k_min
            iu_d = min(iu_d, i_max) - i_min
            ju_d = min(ju_d, j_max) - j_min
            ku_d = min(ku_d, k_max) - k_min

            for q in quantities:
                block_data = filedata['mb_data'][q][block_num]
                if s > 1:
                    block_data = np.repeat(np.repeat(np.repeat(block_data, s, axis=2), s, axis=1), s, axis=0)
                data[q][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_data[kl_s:ku_s, jl_s:ju_s, il_s:iu_s]
        else:
            # Implement restriction logic here (similar to athdf function)
            pass

        if return_levels:
            data['Levels'][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_level

    # Step 4: Finalize data
    if level < max_level and not subsample and not fast_restrict:
        # Remove volume factors from restricted data
        for loc3 in range(lx3):
            for loc2 in range(lx2):
                for loc1 in range(lx1):
                    if restricted_data[loc3, loc2, loc1]:
                        il = loc1 * block_size[0]
                        jl = loc2 * block_size[1]
                        kl = loc3 * block_size[2]
                        iu = il + block_size[0]
                        ju = jl + block_size[1]
                        ku = kl + block_size[2]
                        il = max(il, i_min) - i_min
                        jl = max(jl, j_min) - j_min
                        kl = max(kl, k_min) - k_min
                        iu = min(iu, i_max) - i_min
                        ju = min(ju, j_max) - j_min
                        ku = min(ku, k_max) - k_min
                        for k in range(kl, ku):
                            for j in range(jl, ju):
                                for i in range(il, iu):
                                    x1m, x1p = data['x1f'][i], data['x1f'][i+1]
                                    x2m, x2p = data['x2f'][j], data['x2f'][j+1]
                                    x3m, x3p = data['x3f'][k], data['x3f'][k+1]
                                    vol = vol_func(x1m, x1p, x2m, x2p, x3m, x3p)
                                    for q in quantities:
                                        data[q][k, j, i] /= vol

    # Add metadata
    data['Time'] = filedata['time']
    data['NumCycles'] = filedata['cycle']
    data['MaxLevel'] = max_level

    return data

def read_all_ranks_coarsened_binary_as_athdf(rank0_filename, raw=False, data=None, quantities=None, dtype=None, level=None,
                         return_levels=False, subsample=False, fast_restrict=False, x1_min=None,
                         x1_max=None, x2_min=None, x2_max=None, x3_min=None, x3_max=None,
                         vol_func=None, vol_params=None, face_func_1=None, face_func_2=None,
                         face_func_3=None, center_func_1=None, center_func_2=None,
                         center_func_3=None, num_ghost=0):
    """
    Reads a bin file and organizes data similar to athdf format without writing to file.
    """
    # Step 1: Read binary data
    filedata = read_all_ranks_coarsened_binary(rank0_filename)

    # Step 2: Organize data similar to athdf
    if raw:
        return filedata

    # Prepare dictionary for results
    if data is None:
        data = {}
        new_data = True
    else:
        new_data = False

    # Extract size information
    max_level = max(filedata['mb_logical'][:, 3])
    if level is None:
        level = max_level
    block_size = [filedata['nx1_mb'], filedata['nx2_mb'], filedata['nx3_mb']]
    root_grid_size = [filedata['Nx1'], filedata['Nx2'], filedata['Nx3']]
    levels = filedata['mb_logical'][:, 3]
    logical_locations = filedata['mb_logical'][:, :3]
    if dtype is None:
        dtype = np.float32

    # Calculate nx_vals
    nx_vals = []
    for d in range(3):
        if block_size[d] == 1 and root_grid_size[d] > 1:
            # Implement logic for sum or slice as in athdf
            nx_vals.append(root_grid_size[d] * 2**level)
        elif block_size[d] == 1:
            nx_vals.append(1)
        else:
            nx_vals.append(root_grid_size[d] * 2**level + 2 * num_ghost)
    nx1, nx2, nx3 = nx_vals
    lx1, lx2, lx3 = [nx // bs for nx, bs in zip(nx_vals, block_size)]

    # Set coordinate system and related functions
    coord = 'cartesian'  # Adjust based on your data
    if vol_func is None:
        def vol_func(xm, xp, ym, yp, zm, zp):
            return (xp-xm) * (yp-ym) * (zp-zm)
    # Define center functions if not provided
    if center_func_1 is None:
        def center_func_1(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_2 is None:
        def center_func_2(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_3 is None:
        def center_func_3(xm, xp):
            return 0.5 * (xm + xp)

    # Populate coordinate arrays
    center_funcs = [center_func_1, center_func_2, center_func_3]
    for d in range(1, 4):
        xf = f'x{d}f'
        xv = f'x{d}v'
        nx = nx_vals[d-1]
        if nx == 1:
            xmin = filedata[f'x{d}min']
            xmax = filedata[f'x{d}max']
            data[xf] = np.array([xmin, xmax], dtype=dtype)
        else:
            xmin = filedata[f'x{d}min']
            xmax = filedata[f'x{d}max']
            data[xf] = np.linspace(xmin, xmax, nx + 1, dtype=dtype)
        data[xv] = np.empty(nx, dtype=dtype)
        for i in range(nx):
            data[xv][i] = center_funcs[d-1](data[xf][i], data[xf][i+1])

    # Create list of quantities
    if quantities is None:
        quantities = filedata['var_names']

    # Account for selection
    i_min, i_max = 0, nx1
    j_min, j_max = 0, nx2
    k_min, k_max = 0, nx3
    if x1_min is not None:
        i_min = max(i_min, np.searchsorted(data['x1f'], x1_min))
    if x1_max is not None:
        i_max = min(i_max, np.searchsorted(data['x1f'], x1_max))
    if x2_min is not None:
        j_min = max(j_min, np.searchsorted(data['x2f'], x2_min))
    if x2_max is not None:
        j_max = min(j_max, np.searchsorted(data['x2f'], x2_max))
    if x3_min is not None:
        k_min = max(k_min, np.searchsorted(data['x3f'], x3_min))
    if x3_max is not None:
        k_max = min(k_max, np.searchsorted(data['x3f'], x3_max))

    # Prepare arrays for data and bookkeeping
    if new_data:
        for q in quantities:
            data[q] = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min), dtype=dtype)
        if return_levels:
            data['Levels'] = np.empty((k_max-k_min, j_max-j_min, i_max-i_min), dtype=np.int32)
    else:
        for q in quantities:
            data[q].fill(0.0)
    if not subsample and not fast_restrict and max_level > level:
        restricted_data = np.zeros((lx3, lx2, lx1), dtype=bool)

    # Step 3: Process each block
    for block_num in range(filedata['n_mbs']):
        block_level = levels[block_num]
        block_location = logical_locations[block_num]

        # Implement logic for prolongation, restriction, subsampling
        if block_level <= level:
            s = 2 ** (level - block_level)
            il_d = block_location[0] * block_size[0] * s if nx1 > 1 else 0
            jl_d = block_location[1] * block_size[1] * s if nx2 > 1 else 0
            kl_d = block_location[2] * block_size[2] * s if nx3 > 1 else 0
            iu_d = il_d + block_size[0] * s if nx1 > 1 else 1
            ju_d = jl_d + block_size[1] * s if nx2 > 1 else 1
            ku_d = kl_d + block_size[2] * s if nx3 > 1 else 1

            il_s = max(il_d, i_min) - il_d
            jl_s = max(jl_d, j_min) - jl_d
            kl_s = max(kl_d, k_min) - kl_d
            iu_s = min(iu_d, i_max) - il_d
            ju_s = min(ju_d, j_max) - jl_d
            ku_s = min(ku_d, k_max) - kl_d

            if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                continue

            il_d = max(il_d, i_min) - i_min
            jl_d = max(jl_d, j_min) - j_min
            kl_d = max(kl_d, k_min) - k_min
            iu_d = min(iu_d, i_max) - i_min
            ju_d = min(ju_d, j_max) - j_min
            ku_d = min(ku_d, k_max) - k_min

            for q in quantities:
                block_data = filedata['mb_data'][q][block_num]
                if s > 1:
                    block_data = np.repeat(np.repeat(np.repeat(block_data, s, axis=2), s, axis=1), s, axis=0)
                data[q][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_data[kl_s:ku_s, jl_s:ju_s, il_s:iu_s]
        else:
            # Implement restriction logic here (similar to athdf function)
            pass

        if return_levels:
            data['Levels'][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_level

    # Step 4: Finalize data
    if level < max_level and not subsample and not fast_restrict:
        # Remove volume factors from restricted data
        for loc3 in range(lx3):
            for loc2 in range(lx2):
                for loc1 in range(lx1):
                    if restricted_data[loc3, loc2, loc1]:
                        il = loc1 * block_size[0]
                        jl = loc2 * block_size[1]
                        kl = loc3 * block_size[2]
                        iu = il + block_size[0]
                        ju = jl + block_size[1]
                        ku = kl + block_size[2]
                        il = max(il, i_min) - i_min
                        jl = max(jl, j_min) - j_min
                        kl = max(kl, k_min) - k_min
                        iu = min(iu, i_max) - i_min
                        ju = min(ju, j_max) - j_min
                        ku = min(ku, k_max) - k_min
                        for k in range(kl, ku):
                            for j in range(jl, ju):
                                for i in range(il, iu):
                                    x1m, x1p = data['x1f'][i], data['x1f'][i+1]
                                    x2m, x2p = data['x2f'][j], data['x2f'][j+1]
                                    x3m, x3p = data['x3f'][k], data['x3f'][k+1]
                                    vol = vol_func(x1m, x1p, x2m, x2p, x3m, x3p)
                                    for q in quantities:
                                        data[q][k, j, i] /= vol

    # Add metadata
    data['Time'] = filedata['time']
    data['NumCycles'] = filedata['cycle']
    data['MaxLevel'] = max_level

    return data

def read_single_rank_binary_as_athdf(filename, meshblock_index_in_file, raw=False, data=None, quantities=None, dtype=None,
                                     return_levels=False, x1_min=None, x1_max=None, x2_min=None,
                                     x2_max=None, x3_min=None, x3_max=None, vol_func=None,
                                     center_func_1=None, center_func_2=None, center_func_3=None):
    """
    Reads a single rank binary file and organizes data similar to athdf format without writing to file.
    """
    # Step 1: Read binary data for a single rank
    filedata = read_binary(filename)

    if raw:
        return filedata

    # Prepare dictionary for results
    if data is None:
        data = {}
        new_data = True
    else:
        new_data = False

    # Extract size information
    block_size = [filedata['nx1_mb'], filedata['nx2_mb'], filedata['nx3_mb']]
    if dtype is None:
        dtype = np.float32

    # Set coordinate system and related functions
    if vol_func is None:
        def vol_func(xm, xp, ym, yp, zm, zp):
            return (xp-xm) * (yp-ym) * (zp-zm)
    if center_func_1 is None:
        def center_func_1(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_2 is None:
        def center_func_2(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_3 is None:
        def center_func_3(xm, xp):
            return 0.5 * (xm + xp)

    # Populate coordinate arrays
    center_funcs = [center_func_1, center_func_2, center_func_3]
    for d in range(1, 4):
        xf = f'x{d}f'
        xv = f'x{d}v'
        nx = block_size[d-1]

        # Use the meshblock geometry for local min and max
        xmin = filedata['mb_geometry'][meshblock_index_in_file, (d-1)*2]
        xmax = filedata['mb_geometry'][meshblock_index_in_file, (d-1)*2 + 1]

        data[xf] = np.linspace(xmin, xmax, nx + 1, dtype=dtype)
        data[xv] = np.empty(nx, dtype=dtype)
        for i in range(nx):
            data[xv][i] = center_funcs[d-1](data[xf][i], data[xf][i+1])

    # Create list of quantities
    if quantities is None:
        quantities = filedata['var_names']

    # Account for selection
    i_min, i_max = 0, block_size[0]
    j_min, j_max = 0, block_size[1]
    k_min, k_max = 0, block_size[2]
    if x1_min is not None:
        i_min = max(i_min, np.searchsorted(data['x1f'], x1_min))
    if x1_max is not None:
        i_max = min(i_max, np.searchsorted(data['x1f'], x1_max))
    if x2_min is not None:
        j_min = max(j_min, np.searchsorted(data['x2f'], x2_min))
    if x2_max is not None:
        j_max = min(j_max, np.searchsorted(data['x2f'], x2_max))
    if x3_min is not None:
        k_min = max(k_min, np.searchsorted(data['x3f'], x3_min))
    if x3_max is not None:
        k_max = min(k_max, np.searchsorted(data['x3f'], x3_max))

    # Prepare arrays for data
    if new_data:
        for q in quantities:
            data[q] = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min), dtype=dtype)
        if return_levels:
            data['Levels'] = np.empty((k_max-k_min, j_max-j_min, i_max-i_min), dtype=np.int32)
    else:
        for q in quantities:
            data[q].fill(0.0)

    # Process the single block
    for q in quantities:
        block_data = filedata['mb_data'][q][meshblock_index_in_file]  # Single rank, so only one block
        data[q] = block_data[k_min:k_max, j_min:j_max, i_min:i_max]

    if return_levels:
        data['Levels'].fill(filedata['mb_logical'][meshblock_index_in_file, 3])  # Level of the single block

    # Add metadata
    data['Time'] = filedata['time']
    data['NumCycles'] = filedata['cycle']
    data['MaxLevel'] = filedata['mb_logical'][meshblock_index_in_file, 3]

    return data

def read_coarsened_binary_as_athdf(filename, raw=False, data=None, quantities=None, dtype=None, level=None,
                         return_levels=False, subsample=False, fast_restrict=False, x1_min=None,
                         x1_max=None, x2_min=None, x2_max=None, x3_min=None, x3_max=None,
                         vol_func=None, vol_params=None, face_func_1=None, face_func_2=None,
                         face_func_3=None, center_func_1=None, center_func_2=None,
                         center_func_3=None, num_ghost=0):
    """
    Reads a bin file and organizes data similar to athdf format without writing to file.
    """
    # Step 1: Read binary data
    filedata = read_coarsened_binary(filename)

    # Step 2: Organize data similar to athdf
    if raw:
        return filedata

    # Prepare dictionary for results
    if data is None:
        data = {}
        new_data = True
    else:
        new_data = False

    # Extract size information
    max_level = max(filedata['mb_logical'][:, 3])
    if level is None:
        level = max_level
    block_size = [filedata['nx1_mb'], filedata['nx2_mb'], filedata['nx3_mb']]
    root_grid_size = [filedata['Nx1'], filedata['Nx2'], filedata['Nx3']]
    levels = filedata['mb_logical'][:, 3]
    logical_locations = filedata['mb_logical'][:, :3]
    if dtype is None:
        dtype = np.float32

    # Calculate nx_vals
    nx_vals = []
    for d in range(3):
        if block_size[d] == 1 and root_grid_size[d] > 1:
            # Implement logic for sum or slice as in athdf
            nx_vals.append(root_grid_size[d] * 2**level)
        elif block_size[d] == 1:
            nx_vals.append(1)
        else:
            nx_vals.append(root_grid_size[d] * 2**level + 2 * num_ghost)
    nx1, nx2, nx3 = nx_vals
    lx1, lx2, lx3 = [nx // bs for nx, bs in zip(nx_vals, block_size)]

    # Set coordinate system and related functions
    coord = 'cartesian'  # Adjust based on your data
    if vol_func is None:
        def vol_func(xm, xp, ym, yp, zm, zp):
            return (xp-xm) * (yp-ym) * (zp-zm)
    # Define center functions if not provided
    if center_func_1 is None:
        def center_func_1(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_2 is None:
        def center_func_2(xm, xp):
            return 0.5 * (xm + xp)
    if center_func_3 is None:
        def center_func_3(xm, xp):
            return 0.5 * (xm + xp)

    # Populate coordinate arrays
    center_funcs = [center_func_1, center_func_2, center_func_3]
    for d in range(1, 4):
        xf = f'x{d}f'
        xv = f'x{d}v'
        nx = nx_vals[d-1]
        if nx == 1:
            xmin = filedata[f'x{d}min']
            xmax = filedata[f'x{d}max']
            data[xf] = np.array([xmin, xmax], dtype=dtype)
        else:
            xmin = filedata[f'x{d}min']
            xmax = filedata[f'x{d}max']
            data[xf] = np.linspace(xmin, xmax, nx + 1, dtype=dtype)
        data[xv] = np.empty(nx, dtype=dtype)
        for i in range(nx):
            data[xv][i] = center_funcs[d-1](data[xf][i], data[xf][i+1])

    # Create list of quantities
    if quantities is None:
        quantities = filedata['var_names']

    # Account for selection
    i_min, i_max = 0, nx1
    j_min, j_max = 0, nx2
    k_min, k_max = 0, nx3
    if x1_min is not None:
        i_min = max(i_min, np.searchsorted(data['x1f'], x1_min))
    if x1_max is not None:
        i_max = min(i_max, np.searchsorted(data['x1f'], x1_max))
    if x2_min is not None:
        j_min = max(j_min, np.searchsorted(data['x2f'], x2_min))
    if x2_max is not None:
        j_max = min(j_max, np.searchsorted(data['x2f'], x2_max))
    if x3_min is not None:
        k_min = max(k_min, np.searchsorted(data['x3f'], x3_min))
    if x3_max is not None:
        k_max = min(k_max, np.searchsorted(data['x3f'], x3_max))

    # Prepare arrays for data and bookkeeping
    if new_data:
        for q in quantities:
            data[q] = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min), dtype=dtype)
        if return_levels:
            data['Levels'] = np.empty((k_max-k_min, j_max-j_min, i_max-i_min), dtype=np.int32)
    else:
        for q in quantities:
            data[q].fill(0.0)
    if not subsample and not fast_restrict and max_level > level:
        restricted_data = np.zeros((lx3, lx2, lx1), dtype=bool)

    # Step 3: Process each block
    for block_num in range(filedata['n_mbs']):
        block_level = levels[block_num]
        block_location = logical_locations[block_num]

        # Implement logic for prolongation, restriction, subsampling
        if block_level <= level:
            s = 2 ** (level - block_level)
            il_d = block_location[0] * block_size[0] * s if nx1 > 1 else 0
            jl_d = block_location[1] * block_size[1] * s if nx2 > 1 else 0
            kl_d = block_location[2] * block_size[2] * s if nx3 > 1 else 0
            iu_d = il_d + block_size[0] * s if nx1 > 1 else 1
            ju_d = jl_d + block_size[1] * s if nx2 > 1 else 1
            ku_d = kl_d + block_size[2] * s if nx3 > 1 else 1

            il_s = max(il_d, i_min) - il_d
            jl_s = max(jl_d, j_min) - jl_d
            kl_s = max(kl_d, k_min) - kl_d
            iu_s = min(iu_d, i_max) - il_d
            ju_s = min(ju_d, j_max) - jl_d
            ku_s = min(ku_d, k_max) - kl_d

            if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                continue

            il_d = max(il_d, i_min) - i_min
            jl_d = max(jl_d, j_min) - j_min
            kl_d = max(kl_d, k_min) - k_min
            iu_d = min(iu_d, i_max) - i_min
            ju_d = min(ju_d, j_max) - j_min
            ku_d = min(ku_d, k_max) - k_min

            for q in quantities:
                block_data = filedata['mb_data'][q][block_num]
                if s > 1:
                    block_data = np.repeat(np.repeat(np.repeat(block_data, s, axis=2), s, axis=1), s, axis=0)
                data[q][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_data[kl_s:ku_s, jl_s:ju_s, il_s:iu_s]
        else:
            # Implement restriction logic here (similar to athdf function)
            pass

        if return_levels:
            data['Levels'][kl_d:ku_d, jl_d:ju_d, il_d:iu_d] = block_level

    # Step 4: Finalize data
    if level < max_level and not subsample and not fast_restrict:
        # Remove volume factors from restricted data
        for loc3 in range(lx3):
            for loc2 in range(lx2):
                for loc1 in range(lx1):
                    if restricted_data[loc3, loc2, loc1]:
                        il = loc1 * block_size[0]
                        jl = loc2 * block_size[1]
                        kl = loc3 * block_size[2]
                        iu = il + block_size[0]
                        ju = jl + block_size[1]
                        ku = kl + block_size[2]
                        il = max(il, i_min) - i_min
                        jl = max(jl, j_min) - j_min
                        kl = max(kl, k_min) - k_min
                        iu = min(iu, i_max) - i_min
                        ju = min(ju, j_max) - j_min
                        ku = min(ku, k_max) - k_min
                        for k in range(kl, ku):
                            for j in range(jl, ju):
                                for i in range(il, iu):
                                    x1m, x1p = data['x1f'][i], data['x1f'][i+1]
                                    x2m, x2p = data['x2f'][j], data['x2f'][j+1]
                                    x3m, x3p = data['x3f'][k], data['x3f'][k+1]
                                    vol = vol_func(x1m, x1p, x2m, x2p, x3m, x3p)
                                    for q in quantities:
                                        data[q][k, j, i] /= vol

    # Add metadata
    data['Time'] = filedata['time']
    data['NumCycles'] = filedata['cycle']
    data['MaxLevel'] = max_level

    return data

def write_xdmf_for(xdmfname: str, dumpname: str, fdata: Dict[str, Any], mode: str = "auto") -> None:
    """
    Writes an xdmf file for a fluid snapshot file.

    args:
      xdmfname - string
          name of xdmf file
      dumpname - string
          location of fluid data file relative to xdmfname directory
      fdata    - dict
          dictionary of fluid file data, e.g., as loaded from read_binary(...)
      mode     - string (unimplemented)
          force xdmf for format (auto sets by extension)
    """

    # Using a context manager ensures the file handle is properly closed even if
    # an exception is raised during the write process.
    with open(xdmfname, "w") as fp:

        def write_meshblock(fp, mb, nx1, nx2, nx3, nmb, dumpname, vars_no_b, vars_w_b):
            fp.write(f"""  <Grid Name=\"MeshBlock{mb}\" GridType=\"Uniform\">\n""")
            fp.write("""   <Topology TopologyType=\"3DRectMesh\" """)
            fp.write(f""" NumberOfElements=\"{nx3+1} {nx2+1} {nx1+1}\"/>\n""")
            fp.write("""   <Geometry GeometryType=\"VXVYVZ\">\n""")
            fp.write(
                f"""    <DataItem ItemType=\"HyperSlab\" Dimensions=\"{nx1+1}\">\n     <DataItem Dimensions=\"3 2\" NumberType=\"Int\"> {mb} 0 1 1 1 {nx1+1} </DataItem>\n     <DataItem Dimensions=\"{nmb} {nx1+1}\" Format=\"HDF\"> {dumpname}:/x1f </DataItem>\n    </DataItem>\n    <DataItem ItemType=\"HyperSlab\" Dimensions=\"{nx2+1}\">\n     <DataItem Dimensions=\"3 2\" NumberType=\"Int\"> {mb} 0 1 1 1 {nx2+1} </DataItem>\n     <DataItem Dimensions=\"{nmb} {nx2+1}\" Format=\"HDF\"> {dumpname}:/x2f </DataItem>\n    </DataItem>\n    <DataItem ItemType=\"HyperSlab\" Dimensions=\"{nx3+1}\">\n     <DataItem Dimensions=\"3 2\" NumberType=\"Int\"> {mb} 0 1 1 1 {nx3+1} </DataItem>\n     <DataItem Dimensions=\"{nmb} {nx3+1}\" Format=\"HDF\"> {dumpname}:/x3f </DataItem>\n    </DataItem>\n   </Geometry>\n"""
            )

            nvar_no_b = len(vars_no_b)
            for vi, var_name in enumerate(vars_no_b):
                fp.write(
                    f"""   <Attribute Name=\"{var_name}\" Center=\"Cell\">\n    <DataItem ItemType=\"HyperSlab\" Dimensions=\"{nx3} {nx2} {nx1}\">\n     <DataItem Dimensions=\"3 5\" NumberType=\"Int\">\n      {vi} {mb} 0 0 0 1 1 1 1 1 1 1 {nx3} {nx2} {nx1}\n     </DataItem>\n     <DataItem Dimensions=\"{nvar_no_b} {nmb} {nx3} {nx2} {nx1}\" Format=\"HDF\">\n      {dumpname}:/uov\n     </DataItem>\n    </DataItem>\n   </Attribute>\n"""
                )

            if vars_w_b:
                nvar_w_b = len(vars_w_b)
                for vi, var_name in enumerate(vars_w_b):
                    fp.write(
                        f"""   <Attribute Name=\"{var_name}\" Center=\"Cell\">\n        <DataItem ItemType=\"HyperSlab\" Dimensions=\"{nx3} {nx2} {nx1}\">\n         <DataItem Dimensions=\"3 5\" NumberType=\"Int\">\n          {vi} {mb} 0 0 0 1 1 1 1 1 1 1 {nx3} {nx2} {nx1}\n         </DataItem>\n         <DataItem Dimensions=\"{nvar_w_b} {nmb} {nx3} {nx2} {nx1}\" Format=\"HDF\">\n          {dumpname}:/B\n         </DataItem>\n        </DataItem>\n       </Attribute>\n"""
                    )

            fp.write("""  </Grid>\n""")

        # ----------------------------------------------------------------------------------
        # File header ---------------------------------------------------------------------
        # ----------------------------------------------------------------------------------
        fp.write(
            """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Information Name="TimeVaryingMetaData" Value="True"/>
"""
        )
        fp.write("""<Domain>\n""")
        fp.write("""<Grid Name="Mesh" GridType="Collection">\n""")
        fp.write(f""" <Time Value="{fdata['time']}"/>\n""")

        vars_without_b = [v for v in fdata["var_names"] if "bcc" not in v]
        vars_only_b = [v for v in fdata["var_names"] if v not in vars_without_b]

        nx1 = fdata["nx1_out_mb"]
        nx2 = fdata["nx2_out_mb"]
        nx3 = fdata["nx3_out_mb"]
        nmb = fdata["n_mbs"]

        for mb in range(nmb):
            write_meshblock(
                fp, mb, nx1, nx2, nx3, nmb, dumpname, vars_without_b, vars_only_b
            )

        fp.write("""</Grid>\n""")
        fp.write("""</Domain>\n""")
        fp.write("""</Xdmf>\n""")

def convert_file(binary_fname: str) -> None:
    """
    Converts a single file.

    args:
      binary_filename - string
        filename of bin file to convert

    This will create new files "binary_data.bin" -> "binary_data.athdf" and
    "binary_data.athdf.xdmf"
    """
    athdf_fname = binary_fname.replace(".bin", "") + ".athdf"
    xdmf_fname = athdf_fname + ".xdmf"
    filedata = read_binary(binary_fname)
    write_athdf(athdf_fname, filedata)
    write_xdmf_for(xdmf_fname, os.path.basename(athdf_fname), filedata)

def read_rank_binary_as_athdf(
    filename: str,
    raw: bool = False,
    data: Optional[Dict[str, Any]] = None,
    quantities: Optional[Sequence[str]] = None,
    dtype: Optional[np.dtype] = None,
    return_levels: bool = False,
) -> Dict[str, Any]:
    """
    Reads a single rank binary file and combines all meshblocks into a unified grid covering the full extent of the rank's data.
    Returns an athdf-like dictionary.
    """
    filedata = read_binary(filename)
    if raw:
        return filedata

    # Determine the full grid extents
    Nx1 = filedata['Nx1']
    Nx2 = filedata['Nx2']
    Nx3 = filedata['Nx3']
    x1min = filedata['x1min']
    x1max = filedata['x1max']
    x2min = filedata['x2min']
    x2max = filedata['x2max']
    x3min = filedata['x3min']
    x3max = filedata['x3max']
    n_mbs = filedata['n_mbs']
    nx1_mb = filedata['nx1_mb']
    nx2_mb = filedata['nx2_mb']
    nx3_mb = filedata['nx3_mb']

    if dtype is None:
        dtype = np.float32

    # Populate coordinate arrays for the unified grid
    def center_func(xm, xp):
        return 0.5 * (xm + xp)

    x1f = np.linspace(x1min, x1max, Nx1 + 1, dtype=dtype)
    x1v = np.array([center_func(xm, xp) for xm, xp in zip(x1f[:-1], x1f[1:])], dtype=dtype)
    x2f = np.linspace(x2min, x2max, Nx2 + 1, dtype=dtype)
    x2v = np.array([center_func(xm, xp) for xm, xp in zip(x2f[:-1], x2f[1:])], dtype=dtype)
    x3f = np.linspace(x3min, x3max, Nx3 + 1, dtype=dtype)
    x3v = np.array([center_func(xm, xp) for xm, xp in zip(x3f[:-1], x3f[1:])], dtype=dtype)

    # Create list of quantities
    if quantities is None:
        quantities = filedata['var_names']

    # Allocate arrays for the full grid
    data = {}
    for q in quantities:
        data[q] = np.zeros((Nx3, Nx2, Nx1), dtype=dtype)
    if return_levels:
        data['Levels'] = np.zeros((Nx3, Nx2, Nx1), dtype=np.int32)

    # Insert each meshblock's data into the unified grid
    for mb in range(n_mbs):
        mb_index = filedata['mb_index'][mb]
        is_, ie, js, je, ks, ke = mb_index
        # Clamp indices to grid bounds
        is_, ie = max(is_, 0), min(ie, Nx1-1)
        js, je = max(js, 0), min(je, Nx2-1)
        ks, ke = max(ks, 0), min(ke, Nx3-1)
        # Compute slices
        i_slice = slice(is_, ie+1)
        j_slice = slice(js, je+1)
        k_slice = slice(ks, ke+1)
        # Insert data for each variable
        for q in quantities:
            block_data = filedata['mb_data'][q][mb]
            data[q][k_slice, j_slice, i_slice] = block_data[:ke-ks+1, :je-js+1, :ie-is_+1]
        if return_levels:
            level = filedata['mb_logical'][mb][3]
            data['Levels'][k_slice, j_slice, i_slice] = level

    # Add coordinate arrays
    data['x1f'] = x1f
    data['x1v'] = x1v
    data['x2f'] = x2f
    data['x2v'] = x2v
    data['x3f'] = x3f
    data['x3v'] = x3v
    data['Time'] = filedata['time']
    data['NumCycles'] = filedata['cycle']
    data['MaxLevel'] = np.max(filedata['mb_logical'][:, 3])
    return data



# ========================================================================================

def athinput(filename: str) -> Dict[str, Dict[str, Any]]:
    """Read athinput file and returns a dictionary of dictionaries."""

    # Read data
    with open(filename, 'r') as athinput:
        # remove comments, extra whitespace, and empty lines
        lines = filter(None, [i.split('#')[0].strip() for i in athinput.readlines()])
    data = {}
    # split into blocks, first element will be empty
    blocks = ('\n'.join(lines)).split('<')[1:]

    # Function for interpreting strings numerically
    def typecast(x):
        if '_' in x:
            return x
        try:
            return int(x)
        except ValueError:
            pass
        try:
            return float(x)
        except ValueError:
            pass
        try:
            return complex(x)
        except ValueError:
            pass
        return x

    # Function for parsing assignment based on first '='
    def parse_line(line):
        out = [i.strip() for i in line.split('=')]
        out[1] = '='.join(out[1:])
        out[1] = typecast(out[1])
        return out[:2]

    # Assign values into dictionaries
    for block in blocks:
        info = list(filter(None, block.split('\n')))
        key = info.pop(0)[:-1]  # last character is '>'
        data[key] = dict(map(parse_line, info))
    return data

# --------------------------------------------------------------------------------------
# Shared low-level binary reader --------------------------------------------------------
# --------------------------------------------------------------------------------------

def _read_binary_impl(filename: str, *, coarsened: bool = False) -> Dict[str, Any]:
    """Parse a single Athena++ ``*.bin`` dump.

    This consolidates the duplicated logic previously found in
    ``read_binary`` and ``read_coarsened_binary``.  Both public wrappers now
    delegate to this function so that format tweaks touch just one place.
    """

    # Load file into an in-memory buffer for fast seeks
    with open(filename, "rb") as fh:
        raw_bytes = fh.read()

    fp = io.BytesIO(raw_bytes)
    filesize = len(raw_bytes)

    # ----------------------------- Header tokens -----------------------------
    tokens = fp.readline().split()
    if not tokens or tokens[0] != CODE_HEADER_EXPECTED:
        raise TypeError("Not an Athena++ binary dump (missing magic header)")

    version = tokens[-1].split(b"=")[-1]
    if version != SUPPORTED_VERSION:
        raise TypeError(
            f"Unsupported Athena++ binary version {version.decode()} (expected {SUPPORTED_VERSION.decode()})"
        )

    # Parameter header lines
    pheader_lines = int(fp.readline().split(b"=")[-1])
    pheader: Dict[str, str] = {}
    for _ in range(pheader_lines - 1):
        k, v = [s.strip() for s in fp.readline().decode().split("=")]
        pheader[k] = v

    # Scalar meta-data
    time = float(pheader["time"])
    cycle = int(pheader["cycle"])
    locsizebytes = int(pheader["size of location"])
    varsizebytes = int(pheader["size of variable"])
    coarsen_factor = int(pheader.get("coarsening factor", "1"))

    if coarsened and coarsen_factor == 1:
        raise ValueError("Reader asked for a coarsened dump but header lacks coarsening factor")

    # Variable list
    nvars = int(fp.readline().split(b"=")[-1])
    var_names: List[str] = [tok.decode() for tok in fp.readline().split()[1:]]

    # Simulation header (ASCII)
    ascii_header_size = int(fp.readline().split(b"=")[-1])
    header_bytes = fp.read(ascii_header_size)
    header_lines = [ln.decode().split("#")[0].strip() for ln in header_bytes.split(b"\n") if ln]

    # Validate float sizes
    if locsizebytes not in (4, 8) or varsizebytes not in (4, 8):
        raise ValueError("Unsupported float byte sizes in header")

    dtype_loc = np.float64 if locsizebytes == 8 else np.float32
    dtype_var = np.float64 if varsizebytes == 8 else np.float32

    # Mesh sizes
    Nx1 = int(_get_from_header(header_lines, "<mesh>", "nx1"))
    Nx2 = int(_get_from_header(header_lines, "<mesh>", "nx2"))
    Nx3 = int(_get_from_header(header_lines, "<mesh>", "nx3"))
    nx1 = int(_get_from_header(header_lines, "<meshblock>", "nx1"))
    nx2 = int(_get_from_header(header_lines, "<meshblock>", "nx2"))
    nx3 = int(_get_from_header(header_lines, "<meshblock>", "nx3"))
    nghost = int(_get_from_header(header_lines, "<mesh>", "nghost"))

    x1min = float(_get_from_header(header_lines, "<mesh>", "x1min"))
    x1max = float(_get_from_header(header_lines, "<mesh>", "x1max"))
    x2min = float(_get_from_header(header_lines, "<mesh>", "x2min"))
    x2max = float(_get_from_header(header_lines, "<mesh>", "x2max"))
    x3min = float(_get_from_header(header_lines, "<mesh>", "x3min"))
    x3max = float(_get_from_header(header_lines, "<mesh>", "x3max"))

    # ----------------------------- Mesh blocks -----------------------------
    mb_index, mb_logical, mb_geometry = [], [], []
    mb_data: Dict[str, List[np.ndarray]] = {v: [] for v in var_names}

    while fp.tell() < filesize:
        mb_index.append(np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64) - nghost)
        nx1_out = (mb_index[-1][1] - mb_index[-1][0]) + 1
        nx2_out = (mb_index[-1][3] - mb_index[-1][2]) + 1
        nx3_out = (mb_index[-1][5] - mb_index[-1][4]) + 1

        mb_logical.append(np.frombuffer(fp.read(16), dtype=np.int32))
        mb_geometry.append(np.frombuffer(fp.read(6 * locsizebytes), dtype=dtype_loc))

        # Cell data block
        cells = nx1_out * nx2_out * nx3_out * nvars
        bytes_needed = cells * np.dtype(dtype_var).itemsize
        block_bytes = fp.read(bytes_needed)
        if len(block_bytes) != bytes_needed:
            raise IOError("Unexpected EOF while reading mesh-block data; binary dump may be corrupted")

        block = np.frombuffer(block_bytes, dtype=dtype_var, count=cells).reshape(
            nvars, nx3_out, nx2_out, nx1_out
        )
        for vi, v in enumerate(var_names):
            mb_data[v].append(block[vi])

    # Coarsening factor handling
    factor = coarsen_factor if coarsened else 1

    result: Dict[str, Any] = {
        "header": header_lines,
        "time": time,
        "cycle": cycle,
        "var_names": var_names,
        "nvars": nvars,
        "Nx1": Nx1 // factor,
        "Nx2": Nx2 // factor,
        "Nx3": Nx3 // factor,
        "x1min": x1min,
        "x1max": x1max,
        "x2min": x2min,
        "x2max": x2max,
        "x3min": x3min,
        "x3max": x3max,
        "n_mbs": len(mb_index),
        "nx1_mb": nx1 // factor,
        "nx2_mb": nx2 // factor,
        "nx3_mb": nx3 // factor,
        "nx1_out_mb": (mb_index[0][1] - mb_index[0][0]) + 1,
        "nx2_out_mb": (mb_index[0][3] - mb_index[0][2]) + 1,
        "nx3_out_mb": (mb_index[0][5] - mb_index[0][4]) + 1,
        "mb_index": np.array(mb_index),
        "mb_logical": np.array(mb_logical),
        "mb_geometry": np.array(mb_geometry),
        "mb_data": mb_data,
    }
    if coarsened:
        result["number_of_moments"] = int(pheader["number of moments"])
    return result

__all__ = [
    "read_binary",
    "read_coarsened_binary",
    "read_all_ranks_binary",
    "read_all_ranks_coarsened_binary",
    "read_binary_as_athdf",
    "read_all_ranks_binary_as_athdf",
    "read_all_ranks_coarsened_binary_as_athdf",
    "read_single_rank_binary_as_athdf",
    "read_coarsened_binary_as_athdf",
    "write_xdmf_for",
    "convert_file",
    "read_rank_binary_as_athdf",
    "athinput",
]


