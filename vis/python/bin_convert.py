"""
Functions to:
  (1) convert bin --> Python dictionary
  (2) convert Python dictionary --> athdf(xdmf) files

This module contains a collection of helper functions for readng and
writing athena file data formats. More information is provided in the
function docstrings.

----

In order to translate a binary file into athdf and corresponding xdmf
files, you could do the following:

  import bin_convert
  import os

  binary_fname = "path/to/file.bin"
  athdf_fname = binary_fname.replace(".bin", ".athdf")
  xdmf_fname = athdf_fname + ".xdmf"
  filedata = bin_convert.read_binary(binary_fname)
  bin_convert.write_athdf(athdf_fname, filedata)
  bin_convert.write_xdmf_for(xdmf_fname, os.path.basename(athdf_fname), filedata)

Notice that write_xdmf_for(...) function expects the relative path to
the athdf file from the xdmf, so please be aware of this requirement.

----

The read_*(...) functions return a filedata dictionary-like object with

    filedata['header'] = array of strings
        ordered array of header, including all the header information
    filedata['time'] = float
        time from input file
    filedata['cycle'] = int
        cycle from input file
    filedata['var_names'] = array of strings
        ordered array of variable names, like ['dens', 'eint', ...]
    filedata['n_mbs'] = int
        total number of meshblocks in the file
    filedata['nx1_mb'] = int
        number of cells in x1 direction in MeshBlock
    filedata['nx2_mb'] = int
        number of cells in x2 direction in MeshBlock
    filedata['nx3_mb'] = int
        number of cells in x3 direction in MeshBlock
    filedata['nx1_out_mb'] = int
        number of output cells in x1 direction in MeshBlock (useful for slicing)
    filedata['nx2_out_mb'] = int
        number of output cells in x2 direction in MeshBlock (useful for slicing)
    filedata['nx3_out_mb'] = int
        number of output cells in x3 direction in MeshBlock (useful for slicing)
    filedata['Nx1'] = int
        total number of cell in x1 direction in root grid
    filedata['Nx2'] = int
        total number of cell in x2 direction in root grid
    filedata['Nx3'] = int
        total number of cell in x3 direction in root grid
    filedata['x1min'] = float
        coordinate minimum of root grid in x1 direction
    filedata['x1max'] = float
        coordinate maximum of root grid in x1 direction
    filedata['x2min'] = float
        coordinate minimum of root grid in x2 direction
    filedata['x2max'] = float
        coordinate maximum of root grid in x2 direction
    filedata['x3min'] = float
        coordinate minimum of root grid in x3 direction
    filedata['x3max'] = float
        coordinate maximum of root grid in x3 direction
    filedata['nvars'] = int
        number of output variables (including magnetic field if it exists)
    filedata['mb_index'] = array with shape [n_mbs, 6]
        is,ie,js,je,ks,ke range for output MeshBlock indexing (useful for slicing)
    filedata['mb_logical'] = array with shape [n_mbs, 4]
        i,j,k,level coordinates for each MeshBlock
    filedata['mb_geometry'] = array with shape [n_mbs, 6]
        x1i,x2i,x3i,dx1,dx2,dx3 including cell-centered location of left-most
        cell and offsets between cells
    filedata['mb_data'] = dict of arrays with shape [n_mbs, nx3, nx2, nx1]
        {'var1':var1_array, 'var2':var2_array, ...} dictionary of fluid data arrays
        for each variable in var_names
"""

import numpy as np
import struct
import h5py
import os
import glob

def read_binary(filename):
    """
    Reads a bin file from filename to dictionary.

    Originally written by Lev Arzamasskiy (leva@ias.edu) on 11/15/2021
    Updated to support mesh refinement by George Wong (gnwong@ias.edu) on 01/27/2022
    Made faster by Drummond Fielding on 09/09/2024

    args:
      filename - string
          filename of bin file to read

    returns:
      filedata - dict
          dictionary of fluid file data
    """

    filedata = {}

    # load file and get size
    fp = open(filename, "rb")
    fp.seek(0, 2)
    filesize = fp.tell()
    fp.seek(0, 0)

    # load header information and validate file format
    code_header = fp.readline().split()
    if len(code_header) < 1:
        raise TypeError("unknown file format")
    if code_header[0] != b"Athena":
        raise TypeError(
            f"bad file format \"{code_header[0].decode('utf-8')}\" "
            + '(should be "Athena")'
        )
    version = code_header[-1].split(b"=")[-1]
    if version != b"1.1":
        raise TypeError(f"unsupported file format version {version.decode('utf-8')}")

    pheader_count = int(fp.readline().split(b"=")[-1])
    pheader = {}
    for _ in range(pheader_count - 1):
        key, val = [x.strip() for x in fp.readline().decode("utf-8").split("=")]
        pheader[key] = val
    time = float(pheader["time"])
    cycle = int(pheader["cycle"])
    locsizebytes = int(pheader["size of location"])
    varsizebytes = int(pheader["size of variable"])

    nvars = int(fp.readline().split(b"=")[-1])
    var_list = [v.decode("utf-8") for v in fp.readline().split()[1:]]
    header_size = int(fp.readline().split(b"=")[-1])
    header = [
        line.decode("utf-8").split("#")[0].strip()
        for line in fp.read(header_size).split(b"\n")
    ]
    header = [line for line in header if len(line) > 0]

    if locsizebytes not in [4, 8]:
        raise ValueError(f"unsupported location size (in bytes) {locsizebytes}")
    if varsizebytes not in [4, 8]:
        raise ValueError(f"unsupported variable size (in bytes) {varsizebytes}")

    locfmt = "d" if locsizebytes == 8 else "f"
    varfmt = "d" if varsizebytes == 8 else "f"

    # load grid information from header and validate
    def get_from_header(header, blockname, keyname):
        blockname = blockname.strip()
        keyname = keyname.strip()
        if not blockname.startswith("<"):
            blockname = "<" + blockname
        if blockname[-1] != ">":
            blockname += ">"
        block = "<none>"
        for line in [entry for entry in header]:
            if line.startswith("<"):
                block = line
                continue
            key, value = line.split("=")
            if block == blockname and key.strip() == keyname:
                return value
        raise KeyError(f"no parameter called {blockname}/{keyname}")

    Nx1 = int(get_from_header(header, "<mesh>", "nx1"))
    Nx2 = int(get_from_header(header, "<mesh>", "nx2"))
    Nx3 = int(get_from_header(header, "<mesh>", "nx3"))
    nx1 = int(get_from_header(header, "<meshblock>", "nx1"))
    nx2 = int(get_from_header(header, "<meshblock>", "nx2"))
    nx3 = int(get_from_header(header, "<meshblock>", "nx3"))

    nghost = int(get_from_header(header, "<mesh>", "nghost"))

    x1min = float(get_from_header(header, "<mesh>", "x1min"))
    x1max = float(get_from_header(header, "<mesh>", "x1max"))
    x2min = float(get_from_header(header, "<mesh>", "x2min"))
    x2max = float(get_from_header(header, "<mesh>", "x2max"))
    x3min = float(get_from_header(header, "<mesh>", "x3min"))
    x3max = float(get_from_header(header, "<mesh>", "x3max"))

    # load data from each meshblock
    n_vars = len(var_list)
    mb_count = 0

    mb_index = []
    mb_logical = []
    mb_geometry = []

    mb_data = {}
    for var in var_list:
        mb_data[var] = []

    while fp.tell() < filesize:
        mb_index.append(np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64) - nghost)
        nx1_out = (mb_index[mb_count][1] - mb_index[mb_count][0]) + 1
        nx2_out = (mb_index[mb_count][3] - mb_index[mb_count][2]) + 1
        nx3_out = (mb_index[mb_count][5] - mb_index[mb_count][4]) + 1

        mb_logical.append(np.frombuffer(fp.read(16), dtype=np.int32))
        mb_geometry.append(
            np.frombuffer(fp.read(6 * locsizebytes), dtype=np.float64 if locfmt == 'd' else np.float32)
        )

        data = np.fromfile(fp, dtype=np.float64 if varfmt == 'd' else np.float32, count=nx1_out*nx2_out*nx3_out*n_vars)
        data = data.reshape(nvars, nx3_out, nx2_out, nx1_out)  # Correctly reshape based on actual sizes
        for vari, var in enumerate(var_list):
            mb_data[var].append(data[vari])
        mb_count += 1

    fp.close()

    filedata["header"] = header
    filedata["time"] = time
    filedata["cycle"] = cycle
    filedata["var_names"] = var_list

    filedata["Nx1"] = Nx1
    filedata["Nx2"] = Nx2
    filedata["Nx3"] = Nx3
    filedata["nvars"] = nvars

    filedata["x1min"] = x1min
    filedata["x1max"] = x1max
    filedata["x2min"] = x2min
    filedata["x2max"] = x2max
    filedata["x3min"] = x3min
    filedata["x3max"] = x3max

    filedata["n_mbs"] = mb_count
    filedata["nx1_mb"] = nx1
    filedata["nx2_mb"] = nx2
    filedata["nx3_mb"] = nx3
    filedata["nx1_out_mb"] = (mb_index[0][1] - mb_index[0][0]) + 1
    filedata["nx2_out_mb"] = (mb_index[0][3] - mb_index[0][2]) + 1
    filedata["nx3_out_mb"] = (mb_index[0][5] - mb_index[0][4]) + 1

    filedata["mb_index"] = np.array(mb_index)
    filedata["mb_logical"] = np.array(mb_logical)
    filedata["mb_geometry"] = np.array(mb_geometry)
    filedata["mb_data"] = mb_data

    return filedata

def read_coarsened_binary(filename):
    """
    Reads a coarsened bin file from filename to dictionary.

    Originally written by Lev Arzamasskiy (leva@ias.edu) on 11/15/2021
    Updated to support mesh refinement by George Wong (gnwong@ias.edu) on 01/27/2022
    Updated to support coarsened outputs and for speed by Drummond Fielding on 09/09/2024

    args:
      filename - string
          filename of bin file to read

    returns:
      filedata - dict
          dictionary of fluid file data
    """

    filedata = {}

    # load file and get size
    fp = open(filename, "rb")
    fp.seek(0, 2)
    filesize = fp.tell()
    fp.seek(0, 0)

    # load header information and validate file format
    code_header = fp.readline().split()
    if len(code_header) < 1:
        raise TypeError("unknown file format")
    if code_header[0] != b"Athena":
        raise TypeError(
            f"bad file format \"{code_header[0].decode('utf-8')}\" "
            + '(should be "Athena")'
        )
    version = code_header[-1].split(b"=")[-1]
    if version != b"1.1":
        raise TypeError(f"unsupported file format version {version.decode('utf-8')}")

    pheader_count = int(fp.readline().split(b"=")[-1])
    pheader = {}
    for _ in range(pheader_count - 1):
        key, val = [x.strip() for x in fp.readline().decode("utf-8").split("=")]
        pheader[key] = val
    time = float(pheader["time"])
    cycle = int(pheader["cycle"])
    locsizebytes = int(pheader["size of location"])
    varsizebytes = int(pheader["size of variable"])
    coarsen_factor = int(pheader["coarsening factor"])

    nvars = int(fp.readline().split(b"=")[-1])
    var_list = [v.decode("utf-8") for v in fp.readline().split()[1:]]
    header_size = int(fp.readline().split(b"=")[-1])
    header = [
        line.decode("utf-8").split("#")[0].strip()
        for line in fp.read(header_size).split(b"\n")
    ]
    header = [line for line in header if len(line) > 0]

    if locsizebytes not in [4, 8]:
        raise ValueError(f"unsupported location size (in bytes) {locsizebytes}")
    if varsizebytes not in [4, 8]:
        raise ValueError(f"unsupported variable size (in bytes) {varsizebytes}")

    locfmt = "d" if locsizebytes == 8 else "f"
    varfmt = "d" if varsizebytes == 8 else "f"

    # load grid information from header and validate
    def get_from_header(header, blockname, keyname):
        blockname = blockname.strip()
        keyname = keyname.strip()
        if not blockname.startswith("<"):
            blockname = "<" + blockname
        if blockname[-1] != ">":
            blockname += ">"
        block = "<none>"
        for line in [entry for entry in header]:
            if line.startswith("<"):
                block = line
                continue
            key, value = line.split("=")
            if block == blockname and key.strip() == keyname:
                return value
        raise KeyError(f"no parameter called {blockname}/{keyname}")

    Nx1 = int(get_from_header(header, "<mesh>", "nx1"))
    Nx2 = int(get_from_header(header, "<mesh>", "nx2"))
    Nx3 = int(get_from_header(header, "<mesh>", "nx3"))
    nx1 = int(get_from_header(header, "<meshblock>", "nx1"))
    nx2 = int(get_from_header(header, "<meshblock>", "nx2"))
    nx3 = int(get_from_header(header, "<meshblock>", "nx3"))

    nghost = int(get_from_header(header, "<mesh>", "nghost"))

    x1min = float(get_from_header(header, "<mesh>", "x1min"))
    x1max = float(get_from_header(header, "<mesh>", "x1max"))
    x2min = float(get_from_header(header, "<mesh>", "x2min"))
    x2max = float(get_from_header(header, "<mesh>", "x2max"))
    x3min = float(get_from_header(header, "<mesh>", "x3min"))
    x3max = float(get_from_header(header, "<mesh>", "x3max"))

    # load data from each meshblock
    n_vars = len(var_list)
    mb_count = 0

    mb_index = []
    mb_logical = []
    mb_geometry = []

    mb_data = {}
    for var in var_list:
        mb_data[var] = []

    while fp.tell() < filesize:
        mb_index.append(np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64) - nghost)
        nx1_out = (mb_index[mb_count][1] - mb_index[mb_count][0]) + 1
        nx2_out = (mb_index[mb_count][3] - mb_index[mb_count][2]) + 1
        nx3_out = (mb_index[mb_count][5] - mb_index[mb_count][4]) + 1

        mb_logical.append(np.frombuffer(fp.read(16), dtype=np.int32))
        mb_geometry.append(
            np.frombuffer(fp.read(6 * locsizebytes), dtype=np.float64 if locfmt == 'd' else np.float32)
        )

        data = np.fromfile(fp, dtype=np.float64 if varfmt == 'd' else np.float32, count=nx1_out*nx2_out*nx3_out*n_vars)
        data = data.reshape(nvars, nx3_out, nx2_out, nx1_out)
        for vari, var in enumerate(var_list):
            mb_data[var].append(data[vari])
        mb_count += 1

    fp.close()

    filedata["header"] = header
    filedata["time"] = time
    filedata["cycle"] = cycle
    filedata["var_names"] = var_list

    filedata["Nx1"] = Nx1 // coarsen_factor
    filedata["Nx2"] = Nx2 // coarsen_factor
    filedata["Nx3"] = Nx3 // coarsen_factor
    filedata["nvars"] = nvars
    filedata["number_of_moments"] = int(pheader["number of moments"])

    filedata["x1min"] = x1min
    filedata["x1max"] = x1max
    filedata["x2min"] = x2min
    filedata["x2max"] = x2max
    filedata["x3min"] = x3min
    filedata["x3max"] = x3max

    filedata["n_mbs"] = mb_count
    filedata["nx1_mb"] = nx1 // coarsen_factor
    filedata["nx2_mb"] = nx2 // coarsen_factor
    filedata["nx3_mb"] = nx3 // coarsen_factor
    filedata["nx1_out_mb"] = (mb_index[0][1] - mb_index[0][0]) + 1
    filedata["nx2_out_mb"] = (mb_index[0][3] - mb_index[0][2]) + 1
    filedata["nx3_out_mb"] = (mb_index[0][5] - mb_index[0][4]) + 1

    filedata["mb_index"] = np.array(mb_index)
    filedata["mb_logical"] = np.array(mb_logical)
    filedata["mb_geometry"] = np.array(mb_geometry)
    filedata["mb_data"] = mb_data

    return filedata

def read_all_ranks_binary(rank0_filename):
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
        print("Files are not the same size! you are probably trying to read a slice written with single_file_per_rank=True")
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

def read_all_ranks_coarsened_binary(rank0_filename):
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

def read_single_rank_binary_as_athdf(filename, raw=False, data=None, quantities=None, dtype=None,
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
        xmin = filedata['mb_geometry'][0, (d-1)*2]
        xmax = filedata['mb_geometry'][0, (d-1)*2 + 1]

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
        block_data = filedata['mb_data'][q][0]  # Single rank, so only one block
        data[q] = block_data[k_min:k_max, j_min:j_max, i_min:i_max]

    if return_levels:
        data['Levels'].fill(filedata['mb_logical'][0, 3])  # Level of the single block

    # Add metadata
    data['Time'] = filedata['time']
    data['NumCycles'] = filedata['cycle']
    data['MaxLevel'] = filedata['mb_logical'][0, 3]

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

def write_xdmf_for(xdmfname, dumpname, fdata, mode="auto"):
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

    fp = open(xdmfname, "w")

    def write_meshblock(fp, mb, nx1, nx2, nx3, nmb, dumpname, vars_no_b, vars_w_b):
        fp.write(f"""  <Grid Name="MeshBlock{mb}" GridType="Uniform">\n""")
        fp.write("""   <Topology TopologyType="3DRectMesh" """)
        fp.write(f""" NumberOfElements="{nx3+1} {nx2+1} {nx1+1}"/>\n""")
        fp.write("""   <Geometry GeometryType="VXVYVZ">\n""")
        fp.write(
            f"""    <DataItem ItemType="HyperSlab" Dimensions="{nx1+1}">
     <DataItem Dimensions="3 2" NumberType="Int"> {mb} 0 1 1 1 {nx1+1} </DataItem>
     <DataItem Dimensions="{nmb} {nx1+1}" Format="HDF"> {dumpname}:/x1f </DataItem>
    </DataItem>
    <DataItem ItemType="HyperSlab" Dimensions="{nx2+1}">
     <DataItem Dimensions="3 2" NumberType="Int"> {mb} 0 1 1 1 {nx2+1} </DataItem>
     <DataItem Dimensions="{nmb} {nx2+1}" Format="HDF"> {dumpname}:/x2f </DataItem>
    </DataItem>
    <DataItem ItemType="HyperSlab" Dimensions="{nx3+1}">
     <DataItem Dimensions="3 2" NumberType="Int"> {mb} 0 1 1 1 {nx3+1} </DataItem>
     <DataItem Dimensions="{nmb} {nx3+1}" Format="HDF"> {dumpname}:/x3f </DataItem>
    </DataItem>
   </Geometry>\n"""
        )

        nvar_no_b = len(vars_no_b)
        for vi, var_name in enumerate(vars_no_b):
            fp.write(
                f"""   <Attribute Name="{var_name}" Center="Cell">
    <DataItem ItemType="HyperSlab" Dimensions="{nx3} {nx2} {nx1}">
     <DataItem Dimensions="3 5" NumberType="Int">
      {vi} {mb} 0 0 0 1 1 1 1 1 1 1 {nx3} {nx2} {nx1}
     </DataItem>
     <DataItem Dimensions="{nvar_no_b} {nmb} {nx3} {nx2} {nx1}" Format="HDF">
      {dumpname}:/uov
     </DataItem>
    </DataItem>
   </Attribute>\n"""
            )

        nvar_w_b = len(vars_w_b)
        if nvar_w_b > 0:
            for vi, var_name in enumerate(vars_w_b):
                fp.write(
                    f"""   <Attribute Name="{var_name}" Center="Cell">
        <DataItem ItemType="HyperSlab" Dimensions="{nx3} {nx2} {nx1}">
         <DataItem Dimensions="3 5" NumberType="Int">
          {vi} {mb} 0 0 0 1 1 1 1 1 1 1 {nx3} {nx2} {nx1}
         </DataItem>
         <DataItem Dimensions="{nvar_w_b} {nmb} {nx3} {nx2} {nx1}" Format="HDF">
          {dumpname}:/B
         </DataItem>
        </DataItem>
       </Attribute>\n"""
                )

        fp.write("""  </Grid>\n""")

    fp.write(
        """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Information Name="TimeVaryingMetaData" Value="True"/>\n"""
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

    fp.close()

def convert_file(binary_fname):
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

if __name__ == "__main__":
    import sys
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        def tqdm(L):
            for x in L:
                print(x)
                yield x

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} output_file_1.bin [output_file_2.bin [...]]")
        exit(1)

    for binary_fname in tqdm(sys.argv[1:]):
        convert_file(binary_fname)



























# def read_single_rank_binary_as_athdf(filename, raw=False, data=None, quantities=None, dtype=None,
#                                      return_levels=False, x1_min=None, x1_max=None, x2_min=None,
#                                      x2_max=None, x3_min=None, x3_max=None, vol_func=None,
#                                      center_func_1=None, center_func_2=None, center_func_3=None):
#     """
#     Reads a single rank binary file and organizes data similar to athdf format without writing to file.
#     Handles multiple meshblocks by merging them into a unified grid covering the combined region.

#     args:
#       filename - string
#           filename of bin file to read

#       raw - bool (default=False)
#           If True, returns the raw filedata without processing.

#       data - dict (default=None)
#           Dictionary to store the organized data.

#       quantities - list (default=None)
#           List of variables to extract. If None, all variables are extracted.

#       dtype - numpy dtype (default=None)
#           Data type for the output arrays. Defaults to np.float32.

#       return_levels - bool (default=False)
#           If True, includes the hierarchical level information of each cell.

#       x1_min, x1_max, x2_min, x2_max, x3_min, x3_max - float (default=None)
#           Spatial bounds for selecting a subset of the data.

#       vol_func - function (default=None)
#           Function to compute the volume for each cell. Defaults to a simple Cartesian volume.

#       center_func_1, center_func_2, center_func_3 - functions (default=None)
#           Functions to compute the center coordinates for each dimension. Defaults to midpoint.

#     returns:
#       data - dict
#           Organized data in a grid spanning all meshblocks in the rank.
#     """

#     # Step 1: Read binary data for a single rank
#     filedata = read_binary(filename)

#     if raw:
#         return filedata

#     # Prepare dictionary for results
#     if data is None:
#         data = {}
#         new_data = True
#     else:
#         new_data = False

#     # Extract size and spatial information from all meshblocks
#     x1min_all = np.min(filedata['mb_geometry'][:, 0])
#     x1max_all = np.max(filedata['mb_geometry'][:, 0] + filedata['mb_geometry'][:, 3])
#     x2min_all = np.min(filedata['mb_geometry'][:, 1])
#     x2max_all = np.max(filedata['mb_geometry'][:, 1] + filedata['mb_geometry'][:, 4])
#     x3min_all = np.min(filedata['mb_geometry'][:, 2])
#     x3max_all = np.max(filedata['mb_geometry'][:, 2] + filedata['mb_geometry'][:, 5])

#     # Initialize overall grid based on all meshblocks
#     if dtype is None:
#         dtype = np.float32

#     # Populate coordinate arrays for the unified grid
#     if center_func_1 is None:
#         def center_func_1(xm, xp):
#             return 0.5 * (xm + xp)
#     if center_func_2 is None:
#         def center_func_2(xm, xp):
#             return 0.5 * (xm + xp)
#     if center_func_3 is None:
#         def center_func_3(xm, xp):
#             return 0.5 * (xm + xp)

#     center_funcs = [center_func_1, center_func_2, center_func_3]

#     # Define the unified grid boundaries
#     data['x1f'] = np.linspace(x1min_all, x1max_all, filedata['Nx1'] + 1, dtype=dtype)
#     data['x1v'] = np.array([center_func_1(xm, xp) for xm, xp in zip(data['x1f'][:-1], data['x1f'][1:])], dtype=dtype)

#     data['x2f'] = np.linspace(x2min_all, x2max_all, filedata['Nx2'] + 1, dtype=dtype)
#     data['x2v'] = np.array([center_func_2(xm, xp) for xm, xp in zip(data['x2f'][:-1], data['x2f'][1:])], dtype=dtype)

#     data['x3f'] = np.linspace(x3min_all, x3max_all, filedata['Nx3'] + 1, dtype=dtype)
#     data['x3v'] = np.array([center_func_3(xm, xp) for xm, xp in zip(data['x3f'][:-1], data['x3f'][1:])], dtype=dtype)

#     # Initialize data arrays for variables
#     if quantities is None:
#         quantities = filedata['var_names']

#     for q in quantities:
#         data[q] = np.zeros((filedata['Nx3'], filedata['Nx2'], filedata['Nx1']), dtype=dtype)

#     if return_levels:
#         data['Levels'] = np.zeros((filedata['Nx3'], filedata['Nx2'], filedata['Nx1']), dtype=np.int32)

#     # Step 2: Insert data from each meshblock into the unified grid
#     for mb in range(filedata['n_mbs']):
#         # Get meshblock geometry
#         mb_geom = filedata['mb_geometry'][mb]
#         mb_x1i, mb_x2i, mb_x3i = mb_geom[0], mb_geom[1], mb_geom[2]
#         mb_dx1, mb_dx2, mb_dx3 = mb_geom[3], mb_geom[4], mb_geom[5]
#         mb_x1f = np.linspace(mb_x1i, mb_x1i + mb_dx1, filedata['nx1_mb'] + 1, dtype=dtype)
#         mb_x2f = np.linspace(mb_x2i, mb_x2i + mb_dx2, filedata['nx2_mb'] + 1, dtype=dtype)
#         mb_x3f = np.linspace(mb_x3i, mb_x3i + mb_dx3, filedata['nx3_mb'] + 1, dtype=dtype)

#         # Find the indices in the unified grid where this meshblock fits
#         i_start = np.searchsorted(data['x1f'], mb_x1f[0], side='left') - 1
#         j_start = np.searchsorted(data['x2f'], mb_x2f[0], side='left') - 1
#         k_start = np.searchsorted(data['x3f'], mb_x3f[0], side='left') - 1

#         i_end = i_start + filedata['nx1_mb']
#         j_end = j_start + filedata['nx2_mb']
#         k_end = k_start + filedata['nx3_mb']

#         # Ensure indices are within bounds
#         i_start = max(i_start, 0)
#         j_start = max(j_start, 0)
#         k_start = max(k_start, 0)
#         i_end = min(i_end, filedata['Nx1'])
#         j_end = min(j_end, filedata['Nx2'])
#         k_end = min(k_end, filedata['Nx3'])

#         # Insert data for each variable
#         for q in quantities:
#             block_data = filedata['mb_data'][q][mb]
#             data[q][k_start:k_end, j_start:j_end, i_start:i_end] = block_data[:k_end - k_start, :j_end - j_start, :i_end - i_start]

#         # Insert level information if required
#         if return_levels:
#             level = filedata['mb_logical'][mb][3]
#             data['Levels'][k_start:k_end, j_start:j_end, i_start:i_end] = level

#     # Step 3: Apply any spatial selections if specified
#     if any(v is not None for v in [x1_min, x1_max, x2_min, x2_max, x3_min, x3_max]):
#         # Determine the slicing indices based on the provided bounds
#         i_min = np.searchsorted(data['x1f'], x1_min) if x1_min is not None else 0
#         i_max = np.searchsorted(data['x1f'], x1_max) if x1_max is not None else filedata['Nx1']
#         j_min = np.searchsorted(data['x2f'], x2_min) if x2_min is not None else 0
#         j_max = np.searchsorted(data['x2f'], x2_max) if x2_max is not None else filedata['Nx2']
#         k_min = np.searchsorted(data['x3f'], x3_min) if x3_min is not None else 0
#         k_max = np.searchsorted(data['x3f'], x3_max) if x3_max is not None else filedata['Nx3']

#         # Slice the data arrays accordingly
#         for q in quantities:
#             data[q] = data[q][k_min:k_max, j_min:j_max, i_min:i_max]
#         if return_levels:
#             data['Levels'] = data['Levels'][k_min:k_max, j_min:j_max, i_min:i_max]
#         data['x1f'] = data['x1f'][i_min:i_max+1]
#         data['x1v'] = data['x1v'][i_min:i_max]
#         data['x2f'] = data['x2f'][j_min:j_max+1]
#         data['x2v'] = data['x2v'][j_min:j_max]
#         data['x3f'] = data['x3f'][k_min:k_max+1]
#         data['x3v'] = data['x3v'][k_min:k_max]

#     # Step 4: Finalize data by computing volume factors if necessary
#     if vol_func is not None:
#         # Compute volumes for normalization if a volume function is provided
#         for q in quantities:
#             # Assuming vol_func can take meshgrid arrays
#             X1M, X2M, X3M = np.meshgrid(data['x1f'][:-1], data['x2f'][:-1], data['x3f'][:-1], indexing='ij')
#             X1P, X2P, X3P = np.meshgrid(data['x1f'][1:], data['x2f'][1:], data['x3f'][1:], indexing='ij')
#             volumes = vol_func(X1M, X1P, X2M, X2P, X3M, X3P)
#             data[q] /= volumes.T  # Transpose to match the data array's axis order

#     # Add metadata
#     data['Time'] = filedata['time']
#     data['NumCycles'] = filedata['cycle']

#     return data