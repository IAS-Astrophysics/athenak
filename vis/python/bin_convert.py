"""
Functions to convert bin -> athdf(xdmf) with mesh refinement support.

This module contains a collection of helper functions for reading and
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

    filedata['time'] = float
        time from input file
    filedata['cycle'] = int
        cycle from input file
    filedata['var_names'] = array of strings
        ordered array of variable names, like ['dens', 'eint', ...]
    filedata['n_meshblocks'] = int
        total number of meshblocks in the file
    filedata['nx1_meshblock'] = int
        number of cells in x1 direction in meshblock
    filedata['nx2_meshblock'] = int
        number of cells in x2 direction in meshblock
    filedata['nx3_meshblock'] = int
        number of cells in x3 direction in meshblock
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
    filedata['meshblock_logical'] = array with shape [n_meshblock, 4]
        i,j,k,level coordinates for each meshblock
    filedata['meshblock_geometry'] = array with shape [n_meshblock, 6]
        x1i,x2i,x3i,dx1,dx2,dx3 including cell-centered location of left-most
        cell and offsets between cells
    filedata['meshblock_data'] = dict of arrays with shape [n_meshblock, nx3, nx2, nx1]
        {'var1':var1_array, 'var2':var2_array, ...} dictionary of fluid data arrays
        for each variable in var_names
"""

import numpy as np
import struct
import h5py


def read_binary(filename):
    """
    Reads a bin file from filename to dictionary.

    args:
      filename - string
          filename of bin file to read

    returns:
      filedata - dict
          dictionary of fluid file data
    """

    filedata = {}

    # load file and get size
    fp = open(filename, 'rb')
    fp.seek(0, 2)
    filesize = fp.tell()
    fp.seek(0, 0)

    # load header information and validate file format
    code_header = fp.readline().split()
    if len(code_header) < 1:
        raise TypeError("unknown file format")
    if code_header[0] != b"Athena":
        raise TypeError(f"bad file format \"{code_header[0].decode('utf-8')}\" " +
                        "(should be \"Athena\")")
    version = code_header[-1].split(b'=')[-1]
    if version != b"1.1":
        raise TypeError(f"unsupported file format version {version.decode('utf-8')}")

    pheader_count = int(fp.readline().split(b'=')[-1])
    pheader = {}
    for _ in range(pheader_count-1):
        key, val = [x.strip() for x in fp.readline().decode('utf-8').split('=')]
        pheader[key] = val
    time = float(pheader['time'])
    cycle = int(pheader['cycle'])
    locsizebytes = int(pheader['size of location'])
    varsizebytes = int(pheader['size of variable'])

    nvars = int(fp.readline().split(b'=')[-1])
    var_list = [v.decode('utf-8') for v in fp.readline().split()[1:]]
    header_size = int(fp.readline().split(b'=')[-1])
    header = [line.decode('utf-8').split('#')[0].strip()
              for line in fp.read(header_size).split(b'\n')]
    header = [line for line in header if len(line) > 0]

    if locsizebytes not in [4, 8]:
        raise ValueError(f"unsupported location size (in bytes) {locsizebytes}")
    if varsizebytes not in [4, 8]:
        raise ValueError(f"unsupported variable size (in bytes) {varsizebytes}")

    locfmt = 'd' if locsizebytes == 8 else 'f'
    varfmt = 'd' if varsizebytes == 8 else 'f'

    # load grid information from header and validate
    def get_from_header(header, blockname, keyname):
        blockname = blockname.strip()
        keyname = keyname.strip()
        if not blockname.startswith('<'):
            blockname = '<' + blockname
        if blockname[-1] != '>':
            blockname += '>'
        block = '<none>'
        for line in [entry for entry in header]:
            if line.startswith('<'):
                block = line
                continue
            key, value = line.split('=')
            if block == blockname and key.strip() == keyname:
                return value
        raise KeyError(f'no parameter called {blockname}/{keyname}')

    Nx1 = int(get_from_header(header, '<mesh>', 'nx1'))
    Nx2 = int(get_from_header(header, '<mesh>', 'nx2'))
    Nx3 = int(get_from_header(header, '<mesh>', 'nx3'))
    nx1 = int(get_from_header(header, '<meshblock>', 'nx1'))
    nx2 = int(get_from_header(header, '<meshblock>', 'nx2'))
    nx3 = int(get_from_header(header, '<meshblock>', 'nx3'))

    x1min = float(get_from_header(header, '<mesh>', 'x1min'))
    x1max = float(get_from_header(header, '<mesh>', 'x1max'))
    x2min = float(get_from_header(header, '<mesh>', 'x2min'))
    x2max = float(get_from_header(header, '<mesh>', 'x2max'))
    x3min = float(get_from_header(header, '<mesh>', 'x3min'))
    x3max = float(get_from_header(header, '<mesh>', 'x3max'))

    # load data from each meshblock
    n_vars = len(var_list)
    mb_fstr = f"={nx1*nx2*nx3*n_vars}" + varfmt
    mb_varsize = varsizebytes*nx1*nx2*nx3*n_vars
    mb_count = 0

    meshblock_logical = []
    meshblock_geometry = []

    meshblock_data = {}
    for var in var_list:
        meshblock_data[var] = []

    while fp.tell() < filesize:
        mb_count += 1

        meshblock_logical.append(np.array(struct.unpack('@4i', fp.read(16))))
        meshblock_geometry.append(np.array(struct.unpack('=6'+locfmt,
                                  fp.read(6*locsizebytes))))

        data = np.array(struct.unpack(mb_fstr, fp.read(mb_varsize)))
        data = data.reshape(nvars, nx3, nx2, nx1)
        for vari, var in enumerate(var_list):
            meshblock_data[var].append(data[vari])

    fp.close()

    filedata['time'] = time
    filedata['cycle'] = cycle
    filedata['var_names'] = var_list

    filedata['Nx1'] = Nx1
    filedata['Nx2'] = Nx2
    filedata['Nx3'] = Nx3
    filedata['nvars'] = nvars

    filedata['x1min'] = x1min
    filedata['x1max'] = x1max
    filedata['x2min'] = x2min
    filedata['x2max'] = x2max
    filedata['x3min'] = x3min
    filedata['x3max'] = x3max

    filedata['n_meshblocks'] = mb_count
    filedata['nx1_meshblock'] = nx1
    filedata['nx2_meshblock'] = nx2
    filedata['nx3_meshblock'] = nx3

    filedata['meshblock_logical'] = np.array(meshblock_logical)
    filedata['meshblock_geometry'] = np.array(meshblock_geometry)
    filedata['meshblock_data'] = meshblock_data

    return filedata


def write_athdf(filename, fdata, varsize_bytes=4, locsize_bytes=8):
    """
    Writes an athdf (hdf5) file from a loaded python filedata object.

    (unimplemented) should save various file data information as attributes.

    args:
      filename      - string
          filename for output athdf (hdf5) file
      fdata         - dict
          dictionary of fluid file data, e.g., as loaded from read_binary(...)
      varsize_bytes - int (default=4, options=4,8)
          number of bytes to use for output variable data
      locsize_bytes - int (default=8, options=4,8)
          number of bytes to use for output location data
    """

    if varsize_bytes not in [4, 8]:
        raise ValueError(f'varsizebytes must be 4 or 8, not {varsize_bytes}')
    if locsize_bytes not in [4, 8]:
        raise ValueError(f'locsizebytes must be 4 or 8, not {locsize_bytes}')
    locfmt = '<f4' if locsize_bytes == 4 else '<f8'
    varfmt = '<f4' if varsize_bytes == 4 else '<f8'

    nmb = fdata['n_meshblocks']
    nx1 = fdata['nx1_meshblock']
    nx2 = fdata['nx2_meshblock']
    nx3 = fdata['nx3_meshblock']

    # keep variable order but separate out magnetic field
    vars_without_b = [v for v in fdata['var_names'] if 'bcc' not in v]
    vars_only_b = [v for v in fdata['var_names'] if v not in vars_without_b]

    B = np.zeros((3, nmb, nx3, nx2, nx1))
    Levels = np.zeros(nmb)
    LogicalLocations = np.zeros((nmb, 3))
    prim = np.zeros((len(vars_without_b), nmb, nx3, nx2, nx1))
    x1f = np.zeros((nmb, nx1+1))
    x1v = np.zeros((nmb, nx1))
    x2f = np.zeros((nmb, nx2+1))
    x2v = np.zeros((nmb, nx2))
    x3f = np.zeros((nmb, nx3+1))
    x3v = np.zeros((nmb, nx3))

    for ivar, var in enumerate(vars_without_b):
        prim[ivar] = fdata['meshblock_data'][var]
    for ibvar, bvar in enumerate(vars_only_b):
        B[ibvar] = fdata['meshblock_data'][bvar]

    for mb in range(nmb):
        logical = fdata['meshblock_logical'][mb]
        LogicalLocations[mb] = logical[:3]
        Levels[mb] = logical[-1]
        geometry = fdata['meshblock_geometry'][mb]
        x1f[mb] = geometry[0] + (np.arange(nx1+1)-0.5)*geometry[3]
        x1v[mb] = geometry[0] + np.arange(nx1)*geometry[3]
        x2f[mb] = geometry[1] + (np.arange(nx2+1)-0.5)*geometry[4]
        x2v[mb] = geometry[1] + np.arange(nx2)*geometry[4]
        x3f[mb] = geometry[2] + (np.arange(nx3+1)-0.5)*geometry[5]
        x3v[mb] = geometry[2] + np.arange(nx3)*geometry[5]

    # TODO, support attributes
    #    ATTRIBUTE "Coordinates"
    #    ATTRIBUTE "DatasetNames"
    #    ATTRIBUTE "MaxLevel"
    #    ATTRIBUTE "MeshBlockSize"
    #    ATTRIBUTE "NumCycles"
    #    ATTRIBUTE "NumMeshBlocks"
    #    ATTRIBUTE "NumVariables"
    #    ATTRIBUTE "RootGridSize"
    #    ATTRIBUTE "RootGridX1"
    #    ATTRIBUTE "RootGridX2"
    #    ATTRIBUTE "RootGridX3"
    #    ATTRIBUTE "Time"
    #    ATTRIBUTE "VariableNames"

    hfp = h5py.File(filename, 'w')
    hfp.create_dataset('B', data=B, dtype=varfmt)
    hfp.create_dataset('Levels', data=Levels, dtype='>i4')
    hfp.create_dataset('LogicalLocations', data=LogicalLocations, dtype='>i8')
    hfp.create_dataset('prim', data=prim, dtype=varfmt)
    hfp.create_dataset('x1f', data=x1f, dtype=locfmt)
    hfp.create_dataset('x1v', data=x1v, dtype=locfmt)
    hfp.create_dataset('x2f', data=x2f, dtype=locfmt)
    hfp.create_dataset('x2v', data=x2v, dtype=locfmt)
    hfp.create_dataset('x3f', data=x3f, dtype=locfmt)
    hfp.create_dataset('x3v', data=x3v, dtype=locfmt)
    hfp.close()


def write_xdmf_for(xdmfname, dumpname, fdata, mode='auto'):
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

    fp = open(xdmfname, 'w')

    def write_meshblock(fp, mb, nx1, nx2, nx3, nmb, dumpname, vars_no_b, vars_w_b):
        fp.write("""  <Grid Name="MeshBlock{mb}" GridType="Uniform">\n""")
        fp.write("""   <Topology TopologyType="3DRectMesh" """)
        fp.write(f""" NumberOfElements="{nx3+1} {nx2+1} {nx1+1}"/>\n""")
        fp.write("""   <Geometry GeometryType="VXVYVZ">\n""")
        fp.write(f"""    <DataItem ItemType="HyperSlab" Dimensions="{nx1+1}">
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
   </Geometry>\n""")

        nvar_no_b = len(vars_no_b)
        for vi, var_name in enumerate(vars_no_b):
            fp.write(f"""   <Attribute Name="{var_name}" Center="Cell">
    <DataItem ItemType="HyperSlab" Dimensions="{nx3} {nx2} {nx1}">
     <DataItem Dimensions="3 {nvar_no_b}" NumberType="Int">
      {vi} {mb} 0 0 0 1 1 1 1 1 1 1 {nx3} {nx2} {nx1}
     </DataItem>
     <DataItem Dimensions="{nvar_no_b} {nmb} {nx3} {nx2} {nx1}" Format="HDF">
      {dumpname}:/prim
     </DataItem>
    </DataItem>
   </Attribute>\n""")

        nvar_w_b = len(vars_w_b)
        if nvar_w_b > 0:
            for vi, var_name in enumerate(vars_w_b):
                fp.write(f"""   <Attribute Name="{var_name}" Center="Cell">
        <DataItem ItemType="HyperSlab" Dimensions="{nx3} {nx2} {nx1}">
         <DataItem Dimensions="3 {nvar_no_b}" NumberType="Int">
          {vi} {mb} 0 0 0 1 1 1 1 1 1 1 {nx3} {nx2} {nx1}
         </DataItem>
         <DataItem Dimensions="{nvar_no_b} {nmb} {nx3} {nx2} {nx1}" Format="HDF">
          {dumpname}:/B
         </DataItem>
        </DataItem>
       </Attribute>\n""")

        fp.write("""  </Grid>\n""")

    fp.write("""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Information Name="TimeVaryingMetaData" Value="True"/>\n""")
    fp.write("""<Domain>\n""")
    fp.write("""<Grid Name="Mesh" GridType="Collection">\n""")
    fp.write(f""" <Time Value="{fdata['time']}"/>\n""")

    vars_without_b = set([v for v in fdata['var_names'] if 'bcc' not in v])
    vars_only_b = sorted(set(fdata['var_names']) - vars_without_b)

    nx1 = fdata['nx1_meshblock']
    nx2 = fdata['nx2_meshblock']
    nx3 = fdata['nx3_meshblock']
    nmb = fdata['n_meshblocks']

    for mb in range(nmb):
        write_meshblock(fp, mb, nx1, nx2, nx3, nmb, dumpname, vars_without_b, vars_only_b)

    fp.write("""</Grid>\n""")
    fp.write("""</Domain>\n""")
    fp.write("""</Xdmf>\n""")

    fp.close()
