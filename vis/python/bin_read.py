"""
Function to read binary (.bin) grid outputs from AthenaK
Originally written by Lev Arzamasskiy (leva@ias.edu) on 11/15/2021

Returns dictionary with cell-centered quantities, grid coordinates and dimensions,
output time and cycle

Current limitations:
- only cartesian grids are supported
- no mesh refinement is supported
- native byte order is assumed, which is big-endian or little-endian depending
  on the system this script is run on, which might be different from endian of
  the machine that produced the output
- most simulation parameters are not returned despite being in .bin file
"""

import numpy as np
import struct

def read_binary(filename):
  f = open(filename,'rb')
  line = f.readline() # line 1 -- code name and current time
  if (line.split()[0] != b"AthenaK"):
    print("Error: unexpected file format")
    return

  time = float((line.split()[4]).split(b'=')[1])
  line = f.readline() # line 2 -- current cycle
  cycle = int(line.split(b"=")[1])
  line = f.readline() # line 3 -- number of variables
  nvars = int(line.split(b"=")[1])
  line = f.readline() # line 4 -- list of variables
  var_list = line.split()[1:nvars+1]
  line = f.readline() # line 5 -- header offset
  header_size = int(line.split(b"=")[1])
  header = f.read(header_size)
  header_lines = header.split(b"\n")

  Nx1 = -1
  Nx2 = -1
  Nx3 = -1
  nx1 = -1
  nx2 = -1
  nx3 = -1
  flag_mesh = False
  flag_meshblock = False
  for i in range(len(header_lines)):
    line = header_lines[i]
    if (len(line.split(b"<")) > 1 and len(line.split(b">")) > 1): # new block in input file
      flag_mesh = False
      flag_meshblock = False
    if (line == b"<mesh>"):
      flag_mesh = True
    if (line == b"<meshblock>"):
      flag_meshblock = True    
        
    if (flag_mesh and line.split()[0] == b'nx1'):
      Nx1 = int(line.split()[2])
    if (flag_mesh and line.split()[0] == b'nx2'):
      Nx2 = int(line.split()[2])
    if (flag_mesh and line.split()[0] == b'nx3'):
      Nx3 = int(line.split()[2])
    if (flag_meshblock and line.split()[0] == b'nx1'):
      nx1 = int(line.split()[2])
    if (flag_meshblock and line.split()[0] == b'nx2'):
      nx2 = int(line.split()[2])
    if (flag_meshblock and line.split()[0] == b'nx3'):
      nx3 = int(line.split()[2])
        
    if (flag_mesh and line.split()[0] == b'x1min'):
      x1min = float(line.split()[2])
    if (flag_mesh and line.split()[0] == b'x2min'):
      x2min = float(line.split()[2])
    if (flag_mesh and line.split()[0] == b'x3min'):
      x3min = float(line.split()[2])
    if (flag_mesh and line.split()[0] == b'x1max'):
      x1max = float(line.split()[2])
    if (flag_mesh and line.split()[0] == b'x2max'):
      x2max = float(line.split()[2])
    if (flag_mesh and line.split()[0] == b'x3max'):
      x3max = float(line.split()[2])
        
  if (Nx1<1 or Nx2<1 or Nx3<1 or nx1<1 or nx2<1 or nx3<1):
    print("Error in read_binary(): unexpected grid dimensions",Nx1,Nx2,Nx3,nx1,nx2,nx3)
    return
    
  x1fc = np.linspace(x1min,x1max,Nx1+1)
  x2fc = np.linspace(x2min,x2max,Nx2+1)
  x3fc = np.linspace(x3min,x3max,Nx3+1)
  x1cc = 0.5*(x1fc[1:]+x1fc[:-1])
  x2cc = 0.5*(x2fc[1:]+x2fc[:-1])
  x3cc = 0.5*(x3fc[1:]+x3fc[:-1])

  # final result = dictionary of arrays
  result = {}
  # add an entry for each variable
  for nv in range(nvars):
    result[var_list[nv]] = np.zeros((Nx3,Nx2,Nx1))
  result['x1cc'] = x1cc
  result['x2cc'] = x2cc
  result['x3cc'] = x3cc
  result['x1fc'] = x1fc
  result['x2fc'] = x2fc
  result['x3fc'] = x3fc
  result['time'] = time
  result['cycle'] = cycle
  result['Nx1'] = Nx1
  result['Nx2'] = Nx2
  result['Nx3'] = Nx3
  result['x1min'] = x1min
  result['x2min'] = x2min
  result['x3min'] = x3min
  result['x1max'] = x1max
  result['x2max'] = x2max
  result['x3max'] = x3max

  # starting indices for each logical location
  islist = np.arange(0,Nx1,nx1)
  jslist = np.arange(0,Nx2,nx2)
  kslist = np.arange(0,Nx3,nx3)

  nbtotal = int(Nx1/nx1)*int(Nx2/nx2)*int(Nx3/nx3)

  # loop over all meshblocks and read all variables
  for nb in range(nbtotal):
    il1 = struct.unpack("@i",f.read(4))[0]
    il2 = struct.unpack("@i",f.read(4))[0]
    il3 = struct.unpack("@i",f.read(4))[0]
    iis = islist[il1]
    iie = iis + nx1
    ijs = jslist[il2]
    ije = ijs + nx2
    iks = kslist[il3]
    ike = iks + nx3
    fmt = "@%df"%(nx1*nx2*nx3)
    for nv in range(nvars):
      tmp = result[var_list[nv]]
      data = np.array(struct.unpack(fmt,f.read(4*nx1*nx2*nx3)))
      data = data.reshape(nx3,nx2,nx1)
      tmp[iks:ike,ijs:ije,iis:iie] = data
       
  # close the file
  f.close()
