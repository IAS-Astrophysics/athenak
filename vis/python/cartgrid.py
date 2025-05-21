"""
Reads CartesianGrid binary format
"""

import numpy as np
from struct import Struct


class CartesianGridData:
    """Class representing a single CartesianGrid dump

    Members are:
    * cycle:int           simulation cycle
    * time:float          simulation time
    * center[3]:float     box center
    * extent[3]:float     box extent
    * numpoints[3]:int    number of grid points in each direction
    * is_cheb:bool        Chebyshev grid
    * variables           dictionary with all grid functions
    """

    def __init__(self, fname):
        mdata = Struct("i7f3i?2i")
        with open(fname, "rb") as f:
            # Parse metadata
            blob = f.read(mdata.size)
            ncycle, time, cx, cy, cz, ex, ey, ez, nx, ny, nz, cheb, nout, nstr = (
                mdata.unpack(blob)
            )
            self.cycle = ncycle
            self.time = time
            self.center = (cx, cy, cz)
            self.extent = (ex, ey, ez)
            self.numpoints = (nx, ny, nz)
            self.is_cheb = cheb
            self.variables = {}

            # Read variable names
            blob = f.read(nstr).decode("ascii")
            names = blob.split()
            if len(names) != nout:
                raise IOError(f"Could not read {fname}")

            # Read variables
            for n in names:
                self.variables[n] = (
                    np.fromfile(f, dtype=np.float32, count=np.prod(self.numpoints))
                    .reshape(self.numpoints)
                    .transpose()
                )

    def coords(self, d=None):
        """Returns the coordinates"""
        if d is None:
            return self.coords(0), self.coords(1), self.coords(2)
        if self.is_cheb:
            from math import pi

            return self.center[d] + self.extent[d] * np.cos(
                np.linspace(0, pi, self.numpoints[d])
            )
        else:
            return self.center[d] + self.extent[d] * np.linspace(
                -1, 1, self.numpoints[d]
            )

    def meshgrid(self):
        """Returns a mesh grid with all the coordinates"""
        x, y, z = self.coords()
        return np.meshgrid(x, y, z, indexing="ij")
