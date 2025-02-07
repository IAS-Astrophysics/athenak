#!/usr/bin/env python3
"""
This script loads the AthenaK horizon dump (a binary file produced by the
HorizonDump module) and converts it to an HDF5 file. It then runs an external
apparent-horizon finder (here using the AHFpy.AHF module) on that data.

The binary dump is assumed to contain:
  - a 4-byte int: common_horizon flag,
  - an 8-byte double: simulation time,
  - a flattened array of 16 × (nx^3) double‐precision values in the following order:
      0:  α         (lapse)
      1–3: βₓ, β_y, β_z   (shift components)
      4–9: gₓₓ, gₓ_y, gₓ_z, g_y_y, g_y_z, g_z_z (3-metric)
      10–15: Kₓₓ, Kₓ_y, Kₓ_z, K_y_y, K_y_z, K_z_z (extrinsic curvature)
      
The script then writes out an HDF5 file with groups “coord”, “metric” and “curvature” and
calls the horizon finder.
"""

import numpy as np
import argparse
import os, sys
import tempfile
import h5py

#--------------------------------------------------------------------------
# 1. Function to load the AthenaK horizon dump binary file.
#--------------------------------------------------------------------------
def load_horizon_dump(filename, nx=10, extent=2.0, center=(0.0, 0.0, 0.0)):
    """
    Reads the binary file and returns a dictionary containing:
      'time', 'coord_x', 'coord_y', 'coord_z', 'common_horizon',
      and the 16 variables (keys: 'alpha', 'betax', 'betay', 'betaz',
      'gxx','gxy','gxz','gyy','gyz','gzz','Kxx','Kxy','Kxz','Kyy','Kyz','Kzz').
    """
    with open(filename, "rb") as f:
        common_horizon = np.frombuffer(f.read(4), dtype=np.int32)[0]
        time_val = np.frombuffer(f.read(8), dtype=np.float64)[0]
        total_vals = 16 * (nx ** 3)
        data_flat = np.frombuffer(f.read(total_vals * 8), dtype=np.float64)
    
    if data_flat.size != total_vals:
        raise ValueError(f"Expected {total_vals} values, got {data_flat.size}")
    
    # Reshape to a (16, nx, nx, nx) array.
    data_array = data_flat.reshape((16, nx, nx, nx))
    
    # Create coordinate arrays from (center - extent) to (center + extent)
    x = np.linspace(center[0] - extent, center[0] + extent, nx)
    y = np.linspace(center[1] - extent, center[1] + extent, nx)
    z = np.linspace(center[2] - extent, center[2] + extent, nx)
    
    keys = [
        'alpha', 'betax', 'betay', 'betaz',
        'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz',
        'Kxx', 'Kxy', 'Kxz', 'Kyy', 'Kyz', 'Kzz'
    ]
    
    data = {'time': time_val,
            'coord_x': x,
            'coord_y': y,
            'coord_z': z,
            'common_horizon': common_horizon}
    
    for i, key in enumerate(keys):
        data[key] = data_array[i]
    
    return data

#--------------------------------------------------------------------------
# 2. Function to write the loaded data to an HDF5 file.
#--------------------------------------------------------------------------
def write_data_to_h5(data, h5_filename):
    """
    Write the dictionary 'data' into an HDF5 file with groups:
      'coord' for coordinate arrays and 'time',
      'metric' for the 3-metric,
      'curvature' for the extrinsic curvature.
    """
    with h5py.File(h5_filename, 'w') as f:
        grp = f.create_group("coord")
        grp.create_dataset("coord_x", data=data["coord_x"])
        grp.create_dataset("coord_y", data=data["coord_y"])
        grp.create_dataset("coord_z", data=data["coord_z"])
        grp.create_dataset("time", data=data["time"])
        
        grp = f.create_group("metric")
        for comp in ['gxx','gxy','gxz','gyy','gyz','gzz']:
            grp.create_dataset(comp, data=data[comp])
        
        grp = f.create_group("curvature")
        for comp in ['Kxx','Kxy','Kxz','Kyy','Kyz','Kzz']:
            grp.create_dataset(comp, data=data[comp])
    print("Wrote HDF5 file:", h5_filename)

#--------------------------------------------------------------------------
# 3. Main routine: load data, write to HDF5, and run horizon finder.
#--------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Load AthenaK horizon dump and run horizon finder."
    )
    parser.add_argument("--binfile", type=str, required=True,
                        help="Path to the AthenaK binary horizon dump file")
    parser.add_argument("--nx", type=int, default=10,
                        help="Grid resolution (horizon_nx) in each dimension (default: 10)")
    parser.add_argument("--extent", type=float, default=2.0,
                        help="Half-extent of the grid (default: 2.0)")
    parser.add_argument("--center", type=str, default="0.0,0.0,0.0",
                        help="Comma-separated center coordinates (default: 0.0,0.0,0.0)")
    args = parser.parse_args()

    # Parse center coordinates
    center = tuple(float(x.strip()) for x in args.center.split(","))
    import glob

    flist = glob.glob(args.binfile+"/output*")
    print(flist)

    for f in flist:
        # Load the binary dump data
        data = load_horizon_dump(f+"/etk_output_file.dat", nx=args.nx, extent=args.extent, center=center)
        print(f"Loaded binary data: time = {data['time']}, common_horizon = {data['common_horizon']}")

        # Write the data into a temporary HDF5 file (which the horizon finder expects)
        temp_h5_file = os.path.join(tempfile.gettempdir(), "temp_horizon.h5")
        write_data_to_h5(data, temp_h5_file)

        #--------------------------------------------------------------------------
        # 4. Run the horizon finder.
        #
        # Here we assume that the horizon finder is implemented in the AHFpy module.
        # In your installation, you should be able to import AHFpy.AHF and then create an AHF instance.
        #
        # For this example we use the average of the α–field (or any other estimate)
        # as the initial guess for the radius.
        #--------------------------------------------------------------------------
        if 'AHFpy' in sys.modules:
            from AHFpy import AHF
        else:
            sys.path.insert(0, os.path.abspath('/home/hz0693/ahfpy/AHFpy'))
            from AHF import AHF

        initial_radius = [2]
        print("Using initial radius =", initial_radius)

        # Instantiate the horizon finder.
        # (Other parameters can be adjusted as desired.)
        ahf = AHF(initial_radius=initial_radius,
                central_points=[center],
                outputdir='horizon_output',
                hmean_target=1e-4)
        
        # Run the horizon finder on the temporary HDF5 file.
        # The run() method expects a list of filenames.
        ahf.run([temp_h5_file],
                input_data_from='HDF5_simple',
                data_sym=None,
                data_options=None)
        
        print("Horizon finder run complete.")
        
        # Optionally, remove the temporary file.
        os.remove(temp_h5_file)
        print("Temporary HDF5 file removed.")

if __name__ == "__main__":
    main()

