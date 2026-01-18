import numpy as np
import h5py
import glob
import argparse
import sys
import re
import os
from multiprocessing import Pool

# ==============================================================================
# 1. Parameter Parsing & Physics Helpers
# ==============================================================================

def read_parfile(par_file="parfile.par"):
    """
    Parses the AthenaK/Athena++ input file into a nested dictionary.
    Returns a structure like: params['block']['key'] = value
    """
    params = {}
    current_block = None
    
    if not os.path.exists(par_file):
        print(f"[Error] Parfile '{par_file}' not found. Using defaults.")
        return {}

    try:
        with open(par_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Detect Block Headers <blockname>
            if line.startswith('<') and line.endswith('>'):
                current_block = line[1:-1]
                params[current_block] = {}
                continue
                
            # Parse Key = Value
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.split('#')[0].strip() # Remove inline comments
                
                # Type conversion
                if val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
                else:
                    try:
                        if '.' in val or 'e' in val.lower():
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        pass # Keep as string
                
                if current_block:
                    params[current_block][key] = val
                else:
                    params[key] = val
    except Exception as e:
        print(f"[Error] Failed to parse parfile: {e}")
        sys.exit(1)
    
    return params

def get_bh_positions(t, sep, q):
    """
    Calculates analytical BH positions at time t based on circular Keplerian orbit.
    """
    om = sep**(-1.5)
    r_bh1 = (q / (1.0 + q)) * sep
    r_bh2 = (1.0 / (1.0 + q)) * sep 
    
    cos_om_t = np.cos(om * t)
    sin_om_t = np.sin(om * t)
    
    pos1 = np.array([ r_bh1 * cos_om_t, r_bh1 * sin_om_t, 0.0 ])
    pos2 = np.array([ -r_bh2 * cos_om_t, -r_bh2 * sin_om_t, 0.0 ])
    
    return pos1, pos2

def check_geometry_intersection(params, r_min_requested):
    """
    Checks if the excision spheres around the BHs overlap with R_MIN.
    If overlap exists, snaps R_MIN to the next valid flux surface radius.
    """
    sep = params['problem'].get('sep', 25.0)
    q   = params['problem'].get('q', 1.0)
    
    r_exc_1 = params['problem'].get('flux_radius1', 5.0)
    r_exc_2 = params['problem'].get('flux_radius2', 5.0)
    
    # Flux grid parameters for snapping
    # Defaults based on your typical setup if missing in parfile
    flux_start = params['problem'].get('flux_rsurf_inner', 20.0)
    flux_dr    = params['problem'].get('flux_dr_surf', 10.0) 
    
    # Calculate geometric extent
    dist_bh1 = (q / (1.0 + q)) * sep
    dist_bh2 = (1.0 / (1.0 + q)) * sep
    
    max_extent_1 = dist_bh1 + r_exc_1
    max_extent_2 = dist_bh2 + r_exc_2
    global_max_extent = max(max_extent_1, max_extent_2)
    
    # Tiny epsilon for floating point comparison safety
    epsilon = 1e-5
    
    if r_min_requested > (global_max_extent + epsilon):
        return r_min_requested, False, "No intersection detected."
    
    # Logic: Find the smallest valid Flux Radius > global_max_extent
    # R_valid = flux_start + n * flux_dr
    
    current_r = flux_start
    while current_r < (global_max_extent + epsilon):
        current_r += flux_dr
        
    note = (f"Overlap detected: Req R_MIN={r_min_requested} < Extent={global_max_extent:.2f}.\n"
            f"Adjusting R_MIN to {current_r:.1f} (Next valid flux surface).")
            
    return current_r, True, note

# ==============================================================================
# 2. Worker Kernel (Super-sampling Integration)
# ==============================================================================

def integrate_frame(args):
    """
    Worker function to process a single .athdf file with SUPER-SAMPLING logic.
    """
    fname, rmin, rmax, dr, sep, q, r_exc_1, r_exc_2 = args
    
    result_template = (None, None, None, None, None, None, None) 
    
    try:
        with h5py.File(fname, 'r') as f:
            if 'uov' not in f or 'x1v' not in f:
                return (*result_template[:-1], f"Corrupted structure in {fname}")
            uov = f['uov'][:] 
            x1v = f['x1v'][:]
            x2v = f['x2v'][:]
            x3v = f['x3v'][:]
            t_val = f.attrs.get('Time', 0.0)

        bh1_pos, bh2_pos = get_bh_positions(t_val, sep, q)

        # --- 1. Global Shell Setup ---
        bin_edges = np.arange(rmin, rmax + dr * 1e-9, dr)
        n_bins = len(bin_edges) - 1
        radii_axis = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        n_vars, n_blocks, nz, ny, nx = uov.shape
        
        # Accumulators
        shell_integrals = np.zeros((n_vars, n_bins))
        bh1_vol = np.zeros(n_vars)
        bh2_vol = np.zeros(n_vars)
        cavity_vol = np.zeros(n_vars)

        # Loop over MeshBlocks
        for b in range(n_blocks):
            x = x1v[b]
            y = x2v[b]
            z = x3v[b]
            
            # Compute cell size
            dx = x[1] - x[0] if len(x) > 1 else (x[0] if len(x)==1 else 1.0)
            dy = y[1] - y[0] if len(y) > 1 else (y[0] if len(y)==1 else 1.0)
            dz = z[1] - z[0] if len(z) > 1 else (z[0] if len(z)==1 else 1.0)
            dV = dx * dy * dz
            
            # --- Check Super-sampling Condition ---
            # User request: Only do super-sampling if cell size >= dr/5
            use_supersampling = (dx >= dr/5) and (dy >= dr/5) and (dz >= dr/5)

            # Half-diagonal (radius of the bounding sphere of the cell)
            h_r = 0.5 * np.sqrt(dx*dx + dy*dy + dz*dz)

            # 3D Grid of Cell Centers
            Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
            
            # Distance from origin
            R = np.sqrt(X**2 + Y**2 + Z**2)
            
            # Flatten arrays
            flat_uov = uov[:, b].reshape(n_vars, -1)
            X_flat = X.ravel()
            Y_flat = Y.ravel()
            Z_flat = Z.ravel()
            R_flat = R.ravel()

            # --- A. Shell Integration (rmin <= r < rmax) ---
            
            # Determine min/max bin index per cell based on bounding sphere
            idx_min = np.floor((R_flat - h_r - rmin) / dr).astype(int)
            idx_max = np.floor((R_flat + h_r - rmin) / dr).astype(int)
            
            # Valid cells must interact with ROI
            mask_global = (idx_max >= 0) & (idx_min < n_bins)
            
            if np.any(mask_global):
                # Apply mask to data structures
                inds_min = idx_min[mask_global]
                inds_max = idx_max[mask_global]
                data_active = flat_uov[:, mask_global]
                
                # 1. Fast Path
                if use_supersampling:
                    mask_easy = (inds_min == inds_max)
                else:
                    # Treat all as easy, re-calculate bin on center strictly
                    mask_easy = np.ones(inds_min.shape, dtype=bool) 
                    R_active = R_flat[mask_global]
                    simple_bins = np.digitize(R_active, bin_edges) - 1
                    inds_min = simple_bins

                if np.any(mask_easy):
                    final_bins = inds_min[mask_easy]
                    valid_easy = (final_bins >= 0) & (final_bins < n_bins)
                    
                    if np.any(valid_easy):
                        bins_to_count = final_bins[valid_easy]
                        for v in range(n_vars):
                            w = data_active[v, mask_easy][valid_easy] * dV
                            shell_integrals[v] += np.bincount(bins_to_count, weights=w, minlength=n_bins)

                # 2. Slow Path: Boundary Cells (Super-sampling)
                mask_split = (~mask_easy)
                
                if use_supersampling and np.any(mask_split):
                    X_split = X_flat[mask_global][mask_split]
                    Y_split = Y_flat[mask_global][mask_split]
                    Z_split = Z_flat[mask_global][mask_split]
                    data_split = data_active[:, mask_split]
                    
                    off_x, off_y, off_z = 0.25*dx, 0.25*dy, 0.25*dz
                    sub_dV = dV / 8.0
                    
                    offsets = [
                        ( off_x,  off_y,  off_z), ( off_x,  off_y, -off_z),
                        ( off_x, -off_y,  off_z), ( off_x, -off_y, -off_z),
                        (-off_x,  off_y,  off_z), (-off_x,  off_y, -off_z),
                        (-off_x, -off_y,  off_z), (-off_x, -off_y, -off_z)
                    ]
                    
                    for (ox, oy, oz) in offsets:
                        R_sub = np.sqrt((X_split + ox)**2 + (Y_split + oy)**2 + (Z_split + oz)**2)
                        bins_sub = np.floor((R_sub - rmin) / dr).astype(int)
                        
                        mask_valid_sub = (bins_sub >= 0) & (bins_sub < n_bins)
                        
                        if np.any(mask_valid_sub):
                            valid_bins = bins_sub[mask_valid_sub]
                            for v in range(n_vars):
                                w_sub = data_split[v, mask_valid_sub] * sub_dV
                                shell_integrals[v] += np.bincount(valid_bins, weights=w_sub, minlength=n_bins)

            # --- B. Excision Volume ---
            dist_sq_1 = (X - bh1_pos[0])**2 + (Y - bh1_pos[1])**2 + (Z - bh1_pos[2])**2
            mask_bh1 = dist_sq_1 < (r_exc_1**2)
            if np.any(mask_bh1):
                flat_mask1 = mask_bh1.ravel()
                for v in range(n_vars):
                    bh1_vol[v] += np.sum(flat_uov[v, flat_mask1]) * dV
            
            dist_sq_2 = (X - bh2_pos[0])**2 + (Y - bh2_pos[1])**2 + (Z - bh2_pos[2])**2
            mask_bh2 = dist_sq_2 < (r_exc_2**2)
            if np.any(mask_bh2):
                flat_mask2 = mask_bh2.ravel()
                for v in range(n_vars):
                    bh2_vol[v] += np.sum(flat_uov[v, flat_mask2]) * dV

            # --- C. Central Cavity (r < rmin) with Super-sampling ---
            mask_cavity_center = R_flat < rmin
            
            if use_supersampling:
                # Fully inside cells (Conservative check using h_r)
                mask_full_in = (R_flat + h_r) < rmin
                if np.any(mask_full_in):
                     for v in range(n_vars):
                        cavity_vol[v] += np.sum(flat_uov[v, mask_full_in]) * dV
                
                # Overlapping cells (Center might be in or out, but bounding sphere touches boundary)
                # Logic: Is "partially inside"? (R-h < rmin) AND "partially outside"? (R+h >= rmin)
                # Note: mask_full_in is strictly inside. We need the rest.
                mask_overlap = ((R_flat - h_r) < rmin) & (~mask_full_in)
                
                if np.any(mask_overlap):
                    X_ov = X_flat[mask_overlap]
                    Y_ov = Y_flat[mask_overlap]
                    Z_ov = Z_flat[mask_overlap]
                    data_ov = flat_uov[:, mask_overlap]
                    
                    off_x, off_y, off_z = 0.25*dx, 0.25*dy, 0.25*dz
                    sub_dV = dV / 8.0
                    
                    offsets = [
                        ( off_x,  off_y,  off_z), ( off_x,  off_y, -off_z),
                        ( off_x, -off_y,  off_z), ( off_x, -off_y, -off_z),
                        (-off_x,  off_y,  off_z), (-off_x,  off_y, -off_z),
                        (-off_x, -off_y,  off_z), (-off_x, -off_y, -off_z)
                    ]
                    
                    for (ox, oy, oz) in offsets:
                        R_sub = np.sqrt((X_ov + ox)**2 + (Y_ov + oy)**2 + (Z_ov + oz)**2)
                        mask_sub_in = R_sub < rmin
                        if np.any(mask_sub_in):
                            for v in range(n_vars):
                                cavity_vol[v] += np.sum(data_ov[v, mask_sub_in]) * sub_dV
            else:
                # Standard integration
                if np.any(mask_cavity_center):
                    for v in range(n_vars):
                        cavity_vol[v] += np.sum(flat_uov[v, mask_cavity_center]) * dV

        return (t_val, shell_integrals, radii_axis, bh1_vol, bh2_vol, cavity_vol, None)

    except (OSError, KeyError, IOError) as e:
        return (*result_template[:-1], f"Corruption/IO Error: {e}")
    except Exception as e:
        return (*result_template[:-1], f"Unexpected Error: {e}")

# ==============================================================================
# 3. Parallel Processing Driver
# ==============================================================================

def process_files(file_pattern, rmin, rmax, dr, params, output_filename, nproc=1):
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"[Warning] No files found matching: {file_pattern}")
        return

    try:
        sep = params['problem'].get('sep', 25.0)
        q   = params['problem'].get('q', 1.0)
        r_exc_1 = params['problem'].get('flux_radius1', 5.0)
        r_exc_2 = params['problem'].get('flux_radius2', 5.0)
    except KeyError:
        print("[Error] Could not find critical binary parameters in <problem> block.")
        return

    tasks = [(f, rmin, rmax, dr, sep, q, r_exc_1, r_exc_2) for f in files]
    total_files = len(files)
    
    print(f"Starting analysis on {total_files} files -> {output_filename}")
    print(f"Config: Shells=[{rmin}, {rmax}], BH_exc_r=[{r_exc_1}, {r_exc_2}], Cavity < {rmin}")

    raw_results = []
    
    # --- Parallel Execution with Progress Reporting ---
    if nproc > 1:
        with Pool(nproc) as pool:
            # imap returns results as they complete, allowing progress tracking
            for i, result in enumerate(pool.imap(integrate_frame, tasks)):
                raw_results.append(result)
                
                # Progress Bar Logic
                percent = (i + 1) / total_files * 100
                sys.stdout.write(f"\r[Progress] {i + 1}/{total_files} files ({percent:.1f}%)")
                sys.stdout.flush()
    else:
        # Serial execution
        for i, task in enumerate(tasks):
            result = integrate_frame(task)
            raw_results.append(result)
            
            # Progress Bar Logic
            percent = (i + 1) / total_files * 100
            sys.stdout.write(f"\r[Progress] {i + 1}/{total_files} files ({percent:.1f}%)")
            sys.stdout.flush()

    print("") # Move to new line after progress bar

    # Post-process results
    times = []
    shell_data = []
    bh1_data = []
    bh2_data = []
    cavity_data = []
    radii_axis = None
    
    for i, res in enumerate(raw_results):
        t, shells, r, b1, b2, cav, err = res
        
        if err:
            print(f"\n  [Skipping File {i}] {err}")
            continue
            
        times.append(t)
        shell_data.append(shells)
        bh1_data.append(b1)
        bh2_data.append(b2)
        cavity_data.append(cav)
        
        if radii_axis is None:
            radii_axis = r

    if not times:
        print("[Error] No valid data extracted from files.")
        return

    times = np.array(times)
    sort_idx = np.argsort(times)
    
    times = times[sort_idx]
    shell_data = np.array(shell_data)[sort_idx]   
    bh1_data = np.array(bh1_data)[sort_idx]       
    bh2_data = np.array(bh2_data)[sort_idx]       
    cavity_data = np.array(cavity_data)[sort_idx] 

    np.savez(output_filename,
             time=times,
             radius=radii_axis,
             shell_data=shell_data,
             bh1_data=bh1_data,
             bh2_data=bh2_data,
             cavity_data=cavity_data,
             rmin_used=rmin)
    
    print(f"Successfully saved {len(times)} snapshots to {output_filename}\n")

# ==============================================================================
# 4. Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AthenaK Binary Analysis Script")
    parser.add_argument('-n', '--nproc', type=int, default=1, help='Number of processes')
    parser.add_argument('--parfile', type=str, default='parfile.par')
    args = parser.parse_args()

    print(f"--- Reading {args.parfile} ---")
    params = read_parfile(args.parfile)
    if 'problem' not in params:
        print("Error: Parfile missing <problem> block.")
        sys.exit(1)

    r_edge = params['problem'].get('r_edge', 60.0)
    desired_rmin = params['problem'].get('flux_rsurf_inner', 20.0) 
    
    R_MIN, adjusted, note = check_geometry_intersection(params, desired_rmin)
    if adjusted:
        print(f"!!! GEOMETRY WARNING !!!\n{note}\n")
    else:
        print(f"Geometry check passed: R_MIN={R_MIN} clears binary motion.")

    R_MAX = params['problem'].get('flux_rsurf_outer', 400.0)
    DR    = params['problem'].get('flux_dr_surf', 1.0) 
    
    process_files("./bin/torus.torque.*.athdf", R_MIN, R_MAX, DR, 
                  params, "analysis_torque.npz", args.nproc)
    
    process_files("./bin/torus.angular_momentum.*.athdf", R_MIN, R_MAX, DR, 
                  params, "analysis_am.npz", args.nproc)

    print("\n--- Summary ---")
    if os.path.exists("analysis_torque.npz"):
        data = np.load("analysis_torque.npz")
        t_vol = data['time']
        if len(t_vol) > 0:
            print(f"Torque Analysis: {len(t_vol)} snapshots processed.")
            print(f"Time range: t={t_vol[0]:.1f} to {t_vol[-1]:.1f}")
        else:
            print("Torque Analysis produced empty arrays.")
    else:
        print("Torque output file not found.")

    if os.path.exists("analysis_am.npz"):
        data = np.load("analysis_am.npz")
        t_vol = data['time']
        print(f"AM Analysis: {len(t_vol)} snapshots processed.")

    print("\nDone.")