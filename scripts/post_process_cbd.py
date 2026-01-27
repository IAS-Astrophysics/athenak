import numpy as np
import h5py
import glob
import argparse
import sys
import os
from multiprocessing import Pool

# ==============================================================================
# 1. Parameter Parsing & Physics Helpers
# ==============================================================================

def read_parfile(par_file="parfile.par"):
    """
    Parses the AthenaK/Athena++ input file into a nested dictionary.
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
            if not line or line.startswith('#'): continue
            
            if line.startswith('<') and line.endswith('>'):
                current_block = line[1:-1]
                params[current_block] = {}
                continue
                
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.split('#')[0].strip()
                
                if val.lower() == 'true': val = True
                elif val.lower() == 'false': val = False
                else:
                    try:
                        if '.' in val or 'e' in val.lower(): val = float(val)
                        else: val = int(val)
                    except ValueError: pass
                
                if current_block: params[current_block][key] = val
                else: params[key] = val
    except Exception as e:
        print(f"[Error] Failed to parse parfile: {e}")
        sys.exit(1)
    return params

def get_bh_positions(t, sep, q):
    om = sep**(-1.5)
    r_bh1 = (q / (1.0 + q)) * sep
    r_bh2 = (1.0 / (1.0 + q)) * sep 
    cos_om_t = np.cos(om * t)
    sin_om_t = np.sin(om * t)
    pos1 = np.array([ r_bh1 * cos_om_t, r_bh1 * sin_om_t, 0.0 ])
    pos2 = np.array([ -r_bh2 * cos_om_t, -r_bh2 * sin_om_t, 0.0 ])
    return pos1, pos2

def check_geometry_intersection(params, r_min_requested):
    sep = params['problem'].get('sep', 25.0)
    q   = params['problem'].get('q', 1.0)
    r_exc_1 = params['problem'].get('flux_radius1', 5.0)
    r_exc_2 = params['problem'].get('flux_radius2', 5.0)
    flux_start = params['problem'].get('flux_rsurf_inner', 20.0)
    flux_dr    = params['problem'].get('flux_dr_surf', 10.0) 
    
    dist_bh1 = (q / (1.0 + q)) * sep
    dist_bh2 = (1.0 / (1.0 + q)) * sep
    global_max_extent = max(dist_bh1 + r_exc_1, dist_bh2 + r_exc_2)
    
    if r_min_requested > (global_max_extent + 1e-5):
        return r_min_requested, False, "No intersection detected."
    
    current_r = flux_start
    while current_r < (global_max_extent + 1e-5):
        current_r += flux_dr
            
    return current_r, True, f"Overlap detected. Adjusting R_MIN to {current_r:.1f}"

# ==============================================================================
# 2. Worker Kernel (Memory Optimized & Verbose)
# ==============================================================================

def integrate_and_save(args):
    """
    Worker function to process a single .athdf file AND save the .npz.
    Reads data block-by-block from disk to save RAM.
    """
    fname, npz_out, rmin, rmax, dr, sep, q, r_exc_1, r_exc_2, delete_source = args
    
    try:
        with h5py.File(fname, 'r') as f:
            if 'uov' not in f or 'x1v' not in f:
                return (False, f"Corrupted structure in {fname}")
            
            # --- LAZY LOADING SETUP ---
            ds_uov = f['uov'] 
            x1v = f['x1v'][:]
            x2v = f['x2v'][:]
            x3v = f['x3v'][:]
            t_val = f.attrs.get('Time', 0.0)
            
            n_vars, n_blocks, nz, ny, nx = ds_uov.shape
            
            bh1_pos, bh2_pos = get_bh_positions(t_val, sep, q)

            # Global Shell Setup
            bin_edges = np.arange(rmin, rmax + dr * 1e-9, dr)
            n_bins = len(bin_edges) - 1
            radii_axis = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # Accumulators
            shell_integrals = np.zeros((n_vars, n_bins))
            bh1_vol = np.zeros(n_vars)
            bh2_vol = np.zeros(n_vars)
            cavity_vol = np.zeros(n_vars)

            # Loop over MeshBlocks
            for b in range(n_blocks):
                # --- MEMORY EFFICIENT READ ---
                # Load ONLY the current block
                uov_block = ds_uov[:, b, :, :, :] 
                
                x = x1v[b]
                y = x2v[b]
                z = x3v[b]
                
                dx = x[1] - x[0] if len(x) > 1 else (x[0] if len(x)==1 else 1.0)
                dy = y[1] - y[0] if len(y) > 1 else (y[0] if len(y)==1 else 1.0)
                dz = z[1] - z[0] if len(z) > 1 else (z[0] if len(z)==1 else 1.0)
                dV = dx * dy * dz
                
                use_supersampling = (dx >= dr/5) and (dy >= dr/5) and (dz >= dr/5)
                h_r = 0.5 * np.sqrt(dx*dx + dy*dy + dz*dz)

                # 3D Grid of Cell Centers
                Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
                R = np.sqrt(X**2 + Y**2 + Z**2)
                
                flat_uov = uov_block.reshape(n_vars, -1)
                X_flat = X.ravel()
                Y_flat = Y.ravel()
                Z_flat = Z.ravel()
                R_flat = R.ravel()

                # --- A. Shell Integration ---
                idx_min = np.floor((R_flat - h_r - rmin) / dr).astype(int)
                idx_max = np.floor((R_flat + h_r - rmin) / dr).astype(int)
                mask_global = (idx_max >= 0) & (idx_min < n_bins)
                
                if np.any(mask_global):
                    inds_min = idx_min[mask_global]
                    inds_max = idx_max[mask_global]
                    data_active = flat_uov[:, mask_global]
                    
                    if use_supersampling:
                        mask_easy = (inds_min == inds_max)
                    else:
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

                # --- C. Central Cavity ---
                mask_cavity_center = R_flat < rmin
                if use_supersampling:
                    mask_full_in = (R_flat + h_r) < rmin
                    if np.any(mask_full_in):
                          for v in range(n_vars):
                            cavity_vol[v] += np.sum(flat_uov[v, mask_full_in]) * dV
                    
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
                    if np.any(mask_cavity_center):
                        for v in range(n_vars):
                            cavity_vol[v] += np.sum(flat_uov[v, mask_cavity_center]) * dV
                
                # Cleanup
                del uov_block, flat_uov, X, Y, Z, R

        # --- SAVE RESULT ---
        np.savez(npz_out,
                 time=np.array([t_val]),
                 radius=radii_axis,
                 shell_data=np.array([shell_integrals]),
                 bh1_data=np.array([bh1_vol]),
                 bh2_data=np.array([bh2_vol]),
                 cavity_data=np.array([cavity_vol]),
                 rmin_used=rmin)
                 
        if delete_source:
            try:
                os.remove(fname)
            except OSError as e:
                return (True, f"Processed, but failed to delete source: {e}")

        return (True, f"Saved {os.path.basename(npz_out)}")

    except (OSError, KeyError, IOError) as e:
        return (False, f"Corruption/IO Error in {fname}: {e}")
    except Exception as e:
        return (False, f"Unexpected Error in {fname}: {e}")

# ==============================================================================
# 3. Stitching Helper
# ==============================================================================

def stitch_archives(file_pattern, output_filename):
    """
    Aggregates individual small NPZ files into one master archive.
    Mimics the output format of Code 1.
    """
    if file_pattern.endswith('.athdf'):
        search_pattern = file_pattern.replace('.athdf', '.npz')
        if not glob.glob(search_pattern):
             search_pattern = file_pattern + ".npz"
    else:
        search_pattern = file_pattern

    files = sorted(glob.glob(search_pattern))
    if not files:
        print(f"[Stitcher] No processed .npz files found matching {search_pattern}")
        return

    print(f"[Stitcher] Found {len(files)} files. Stitching into {output_filename}...")

    times = []
    shell_data = []
    bh1_data = []
    bh2_data = []
    cavity_data = []
    radii_axis = None
    rmin_used = None

    for i, f in enumerate(files):
        try:
            with np.load(f) as data:
                t = data['time'][0] 
                shells = data['shell_data'][0]
                b1 = data['bh1_data'][0]
                b2 = data['bh2_data'][0]
                cav = data['cavity_data'][0]
                
                if radii_axis is None:
                    radii_axis = data['radius']
                    rmin_used = data['rmin_used']
                
                times.append(t)
                shell_data.append(shells)
                bh1_data.append(b1)
                bh2_data.append(b2)
                cavity_data.append(cav)
                
        except Exception as e:
            print(f"[Stitcher] Warning: Failed to read {f}: {e}")

    if not times:
        print("[Stitcher] No valid data collected.")
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
             rmin_used=rmin_used)
             
    print(f"[Stitcher] Successfully saved {len(times)} snapshots to {output_filename}")


# ==============================================================================
# 4. Parallel Processing Driver
# ==============================================================================

def process_files(file_pattern, rmin, rmax, dr, params, output_filename, delete_source=False, nproc=1):
    all_files = sorted(glob.glob(file_pattern))
    if not all_files:
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

    tasks = []
    skipped_count = 0
    deprecated_cleaned = 0
    
    # 1. Generate Tasks
    for f in all_files:
        if f.endswith('.athdf'):
            npz_out = f[:-6] + ".npz"
        else:
            npz_out = f + ".npz"
            
        should_process = True
        
        if os.path.exists(npz_out):
            t_src = os.path.getmtime(f)
            t_npz = os.path.getmtime(npz_out)
            
            if t_npz > t_src:
                should_process = False
                skipped_count += 1
                
                # --- DEPRECATED CLEANING FEATURE ---
                # Case: We have an up-to-date NPZ, but the ATHDF is still here.
                # If --delete is enabled, this is "deprecated" trash. Delete it.
                if delete_source:
                    try:
                        os.remove(f)
                        deprecated_cleaned += 1
                    except OSError as e:
                        print(f"[Warning] Failed to clean deprecated file {f}: {e}")

            else:
                # NPZ is older than ATHDF -> Re-process
                try:
                    os.remove(npz_out)
                except OSError:
                    pass
                    
        if should_process:
            tasks.append((f, npz_out, rmin, rmax, dr, sep, q, r_exc_1, r_exc_2, delete_source))

    # 2. Run Processing
    total_tasks = len(tasks)
    print(f"\nScanning: {len(all_files)} total files.")
    print(f"Skipped:  {skipped_count} (Up-to-date)")
    if deprecated_cleaned > 0:
        print(f"Cleaned:  {deprecated_cleaned} deprecated source files.")
    print(f"Queue:    {total_tasks} files to process")
    
    if total_tasks > 0:
        processed_count = 0
        if nproc > 1:
            with Pool(nproc) as pool:
                for i, result in enumerate(pool.imap(integrate_and_save, tasks)):
                    success, msg = result
                    processed_count += 1
                    status = "OK" if success else "ERR"
                    sys.stdout.write(f"\r[Proc] {processed_count}/{total_tasks} | Last: {status} {msg[:40]}...")
                    sys.stdout.flush()
        else:
            for i, task in enumerate(tasks):
                result = integrate_and_save(task)
                success, msg = result
                processed_count += 1
                status = "OK" if success else "ERR"
                sys.stdout.write(f"\r[Proc] {processed_count}/{total_tasks} | Last: {status} {msg[:40]}...")
                sys.stdout.flush()
        print("\nBatch processing complete.")
    
    # 3. Trigger Stitching
    stitch_archives(file_pattern, output_filename)


# ==============================================================================
# 5. Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AthenaK Binary Analysis Script")
    parser.add_argument('-n', '--nproc', type=int, default=1, help='Number of processes')
    parser.add_argument('--parfile', type=str, default='parfile.par')
    parser.add_argument('--delete', action='store_true', default=False, 
                        help='Delete source .athdf file after successful processing')
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
    
    print("\n--- Processing Torque Output ---")
    process_files("./bin/torus.torque.*.athdf", R_MIN, R_MAX, DR, 
                  params, "analysis_torque.npz", delete_source=args.delete, nproc=args.nproc)
    
    print("\n--- Processing AM Output ---")
    process_files("./bin/torus.angular_momentum.*.athdf", R_MIN, R_MAX, DR, 
                  params, "analysis_am.npz", delete_source=args.delete, nproc=args.nproc)

    print("\nDone.")