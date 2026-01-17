import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
from multiprocessing import Pool
import os

# ============================================================
# 1. Shell integration with partial-cell volume fractions
# ============================================================

def integrate_between_shells(uov, x1v, x2v, x3v, rmin, rmax, dr):
    """
    Integrate variables in uov over spherical shells [rmin, rmax) with width dr.
    (Logic unchanged from original)
    """
    # --- radial shells ---
    bin_edges = np.arange(rmin, rmax + dr * 1e-9, dr)
    n_bins = len(bin_edges) - 1
    radii = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_vars, n_blocks, nz, ny, nx = uov.shape
    results = np.zeros((n_vars, n_bins))

    for b in range(n_blocks):
        x = x1v[b]
        y = x2v[b]
        z = x3v[b]

        # --- cell sizes & volume (Cartesian grid) ---
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0
        dz = z[1] - z[0] if len(z) > 1 else 1.0
        dV = dx * dy * dz

        # Effective radial half-thickness
        h_r = 0.5 * np.sqrt(dx*dx + dy*dy + dz*dz)
        seg_len = 2.0 * h_r

        # Quick block-level culling
        corners_x = np.array([x[0] - dx/2, x[-1] + dx/2])
        corners_y = np.array([y[0] - dy/2, y[-1] + dy/2])
        corners_z = np.array([z[0] - dz/2, z[-1] + dz/2])
        cx, cy, cz = np.meshgrid(corners_x, corners_y, corners_z, indexing='ij')
        cr = np.sqrt(cx**2 + cy**2 + cz**2)

        if cr.max() < rmin or cr.min() > rmax:
            continue

        # Full 3D coordinates & radius
        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')      # (nz, ny, nx)
        R = np.sqrt(X**2 + Y**2 + Z**2)                    # (nz, ny, nx)
        R_flat = R.ravel()                                 # (n_cell,)

        # Radial segment per cell
        r_min_cell = R_flat - h_r
        r_max_cell = R_flat + h_r

        # Cells that intersect the [rmin, rmax] band at all
        mask_band = (r_max_cell > rmin) & (r_min_cell < rmax)
        if not np.any(mask_band):
            continue

        R_valid     = R_flat[mask_band]
        r_min_valid = r_min_cell[mask_band]
        r_max_valid = r_max_cell[mask_band]

        # Bin according to cell-center radius
        bin_center = np.digitize(R_valid, bin_edges) - 1    # 0..n_bins-1
        mask_valid_bins = (bin_center >= 0) & (bin_center < n_bins)
        if not np.any(mask_valid_bins):
            continue

        R_valid      = R_valid[mask_valid_bins]
        r_min_valid  = r_min_valid[mask_valid_bins]
        r_max_valid  = r_max_valid[mask_valid_bins]
        bins_center  = bin_center[mask_valid_bins]
        n_valid      = R_valid.size

        # Flatten data for this block
        data_block = uov[:, b].reshape(n_vars, -1)
        data_valid = data_block[:, mask_band][:, mask_valid_bins]

        left_edge  = bin_edges[bins_center]
        right_edge = bin_edges[bins_center + 1]

        inside_mask = (r_min_valid >= left_edge) & (r_max_valid <= right_edge)
        boundary_mask = ~inside_mask

        # 1) Fully interior cells
        if np.any(inside_mask):
            idx_inside = bins_center[inside_mask]
            for v in range(n_vars):
                w = data_valid[v, inside_mask] * dV
                results[v] += np.bincount(idx_inside, weights=w, minlength=n_bins)

        # 2) Boundary cells
        if np.any(boundary_mask):
            bins_b   = bins_center[boundary_mask]
            rmin_b   = r_min_valid[boundary_mask]
            rmax_b   = r_max_valid[boundary_mask]
            left_b   = left_edge[boundary_mask]
            right_b  = right_edge[boundary_mask]
            data_b   = data_valid[:, boundary_mask]

            # (a) Overlap with center shell
            overlap_center = np.clip(
                np.minimum(rmax_b, right_b) - np.maximum(rmin_b, left_b), 0.0, None
            )
            frac_center = overlap_center / seg_len

            for v in range(n_vars):
                w_center = data_b[v] * dV * frac_center
                results[v] += np.bincount(bins_b, weights=w_center, minlength=n_bins)

            # (b) Overlap with right neighbor
            mask_right = (rmax_b > right_b) & (bins_b < n_bins - 1)
            if np.any(mask_right):
                bins_r = bins_b[mask_right] + 1
                rmin_r = rmin_b[mask_right]
                rmax_r = rmax_b[mask_right]
                right_b_r = right_b[mask_right]
                right_edge_next = bin_edges[bins_r + 1]

                overlap_r = np.clip(
                    np.minimum(rmax_r, right_edge_next) - np.maximum(rmin_r, right_b_r), 0.0, None
                )
                frac_r = overlap_r / seg_len

                for v in range(n_vars):
                    w_r = data_b[v, mask_right] * dV * frac_r
                    results[v] += np.bincount(bins_r, weights=w_r, minlength=n_bins)

            # (c) Overlap with left neighbor
            mask_left = (rmin_b < left_b) & (bins_b > 0)
            if np.any(mask_left):
                bins_l = bins_b[mask_left] - 1
                rmin_l = rmin_b[mask_left]
                rmax_l = rmax_b[mask_left]
                left_b_l = left_b[mask_left]
                left_edge_prev = bin_edges[bins_l]

                overlap_l = np.clip(
                    np.minimum(rmax_l, left_b_l) - np.maximum(rmin_l, left_edge_prev), 0.0, None
                )
                frac_l = overlap_l / seg_len

                for v in range(n_vars):
                    w_l = data_b[v, mask_left] * dV * frac_l
                    results[v] += np.bincount(bins_l, weights=w_l, minlength=n_bins)

    return results, radii


# ============================================================
# 2. Parallel Batch processing logic
# ============================================================

def process_single_file(args):
    """
    Worker function to process one file.
    Args tuple: (fname, rmin, rmax, dr, file_index)
    """
    fname, rmin, rmax, dr, idx = args
    
    try:
        with h5py.File(fname, 'r') as f:
            uov = f['uov'][:] 
            x1v = f['x1v'][:]
            x2v = f['x2v'][:]
            x3v = f['x3v'][:]
            t_val = f.attrs.get('Time', float(idx))
            
        integ, radii = integrate_between_shells(uov, x1v, x2v, x3v, rmin, rmax, dr)
        return (t_val, integ, radii, None) # None = no error
    except Exception as e:
        return (None, None, None, f"Error processing {fname}: {str(e)}")


def process_files_parallel(file_pattern, rmin, rmax, dr, output_npz, nproc=1, var_idx_for_quick_plot=None):
    """
    Parallel version of process_files.
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"[process_files] No files found matching {file_pattern}")
        return

    n_files = len(files)
    print(f"[process_files] Found {n_files} files. Starting processing with {nproc} threads...")

    # Prepare arguments for the worker
    tasks = [(f, rmin, rmax, dr, i) for i, f in enumerate(files)]

    results = []
    if nproc > 1:
        with Pool(nproc) as p:
            # Map returns results in order
            raw_results = p.map(process_single_file, tasks)
            results = raw_results
    else:
        # Serial fallback
        for t in tasks:
            results.append(process_single_file(t))

    # Unpack results
    times = []
    data_buffer = []
    radii_axis = None

    for res in results:
        t_val, integ, radii, error = res
        if error:
            print(error)
            continue
        
        times.append(t_val)
        data_buffer.append(integ)
        if radii_axis is None:
            radii_axis = radii

    print(f"\n[process_files] Integration complete for {len(times)}/{n_files} files.")
    
    if len(times) == 0:
        print("No valid data collected. Exiting this batch.")
        return

    # Convert to arrays and sort by time (just in case)
    times = np.array(times)
    all_data = np.array(data_buffer) # (Nt, n_vars, Nshell)

    # Ensure time ordering
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    all_data = all_data[sort_idx]

    print(f"[process_files] Saving to {output_npz} ...")
    np.savez(output_npz,
             data=all_data,
             radius=radii_axis,
             time=times)

    # --- Quick Plot (Serial) ---
    if var_idx_for_quick_plot is not None:
        print("[process_files] Generating quick radial evolution plot...")
        plt.figure(figsize=(9, 6))
        ax = plt.gca()

        if len(times) > 1:
            colors = cm.viridis(np.linspace(0, 1, len(times)))
        else:
            colors = ['C0']

        var_data = all_data[:, var_idx_for_quick_plot, :]

        for t_idx, profile in enumerate(var_data):
            c = colors[t_idx] if len(times) > 1 else 'C0'
            plt.plot(radii_axis, profile, color=c, label=f"t={times[t_idx]:.3f}")

        plt.xlabel("Radius")
        plt.ylabel(f"Integrated var index {var_idx_for_quick_plot}")
        plt.title(f"Radial profile evolution (dr={dr})")
        if len(times) <= 10:
            plt.legend()
        else:
            sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=times.min(), vmax=times.max()))
            plt.colorbar(sm, ax=ax, label="time")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_npz.replace(".npz", "_radial_evolution.png"), dpi=150)
        plt.close()


# ============================================================
# 3. Shell-wise angular momentum balance (Unchanged)
# ============================================================

def shell_imbalance(time, radius, angular_momentum_profile, torque_profile, flux_array, t_idx, comp=2, flux_idx=11):
    Nt = time.shape[0]
    # Check bounds
    if t_idx >= Nt - 1:
        print(f"Warning: t_idx {t_idx} is out of bounds for derivative (Nt={Nt}). Skipping.")
        return np.zeros_like(radius), np.zeros_like(radius), np.zeros_like(radius), np.zeros_like(radius)

    dt = time[t_idx + 1] - time[t_idx]

    L_now  = angular_momentum_profile[t_idx,   comp, :]
    L_next = angular_momentum_profile[t_idx+1, comp, :]
    dLdt = (L_next - L_now) / dt

    F_now  = flux_array[t_idx,   :, flux_idx]
    F_next = flux_array[t_idx+1, :, flux_idx]
    F_mid  = 0.5 * (F_now + F_next)
    
    advective_term = -(F_mid[1:] - F_mid[:-1])

    T_now  = torque_profile[t_idx,   comp, :]
    T_next = torque_profile[t_idx+1, comp, :]
    T_mid  = 0.5 * (T_now + T_next)

    residual = dLdt + advective_term - T_mid
    return dLdt, advective_term, T_mid, residual


# ============================================================
# 4. Main Execution
# ============================================================

if __name__ == "__main__":
    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Parallel Shell Integration Analysis")
    parser.add_argument('-n', '--nproc', type=int, default=1, help='Number of parallel processes (default: 1)')
    args = parser.parse_args()

    # --- Configurable parameters ---
    R_MIN = 20.0
    R_MAX = 300.0
    DR    = 10.0

    # File patterns
    FILE_PATTERN_AM     = "./bin/torus.angular_momentum.*.athdf"
    FILE_PATTERN_TORQUE = "./bin/torus.torque.*.athdf"
    OUTPUT_AM           = "angular_momentum_shells.npz"
    OUTPUT_TORQUE       = "torque_shells.npz"

    # 4.1 Process angular momentum and torque athdf series (Parallel)
    process_files_parallel(FILE_PATTERN_AM, R_MIN, R_MAX, DR, OUTPUT_AM, 
                           nproc=args.nproc, var_idx_for_quick_plot=2)
    
    process_files_parallel(FILE_PATTERN_TORQUE, R_MIN, R_MAX, DR, OUTPUT_TORQUE, 
                           nproc=args.nproc, var_idx_for_quick_plot=2)

    # 4.2 Load shell-integrated data (Serial Analysis)
    if not os.path.exists(OUTPUT_AM) or not os.path.exists(OUTPUT_TORQUE):
        print("Output NPZ files not found. Exiting.")
        sys.exit(1)

    am_data   = np.load(OUTPUT_AM)
    torque_data = np.load(OUTPUT_TORQUE)

    angular_momentum_profile = am_data["data"]   
    radius   = am_data["radius"]                 
    time     = am_data["time"]                   
    torque_profile = torque_data["data"]             

    # 4.3 Load flux history
    if os.path.exists("torus.user.hst"):
        flux_raw = np.loadtxt("torus.user.hst")
        time_hst = flux_raw[:, 0]
        flux_flat = flux_raw[:, 2:]
        # Check dimensions
        if flux_flat.shape[1] % 17 == 0:
            Nsurf = flux_flat.shape[1] // 17
            flux_array = flux_flat.reshape(flux_flat.shape[0], Nsurf, 17)

            # 4.4 Check balance
            t_idx = 0      
            comp = 2       
            flux_idx = 11  

            dLdt, adv_flux, torque_mid, resid = shell_imbalance(
                time=time,
                radius=radius,
                angular_momentum_profile=angular_momentum_profile,
                torque_profile=torque_profile,
                flux_array=flux_array,
                t_idx=t_idx,
                comp=comp,
                flux_idx=flux_idx
            )

            # 4.5 Plot
            plt.figure(figsize=(9, 6))
            plt.plot(radius, dLdt,     label="dL/dt (shell Lz)")
            plt.plot(radius, adv_flux, label="Advective flux term")
            plt.plot(radius, torque_mid, label="Torque term")
            plt.plot(radius, resid,    label="Residual", linestyle='--')
            plt.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
            plt.xlabel("r")
            plt.ylabel("Torque density (code units)")
            plt.title(f"Angular momentum balance, t_idx={t_idx}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig("angular_momentum_balance_tidx%d.png" % t_idx, dpi=150)
            print(f"Analysis plot saved to angular_momentum_balance_tidx{t_idx}.png")
        else:
            print("torus.user.hst shape does not match expected 17 vars/surf.")
    else:
        print("torus.user.hst not found. Skipping balance check.")
