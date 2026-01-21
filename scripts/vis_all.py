import yt
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import argparse
from multiprocessing import Pool
import sys
import os
import numpy as np
import functools
from matplotlib.ticker import MaxNLocator

# 1. Prevent matplotlib from trying to open windows
plt.switch_backend('Agg')

# 2. Suppress yt logging
yt.mylog.setLevel(50)

# --- Define the Beta Field Function (User's Convention) ---
def _plasma_beta(field, data):
    """
    Calculates plasma beta = 2 * P / B^2
    Assumes Athena++ variable names.
    Uses .d to strip units and work with raw numpy arrays.
    """
    # 1. Calculate B^2 (using .d to get raw values)
    b2 = (data["athena_pp", "bcc1"].d**2 +
          data["athena_pp", "bcc2"].d**2 +
          data["athena_pp", "bcc3"].d**2)

    # 2. Get Pressure directly (using .d)
    press = data["athena_pp", "press"].d

    # 3. Calculate Beta = 2 * P / B^2
    # Add epsilon to prevent division by zero
    return 2.0 * press / (b2 + 1e-30)

def process_frame(f_xy):
    """
    Worker function to process slices. Plots 3x2 grid.
    """
    f_xz = f_xy.replace("slice_x3", "slice_x2")

    if not os.path.exists(f_xz):
        print(f"Skipping {f_xy}: Corresponding XZ slice {f_xz} not found.")
        return

    try:
        # Load datasets
        ds_xy = yt.load(f_xy)
        ds_xz = yt.load(f_xz)

        # Register Beta field for both
        for ds in [ds_xy, ds_xz]:
            ds.add_field(
                ("gas", "beta"),
                function=_plasma_beta,
                sampling_type="cell",
                units="",
                display_name=r"\beta"
            )

        # --- Set up 3x2 Grid ---
        # Increased figure width slightly to help with spacing
        fig = plt.figure(figsize=(20, 10)) 
        
        grid = AxesGrid(
            fig,
            (0.05, 0.05, 0.90, 0.85), # Adjusted rect to leave room for suptitle
            nrows_ncols=(2, 3),
            axes_pad=0.4,             # Increased pad to stop overlap
            label_mode="L",
            share_all=True,
            cbar_location="right",
            cbar_mode="each",
            cbar_size="5%",
            cbar_pad="2%",
        )

        # Define Field Names
        dens_field = ("athena_pp", "dens")
        temp_field = ("athena_pp", "temperature")
        beta_field = ("gas", "beta")

        # Define Vector Components
        # B-field components
        bx, by, bz = ("athena_pp", "bcc1"), ("athena_pp", "bcc2"), ("athena_pp", "bcc3")
        # Velocity components
        vx, vy, vz = ("athena_pp", "velx"), ("athena_pp", "vely"), ("athena_pp", "velz")

        # Column Configurations
        # ptype: 0=Density(Fluid Lines), 1=Temp(Contours), 2=Beta(B-Field Lines)
        cols_config = [
            (dens_field, "viridis", (1e-5, 1.0), True, 0),
            (temp_field, "inferno", (1e-4, 1e-1), True, 1), 
            (beta_field, "coolwarm", (1e-2, 1e2), True, 2)
        ]
        
        col_names = ["Density", "Temperature", r"Plasma $\beta$"]

        # ==========================================
        # Loop over Columns
        # ==========================================
        for i, (field, cmap, (zmin, zmax), is_log, ptype) in enumerate(cols_config):
            
            # Helper for contours
            contour_kwargs = {"levels": 5, "plot_args": {"colors": "white", "linewidths": 0.5, "alpha": 0.7}}
            if zmin is not None: contour_kwargs["clim"] = (zmin, zmax)

            # ------------------------------------
            # TOP ROW: XY Plane
            # ------------------------------------
            slc_xy = yt.SlicePlot(ds_xy, "z", field)
            slc_xy.zoom(20)
            slc_xy.set_log(field, is_log)
            if zmin: slc_xy.set_zlim(field, zmin, zmax)
            slc_xy.set_cmap(field, cmap)

            # XY Annotations
            try:
                if ptype == 0:   
                    # Fluid flow lines: vx, vy
                    slc_xy.annotate_streamlines(vx, vy, density=1.0, color='gray', linewidth=0.6)
                elif ptype == 1: 
                    slc_xy.annotate_contour(field, **contour_kwargs)
                elif ptype == 2: 
                    # Magnetic field lines: bx, by
                    slc_xy.annotate_streamlines(bx, by, density=1.0, color='gray', linewidth=0.6)
            except: pass

            # Render XY
            plot_xy = slc_xy.plots[field]
            plot_xy.figure = fig
            plot_xy.axes = grid[i].axes
            plot_xy.cax = grid.cbar_axes[i]
            slc_xy.render()

            # Clean up XY Colorbar
            grid.cbar_axes[i].set_ylabel("") 
            grid.cbar_axes[i].tick_params(labelsize=9) 

            # Reduce Y Ticks (XY)
            grid[i].axes.yaxis.set_major_locator(MaxNLocator(nbins=4))

            # Set Column Title
            grid[i].axes.set_title(col_names[i], fontsize=14, pad=10)
            
            # Axis labels for row 0
            if i == 0: grid[i].axes.set_ylabel(r"$y~(M)$")

            # ------------------------------------
            # BOTTOM ROW: XZ Plane
            # ------------------------------------
            # User convention: Slice 'y', then swap_axes()
            slc_xz = yt.SlicePlot(ds_xz, "y", field)
            slc_xz.swap_axes()
            slc_xz.zoom(20)
            slc_xz.set_log(field, is_log)
            if zmin: slc_xz.set_zlim(field, zmin, zmax)
            slc_xz.set_cmap(field, cmap)

            # XZ Annotations (User convention: bz, bx)
            try:
                if ptype == 0:   
                    # Fluid flow lines: vz, vx (Matching the bz, bx convention)
                    slc_xz.annotate_streamlines(vz, vx, density=1.0, color='gray', linewidth=0.6)
                elif ptype == 1: 
                    slc_xz.annotate_contour(field, **contour_kwargs)
                elif ptype == 2: 
                    # Magnetic field lines: bz, bx (Strict user convention)
                    slc_xz.annotate_streamlines(bz, bx, density=1.0, color='gray', linewidth=0.6)
            except: pass

            # Render XZ
            plot_xz = slc_xz.plots[field]
            plot_xz.figure = fig
            plot_xz.axes = grid[i+3].axes
            plot_xz.cax = grid.cbar_axes[i+3]
            slc_xz.render()

            # Clean up XZ Colorbar
            grid.cbar_axes[i+3].set_ylabel("")
            grid.cbar_axes[i+3].tick_params(labelsize=9)

            # Reduce Y Ticks (XZ)
            grid[i+3].axes.yaxis.set_major_locator(MaxNLocator(nbins=4))

            # Axis labels for row 1
            grid[i+3].axes.set_xlabel(r"$x~(M)$")
            if i == 0: grid[i+3].axes.set_ylabel(r"$z~(M)$")

        # Set Global Time Title
        fig.suptitle(f"Time = {ds_xy.current_time.v:.1f} M", fontsize=16, y=0.95)

        # Save output
        outname = f_xy.replace("slice_x3", "combined_3x2").replace(".athdf", ".png")
        fig.savefig(outname, dpi=150)
        plt.close(fig)

        print(f"Finished: {outname}")

    except Exception as e:
        print(f"Error processing {f_xy}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel plotting of Athena++ slice files.")
    parser.add_argument("-n", "--nproc", type=int, default=1, help="Number of processes to run in parallel")
    args = parser.parse_args()

    athdf_files = sorted(glob.glob("./bin/torus.slice_x3.*.athdf"))

    if not athdf_files:
        print("No slice_x3 files found in ./bin/")
        sys.exit(1)

    print(f"Found {len(athdf_files)} XY slice files.")
    print(f"Processing 3x2 panels with {args.nproc} processes...")

    with Pool(processes=args.nproc) as pool:
        pool.map(process_frame, athdf_files)

    print("All done.")
