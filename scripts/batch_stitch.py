import os
import glob
import re
import subprocess
import sys

# ================= CONFIGURATION =================
# Path to your existing stitch script
STITCHER_PATH = os.path.expanduser("~/athenak/scripts/stitch.py")
# The pattern for your run folders
RUN_DIR_PATTERN = "run_*"
# The subdirectory where AthenaK stores restart files (usually 'rst')
RST_SUBDIR = "rst"
# =================================================

def natural_sort_key(s):
    """Sorts strings with numbers naturally."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def find_latest_rst(base_rst_dir):
    """
    Looks into rank_00000000 of the base_rst_dir (e.g., run_0/rst) 
    to find the restart file with the highest index.
    """
    # We look inside the rank_0 folder to find filenames
    rank0_path = os.path.join(base_rst_dir, "rank_00000000")
    
    if not os.path.exists(rank0_path):
        return None

    # Find all .rst files
    rst_files = glob.glob(os.path.join(rank0_path, "*.rst"))
    
    if not rst_files:
        return None

    # Sort files naturally to find the highest number
    rst_files.sort(key=natural_sort_key)
    
    # Get the basename (e.g., "torus.00135.rst")
    return os.path.basename(rst_files[-1])

def main():
    if not os.path.exists(STITCHER_PATH):
        print(f"Error: Could not find stitcher script at: {STITCHER_PATH}")
        sys.exit(1)

    run_dirs = glob.glob(RUN_DIR_PATTERN)
    run_dirs.sort(key=natural_sort_key)

    if not run_dirs:
        print("No directories matching 'run_*' found.")
        sys.exit(0)

    print(f"Found {len(run_dirs)} run directories. Checking for 'rst' folders...\n")

    for d in run_dirs:
        # Construct the full path to where rank folders actually live: run_X/rst
        rst_dir_path = os.path.join(d, RST_SUBDIR)
        
        # Find the latest file inside run_X/rst/rank_00000000
        latest_file = find_latest_rst(rst_dir_path)
        
        if latest_file:
            print(f"--> Processing {d} (Latest: {latest_file})")
            
            # Pass the directory containing the rank folders (run_X/rst) to stitch.py
            cmd = [
                "python", 
                STITCHER_PATH,
                "-d", rst_dir_path,
                "-f", latest_file
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"[Error] Failed to stitch {d}")
        else:
            print(f"[Skip] {d}: No restart files found in {rst_dir_path}/rank_00000000")
        
        print("-" * 40)

if __name__ == "__main__":
    main()
