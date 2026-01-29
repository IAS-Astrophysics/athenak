import glob
import os
import sys
import argparse
from multiprocessing import Pool

sys.dont_write_bytecode = True
# Ensure the path is correct for your environment
sys.path.insert(0, '/home/hzhu/athenak/vis/python')

import bin_convert

target_dir = 'bin/'

def convert_single_file(filename):
    """
    Worker function to process a single binary file.
    """
    binary_fname = os.path.join(target_dir, filename)
    athdf_fname = binary_fname.replace(".bin", ".athdf")
    xdmf_fname = athdf_fname + ".xdmf"

    print(f"Processing: {filename}")

    try:
        filedata = bin_convert.read_binary(binary_fname)
        bin_convert.write_athdf(athdf_fname, filedata)
        bin_convert.write_xdmf_for(xdmf_fname, os.path.basename(athdf_fname), filedata)

        # --- MODIFICATION START ---
        # Conversion successful, delete the source binary file
        print(f"Finished {filename}. Deleting source binary...")
        if os.path.exists(binary_fname):
            os.remove(binary_fname)
        # --- MODIFICATION END ---

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

def main():
    # 1. Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Parallel binary to athdf converter.")
    parser.add_argument(
        '-n', '--nproc',
        type=int,
        default=1,
        help='Number of parallel processes to use (default: 1)'
    )
    args = parser.parse_args()

    # 2. Get file lists
    bin_files = glob.glob1(target_dir, "*.bin")
    # Using a set for faster lookups
    athdf_files = set(glob.glob1(target_dir, "*.athdf"))

    # 3. Create a list of files that actually need processing
    tasks = []
    for i in bin_files:
        athdf_name = i.replace(".bin", ".athdf")
        
        # Check if the target athdf already exists
        if athdf_name in athdf_files:
            # --- MODIFICATION START ---
            # If both exist, we assume we need to REGENERATE.
            # Delete the existing athdf/xdmf so we can start fresh.
            athdf_path = os.path.join(target_dir, athdf_name)
            xdmf_path = athdf_path + ".xdmf"
            
            print(f"Both binary and athdf exist for {i}. Deleting old athdf to regenerate...")
            try:
                os.remove(athdf_path)
                if os.path.exists(xdmf_path):
                    os.remove(xdmf_path)
            except OSError as e:
                print(f"Error deleting old files for {i}: {e}")
                continue # Skip this file if we can't clean up
            # --- MODIFICATION END ---

        # Add to tasks list (whether it was fresh or we just deleted the old output)
        tasks.append(i)

    # 4. Execute processing
    if not tasks:
        print("No files to convert.")
        return

    if args.nproc > 1:
        print(f"Starting parallel conversion on {args.nproc} processes for {len(tasks)} files...")
        with Pool(args.nproc) as p:
            p.map(convert_single_file, tasks)
    else:
        print(f"Starting serial conversion for {len(tasks)} files...")
        for f in tasks:
            convert_single_file(f)

if __name__ == "__main__":
    main()
