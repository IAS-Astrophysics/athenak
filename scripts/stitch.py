import os
import struct
import argparse
import sys
import glob
import re

def detect_ranks(base_dir):
    """
    Scans the base_dir for folders matching 'rank_XXXXXXXX'.
    Returns the total count and validates continuity.
    """
    # Find all directories starting with 'rank_'
    rank_dirs = glob.glob(os.path.join(base_dir, "rank_*"))
    
    if not rank_dirs:
        raise FileNotFoundError(f"No 'rank_XXXXXXXX' directories found in {base_dir}")

    # Extract rank indices
    ranks = []
    for d in rank_dirs:
        match = re.search(r'rank_(\d+)', d)
        if match:
            ranks.append(int(match.group(1)))
    
    ranks.sort()
    
    if not ranks:
        raise ValueError("Found rank folders but could not extract rank IDs.")

    # Validation: Check if ranks start at 0 and are continuous
    if ranks[0] != 0:
        print(f"[Warning] Ranks do not start at 0 (Starts at {ranks[0]}).")
    
    # Check for gaps
    expected = list(range(ranks[0], ranks[-1] + 1))
    if ranks != expected:
        missing = set(expected) - set(ranks)
        raise ValueError(f"Missing rank directories: {sorted(list(missing))}")

    count = ranks[-1] + 1
    print(f"[Auto-Detect] Found {len(ranks)} rank directories (Max Rank ID: {ranks[-1]}).")
    return count

def find_header_end(file_path):
    """
    Heuristic to find the end of the header and the data_size per block.
    """
    file_size = os.path.getsize(file_path)
    min_header = 100
    max_scan = min(file_size - 8, 10 * 1024 * 1024) 
    
    with open(file_path, 'rb') as f:
        f.seek(0)
        buffer = f.read(max_scan)
        
        # Scan for the data_size variable (unsigned long long, 8 bytes)
        for offset in range(min_header, len(buffer) - 8):
            candidate_size = struct.unpack_from('<Q', buffer, offset)[0]
            
            if candidate_size < 1024:
                continue
                
            header_end = offset + 8
            remaining_data = file_size - header_end
            
            if remaining_data > 0 and (remaining_data % candidate_size == 0):
                num_blocks = remaining_data // candidate_size
                print(f"[Info] Detected header size: {header_end} bytes.")
                print(f"[Info] Block data size: {candidate_size} bytes.")
                return header_end
                
    raise ValueError("Could not auto-detect header size. File structure corrupted?")

def stitch_files(base_dir, basename, output_name, nprocs=None):
    # --- Step 0: Auto-detect N if not provided ---
    if nprocs is None:
        try:
            nprocs = detect_ranks(base_dir)
        except Exception as e:
            print(f"Error during auto-detection: {e}")
            sys.exit(1)

    # --- Step 1: Analyze Rank 0 Header ---
    rank0_dir = os.path.join(base_dir, "rank_00000000")
    rank0_file = os.path.join(rank0_dir, basename)
    
    if not os.path.exists(rank0_file):
        print(f"Error: Rank 0 file not found at {rank0_file}")
        sys.exit(1)
    
    try:
        header_size = find_header_end(rank0_file)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # --- Step 2: Stitch ---
    output_path = os.path.join(base_dir, output_name)
    print(f"Writing stitched output to: {output_path}")
    
    with open(output_path, 'wb') as outfile:
        # Write Header from Rank 0
        with open(rank0_file, 'rb') as f0:
            header_data = f0.read(header_size)
            outfile.write(header_data)
        
        # Append Data from all ranks
        for r in range(nprocs):
            rank_str = f"rank_{r:08d}"
            current_file = os.path.join(base_dir, rank_str, basename)
            
            if not os.path.exists(current_file):
                print(f"Error: Missing file for rank {r}: {current_file}")
                sys.exit(1)
                
            # Efficient copy using shutil (avoids loading full file to RAM)
            with open(current_file, 'rb') as infile:
                infile.seek(header_size)
                import shutil
                shutil.copyfileobj(infile, outfile)
            
            if r % 10 == 0 and r > 0:
                print(f"  ...processed {r}/{nprocs} ranks...", end='\r')

    print(f"\nSuccess! Stitched {nprocs} rank files into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch AthenaK split restart files.")
    parser.add_argument("-d", "--dir", type=str, default="rst", help="Directory containing rank_ folders")
    parser.add_argument("-f", "--file", type=str, required=True, help="Basename (e.g., torus.00000.rst)")
    parser.add_argument("-n", "--nprocs", type=int, help="Optional: Force number of ranks (overrides auto-detect)")
    parser.add_argument("-o", "--output", type=str, help="Output filename")
    
    args = parser.parse_args()
    out_name = args.output if args.output else f"stitched_{args.file}"
    
    stitch_files(args.dir, args.file, out_name, args.nprocs)
