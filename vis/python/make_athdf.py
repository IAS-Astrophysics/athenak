# A simple script for converting a collection of .bin files to .athdf/.xdmf files using
# bin_convert

# Python modules
import os
import argparse
import glob

# AthenaK modules
import bin_convert


# Main function
def main(**kwargs):
    # Get the root name for the file.
    files = glob.glob(kwargs['file_stem'] + '*.bin')
    if len(files) < 1:
        print(f"No files found with stem {kwargs['file_stem']}")
        quit()

    total = len(files)
    count = 1

    for fname in files:
        athdf_name = fname.replace(".bin", ".athdf")
        xdmf_name = athdf_name + ".xdmf"
        filedata = bin_convert.read_binary(fname)
        bin_convert.write_athdf(athdf_name, filedata)
        bin_convert.write_xdmf_for(xdmf_name, os.path.basename(athdf_name), filedata)
        if kwargs['verbose']:
            print(f'Converting {count}/{total}: {fname}')
        count = count+1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_stem', help='path to files, excluding .#.bin')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print file conversion progress')
    args = parser.parse_args()
    main(**vars(args))
