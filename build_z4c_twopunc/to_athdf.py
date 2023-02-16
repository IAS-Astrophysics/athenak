import glob
import os
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/home/hzhu/Desktop/research/gr/athenak_versions/athenak_rebased/vis/python')

import bin_convert

target_dir = 'bin/'
bin_files = glob.glob1(target_dir, "z4c.*.bin")
athdf_files = glob.glob1(target_dir, "z4c.*.athdf")

for i in bin_files:
    binary_fname = target_dir + i
    athdf_fname = binary_fname.replace(".bin", ".athdf")
    xdmf_fname = athdf_fname + ".xdmf"
    if i.replace(".bin", ".athdf") not in athdf_files:
        print(i)
        filedata = bin_convert.read_binary(binary_fname)
        bin_convert.write_athdf(athdf_fname, filedata)
        bin_convert.write_xdmf_for(xdmf_fname, os.path.basename(athdf_fname), filedata)
