# Various functions to read Athena++ output data files

# Python modules
import re
import warnings
from io import open  # Consistent binary I/O from Python 2 and 3

# Other Python modules
import numpy as np

check_nan_flag = False


# Check input NumPy array for the presence of any NaN entries
def check_nan(data):
    if np.isnan(data).any():
        raise FloatingPointError("NaN encountered")
    return


# Wrapper to np.loadtxt() for checks used in regression tests
def error_dat(filename, **kwargs):
    data = np.loadtxt(filename,
                      dtype=np.float64,
                      ndmin=2,  # prevent NumPy from squeezing singleton dim
                      **kwargs)
    if check_nan_flag:
        check_nan(data)
    return data


# Read .tab files and return dict.
def tab(filename):

    # Parse header
    data_dict = {}
    with open(filename, 'r') as data_file:
        line = data_file.readline()
        attributes = re.search(r'time=(\S+)\s+cycle=(\S+)', line)
        line = data_file.readline()
        headings = line.split()[1:]
    data_dict['time'] = float(attributes.group(1))
    data_dict['cycle'] = int(attributes.group(2))
    headings = headings[1:]

    # Go through lines
    data_array = []
    num_lines = 0
    with open(filename, 'r') as data_file:
        first_line = True
        for line in data_file:
            # Skip comments
            if line.split()[0][0] == '#':
                continue

            # Extract cell indices
            vals = line.split()
            if first_line:
                num_entries = len(vals) - 1
                first_line = False

            # Extract cell values
            vals = vals[1:]
            data_array.append([float(val) for val in vals])
            num_lines += 1

    # Reshape array
    array_shape = (num_lines, num_entries)
    array_transpose = (1, 0)
    data_array = np.transpose(np.reshape(data_array, array_shape),
                              array_transpose)

    # Finalize data
    for n, heading in enumerate(headings):
        if check_nan_flag:
            check_nan(data_array[n, ...])
        data_dict[heading] = data_array[n, ...]
    return data_dict


# Read .hst files and return dict of 1D arrays.
# Keyword arguments:
# raw -- if True, do not prune file to remove stale data
# from prev runs (default False)
def hst(filename, raw=False):
    # Read data
    with open(filename, 'r') as data_file:
        # Find header
        header_found = False
        multiple_headers = False
        header_location = None
        line = data_file.readline()
        while len(line) > 0:
            if line == '# Athena++ history data\n':
                if header_found:
                    multiple_headers = True
                else:
                    header_found = True
                header_location = data_file.tell()
            line = data_file.readline()
        if multiple_headers:
            warnings.warn('Multiple headers found; using most recent data')
        if header_location is None:
            raise RuntimeError('athena_read.hst: Could not find header')

        # Parse header
        data_file.seek(header_location)
        header = data_file.readline()
        data_names = re.findall(r'\[\d+\]=(\S+)', header)
        if len(data_names) == 0:
            raise RuntimeError('athena_read.hst: Could not parse header')

        # Prepare dictionary of results
        data = {}
        for name in data_names:
            data[name] = []

        # Read data
        for line in data_file:
            for name, val in zip(data_names, line.split()):
                data[name].append(float(val))

    # Finalize data
    for key, val in data.items():
        data[key] = np.array(val)
    if not raw:
        if data_names[0] != 'time':
            raise AthenaError('Cannot remove spurious data because time '
                              'column could not be identified')
        branches_removed = False
        while not branches_removed:
            branches_removed = True
            for n in range(1, len(data['time'])):
                if data['time'][n] <= data['time'][n-1]:
                    branch_index = np.where((data['time'][:n] >=
                                             data['time'][n]))[0][0]
                    for key, val in data.items():
                        data[key] = np.concatenate((val[:branch_index],
                                                    val[n:]))
                    branches_removed = False
                    break
        if check_nan_flag:
            for key, val in data.items():
                check_nan(val)
    return data


# General exception class for these functions
class AthenaError(RuntimeError):
    pass
