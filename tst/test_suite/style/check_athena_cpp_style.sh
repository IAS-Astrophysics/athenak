#!/usr/bin/env bash

# SCRIPT: check_athena_cpp_style.sh
# AUTHOR: Kyle Gerard Felker - kfelker@princeton.edu
# DATE:   4/18/2018
# PURPOSE:  Wrapper script to ./cpplint.py application to check AthenaK src/ code
#           compliance with C++ style guildes. User's Python and/or Bash shell
#           implementation may not support recursive globbing of src/ subdirectories
#           and files, so this uses "find" cmd w/ non-POSIX Bash process substitution.
#
# USAGE:    ./check_athena_cpp_style.sh
#           Assumes this script is executed from ./tst/style/ with cpplint.py in
#           the same directory, and that CPPLINT.cfg is in root directory.
#           TODO: add explicit check of execution directory
#
# LOG:      Updated by @pdmullen on 1/12/2022 for use in AthenaK

# Obtain Google C++ Style Linter:
echo "Obtaining Google C++ Style cpplint.py test"
curl https://raw.githubusercontent.com/cpplint/cpplint/master/cpplint.py \
--output cpplint.py --silent

# Apply Google C++ Style Linter to all source code files at once:
echo "Starting Google C++ Style cpplint.py test"
# Use "python3 -u" to prevent buffering of sys.stdout,stderr.write()
# calls in cpplint.py and mix-up in Jenkins logs,
find ../../../src -type f \( -name "*.cpp" -o -name "*.hpp" \) \
-print | xargs python3 -u cpplint.py --filter=-build/include_subdir --counting=detailed
if [ $? -ne 0 ]; then echo "ERROR: C++ style errors found"; rm -f cpplint.py; exit 1; fi
rm -f cpplint.py
echo "End of Google C++ Style cpplint.py test"

# Begin custom AthenaK style rules and checks:
echo "Starting \t, closing brace, and #pragma test"
while read -r file
do
    echo "Checking $file...."
    # TYPE 1: may cause bugs, or introduces abhorrent style (e.g. mixing tabs and spaces).
    grep -n "$(printf '\t')" $file
    if [ $? -ne 1 ]; then echo "ERROR: Do not use \t tab characters"; exit 1; fi

    # TYPE 2: purely stylistic inconsistencies.
    # These errors would not cause any changes to code behavior if they were ignored,
    # but they may affect readability.
    grep -nri "}}" "$file" | grep -v "//"
    if [ $? -ne 1 ]; then echo "ERROR: Use single closing brace '}}' per line"; exit 1; fi

    # GNU Grep Extended Regex (ERE) syntax:
    grep -nrEi '^\s+#pragma' "$file"
    if [ $? -ne 1 ]; then echo "ERROR: Left justify any #pragma statements"; exit 1; fi

done < <(find ../../../src -type f \( -name "*.cpp" -o -name "*.hpp" \) -print)
echo "End of \t, closing brace, and #pragma test"

# Search src/ C++ source code for trailing whitespace errors
# (Google C++ Style Linter does not check for this,
# but flake8 via pycodestyle warning W291 will check *.py)
echo "Checking for trailing whitespace in src/"
find ../../../src -type f \( -name "*.cpp" -o -name "*.hpp*" \) \
-exec grep -n -E " +$" {} +
if [ $? -ne 1 ]; then echo "ERROR: Found C++ file(s) in src/ \
with trailing whitespace"; exit 1; fi
echo "End of trailing whitespace test"

# Check that all files in src/ have the correct, non-executable octal permission 644
# Git only tracks permission changes (when core.filemode=true) for the "user/owner"
# executable permissions bit, and ignores the user read/write and all "group" and "other",
# setting file modes to 100644 or 100755 (exec)
echo "Checking for correct file permissions in src/"
git ls-tree -r --full-tree HEAD src/ | awk '{print substr($1,4,5), $4}' | grep -v "644"
if [ $? -ne 1 ]; then echo "ERROR: Found C++ file(s) in src/ \
with executable permission"; exit 1; fi
echo "End of file permissions test"