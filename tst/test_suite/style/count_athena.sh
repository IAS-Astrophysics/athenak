#!/usr/bin/env bash

# SCRIPT: count_athena.sh
# AUTHOR: Kyle Gerard Felker - kfelker@princeton.edu
# DATE:   4/19/2018
#
# PURPOSE:  Count files in each major subdirectory of the Athena++ repo, and
#           sort each list by the number of lines. Excludes data files and
#           external library source code that is packaged with Athena++
#
# USAGE:    ./count_athena.sh
#           Assumes this script is executed from ./tst/scripts/style/
#
# LOG:      Updated by @pdmullen on 1/12/2022 for use in AthenaK

cd ../../../src
echo "Counting src/ files...."
git ls-files | xargs wc -l | sort -k1 -r

cd ../vis/
echo "Counting vis/ files...."
# Exclude vis/visit/ .xml files from count
git ls-files | grep -vE "__init__\.py" | xargs wc -l | sort -k1 -r

cd ../tst/
echo "Counting tst/ files...."
# Exclude Google C++ Style linter cpplint.py and __init__.py files from count
git ls-files | grep -vE "cpplint\.py|__init__\.py" | xargs wc -l | sort -k1 -r
