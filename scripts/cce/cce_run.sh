#!/usr/bin/env bash
set -x

if false
then
cd runs
echo "running preproc...\n"
./PreprocessCceWorldtube --input-file PreprocessCceWorldtube_bin.yaml
echo "-------------------------"
echo "running CCE..."
./CharacteristicExtract --input-file cce_par_bin.yaml

# h5 test
elif false
then
cd runs
echo "running preproc...\n"
./PreprocessCceWorldtube --input-file PreprocessCceWorldtube_h5.yaml
echo "-------------------------"
echo "running CCE..."
./CharacteristicExtract --input-file cce_par_h5.yaml

# q2
elif true
then
cd runs
echo "running preproc...\n"
./PreprocessCceWorldtube --input-file PreprocessCceWorldtube_q2_h5.yaml
echo "-------------------------"
echo "running CCE..."
./CharacteristicExtract --input-file cce_par_q2_h5.yaml

# q2 dev
elif false
then
cd runs
echo "running preproc...\n"
./PreprocessCceWorldtube --input-file PreprocessCceWorldtube_q2_dev_h5.yaml
echo "-------------------------"
echo "running CCE..."
./CharacteristicExtract --input-file cce_par_q2_dev_h5.yaml
fi

