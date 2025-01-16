#!/usr/bin/env bash

cd runs

echo "running preproc...\n"
./PreprocessCceWorldtube --input-file PreprocessCceWorldtube_test.yaml

echo "-------------------------"


echo "running CCE..."
./CharacteristicExtract --input-file cce_par.yaml

