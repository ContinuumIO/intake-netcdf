#!/bin/bash
set -e # exit on error

echo "Installing dependencies."
conda install -c conda-forge conda-build conda-verify jinja2 intake>=0.4.1 xarray>=0.11.2 zarr dask
conda list

echo "Building conda package."
conda build -c conda-forge ./conda

# If tagged, upload package to main channel, otherwise, run tests
if [ -n "$TRAVIS_TAG" ]; then
    echo "Uploading conda package."
    anaconda -t ${ANACONDA_TOKEN} upload -u intake --force `conda build --output ./conda`
fi
