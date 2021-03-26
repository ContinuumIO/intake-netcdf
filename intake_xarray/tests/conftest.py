# -*- coding: utf-8 -*-

import posixpath
import pytest
import shutil
import tempfile
import xarray as xr

from intake_xarray.netcdf import NetCDFSource
from intake_xarray.xzarr import ZarrSource

TEST_DATA_DIR = 'data'
TEST_DATA = 'example_1.nc'

here = posixpath.dirname(__file__)
TEST_URLPATH = posixpath.join(here, TEST_DATA_DIR, TEST_DATA)


@pytest.fixture
def netcdf_source():
    return NetCDFSource(TEST_URLPATH, {})


@pytest.fixture
def dataset():
    return xr.open_dataset(TEST_URLPATH)


@pytest.fixture(scope='module')
def zarr_source():
    pytest.importorskip('zarr')
    try:
        tdir = tempfile.mkdtemp()
        data = xr.open_dataset(TEST_URLPATH)
        data.to_zarr(tdir)
        yield ZarrSource(tdir, chunks={"level": 4})
    finally:
        shutil.rmtree(tdir)


@pytest.fixture
def mock_import_xarray():
    import sys
    import xarray
    try:
        sys.modules['xarray'] = None
        yield
    finally:
        sys.modules['xarray'] = xarray
