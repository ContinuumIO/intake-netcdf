# -*- coding: utf-8 -*-
from .base import XarraySource


class NetCDFSource(XarraySource):
    """Open a netcdf file.

    This creates an xarray.Dataset.

    See http://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html
    for the file formats supported and possible extra arguments.

    NOTE: When reading from OPeNDAP URLs do not set the ``chunks`` option to
    use provided default chunking. If your OPeNDAP implementation requires
    authentication, use the ``opendap`` driver instead.

    Parameters
    ----------
    urlpath : str
        Path to source file. May include glob "*" characters or format
        pattern strings. Accepts un-authenticated OPeNDAP urls without caching.
        Other remote sources can be used but only with caching.
        Must be a format supported by ``xr.open_dataset``.

        Some examples:
            - ``s3://data/*.nc``
            - ``http://thredds.ucar.edu/thredds/dodsC/grib/FNMOC/WW3/Global_1p0deg/Best``
            - ``https://github.com/pydata/xarray-data/blob/master/air_temperature.nc?raw=true``
            - ``{{ CATALOG_DIR }}/data/{site_id}_data.nc``
    xarray_kwargs : dict, optional
        Any further arguments to pass to ``xr.open_dataset``.
    """
    name = 'netcdf'
    __doc__ += XarraySource.__inheritted_parameters_doc__

    def __init__(self, urlpath, chunks=None, reader=None, multireader=None, **kwargs):
        from xarray import open_dataset, open_mfdataset

        super(NetCDFSource, self).__init__(urlpath, chunks, **kwargs)
        self.reader = reader or open_dataset
        self.multireader = multireader or open_mfdataset
