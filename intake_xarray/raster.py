import xarray as xr
import numpy as np
from intake.source.utils import path_to_glob, path_to_pattern, reverse_format
from .base import DataSourceMixin, Schema

import glob


class RasterIOSource(DataSourceMixin):
    """Open a xarray dataset via RasterIO.

    This creates an xarray.array, not a dataset (i.e., there is exactly one
    variable).

    See https://rasterio.readthedocs.io/en/latest/ for the file formats
    supported, particularly GeoTIFF, and
    http://xarray.pydata.org/en/stable/generated/xarray.open_rasterio.html#xarray.open_rasterio
    for possible extra arguments

    Parameters
    ----------
    urlpath: str, location of data
        May be a local path, or remote path if including a protocol specifier
        such as ``'s3://'``. May include glob wildcards or format pattern strings.
        Must be a format supported by rasterIO (normally GeoTiff).
        Some examples:
            - ``{{ CATALOG_DIR }}data/RGB.tif``
            - ``s3://data/*.tif``
            - ``s3://data/landsat8_band{band}.tif``
            - ``s3://data/{location}/landsat8_band{band}.tif``
            - ``{{ CATALOG_DIR }}data/landsat8_{start_date:%Y%m%d}_band{band}.csv``
    chunks: int or dict
        Chunks is used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    path_as_pattern: bool, optional
        Whether to treat the path as a pattern (ie. ``data_{field}.tiff``)
        and create new coodinates in the output corresponding to pattern
        fields. Default is True.
    """
    name = 'rasterio'

    def __init__(self, urlpath, chunks, concat_dim, xarray_kwargs=None,
                 metadata=None, path_as_pattern=True, **kwargs):
        self.urlpath = path_to_glob(urlpath) if path_as_pattern else urlpath
        if path_as_pattern and self.urlpath != urlpath:
            self.pattern = path_to_pattern(urlpath, metadata)
        else:
            self.pattern = None
        self.chunks = chunks
        self.dim = concat_dim
        self._kwargs = xarray_kwargs or {}
        self._ds = None
        super(RasterIOSource, self).__init__(metadata=metadata)

    def _open_files(self, files):
        das = []
        for f in files:
            da = xr.open_rasterio(f, chunks=self.chunks, **self._kwargs)
            dim_shape = da.sizes.get(self.dim, 1)

            coords = {}
            if self.pattern is not None:
                for k, v in reverse_format(self.pattern, f).items():
                    coords[k] = xr.DataArray(
                        np.full(dim_shape, v), dims=self.dim)
            das.append(da.assign_coords(**coords))

        return xr.concat(das, dim=self.dim)

    def _open_dataset(self):
        if '*' in self.urlpath:
            files = sorted(glob.glob(self.urlpath))
            if len(files) == 0:
                raise Exception("No files found at {}".format(self.urlpath))
            self._ds = self._open_files(files)
        elif isinstance(self.urlpath, list):
            self._ds = self._open_files(self.urlpath)
        else:
            self._ds = xr.open_rasterio(self.urlpath, chunks=self.chunks,
                                        **self._kwargs)

    def _get_schema(self):
        """Make schema object, which embeds xarray object and some details"""
        from .xarray_container import serialize_zarr_ds
        import msgpack

        self.urlpath, *_ = self._get_cache(self.urlpath)

        if self._ds is None:
            self._open_dataset()

            ds2 = xr.Dataset({'raster': self._ds})
            metadata = {
                'dims': dict(ds2.dims),
                'data_vars': {k: list(ds2[k].coords)
                              for k in ds2.data_vars.keys()},
                'coords': tuple(ds2.coords.keys()),
                'internal': serialize_zarr_ds(ds2),
                'array': 'raster'
            }
            for k, v in self._ds.attrs.items():
                try:
                    msgpack.packb(v)
                    metadata[k] = v
                except TypeError:
                    pass
            self._schema = Schema(
                datashape=None,
                dtype=str(self._ds.dtype),
                shape=self._ds.shape,
                npartitions=self._ds.data.npartitions,
                extra_metadata=metadata)

        return self._schema
