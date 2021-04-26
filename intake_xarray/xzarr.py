from .base import DataSourceMixin


class ZarrSource(DataSourceMixin):
    """Open a xarray dataset.

    Parameters
    ----------
    urlpath: str
        Path to source. This can be a local directory or a remote data
        service (i.e., with a protocol specifier like ``'s3://``).
    storage_options: dict
        Parameters passed to the backend file-system
    kwargs:
        Further parameters are passed to xr.open_zarr
    """
    name = 'zarr'

    def __init__(self, urlpath, engine='zarr', chunks="auto",
                 concat_dim='concat_dim', xarray_kwargs=None,
                 storage_options=None, metadata=None, **kwargs):
        self.urlpath = urlpath
        self.engine = engine
        self.chunks = chunks
        self.concat_dim = concat_dim
        self.storage_options = storage_options or {}
        self._kwargs = xarray_kwargs or {}
        self._ds = None
        super(ZarrSource, self).__init__(metadata=metadata, **kwargs)

    def _open_dataset(self):
        import xarray as xr
        kwargs = self._kwargs
        if "*" in self.urlpath or isinstance(self.urlpath, list):
            if self.engine not in kwargs.keys():
                kwargs.update(engine=self.engine)
            _open_dataset = xr.open_mfdataset
            if 'concat_dim' not in kwargs.keys():
                kwargs.update(concat_dim=self.concat_dim)
        else:
            _open_dataset = xr.open_dataset
        self._ds = _open_dataset(self.urlpath, chunks=self.chunks, **kwargs)

    def close(self):
        super(ZarrSource, self).close()
        self._fs = None
        self._mapper = None
