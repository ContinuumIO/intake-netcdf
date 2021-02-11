import json

import dask.array

from . import __version__
from intake.source.base import DataSource, Schema


class DataSourceMixin(DataSource):
    """Common behaviours for plugins in this repo"""
    version = __version__
    container = 'xarray'
    partition_access = True

    def _get_schema(self):
        """Make schema object, which embeds xarray object and some details"""
        from .xarray_container import serialize_zarr_ds

        self.urlpath = self._get_cache(self.urlpath)[0]

        if self._ds is None:
            self._open_dataset()

            metadata = {
                'dims': dict(self._ds.dims),
                'data_vars': {k: list(self._ds[k].coords)
                              for k in self._ds.data_vars.keys()},
                'coords': tuple(self._ds.coords.keys()),
            }
            if getattr(self, 'on_server', False):
                serialized = serialize_zarr_ds(self._ds)
                metadata['internal'] = serialized
                # The zarr serialization imposes a certain chunking, which will
                # be reflected in the xarray.Dataset object constructed on the
                # client side. We need to use that same chunking here on the
                # server side. Extract it from the serialized zarr metadata.
                self._chunks = {k.rsplit('/', 1)[0]: json.loads(v.decode())['chunks']
                                for k, v in serialized.items() if k.endswith('/.zarray')}
            metadata.update(self._ds.attrs)
            self._schema = Schema(
                datashape=None,
                dtype=None,
                shape=None,
                npartitions=None,
                extra_metadata=metadata)
        return self._schema

    def read(self):
        """Return a version of the xarray with all the data in memory"""
        self._load_metadata()
        return self._ds.load()

    def read_chunked(self):
        """Return xarray object (which will have chunks)"""
        self._load_metadata()
        return self._ds

    def read_partition(self, i):
        """Fetch one chunk of data at tuple index i
        """
        import numpy as np
        self._load_metadata()
        if not isinstance(i, (tuple, list)):
            raise TypeError('For Xarray sources, must specify partition as '
                            'tuple')
        variable, *part = i
        part = tuple(part)
        if hasattr(self._ds, 'variables') or variable in self._ds.coords:
            arr = self._ds[variable].data
        else:
            arr = self._ds.data
        if isinstance(arr, np.ndarray):
            # Make a dask.array so that we can return the appropriate block.
            arr = dask.array.from_array(arr, chunks=self._chunks[variable])
        return arr.blocks[part].compute()

    def to_dask(self):
        """Return xarray object where variables are dask arrays"""
        return self.read_chunked()

    def close(self):
        """Delete open file from memory"""
        self._ds = None
        self._schema = None
