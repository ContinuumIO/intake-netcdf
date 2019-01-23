
from intake.source.base import PatternMixin
from intake.source.utils import reverse_formats
from .base import DataSourceMixin, Schema


def _coerce_shape(array, shape):
    """ Trim or pad array to match desired shape"""
    import numpy as np

    if len(target_shape) != 2:
        raise ValueError('coerce_shape must be an iterable of len 2')

    target_shape = shape
    actual_shape = array.shape
    ndims = len(actual_shape)

    if actual_shape[:2] == target_shape:
        # no trimming or padding needed
        return array

    # do any necessary trimming first
    for i, (a, t) in enumerate(zip(actual_shape[:2], target_shape)):
        if a > t:
            if i == 0:
                if ndims == 2:
                    array = [:t, :]
                else:
                    array = [:t, :, :]
            else:
                if ndims == 2:
                    array = [:, :t]
                else:
                    array = [:, :t, :]

    if array.shape[:2] == target_shape:
        # only needed trimming
        return array

    # create array of zeros and fill with trimmed value array
    new_array = np.zeros(target_shape)

    if ndims == 2:
        new_array[:array.shape[0], :array.shape[1]] = array
    else:
        new_array[:array.shape[0], :array.shape[1], :] = array

    return new_array


def _dask_imread(files, imread=None, preprocess=None, coerce_shape=None):
    """ Read a stack of images into a dask array """
    from dask.array import Array
    from dask.base import tokenize

    if not imread:
        from skimage.io import imread

    def _imread(open_file):
        with open_file as f:
            return imread(f)

    def add_leading_dimension(x):
        return x[None, ...]

    filenames = [f.path for f in files]

    name = 'imread-%s' % tokenize(filenames)

    with files[0] as f:
        sample = imread(f)
    if preprocess:
        sample = preprocess(sample)

    keys = [(name, i) + (0,) * len(sample.shape)
            for i in range(len(files))]

    if preprocess:
        values = [(add_leading_dimension, (preprocess, (_imread, f)))
                  for f in files]
    else:
        values = [(add_leading_dimension, (_imread, f))
                  for f in files]
    dsk = dict(zip(keys, values))

    chunks = ((1, ) * len(files), ) + tuple((d, ) for d in sample.shape)

    return Array(dsk, name, chunks, sample.dtype)


def reader(file, chunks, imread=None, preprocess=None, **kwargs):
    """Read a file object and output an dask xarray object

    NOTE: inspired by dask.array.image.imread but altering the input to accept
    a just one file object.

    Parameters
    ----------
    file : OpenFile
        File object
    chunks : int or dict
        Chunks is used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    imread : function (optional)
        Optionally provide custom imread function.
        Function should expect a file object and produce a numpy array.
        Defaults to ``skimage.io.imread``.
    preprocess : function (optional)
        Optionally provide custom function to preprocess the image.
        Function should expect a numpy array for a single image.

    Returns
    -------
    Dask xarray.DataArray of all the image. Treated as one chunk unless
    chunks kwarg is specified.
    """
    import numpy as np
    from xarray import DataArray

    if not imread:
        from skimage.io import imread

    with file as f:
        array = imread(f)
    if preprocess:
        array = preprocess(array)

    ny, nx = array.shape[:2]
    coords = {'y': np.arange(ny),
              'x': np.arange(nx)}
    dims = ('y', 'x')

    if len(array.shape) == 3:
        nchannel = array.shape[2]
        coords['channel'] = np.arange(nchannel)
        dims += ('channel',)

    return DataArray(array, coords=coords, dims=dims).chunk(chunks=chunks)


def multireader(files, chunks, concat_dim, **kwargs):
    """Read a stack of images into a dask xarray object

    NOTE: copied from dask.array.image.imread but altering the input to accept
    a list of file objects.

    Parameters
    ----------
    files : iter
        List of file objects
    chunks : int or dict
        Chunks is used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    concat_dim : str or iterable
        Dimension over which to concatenate. If iterable, all fields must be
        part of the the pattern.
    imread : function (optional)
        Optionally provide custom imread function.
        Function should expect a file object and produce a numpy array.
        Defaults to ``skimage.io.imread``.
    preprocess : function (optional)
        Optionally provide custom function to preprocess the image.
        Function should expect a numpy array for a single image.

    Returns
    -------
    Dask xarray.DataArray of all images stacked along the first dimension.
    All images will be treated as individual chunks unless
    chunks kwarg is specified.
    """
    import numpy as np
    from xarray import DataArray

    dask_array = _dask_imread(files, **kwargs)

    ny, nx = dask_array.shape[1:3]
    coords = {'y': np.arange(ny),
              'x': np.arange(nx)}
    if isinstance(concat_dim, list):
        dims = ('dim_0', 'y', 'x')
    else:
        dims = (concat_dim, 'y', 'x')

    if len(dask_array.shape) == 4:
        nchannel = dask_array.shape[3]
        coords['channel'] = np.arange(nchannel)
        dims += ('channel',)

    return DataArray(dask_array, coords=coords, dims=dims).chunk(chunks=chunks)


class ImageSource(DataSourceMixin, PatternMixin):
    """Open a xarray dataset from image files.

    This creates an xarray.DataArray or an xarray.Dataset.
    See http://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread
    for the file formats supported.

    NOTE: Although ``skimage.io.imread`` is used by default, any reader
    function which accepts a file object and outputs a numpy array can be
    used instead.

    Parameters
    ----------
    urlpath : str or iterable, location of data
        May be a local path, or remote path if including a protocol specifier
        such as ``'s3://'``. May include glob wildcards or format pattern
        strings. Must be a format supported by ``skimage.io.imread`` or
        user-supplied ``imread``. Some examples:
            - ``{{ CATALOG_DIR }}/data/RGB.tif``
            - ``s3://data/*.jpeg``
            - ``https://example.com/image.png`
            - ``s3://data/Images/{{ landuse }}/{{ '%02d' % id }}.tif``
    chunks : int or dict
        Chunks is used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    path_as_pattern : bool or str, optional
        Whether to treat the path as a pattern (ie. ``data_{field}.tif``)
        and create new coodinates in the output corresponding to pattern
        fields. If str, is treated as pattern to match on. Default is True.
    concat_dim : str or iterable
        Dimension over which to concatenate. If iterable, all fields must be
        part of the the pattern.
    imread : function (optional)
        Optionally provide custom imread function.
        Function should expect a file object and produce a numpy array.
        Defaults to ``skimage.io.imread``.
    preprocess : function (optional)
        Optionally provide custom function to preprocess the image.
        Function should expect a numpy array for a single image and return
        a numpy array.
    """
    name = 'xarray_image'

    def __init__(self, urlpath, chunks=None, concat_dim='concat_dim',
                 metadata=None, path_as_pattern=True,
                 storage_options=None, **kwargs):
        self.path_as_pattern = path_as_pattern
        self.urlpath = urlpath
        self.chunks = chunks
        self.concat_dim = concat_dim
        self.storage_options = storage_options or {}
        self._kwargs = kwargs
        self._ds = None
        super(ImageSource, self).__init__(metadata=metadata)

    def _open_files(self, files):
        """
        This function is called when the data source refers to more
        than one file either as a list or a glob. It sets up the
        dask graph for opening the files.

        Parameters
        ----------
        files : iter
            List of file objects
        """
        import pandas as pd
        from xarray import DataArray

        out = multireader(files, self.chunks, self.concat_dim, **self._kwargs)
        if not self.pattern:
            return out

        coords = {}
        filenames = [f.path for f in files]
        field_values = reverse_formats(self.pattern, filenames)

        if isinstance(self.concat_dim, list):
            if not set(field_values.keys()).issuperset(set(self.concat_dim)):
                raise KeyError('All concat_dims should be in pattern.')
            index = pd.MultiIndex.from_tuples(
                zip(*(field_values[dim] for dim in self.concat_dim)),
                names=self.concat_dim)
            coords = {
                k: DataArray(v, dims=('dim_0'))
                for k, v in field_values.items() if k not in self.concat_dim
            }
            out = (out.assign_coords(dim_0=index, **coords)  # use the index
                      .unstack().chunk(self.chunks))  # unstack along new index
            return out.transpose(*self.concat_dim,  # reorder dims
                                 *filter(lambda x: x not in self.concat_dim,
                                         out.dims))
        else:
            coords = {
                k: DataArray(v, dims=self.concat_dim)
                for k, v in field_values.items()
            }
            return out.assign_coords(**coords).chunk(self.chunks)

    def _open_dataset(self):
        """
        Main entry function that finds a set of files and passes them to the
        reader.
        """
        from dask.bytes import open_files

        files = open_files(self.urlpath, **self.storage_options)
        if len(files) == 0:
            raise Exception("No files found at {}".format(self.urlpath))
        if len(files) == 1:
            self._ds = reader(files[0], self.chunks, **self._kwargs)
        else:
            self._ds = self._open_files(files)

    def _get_schema(self):
        """Make schema object, which embeds xarray object and some details"""
        import xarray as xr
        import msgpack
        from .xarray_container import serialize_zarr_ds

        self.urlpath, *_ = self._get_cache(self.urlpath)

        if self._ds is None:
            self._open_dataset()

            # convert to dataset for serialization
            ds2 = xr.Dataset({'raster': self._ds})
            metadata = {
                'dims': dict(ds2.dims),
                'data_vars': {k: list(ds2[k].coords)
                              for k in ds2.data_vars.keys()},
                'coords': tuple(ds2.coords.keys()),
                'array': 'raster'
            }
            if getattr(self, 'on_server', False):
                metadata['internal'] = serialize_zarr_ds(ds2)
            for k, v in self._ds.attrs.items():
                try:
                    # ensure only sending serializable attrs from remote
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
