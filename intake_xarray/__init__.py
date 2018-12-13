from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .base import XarraySource
from .image import ImageSource
from .netcdf import NetCDFSource
from .opendap import OpenDapSource
from .raster import RasterIOSource
from .xzarr import ZarrSource
from .xarray_container import RemoteXarray

import intake.container

intake.registry['remote-xarray'] = RemoteXarray
intake.container.container_map['xarray'] = RemoteXarray
