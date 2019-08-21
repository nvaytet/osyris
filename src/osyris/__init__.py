from .load_ramses import RamsesData
from .plot_histogram import plot_histogram
from .plot_slice import plot_slice
from .plot_column_density import plot_column_density
from .plot_3d import plot_volume
from . import ism_physics
from .vtk import to_vtk
from .interpolate import interpolate

# Import the config from "/home/user/.osyris/config if it exists, if not load
# the default
from os.path import expanduser
import sys
sys.path.append(expanduser("~") + "/.osyris")
try:
    import config
except ModuleNotFoundError:
    from . import config
