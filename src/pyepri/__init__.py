# temporary fix for FINUFFT issue #596: force OMP_NUM_THREADS = number of physical cores
#import os
#import psutil
#os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=False))

# read version from installed package
from importlib.metadata import version
__version__ = version("pyepri")
