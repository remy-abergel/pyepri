import os.path

# retrieve absolute path towards the 'datasets' directory of this
# packages (this directory contains the embedded datasets used in the
# reproducible demonstration examples of the documentation)
__PYEPRI_DATA_PATH__ = os.path.dirname(os.path.realpath(__file__))

def get_path(filename:str) -> str:
    """Compute absolute path towards a file comprised in the 'datasets' directory of the PyEPRI package (embedded datasets only)
    
    Example: 
    --------
    
    >>> import pyepri.datasets
    >>> path_proj = pyepri.datasets.get_path('phalanx-20220203-proj.npy')
    >>> path_fgrad = pyepri.datasets.get_path('phalanx-20220203-fgrad.npy')
    >>> print(path_proj) # path towards the embedded file 'phalanx-20220203-proj.npy'
    >>> print(path_fgrad) # path towards the embedded file 'phalanx-20220203-fgrad.npy'
    
    """
    path = os.path.join(__PYEPRI_DATA_PATH__, filename)
    if not os.path.isfile(path):
        raise RuntimeError("file '%s' does not exist." % path)
    return path
