"""
Spade-Club-Diamond shapes (paramagnetic inkjet)
===============================================

2D image reconstruction of spade/club/diamond shapes printed on paper
using a paramagnetic inkjet (see :ref:`here
<dataset_scd-inkjet-20141204>` for the sample and the dataset
description).

"""

# -------------- #
# Import modules #
# -------------- #
import numpy as np
import pyepri.backends as backends
import pyepri.datasets as datasets
import pyepri.displayers as displayers
import pyepri.processing as processing

# -------------- #
# Create backend #
# -------------- #
#
# We create a numpy backend here because it should be available on
# your system (as a mandatory dependency of the PyEPRI
# package).
#
# However, you can also try another backend (if available on your
# system) by uncommenting one of the following commands. Depending on
# your system, using another backend may drastically reduce the
# computation time.
#
backend = backends.create_numpy_backend()
#backend = backends.create_torch_backend('cpu') # uncomment here for torch-cpu backend
#backend = backends.create_cupy_backend() # uncomment here for cupy backend
#backend = backends.create_torch_backend('cuda') # uncomment here for torch-gpu backend

#--------------#
# Load dataset #
#--------------#
#
# We load the ``scd-inkjet-20141204`` dataset (files are embedded with
# the PyEPRI package) in ``float32`` precision (you can also select
# ``float64`` precision by setting ``dtype='float64'``).
#
dtype = 'float32'
path_proj = datasets.get_path('scd-inkjet-20141204-proj.npy')
path_B = datasets.get_path('scd-inkjet-20141204-B.npy')
path_h = datasets.get_path('scd-inkjet-20141204-h.npy')
path_fgrad = datasets.get_path('scd-inkjet-20141204-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# --------------------------------------- #
# Perform monosource image reconstruction #
# --------------------------------------- #

# set mandatory parameters
delta = 1e-2 # sampling step (cm)
out_shape = (150, 300) # output image shape
xgrid = (-(out_shape[1]//2) + backend.arange(out_shape[1], dtype=dtype)) * delta
ygrid = (-(out_shape[0]//2) + backend.arange(out_shape[0], dtype=dtype)) * delta
if backend.lib.__name__ in ['numpy','cupy']: 
    interp1 = lambda xp, fp, x : backend.lib.interp(x, xp.flatten(), fp.flatten(), left=0., right=0.)
else:
    import torchinterp1d
    interp1 = lambda xp, fp, x : torchinterp1d.interp1d(xp, fp, x) * ((x >= xp[0]) & (x <= xp[-1]))

# set optional parameters
verbose = False # disable console verbose mode
video = True # enable video display
Ndisplay = 5 # refresh display rate (iteration per refresh)
frequency_cutoff = .1 # we will apply a frequency cutoff to the
                      # projections (keep only 10% of the frequency
                      # coefficients) during the filter-backprojection
                      # process in order to avoid dramatic noise
                      # amplification caused by the Ram-Lak filter

# customize 2D image displayer (optional, for video mode only)
xgrid_np = backend.to_numpy(xgrid) # X-axis spatial grid (numpy view)
ygrid_np = backend.to_numpy(ygrid) # Y-axis spatial grid (numpy view)
grids_np = (ygrid_np, xgrid_np) # spatial sampling grids
unit = 'cm' # provide length unit associated to the grids
display_labels = True # display axes labels within subplots
adjust_dynamic = True # maximize displayed dynamic at each refresh
boundaries = 'same' #  give all subplots the same axes boundaries
                    #  (ensure same pixel size for each displayed
                    #  slice)
displayFcn = lambda u : np.maximum(u, 0.) # threshold display
                                          # (negative values are
                                          # displayed as 0)
displayer = displayers.create_2d_displayer(nsrc=1,
                                           displayFcn=displayFcn,
                                           units=unit,
                                           adjust_dynamic=adjust_dynamic,
                                           display_labels=display_labels,
                                           boundaries=boundaries,
                                           grids=grids_np)
# run processing
out = processing.eprfbp2d(proj, fgrad, h, B, xgrid, ygrid, interp1,
                          backend=backend,
                          frequency_cutoff=frequency_cutoff,
                          video=video, Ndisplay=Ndisplay,
                          displayer=displayer)
