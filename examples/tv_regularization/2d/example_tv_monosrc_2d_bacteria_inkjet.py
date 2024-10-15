"""
Paramagnetic ink (word "Bacteria" printed on a paper sheet)
===========================================================
 
2D image reconstruction of the word "Bacteria" printed on a paper
sheet using a paramagnetic inkjet (see :ref:`here
<dataset_bacteria-inkjet-20100709>` for the sample and the dataset
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
# We load the ``bacteria-inkjet-20100709`` dataset (files are embedded
# with the PyEPRI package) in ``float32`` precision (you can also
# select ``float64`` precision by setting ``dtype='float64'``).
#
dtype = 'float32'
path_proj = datasets.get_path('bacteria-inkjet-20100709-proj.npy')
path_B = datasets.get_path('bacteria-inkjet-20100709-B.npy')
path_h = datasets.get_path('bacteria-inkjet-20100709-h.npy')
path_fgrad = datasets.get_path('bacteria-inkjet-20100709-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# --------------------------------------- #
# Perform monosource image reconstruction #
# --------------------------------------- #

# set mandatory parameters
out_shape = (100, 250) # output image shape
delta = 1e-2 # sampling step in cm
lbda = 5. # normalized regularity parameter

# set optional parameters
nitermax = 1000 # maximal number of iterations
verbose = False # disable console verbose mode
video = True # enable video display
Ndisplay = 20 # refresh display rate (iteration per refresh)
eval_energy = False # disable TV-regularized least-square energy
                    # evaluation each Ndisplay iteration

# customize 3D image displayer (optional, for video mode only)
xgrid = (-(out_shape[1]//2) + np.arange(out_shape[1])) * delta
ygrid = (-(out_shape[0]//2) + np.arange(out_shape[0])) * delta
grids = (ygrid, xgrid) # provide spatial sampling grids
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
                                           grids=grids)

# run processing
out = processing.tv_monosrc(proj, B, fgrad, delta, h, lbda, out_shape,
                            backend=backend, tol=1e-5,
                            nitermax=nitermax, verbose=verbose,
                            eval_energy=eval_energy, video=video,
                            Ndisplay=Ndisplay, displayer=displayer)
