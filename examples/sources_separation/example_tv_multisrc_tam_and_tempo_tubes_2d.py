"""
Separation of TAM & TEMPO tubes (real 2D dataset)
=================================================

Simultaneous image reconstruction and separation of sample made of one
tube filled with a solution of TAM and one tube filled with a solution
of TEMPO (see :ref:`here <dataset_tam-and-tempo-tubes-2d-20210609>`
for the sample and the dataset description).

.. figure:: ../../_static/tam-and-tempo-tubes-3d-20210609-pic.png
  :width: 48%
  :align: center
  :alt: TAM & TEMPO in two distinct tubes

|

"""

# %%
# Load & display dataset
# ----------------------

# -------------- #
# Import modules #
# -------------- #
import math
import matplotlib.pyplot as plt
import numpy as np
import pyepri.backends as backends
import pyepri.datasets as datasets
import pyepri.displayers as displayers
import pyepri.monosrc as monosrc
import pyepri.processing as processing
plt.ion()

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

#----------------------------------#
# load dataset (real measurements) #
#----------------------------------#

# retrieve paths towards of the different files comprised in the dataset
path_B = datasets.get_path('tam-and-tempo-tubes-2d-20210609-B.npy')
path_htam = datasets.get_path('tam-and-tempo-tubes-2d-20210609-htam.npy')
path_htempo = datasets.get_path('tam-and-tempo-tubes-2d-20210609-htempo.npy')
path_hmixt = datasets.get_path('tam-and-tempo-tubes-2d-20210609-hmixt.npy')
path_fgrad = datasets.get_path('tam-and-tempo-tubes-2d-20210609-fgrad.npy')
path_proj = datasets.get_path('tam-and-tempo-tubes-2d-20210609-proj.npy')

# load the dataset
dtype = 'float32'
B = backend.from_numpy(np.load(path_B).astype(dtype))
h_tam = backend.from_numpy(np.load(path_htam).astype(dtype))
h_tempo = backend.from_numpy(np.load(path_htempo).astype(dtype))
h_mixture = backend.from_numpy(np.load(path_hmixt).astype(dtype))
fg = backend.from_numpy(np.load(path_fgrad).astype(dtype))
proj_mixture = backend.from_numpy(np.load(path_proj).astype(dtype))

#---------------------#
# display projections #
#---------------------#

# prepare display
plt.figure(figsize=(10, 4))
theta = backend.arctan2(fg[1], fg[0])
proj_extent = [t.item() for t in (B[0], B[-1], theta[0]*180./math.pi, theta[-1]*180./math.pi)]

# display reference spectrum of the sample (contains one tube of TAM
# and one tube of TEMPO)
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h_mixture))
plt.xlabel("B: homogeneous magnetic field intensity (G)")
plt.ylabel("measurement (arb. units)")
plt.title("Reference spectrum of the sample")

# display measured projections
plt.subplot(1, 2, 2)
plt.imshow(backend.to_numpy(proj_mixture), extent=proj_extent, aspect='auto')
plt.title("Measured projections")
plt.xlabel("B: homogeneous magnetic field intensity (G)")
_ = plt.ylabel("field gradient orientation (degree)")

# %%
# Perform source separation
# -------------------------
#
# Now let us perform the source separation, that is, the
# reconstruction of one image of the tube of TAM and one image of the
# tube of TEMPO.

# set reconstruction parameters
tam_shape = (100, 100) # required size for the output TAM source
tempo_shape = (100, 100) # required size for the output TEMPO source
out_shape = (tam_shape, tempo_shape) # output image shape
delta = 3e-2 # sampling step of the reconstruction (cm)
lbda = 5 # normalized regularity parameter
proj = (proj_mixture,) # list of input experiments (here only one experiment)
h = ((h_tam, h_tempo),) # list of source spectra associated to each experiment
fgrad = (fg,) # list of field gradient vectors associated to each experiment

# set optional parameters
nitermax = 5000 # maximal number of iterations
verbose = False # disable console verbose mode
video = True # enable video display
Ndisplay = 20 # refresh display rate (iteration per refresh)
eval_energy = False # disable TV-regularized least-square energy
                    # evaluation each Ndisplay iteration

# customize 2D multi-sources image displayer: also optional, customize
# display (when video mode is enabled)
tam_shape = out_shape[0]
tempo_shape = out_shape[1]
xgrid_tam = (-(tam_shape[1]//2) + np.arange(tam_shape[1])) * delta
ygrid_tam = (-(tam_shape[0]//2) + np.arange(tam_shape[0])) * delta
xgrid_tempo = (-(tempo_shape[1]//2) + np.arange(tempo_shape[1])) * delta
ygrid_tempo = (-(tempo_shape[0]//2) + np.arange(tempo_shape[0])) * delta
grid_tam = (ygrid_tam, xgrid_tam)
grid_tempo = (ygrid_tempo, xgrid_tempo)
grids = (grid_tam, grid_tempo) # provide spatial sampling grids for each source
unit = 'cm' # provide length unit associated to the grids
display_labels = True # display axes labels within subplots
adjust_dynamic = True # maximize displayed dynamic at each refresh
boundaries = 'same' #  give all subplots the same axes boundaries
                    #  (ensure same pixel size for each displayed
                    #  slice)
figsize = (10., 4.4) # size of the figure to be displayed
src_labels = ('TAM', 'TEMPO') # source labels (to be included into suptitles)
displayer = displayers.create_2d_displayer(nsrc=2, units=unit,
                                           figsize=figsize,
                                           adjust_dynamic=adjust_dynamic,
                                           display_labels=display_labels,
                                           boundaries=boundaries,
                                           grids=grids,
                                           src_labels=src_labels)
                    
# run processing
out = processing.tv_multisrc(proj, B, fgrad, delta, h, lbda,
                             out_shape, backend=backend, tol=1e-5,
                             nitermax=nitermax,
                             eval_energy=eval_energy, video=video,
                             verbose=verbose, Ndisplay=Ndisplay,
                             displayer=displayer)	   
