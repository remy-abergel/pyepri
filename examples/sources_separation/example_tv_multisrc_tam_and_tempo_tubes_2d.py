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

  Picture of the sample.

"""

# %%
# Import needed modules
# ---------------------

# sphinx_gallery_thumbnail_number = 1
import math # basic math functions
import numpy as np # for array manipulations
import matplotlib.pyplot as plt # tools for data visualization
import pyepri.backends as backends # to instanciate PyEPRI backends
import pyepri.datasets as datasets # to retrieve the path (on your own machine) of the demo dataset
import pyepri.displayers as displayers # tools for displaying images (with update along the computation)
import pyepri.processing as processing # tools for EPR image reconstruction
import pyepri.io as io # tools for loading EPR datasets (in BES3T or Python .PKL format)

# %%
# Create backend
# --------------
#
# We create a numpy backend here because it should be available on
# your system (as a mandatory dependency of the PyEPRI package). You
# can try another backend (if available on your system) by
# uncommenting the appropriate line below (using a GPU backend may
# drastically reduce the computation time).
#
backend = backends.create_numpy_backend() # default numpy backend (CPU)
#backend = backends.create_torch_backend('cpu') # uncomment here for torch-cpu backend (CPU)
#backend = backends.create_cupy_backend() # uncomment here for cupy backend (GPU)
#backend = backends.create_torch_backend('cuda') # uncomment here for torch-gpu backend (GPU)

# %%
# Load the input dataset
# ----------------------
#
# We load the ``tam-and-tempo-tubes-2d-20210609`` dataset (embedded
# with the PyEPRI package) in ``float32`` precision. Take a look to
# the comments for changing the precision to ``float64`` or replacing
# the embedded dataset by one of your own dataset. Note that to adapt
# this example to your own dataset, you will need to provide
# 
# * the sequence of measured projection of the sample (mixture of
#   multiple EPR species);
#
# * the reference spectrum of each EPR species present in the sample
#   (must be calculated beforehand, preferably extracted from the
#   reference spectrum of the mixture).
#

# set the path towards the measured projections (mixture of TAM & TEMPO)
path_proj = datasets.get_path('tam-and-tempo-tubes-2d-20210609-proj.pkl') # or use your own dataset, e.g., path_proj = '~/my_projections.DSC'

# set the paths towards the reference spectra of each EPR species
# present in the mixture (in this dataset, the reference spectra of
# the single species were fited from the ref. spectrum of the mixture)
path_htam = datasets.get_path('tam-and-tempo-tubes-2d-20210609-htam.pkl') # or use your own dataset, e.g., path_htam = '~/my_tam_spectrum.DSC'
path_htempo = datasets.get_path('tam-and-tempo-tubes-2d-20210609-htempo.pkl') # or use your own dataset, e.g., path_htempo = '~/my_tempo_spectrum.DSC'

# load the dataset in float32 precision
dtype = 'float32' # use 'float32' for single (32 bit) precision and 'float64' for double (64 bit) precision
dataset_proj = io.load(path_proj, backend=backend, dtype=dtype) # load the dataset containing the projections
dataset_htam = io.load(path_htam, backend=backend, dtype=dtype) # load the dataset containing the TAM reference spectrum
dataset_htempo = io.load(path_htempo, backend=backend, dtype=dtype) # load the dataset containing the TEMPO reference spectrum

# extract data from the loaded datasets
B = dataset_proj['B'] # B sampling nodes
proj_mixture = dataset_proj['DAT'] # projections data
fg = dataset_proj['FGRAD'] # field gradient vectors coordinates
h_tam = dataset_htam['DAT'] # reference spectrum of the TAM
h_tempo = dataset_htempo['DAT'] # reference spectrum of the TEMPO

# %%
# Display the input dataset
# -------------------------
#
# Before addressing the source separation problem, let us take a look
# at the measured reference spectrum of the sample (TAM + TEMPO
# mixture) as well as to the measured projections.
#

# prepare display
plt.figure(figsize=(10, 4))
theta = backend.arctan2(fg[1], fg[0])
proj_extent = [t.item() for t in (B[0], B[-1], theta[0]*180./math.pi, theta[-1]*180./math.pi)]

# display reference spectrum of the sample (contains one tube of TAM
# and one tube of TEMPO)
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h_tam + h_tempo))
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
#

# ----------------------------- #
# Set reconstruction parameters #
# ----------------------------- #
tam_shape = (100, 100) # required shape (number of pixels along each axe) for the output TAM source
tempo_shape = (100, 100) # required shape (number of pixels along each axe) for the output TEMPO source
out_shape = (tam_shape, tempo_shape) # output multisource image shape
delta = 3e-2 # sampling step in the same length unit as the provided field gradient coordinates (here cm)
lbda = 5 # normalized regularity parameter (arbitrary unit)
proj = (proj_mixture,) # list of input experiments (here only one experiment)
h = ((h_tam, h_tempo),) # list of source spectra associated to each experiment
fgrad = (fg,) # list of field gradient vectors associated to each experiment

# ----------------------- #
# Set optional parameters #
# ----------------------- #
nitermax = 5000 # maximal number of iterations
verbose = False # disable console verbose mode
video = True # enable video display
Ndisplay = 20 # refresh display rate (iteration per refresh)
eval_energy = False # disable TV-regularized least-square energy
                    # evaluation each Ndisplay iteration

# ---------------------------------------------------------- #
# Customize 2D multi-sources image displayer (optional, used #
# only when video=True above)                                #
# ---------------------------------------------------------- #
tam_shape = out_shape[0]
tempo_shape = out_shape[1]
xgrid_tam = (-(tam_shape[1]//2) + np.arange(tam_shape[1])) * delta
ygrid_tam = (-(tam_shape[0]//2) + np.arange(tam_shape[0])) * delta
xgrid_tempo = (-(tempo_shape[1]//2) + np.arange(tempo_shape[1])) * delta
ygrid_tempo = (-(tempo_shape[0]//2) + np.arange(tempo_shape[0])) * delta
grid_tam = (ygrid_tam, xgrid_tam)
grid_tempo = (ygrid_tempo, xgrid_tempo)
grids = (grid_tam, grid_tempo) # provide spatial sampling grids for each source
unit = 'cm' # provide length unit associated to the grids (used to label the image axes)
display_labels = True # display axes labels within subplots
adjust_dynamic = True # maximize displayed dynamic at each refresh
boundaries = 'same' # give all subplots the same axes boundaries (ensure same pixel size for
                    # each displayed slice)
figsize = (10., 4.4) # size of the figure to be displayed
src_labels = ('TAM', 'TEMPO') # source labels (to be included into suptitles)
displayer = displayers.create_2d_displayer(nsrc=2,
                                           units=unit,
                                           figsize=figsize,
                                           adjust_dynamic=adjust_dynamic,
                                           display_labels=display_labels,
                                           boundaries=boundaries,
                                           grids=grids,
                                           src_labels=src_labels)

# --------------------------------------------------------------------- #
# Configure and run the TV-regularized multisource image reconstruction #
# --------------------------------------------------------------------- #
out = processing.tv_multisrc(proj, B, fgrad, delta, h, lbda,
                             out_shape, backend=backend, tol=1e-5,
                             nitermax=nitermax,
                             eval_energy=eval_energy, video=video,
                             verbose=verbose, Ndisplay=Ndisplay,
                             displayer=displayer)
