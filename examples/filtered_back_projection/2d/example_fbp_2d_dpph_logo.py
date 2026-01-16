"""
DPPH cristal powder in rubber (former CNRS logo)
================================================

2D image reconstruction of DPPH crystals powder in rubber (former CNRS
logo) using TV-regularized least-squares (see :ref:`here
<dataset_dpph-logo-20080402>` for the sample and the dataset
description).


.. figure:: ../../../_static/dpph-logo-20080402-pic.png
  :width: 50%
  :align: center
  :alt: (Former) CNRS logo has been engraved in rubber, then filled with DDPH powder

  Picture of the sample.

"""


# %%
# Import needed modules
# ---------------------

# sphinx_gallery_thumbnail_number = -1
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
# Load and display the input dataset
# ----------------------------------
#
# We load the ``dpph-logo-20080402`` dataset (embedded with the PyEPRI
# package) in ``float32`` precision. Take a look to the comments for
# changing the precision to ``float64`` or replacing the embedded
# dataset by one of your own dataset.
#

# ---------------------- #
# Load the input dataset #
# ---------------------- #
dtype = 'float32'
path_proj = datasets.get_path('dpph-logo-20080402-proj.pkl') # or use your own dataset, e.g., path_proj = '~/my_projections.DSC'
path_h = datasets.get_path('dpph-logo-20080402-h.pkl') # or use your own dataset, e.g., path_h = '~/my_spectrum.DSC'
dataset_proj = io.load(path_proj, backend=backend, dtype=dtype) # load the dataset containing the projections
dataset_h = io.load(path_h, backend=backend, dtype=dtype) # load the dataset containing the reference spectrum
B = dataset_proj['B'] # get B nodes from the loaded dataset
proj = dataset_proj['DAT'] # get projections data from the loaded dataset
fgrad = dataset_proj['FGRAD'] # get field gradient data from the loaded dataset
h = dataset_h['DAT'] # get reference spectrum data from the loaded dataset

# -------------------------------------------------------- #
# Display the retrieved projections and reference spectrum #
# -------------------------------------------------------- #

# plot the reference spectrum
fig = plt.figure(figsize=(12, 5))
fig.add_subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.grid(linestyle=':')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')

# plot the projections
fig.add_subplot(1, 2, 2)
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
im_hdl = plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
plt.title('projections (proj)')

# add suptitle and display the figure
plt.suptitle("Input dataset",  weight='demibold');
plt.show() # to keep the display persistent when the code is executed as a script

# %%
# Configure and run the filtered backprojection image reconstruction
# ------------------------------------------------------------------

# ------------------------ #
# Set mandatory parameters #
# ------------------------ #
delta = 1e-2; # sampling step in the same length unit as the provided field gradient coordinates (here cm)
out_shape = (250, 300) # output image shape (number of pixels along each axes)
xgrid = (-(out_shape[1]//2) + backend.arange(out_shape[1], dtype=dtype)) * delta # X-axis (horizontal) sampling nodes
ygrid = (-(out_shape[0]//2) + backend.arange(out_shape[0], dtype=dtype)) * delta # Y-axis (vertical) sampling nodes
if backend.lib.__name__ in ['numpy','cupy']: 
    interp1 = lambda xp, fp, x : backend.lib.interp(x, xp.flatten(), fp.flatten(), left=0., right=0.)
else:
    import torchinterp1d
    interp1 = lambda xp, fp, x : torchinterp1d.interp1d(xp, fp, x) * ((x >= xp[0]) & (x <= xp[-1]))

# ----------------------- #
# Set optional parameters #
# ----------------------- #
verbose = False # disable console verbose mode
video = True # enable video display
Ndisplay = 5 # refresh display rate (iteration per refresh)
frequency_cutoff = .1 # frequency cutoff to apply to the projections (here, keep only 10% of 
                      # the frequency coefficients) during the filter-backprojection process
                      # to avoid dramatic noise amplification caused by the Ram-Lak filter

# ------------------------------------------------------------------------ #
# Customize 2D image displayer (optional, used only when video=True above) #
# ------------------------------------------------------------------------ #
xgrid_np = backend.to_numpy(xgrid) # X-axis spatial grid (numpy view)
ygrid_np = backend.to_numpy(ygrid) # Y-axis spatial grid (numpy view)
grids_np = (ygrid_np, xgrid_np) # spatial sampling grids
unit = 'cm' # provide length unit associated to the grids (used to label the image axes)
display_labels = True # display axes labels within subplots
adjust_dynamic = True # maximize displayed dynamic at each refresh
boundaries = 'same' # give all subplots the same axes boundaries (ensure same pixel size for
                    # each displayed slice)
displayFcn = lambda u : np.maximum(u, 0.) # threshold display (negative values are displayed as 0)
displayer = displayers.create_2d_displayer(nsrc=1,
                                           displayFcn=displayFcn,
                                           units=unit,
                                           adjust_dynamic=adjust_dynamic,
                                           display_labels=display_labels,
                                           boundaries=boundaries,
                                           grids=grids_np)

# ------------------------------------------------------------------- #
# Perform filtered back-projection (image reconstruction starts here) #
# ------------------------------------------------------------------- #
out = processing.eprfbp2d(proj, fgrad, h, B, xgrid, ygrid, interp1,
                          backend=backend,
                          frequency_cutoff=frequency_cutoff,
                          video=video, Ndisplay=Ndisplay,
                          displayer=displayer)
plt.show() # to keep the display persistent when the code is executed as a script
