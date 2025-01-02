"""
Fusillo soaked with 40H-TEMPO
=============================

3D image reconstruction of a 'Fusillo' (pasta with helical shape)
soaked with an aqueous 4OH-TEMPO solution (see :ref:`here
<dataset_fusillo-20091002>` for the sample and the dataset
description). The reconstruction is performed using filtered
backprojection.


.. figure:: ../../../_static/fusillo-20091002-pic.png
  :width: 50%
  :align: center
  :alt: Fusillo soaked with TEMPO

|

"""

# %%
# Image reconstruction
# --------------------


# sphinx_gallery_thumbnail_number = -1
# -------------- #
# Import modules #
# -------------- #
import numpy as np
import pyepri.backends as backends
import pyepri.datasets as datasets
import pyepri.displayers as displayers
import pyepri.processing as processing
import pyvista as pv

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
# We load the ``fusillo-20091002`` dataset (files are embedded with
# the PyEPRI package) in ``float32`` precision (you can also select
# ``float64`` precision by setting ``dtype='float64'``).
#
dtype = 'float32'
path_proj = datasets.get_path('fusillo-20091002-proj.npy')
path_B = datasets.get_path('fusillo-20091002-B.npy')
path_h = datasets.get_path('fusillo-20091002-h.npy')
path_fgrad = datasets.get_path('fusillo-20091002-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ----------------------------------------- #
# Configure and run filtered backprojection #
# ----------------------------------------- #

# set mandatory parameters
delta = .05; # sampling step (cm)
out_shape = (100, 50, 50) # output image shape
xgrid = (-(out_shape[1]//2) + backend.arange(out_shape[1], dtype=dtype)) * delta
ygrid = (-(out_shape[0]//2) + backend.arange(out_shape[0], dtype=dtype)) * delta
zgrid = (-(out_shape[2]//2) + backend.arange(out_shape[2], dtype=dtype)) * delta
if backend.lib.__name__ in ['numpy','cupy']: 
    interp1 = lambda xp, fp, x : backend.lib.interp(x, xp.flatten(), fp.flatten(), left=0., right=0.)
else:
    import torchinterp1d
    interp1 = lambda xp, fp, x : torchinterp1d.interp1d(xp, fp, x) * ((x >= xp[0]) & (x <= xp[-1]))

# set optional parameters
verbose = False # disable console verbose mode
video = True # enable video display
Ndisplay = 100 # refresh display rate (iteration per refresh)
frequency_cutoff = .1 # we will apply a frequency cutoff to the
                      # projections (keep only 10% of the frequency
                      # coefficients) during the filter-backprojection
                      # process in order to avoid dramatic noise
                      # amplification caused by the Ram-Lak filter

# customize 3D image displayer (optional, for video mode only)
xgrid_np = backend.to_numpy(xgrid) # X-axis spatial grid (numpy view)
ygrid_np = backend.to_numpy(ygrid) # Y-axis spatial grid (numpy view)
zgrid_np = backend.to_numpy(zgrid) # Z-axis spatial grid (numpy view)
grids_np = (ygrid_np, xgrid_np, zgrid_np) # spatial sampling grids
unit = 'cm' # provide length unit associated to the grids
display_labels = True # display axes labels within subplots
adjust_dynamic = True # maximize displayed dynamic at each refresh
boundaries = 'same' #  give all subplots the same axes boundaries
                    #  (ensure same pixel size for each displayed
                    #  slice)
displayFcn = lambda u : np.maximum(u, 0.) # threshold display
                                          # (negative values are
                                          # displayed as 0)
figsize=(11., 6.)
displayer = displayers.create_3d_displayer(nsrc=1,figsize=figsize,
                                           displayFcn=displayFcn,
                                           units=unit,
                                           adjust_dynamic=adjust_dynamic,
                                           display_labels=display_labels,
                                           boundaries=boundaries,
                                           grids=grids_np)

# perform filtered back-projection
out = processing.eprfbp3d(proj, fgrad, h, B, xgrid, ygrid, zgrid,
                          interp1, backend=backend,
                          frequency_cutoff=frequency_cutoff,
                          video=video, Ndisplay=Ndisplay,
                          displayer=displayer)

# %%
# Isosurface rendering
# --------------------

# prepare isosurface display
x, y, z = np.meshgrid(xgrid_np, ygrid_np, zgrid_np, indexing='xy')
grid = pv.StructuredGrid(x, y, z)

# compute isosurface
vol = np.moveaxis(backend.to_numpy(out), (0,1,2), (2,1,0))
grid["vol"] = vol.flatten()
l1 = vol.max()
l0 = .2 * l1
isolevels = np.linspace(l0, l1, 10)
contours = grid.contour(isolevels)

# display isosurface
cpos = [(-8.2, -3.1, 4.1), (0.0, 0.0, 0.0), (0.3, -0.95, -0.14)]
p = pv.Plotter()
p.camera_position = cpos
labels = dict(ztitle='Z', xtitle='X', ytitle='Y')
p.add_mesh(contours, show_scalar_bar=False, color='#f7fe00')
p.show_grid(**labels)
p.show()
