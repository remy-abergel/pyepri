"""
Tubes filled with TAM
=====================

3D image reconstruction of tubes filled with a solution of TAM (see
:ref:`here <dataset_tamtubes-20211201>` for the sample and the dataset
description).

.. figure:: ../../../_static/tamtubes-20211201-pic.png
  :width: 60%
  :align: center
  :alt: Tubes filled with a TAM solution

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
# We load the ``tamtubes-20211201`` dataset (files are embedded with
# the PyEPRI package) in ``float32`` precision (you can also select
# ``float64`` precision by setting ``dtype='float64'``).
#
dtype = 'float32'
path_proj = datasets.get_path('tamtubes-20211201-proj.npy')
path_B = datasets.get_path('tamtubes-20211201-B.npy')
path_h = datasets.get_path('tamtubes-20211201-h.npy')
path_fgrad = datasets.get_path('tamtubes-20211201-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# --------------------------------------- #
# Perform monosource image reconstruction #
# --------------------------------------- #

# set mandatory parameters
out_shape = (55, 55, 75)
delta = .02 # sampling step (cm)
lbda = 5.

# set optional parameters
nitermax = 1000 # maximal number of iterations
verbose = False # disable console verbose mode
video = True # enable video display
Ndisplay = 20 # refresh display rate (iteration per refresh)
eval_energy = False # disable TV-regularized least-square energy
                    # evaluation each Ndisplay iteration

# customize 3D image displayer (optional also, for video mode only)
xgrid = (-(out_shape[1]//2) + np.arange(out_shape[1])) * delta
ygrid = (-(out_shape[0]//2) + np.arange(out_shape[0])) * delta
zgrid = (-(out_shape[2]//2) + np.arange(out_shape[2])) * delta
grids = (ygrid, xgrid, zgrid) # provide spatial sampling grids
unit = 'cm' # provide length unit associated to the grids
display_labels = True # display axes labels within subplots
adjust_dynamic = True # maximize displayed dynamic at each refresh
boundaries = 'same' #  give all subplots the same axes boundaries
                    #  (ensure same pixel size for each displayed
                    #  slice)
displayFcn = lambda u : np.maximum(u, 0.) # threshold display
                                          # (negative values are
                                          # displayed as 0)
figsize=(17.5, 4.5)
displayer = displayers.create_3d_displayer(nsrc=1,figsize=figsize,
                                           displayFcn=displayFcn,
                                           units=unit,
                                           adjust_dynamic=adjust_dynamic,
                                           display_labels=display_labels,
                                           boundaries=boundaries,
                                           grids=grids)

# run processing
out = processing.tv_monosrc(proj, B, fgrad, delta, h, lbda, out_shape,
                            backend=backend, init=None, tol=1e-4,
                            nitermax=nitermax,
                            eval_energy=eval_energy, verbose=verbose,
                            video=video, Ndisplay=Ndisplay,
                            displayer=displayer)

# %%
# Isosurface rendering
# --------------------

# prepare isosurface display
x, y, z = np.meshgrid(xgrid, ygrid, zgrid, indexing='xy')
grid = pv.StructuredGrid(x, y, z)

# compute isosurface
vol = np.moveaxis(backend.to_numpy(out), (0,1,2), (2,1,0))
grid["vol"] = vol.flatten()
l1 = vol.max()
l0 = .2 * l1
isolevels = np.linspace(l0, l1, 10)
contours = grid.contour(isolevels)

# display isosurface
p = pv.Plotter()
cpos = [(-2.26, 1.84, -2.), (0, 0, 0), (0, 0, -1)]
p.camera_position = cpos
labels = dict(ztitle='Z', xtitle='X', ytitle='Y')
p.add_mesh(contours, show_scalar_bar=False, color='#01b517')
p.show_grid(**labels)
p.show()
