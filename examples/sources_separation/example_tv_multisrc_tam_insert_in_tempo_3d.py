"""
Separation of a TAM insert in a TEMPO solution (real 3D dataset)
================================================================

Simultaneous image reconstruction and separation of sample made of one
small eppendorf filled with a solution of TAM inserted into a larger
eppendorf filled with a solution of TEMPO (see :ref:`here
<dataset_tam-insert-in-tempo-20230929>` for the sample and the dataset
description).

.. figure:: ../../_static/tam-insert-in-tempo-20230929-pic.png
   :width: 48%
   :align: center
   :alt: TAM insert in TEMPO solution

|

"""

# %%
# Load & display dataset
# ----------------------

# sphinx_gallery_thumbnail_number = -1
# -------------- #
# Import modules #
# -------------- #
import math
import matplotlib.pyplot as plt
import numpy as np
import pyepri.backends as backends
import pyepri.displayers as displayers
import pyepri.datasets as datasets
import pyepri.processing as processing
import pyvista as pv
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
path_B = datasets.get_path('tam-insert-in-tempo-20230929-B.npy')
path_htam = datasets.get_path('tam-insert-in-tempo-20230929-htam.npy')
path_htempo = datasets.get_path('tam-insert-in-tempo-20230929-htempo.npy')
path_fgrad = datasets.get_path('tam-insert-in-tempo-20230929-fgrad.npy')
path_proj = datasets.get_path('tam-insert-in-tempo-20230929-proj.npy')

# load the dataset
dtype = 'float32'
B = backend.from_numpy(np.load(path_B).astype(dtype))
h_tam = backend.from_numpy(np.load(path_htam).astype(dtype))
h_tempo = backend.from_numpy(np.load(path_htempo).astype(dtype))
fg = backend.from_numpy(np.load(path_fgrad).astype(dtype))
proj_mixture = backend.from_numpy(np.load(path_proj).astype(dtype))


#---------------------#
# display projections #
#---------------------#

# prepare display
plt.figure(figsize=(10, 4))
theta = backend.arctan2(fg[1], fg[0])
proj_extent = [B[0].item(), B[-1].item(), proj_mixture.shape[0] - 1, 0]

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
_ = plt.ylabel("projection indexes")


# %%
# Perform source separation
# -------------------------
#
# Now let us perform the source separation, that is, the
# reconstruction of one image of the tube of TAM and one image of the
# tube of TEMPO.

# set reconstruction parameters
tam_shape = (30, 15, 15) # required size for the output TAM source
tempo_shape = (50, 20, 20) # required size for the output TEMPO source
delta = .1 # sampling step of the reconstruction (cm)
lbda = 2e-5 # regularity parameter
out_shape = (tam_shape, tempo_shape) # output image shape
proj = (proj_mixture,) # list of input experiments (here only one experiment)
h = ((h_tam, h_tempo),) # list of source spectra associated to each experiment
fgrad = (fg,) # list of field gradient vectors associated to each experiment

# set optional parameters
nitermax = 1000 # maximal number of iterations
verbose = False # disable console verbose mode
video = True # enable video display
Ndisplay = 10 # refresh display rate (iteration per refresh)
eval_energy = False # disable TV-regularized least-square energy
                    # evaluation each Ndisplay iteration

# customize 2D multi-sources image displayer: also optional, customize
# display (when video mode is enabled)
tam_shape = out_shape[0]
tempo_shape = out_shape[1]
xgrid_tam = (-(tam_shape[1]//2) + np.arange(tam_shape[1])) * delta
ygrid_tam = (-(tam_shape[0]//2) + np.arange(tam_shape[0])) * delta
zgrid_tam = (-(tam_shape[2]//2) + np.arange(tam_shape[2])) * delta
xgrid_tempo = (-(tempo_shape[1]//2) + np.arange(tempo_shape[1])) * delta
ygrid_tempo = (-(tempo_shape[0]//2) + np.arange(tempo_shape[0])) * delta
zgrid_tempo = (-(tempo_shape[2]//2) + np.arange(tempo_shape[2])) * delta
grid_tam = (ygrid_tam, xgrid_tam, zgrid_tam)
grid_tempo = (ygrid_tempo, xgrid_tempo, zgrid_tempo)
grids = (grid_tam, grid_tempo) # provide spatial sampling grids for each source
unit = 'cm' # provide length unit associated to the grids
display_labels = True # display axes labels within subplots
adjust_dynamic = True # maximize displayed dynamic at each refresh
origin = "lower"
boundaries = 'same' #  give all subplots the same axes boundaries
                    #  (ensure same pixel size for each displayed
                    #  slice)
src_labels = ('TAM', 'TEMPO') # source labels (to be included into suptitles)
figsize = (8., 8.) # size of the figure to be displayed
displayer = displayers.create_3d_displayer(nsrc=2, units=unit,
                                           figsize=figsize,
                                           adjust_dynamic=adjust_dynamic,
                                           display_labels=display_labels,
                                           boundaries=boundaries,
                                           origin=origin, grids=grids,
                                           src_labels=src_labels)

# run processing
out = processing.tv_multisrc(proj, B, fgrad, delta, h, lbda,
                             out_shape, backend=backend, tol=1e-5,
                             nitermax=nitermax,
                             eval_energy=eval_energy, video=video,
                             verbose=verbose, Ndisplay=Ndisplay,
                             displayer=displayer)

# %%
# Isosurface rendering
# --------------------
#
# Let us display isosurfaces of the reconstructed TAM and TEMPO source
# images (TAM is displayed in red color and TEMPO is displayed in
# green color). The TEMPO source image is masked over half a plane so
# that we can visualize the TAM insert.
#

# prepare isosurface display
#pv.set_jupyter_backend('static')
#PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
x_tam, y_tam, z_tam = np.meshgrid(xgrid_tam, ygrid_tam, zgrid_tam, indexing='xy')
x_tempo, y_tempo, z_tempo = np.meshgrid(xgrid_tempo, ygrid_tempo, zgrid_tempo, indexing='xy')
grid_tam = pv.StructuredGrid(x_tam, y_tam, z_tam)
grid_tempo = pv.StructuredGrid(x_tempo, y_tempo, z_tempo)

# compute TAM isosurface
vol = np.moveaxis(backend.to_numpy(out[0]), (0,1,2), (2,1,0))
grid_tam["vol"] = vol.flatten()
l1 = vol.max()
l0 = .5 * l1
isolevels = np.linspace(l0, l1, 10)
contours_tam = grid_tam.contour(isolevels)

# compute TEMPO isosurface (mask half of the reconstructed volume so
# that we can see inside)
mask = (x_tempo > -.3).astype(dtype)
vol_left = backend.to_numpy(out[1]) * mask
vol_left = np.moveaxis(vol_left, (0,1,2), (2,1,0))
grid_tempo["vol"] = vol_left.flatten()
l0 = 0.11 * l1
isolevels = np.linspace(l0, l1, 10)
contours_tempo = grid_tempo.contour(isolevels)

# display isosurfaces (green = TAM, red = TEMPO)
p = pv.Plotter()
cpos = [(-5.92, -0.11, -1.35), (-0.02, -0.62, 0.1), (0.22, -0.22, -0.95)]
p.camera_position = cpos
labels = dict(ztitle='Z', xtitle='X', ytitle='Y')
p.add_mesh(contours_tam, show_scalar_bar=False, color='#db0404', label=' TAM')
p.add_mesh(contours_tempo, show_scalar_bar=False, color='#01b517', label=' TEMPO')
p.show_grid(**labels)
p.add_legend(face='r')
p.show()



