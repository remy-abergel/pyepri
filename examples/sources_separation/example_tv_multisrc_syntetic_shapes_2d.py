"""
Separation of TAM & TEMPO (synthetic 2D experiment)
===================================================

Simultaneous image reconstruction and separation of a mixture of TAM &
TEMPO using TV regularized least-squares (synthetic experiment).

"""

# %% Generate synthetic projections (mixture of TAM & TEMPO)

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

#--------------------------#
# Compute synthetic images # 
#--------------------------#

# set datatype, pixel-size & dimensions of the source images
dtype = 'float32'
delta = 1e-2 # spatial sampling step (cm)
Nx, Ny = 300, 300 # width & height of the source images

# compute spatial coordinates
x = (-(Nx//2) + backend.arange(Nx)) * delta
y = (-(Ny//2) + backend.arange(Ny)) * delta
X, Y = backend.meshgrid(x, y)

# compute TAM source image (disk)
D = .8 # disc diameter (cm)
x0, y0 = -0.1, .25 # center coordinates
u_tam = backend.cast((X - x0)**2 + (Y - y0)**2 <= (D/2)**2, dtype)

# compute TEMPO source image (filled rectangle)
rx, ry = 1.5, .75 # rectangle size along each axis (cm)
u_tempo = backend.cast((backend.abs(X) <= rx/2.) &
                       (backend.abs(Y) <= ry/2.), dtype)

#--------------------------------------------------------#
# load TAM & TEMPO reference spectra (real measurements) #
#--------------------------------------------------------#

# retrieve paths towards of the different files comprised in the dataset
path_B = datasets.get_path('tam-and-tempo-tubes-2d-20210609-B.npy')
path_htam = datasets.get_path('tam-and-tempo-tubes-2d-20210609-htam.npy')
path_htempo = datasets.get_path('tam-and-tempo-tubes-2d-20210609-htempo.npy')

# load the dataset
dtype = 'float32'
B = backend.from_numpy(np.load(path_B).astype(dtype))
h_tam = backend.from_numpy(np.load(path_htam).astype(dtype))
h_tempo = backend.from_numpy(np.load(path_htempo).astype(dtype))

#--------------------------------#
# generate synthetic projections #
#--------------------------------#

# generate field gradient vector 
Gampl = 10 # gradient amplitude (G/cm)
Nproj = 40 # number of projections to simulate
theta = backend.linspace(0, math.pi, Nproj, dtype=dtype) # Gradient orientations (rad)
Gx = Gampl * backend.cos(theta)
Gy = Gampl * backend.sin(theta)
fg = backend.stack([Gx, Gy])

# compute synthetic projections as the sum between the projections of
# the TAM source and the projections of the TEMPO source
proj_tam = monosrc.proj2d(u_tam, delta, B, h_tam, fg, backend=backend)
proj_tempo = monosrc.proj2d(u_tempo, delta, B, h_tempo, fg, backend=backend)
proj_mixture = proj_tam + proj_tempo

# add synthetic noise (Gaussian)
proj_mixture += backend.randn(proj_mixture.shape, std=5e3)

#-------------------------------------------------#
# display source images and synthetic projections #
#-------------------------------------------------#

# prepare display
plt.figure(figsize=(15, 4))
img_extent = [t.item() for t in (x[0], x[-1], y[0], y[-1])]
proj_extent = [t.item() for t in (B[0], B[-1], theta[0]*180./math.pi, theta[-1]*180./math.pi)]

# display TAM source image
plt.subplot(1, 3, 1)
plt.imshow(backend.to_numpy(u_tam), extent=img_extent, origin='lower')
plt.title("(a) TAM source")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")

# display TEMPO source image
plt.subplot(1, 3, 2)
plt.imshow(backend.to_numpy(u_tempo), extent=img_extent, origin='lower')
plt.title("(b) TEMPO source")
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")

# synthetic projections of the mixture of TAM + TEMPO
plt.subplot(1, 3, 3)
plt.imshow(backend.to_numpy(proj_mixture), extent=proj_extent, aspect='auto')
plt.title("Synthetic projections of the mixture (a) + (b)")
plt.xlabel("B: homogeneous magnetic field intensity (G)")
_ = plt.ylabel("field gradient orientation (degree)")

# %% Source separation (reconstruction of the TAM & TEMPO source images)

# set reconstruction parameters
tam_shape = (100, 100) # required size for the output TAM source
tempo_shape = (100, 100) # required size for the output TEMPO source
out_shape = (tam_shape, tempo_shape) # output image shape
delta = 3e-2 # sampling step of the reconstruction (cm)
lbda = 10 # normalized regularity parameter
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
displayFcn = lambda u : [np.maximum(im, 0) for im in u] # display
                                                        # positive
                                                        # part of each
                                                        # source image
figsize = (10., 4.4) # size of the figure to be displayed
src_labels = ('TAM', 'TEMPO') # source labels (to be included into suptitles)
displayer = displayers.create_2d_displayer(nsrc=2, units=unit,
                                           figsize=figsize,
                                           adjust_dynamic=adjust_dynamic,
                                           display_labels=display_labels,
                                           boundaries=boundaries,
                                           grids=grids,
                                           src_labels=src_labels,
                                           displayFcn=displayFcn)
                    
# run processing
out = processing.tv_multisrc(proj, B, fgrad, delta, h, lbda,
                             out_shape, backend=backend, tol=1e-5,
                             nitermax=nitermax,
                             eval_energy=eval_energy, video=video,
                             verbose=verbose, Ndisplay=Ndisplay,
                             displayer=displayer)	   
