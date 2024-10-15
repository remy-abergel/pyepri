"""
Embedded demonstration EPR datasets
===================================

Presentation of the EPR datasets embedded with this package.

"""

# sphinx_gallery_thumbnail_path = '/tmp/database-file-icon2.png' 
# -------------- #
# Import modules #
# -------------- #
import numpy as np
import matplotlib.pyplot as plt
import pyepri.backends as backends
import pyepri.datasets as datasets
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

# %%
#
# We provide in this page some short descriptions of all demonstration
# EPR datasets embedded with the PyEPRI package (and used in the
# demonstration examples). We include load and display command line
# instructions. You can use the links listed below to jump to a
# specific dataset.
#
# **Two-dimensional EPR imaging datasets:**
#
# + :ref:`irradiated phalanx (phalanx-20220203)
#   <dataset_phalanx-20220203>`
#
# + :ref:`Spade-Club-Diamond shapes printed with paramagnetic ink
#   (scd-inkjet-20141204) <dataset_scd-inkjet-20141204>`
#
# + :ref:`Word "Bacteria" printed with paramagnetic ink
#   (bacteria-inkjet-20100709) <dataset_bacteria-inkjet-20100709>`
#
# + :ref:`Word "CNRS" printed with paramagnetic ink
#   (cnrs-inkjet-20110614) <dataset_cnrs-inkjet-20110614>`
#
# + :ref:`Mouse knee phantom (beads-phantom-20080313)
#   <dataset_beads-phantom-20080313>`
#
# + :ref:`Plastic beads in a thin tube filled with TAM
#   (beads-thintubes-20081017) <dataset_beads-thintubes-20081017>`
#
# + :ref:`DPPH crystal powder in rubber (dpph-logo-20080402)
#   <dataset_dpph-logo-20080402>`
#
# + :ref:`Solutions of TAM & TEMPO in two distinct tubes
#   (tam-and-tempo-tubes-2d-20210609)
#   <dataset_tam-and-tempo-tubes-2d-20210609>`
# 
# **Three-dimensional EPR imaging datasets:**
#
# + :ref:`Fusillo soaked with 4OH-TEMPO (fusillo-20091002)
#   <dataset_fusillo-20091002>`
# 
# + :ref:`TAM solution in tubes with various diameters
#   (tamtubes-20211201) <dataset_tamtubes-20211201>`
#
# + :ref:`Solutions of TAM & TEMPO in two distinct tubes
#   (tam-and-tempo-tubes-3d-20210609)
#   <dataset_tam-and-tempo-tubes-3d-20210609>`
#
# + :ref:`TAM solution insert into a TEMPO solution
#   (tam-insert-in-tempo-20230929)
#   <dataset_tam-insert-in-tempo-20230929>`
#

# %%
#
# .. _dataset_phalanx-20220203:
#
# Irradiated phalanx (2D)
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The sample is an irradiated distal phalanx. It was kindly provided
# to `LCBPT <https://lcbpt.biomedicale.parisdescartes.fr/>`_ by our
# colleague `Pr. Bernard Gallez (UC Louvain)
# <https://uclouvain.be/fr/repertoires/bernard.gallez>`_.
#
#
# .. figure:: ../../../_static/phalanx-20220203-pic.png
#   :width: 20%
#   :align: center
#   :alt: Irradiated distal phalanx
#
# |
#
# Irradiations generate paramagnetic damages of the bone lattice
# microarchitecture that can be observed using EPR imaging.
#
# The dataset was acquired at `LCBPT
# <https://lcbpt.biomedicale.parisdescartes.fr/>`_ using an X-band
# Bruker spectrometer. It comprised of several files in ``.npy``
# format.
#
# + ``phalanx-20220203-h.npy``: reference spectrum (1D spectrum);
#
# + ``phalanx-20220203-proj.npy``: 2D projections (= 2D sinogram);
#
# + ``phalanx-20220203-fgrad.npy``: 2D coordinates (in the image
#   plane) of the field gradient vectors associated to the projections;
#
# + ``phalanx-20220203-B.npy``: sampling grid (homogeneous magnetic
#   field intensity) used to acquire the projections and the reference
#   spectrum.
#
# The dataset contains 113 projections acquired with constant field
# gradient magnitude equal to 168 G/cm and orientation uniformly
# sampled between 0 and 180°. Each projection contains 2000
# measurement points (same for the reference spectrum). The next
# commands show how to open and display the dataset.
#
#

#--------------#
# Load dataset #
#--------------#
#
# We load the ``phalanx-20220203`` dataset in ``float32`` precision
# (you can also select ``float64`` precision by setting
# ``dtype='float64'``).
#
dtype = 'float32'
path_proj = datasets.get_path('phalanx-20220203-proj.npy')
path_B = datasets.get_path('phalanx-20220203-B.npy')
path_h = datasets.get_path('phalanx-20220203-h.npy')
path_fgrad = datasets.get_path('phalanx-20220203-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum 
plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')

# display the field gradient vector associated to the measured projections
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), 'go', markersize=1)
plt.gca().set_aspect('equal')
plt.xlabel("horizontal axis (G/cm)")
plt.ylabel("vertical axis (G/cm)")
_ = plt.title("2D field gradient vectors")

# display the projections
plt.figure(figsize=(13, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)')


# %% 
#
# **Note**: this dataset is also available in ASCII format (files
# ``phalanx-20220203-h.txt``, ``phalanx-20220203-proj.txt``,
# ``phalanx-20220203-fgrad.txt``, ``phalanx-20220203-B.txt``) and in
# the original BES3T format (files ``phalanx-20220203-h.DTA``,
# ``phalanx-20220203-h.DSC``, ``phalanx-20220203-proj.DTA`` and
# ``phalanx-20220203-proj.DSC``). See :ref:`here <tutorial_io>` how to
# deal with those kinds of files.
#

# %%
#
# Paramagnetic ink (2D)
# ~~~~~~~~~~~~~~~~~~~~~
#
# The three next datasets correspond to paramagnetic ink printed on
# paper sheets.
#
#

# %%
#
# .. _dataset_scd-inkjet-20141204:
#
# Spade-Club-Diamond shapes printed on a paper sheet
# ++++++++++++++++++++++++++++++++++++++++++++++++++
#
# We present below the ``scd-inkjet-20141204`` dataset, acquired at
# `LCBPT <https://lcbpt.biomedicale.parisdescartes.fr/>`_ using an
# X-band Bruker spectrometer.
#
# .. figure:: ../../../_static/scd-inkjet-20141204-pic.png 
#   :width: 60% 
#   :align: center
#   :alt: Spade-Club-Diamond shapes 
#
# |
#
# The dataset is comprised of ``.npy`` files stored using the same
# suffix naming conventions as the other datasets
# (``datasetname-{proj,h,B,fgrad}.npy``). It contains 94 projections
# acquired with constant field gradient magnitude equal to 57.6 G/cm
# and orientation uniformly sampled between -90° and 90°. Each
# projection contains 1300 measurement points (same for the reference
# spectrum). The next commands show how to open and display the
# dataset.
#

#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'scd-inkjet-20141204'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_h = datasets.get_path(datasetname + '-h.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum 
plt.figure(figsize=(15, 5))
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold');
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')

# display the field gradient vector associated to the measured projections
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), 'go', markersize=1)
plt.gca().set_aspect('equal')
plt.xlabel("horizontal axis (G/cm)")
plt.ylabel("vertical axis (G/cm)")
_ = plt.title("2D field gradient vectors")

# display the projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)')


# %%
#
# .. _dataset_bacteria-inkjet-20100709:
#
# Word "Bacteria" printed on a paper sheet
# ++++++++++++++++++++++++++++++++++++++++
#
# Here is the ``bacteria-inkjet-20100709`` dataset, acquired at `LCBPT
# <https://lcbpt.biomedicale.parisdescartes.fr/>`_ using an X-band
# Bruker spectrometer.
#
# .. figure:: ../../../_static/bacteria-inkjet-20100709-pic.png
#   :width: 50%
#   :align: center
#   :alt: printed "Bacteria" word
#
# |
#

#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'bacteria-inkjet-20100709'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_h = datasets.get_path(datasetname + '-h.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum 
plt.figure(figsize=(10, 5))
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold');
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')

# display the projections
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), 'go', markersize=1)
plt.gca().set_aspect('equal')
plt.xlabel("horizontal axis (G/cm)")
plt.ylabel("vertical axis (G/cm)")
_ = plt.title("2D field gradient vectors")

# display the field gradient vector associated to the measured projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)')

# %%
# 
# .. _dataset_cnrs-inkjet-20110614:
#
# Word "CNRS" printed on a paper sheet
# ++++++++++++++++++++++++++++++++++++
#
# The last dataset of this series of printed paramagnetic ink is the
# ``cnrs-inkjet-20110614`` dataset, acquired at `LCBPT
# <https://lcbpt.biomedicale.parisdescartes.fr/>`_ using an X-band
# Bruker spectrometer. The sample is a sheet of paper on which the
# word "CNRS" is printed with paramagnetic ink. Unfortunately no
# picture of the sample is currently available.
#

#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'cnrs-inkjet-20110614'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_h = datasets.get_path(datasetname + '-h.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum 
plt.figure(figsize=(10, 5))
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold');
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')

# display the projections
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), 'go', markersize=1)
plt.gca().set_aspect('equal')
plt.xlabel("horizontal axis (G/cm)")
plt.ylabel("vertical axis (G/cm)")
_ = plt.title("2D field gradient vectors")

# display the field gradient vector associated to the measured projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
im_hdl = plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cmax = backend.quantile(proj, .9995) 
im_hdl.set_clim(proj.min().item(), cmax.item()) # saturate 0.05% of the highest values
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)\n(saturation of 0.05% of the max. values)')

# %%
#
# Plastic beads in TAM solution (2D)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The two next datasets are made of plastic beads placed in a tube
# filled with a solution of TAM.
#

# %%
#
# .. _dataset_beads-phantom-20080313:
#
# Mouse knee phantom (2D)
# +++++++++++++++++++++++
#
# The sample in the ``beads-phantom-20080313`` dataset is made of a
# single EPR tube (with 3 mm internal diameter) containing plastic
# beads with diameter 2.5 mm and a TAM aqueous solution with
# concentration 5 mM. The sample has been designed to represent a
# mouse knee phantom. 
#
# .. figure:: ../../../_static/beads-phantom-20080313-pic.png
#   :width: 50%
#   :align: center
#   :alt: Mouse knee phantom
#
# |
#
# This ``beads-phantom-20080313```` dataset has been acquired at
# `LCBPT <https://lcbpt.biomedicale.parisdescartes.fr/>`_ using an
# L-band Bruker spectrometer.


#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'beads-phantom-20080313'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_h = datasets.get_path(datasetname + '-h.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum 
plt.figure(figsize=(10, 5))
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold');
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')

# display the projections
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), 'go', markersize=1)
plt.gca().set_aspect('equal')
plt.xlabel("horizontal axis (G/cm)")
plt.ylabel("vertical axis (G/cm)")
_ = plt.title("2D field gradient vectors")

# display the field gradient vector associated to the measured projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)')

# %%
#
# .. _dataset_beads-thintubes-20081017:
#
# Plastic beads in a thin tube (2D)
# +++++++++++++++++++++++++++++++++
#
# The sample in the ``beads-thintubes-20081017`` dataset is made of a
# single capillary tube with 0.5 mm diameter containing TAM aqueous
# solution with concentration 5 mM and some plastic beads with 0.4 mm
# diameter. 
#
# .. figure:: ../../../_static/beads-thintubes-20081017-pic.png
#   :width: 50%
#   :align: center
#   :alt: Thin tube filled with TAM and plastic beads
#
# |
#
# This ``beads-thintubes-20081017`` dataset has been acquired at
# `LCBPT <https://lcbpt.biomedicale.parisdescartes.fr/>`_ using an
# X-band Bruker spectrometer .


#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'beads-thintubes-20081017'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_h = datasets.get_path(datasetname + '-h.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum 
plt.figure(figsize=(10, 5))
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold');
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')


# display the projections
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), 'go', markersize=1)
plt.gca().set_aspect('equal')
plt.xlabel("horizontal axis (G/cm)")
plt.ylabel("vertical axis (G/cm)")
_ = plt.title("2D field gradient vectors")

# display the field gradient vector associated to the measured projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
im_hdl = plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cmin = backend.quantile(proj, .005) 
cmax = backend.quantile(proj, .995) 
im_hdl.set_clim(cmin.item(), cmax.item()) # saturate .5% of the lowest and highest values
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)\n(saturation of 0.5% of the min/max values)')

# %%
#
# .. _dataset_dpph-logo-20080402:
#
# DPPH cristal powder in rubber (2D)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The (former) CNRS logo has been engraved in rubber, then filled with DDPH powder. 
#
# .. figure:: ../../../_static/dpph-logo-20080402-pic.png
#   :width: 50%
#   :align: center
#   :alt: Thin tube filled with TAM and plastic beads
#
# |
#
# Then, the ``dpph-logo-20080402`` dataset has been acquired at `LCBPT
# <https://lcbpt.biomedicale.parisdescartes.fr/>`_ using an L-band
# Bruker spectrometer .
#

#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'dpph-logo-20080402'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_h = datasets.get_path(datasetname + '-h.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum 
plt.figure(figsize=(10, 5))
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold');
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')

# display the field gradient vector associated to the measured projections
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), 'go', markersize=1)
plt.gca().set_aspect('equal')
plt.xlabel("horizontal axis (G/cm)")
plt.ylabel("vertical axis (G/cm)")
_ = plt.title("2D field gradient vectors")

# display the projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
im_hdl = plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)')

# %%
#
# .. _dataset_fusillo-20091002:
#
# Fusillo soaked with TEMPO (3D)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The next sample is made of a fusillo with helical shape which was
# put into a 4 mM aqueous solution of 4OH-TEMPO for a night, then
# dried using paper.
#
# .. figure:: ../../../_static/fusillo-20091002-pic.png
#   :width: 50%
#   :align: center
#   :alt: Fusillo soaked with TEMPO
#
# |
#
# The dataset ``fusillo-20091002`` is three dimensional and has been
# acquired at `LCBPT <https://lcbpt.biomedicale.parisdescartes.fr/>`_
# using an L-band Bruker spectrometer.

#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'fusillo-20091002'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_h = datasets.get_path(datasetname + '-h.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2 + fgrad[2]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum 
fig = plt.figure(figsize=(10, 5))
fig.add_subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.grid(linestyle=':')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold');

# display the field gradient vector associated to the measured projections
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), backend.to_numpy(fgrad[2]))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_zlim(-15, 15)
ax.set_aspect('equal', 'box')
ax.set_xlabel("X axis (G/cm)")
ax.set_ylabel("Y axis (G/cm)")
ax.set_zlabel("Z axis (G/cm)")
_ = plt.title('magnetic field gradient samples')

# display the projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
im_hdl = plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)')

# %%
#
# .. _dataset_tamtubes-20211201:
#
# TAM solution in tubes with various diameters (3D)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The sample is made of six capillary tubes filled with an aqueous TAM
# solution with concentration 1 mM. The tubes were sealed using wax
# sealed plate (Fischer scientific) and attached together using
# masking tape.
#
# .. figure:: ../../../_static/tamtubes-20211201-pic.png
#   :width: 60%
#   :align: center
#   :alt: Tubes filled with a TAM solution
#
# |
#
# The dataset ``tamtubes-20211201`` is three dimensional and has been
# acquired at `LCBPT <https://lcbpt.biomedicale.parisdescartes.fr/>`_
# using an L-band Bruker spectrometer.

#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'tamtubes-20211201'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_h = datasets.get_path(datasetname + '-h.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
h = backend.from_numpy(np.load(path_h).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2 + fgrad[2]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum 
fig = plt.figure(figsize=(19, 9))
ax = fig.add_subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(h))
plt.grid(linestyle=':')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum (h)')
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold', fontsize=20);
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(16)

# display the field gradient vector associated to the measured projections
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), backend.to_numpy(fgrad[2]))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_zlim(-15, 15)
ax.set_aspect('equal', 'box')
ax.set_xlabel("X axis (G/cm)")
ax.set_ylabel("Y axis (G/cm)")
ax.set_zlabel("Z axis (G/cm)")
_ = plt.title('magnetic field gradient samples')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label,
              ax.zaxis.label] + ax.get_xticklabels() +
             ax.get_yticklabels() + ax.get_zticklabels()):
    item.set_fontsize(16)
    
# display the projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
im_hdl = plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cmin = backend.quantile(proj, .001) 
cmax = backend.quantile(proj, .999) 
im_hdl.set_clim(cmin.item(), cmax.item()) # saturate 0.1% of the highest values
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)\n(saturation of 0.1% of the min/max values)')

# %%
#
# TAM and TEMPO (2D \& 3D)
# ~~~~~~~~~~~~~~~~~~~~~~~~
#

# %%
#
# .. _dataset_tam-and-tempo-tubes-2d-20210609:
#
# Solutions of TAM & TEMPO in two distinct tubes (2D)
# +++++++++++++++++++++++++++++++++++++++++++++++++++
#
# The sample is made of two tubes. One containing 500 µL of a 2 mM TAM
# solution, the other one containing a 5 mM TEMPO solution. The tubes
# were placed in the cavity roughly a centimeter apart using a plastic
# holder.
#
# .. figure:: ../../../_static/tam-and-tempo-tubes-3d-20210609-pic.png
#   :width: 48%
#   :align: center
#   :alt: TAM & TEMPO in two distinct tubes
#
# |
#
# The dataset ``tam-and-tempo-tubes-2d-20210609`` is two dimensional
# and has been acquired at `LCBPT
# <https://lcbpt.biomedicale.parisdescartes.fr/>`_ using an L-band
# Bruker spectrometer.

#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'tam-and-tempo-tubes-2d-20210609'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_hmixt = datasets.get_path(datasetname + '-hmixt.npy')
path_htam = datasets.get_path(datasetname + '-htam.npy')
path_htempo = datasets.get_path(datasetname + '-htempo.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
hmixt = backend.from_numpy(np.load(path_hmixt).astype(dtype))
htam = backend.from_numpy(np.load(path_htam).astype(dtype))
htempo = backend.from_numpy(np.load(path_htempo).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum of the whole sample (TAM + TEMPO)
plt.figure(figsize=(17, 9))
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(hmixt))
ax1 = plt.gca()
plt.grid(linestyle=':')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum of the sample (TAM+TEMPO)\n')
for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(16)

# display the reference spectrum of the separated sources (TAM & TEMPO)
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(B), backend.to_numpy(htam))
plt.plot(backend.to_numpy(B), backend.to_numpy(htempo))
ax2 = plt.gca()
ax2.set_ylim(ax1.get_ylim())
plt.grid(linestyle=':')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('separated reference spectra of the\ntwo sources (TAM & TEMPO)')
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold', fontsize=20);
for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(16)
ax2.legend(['TAM (htam)', 'TEMPO (htempo)'], fontsize="16")

# display the projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
im_hdl = plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
plt.title('projections (proj)')

# display the field gradient vector associated to the measured projections
plt.figure(figsize=(7, 5))
plt.plot(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), 'go', markersize=1)
plt.gca().set_aspect('equal')
plt.xlabel("horizontal axis (G/cm)")
plt.ylabel("vertical axis (G/cm)")
_ = plt.title("2D field gradient vectors")

# %%
#
# .. _dataset_tam-and-tempo-tubes-3d-20210609:
#
# Solutions of TAM & TEMPO in two distinct tubes (3D)
# +++++++++++++++++++++++++++++++++++++++++++++++++++
#
# The sample is the same as that presented in the :ref:`previous
# section <dataset_tam-and-tempo-tubes-2d-20210609>`.  The dataset
# ``tam-and-tempo-tubes-3d-20210609`` is three dimensional and has
# been acquired at `LCBPT
# <https://lcbpt.biomedicale.parisdescartes.fr/>`_ using an L-band
# Bruker spectrometer.


#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'tam-and-tempo-tubes-3d-20210609'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_htam = datasets.get_path(datasetname + '-htam.npy')
path_htempo = datasets.get_path(datasetname + '-htempo.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
htam = backend.from_numpy(np.load(path_htam).astype(dtype))
htempo = backend.from_numpy(np.load(path_htempo).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2 + fgrad[2]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum of the first source (TAM)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(htam))
ax1 = plt.gca()
plt.grid(linestyle=':')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum of the first source (TAM)')

# display the reference spectrum of the first source (TEMPO)
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(B), backend.to_numpy(htempo))
ax2 = plt.gca()
ax2.set_ylim(ax1.get_ylim())
plt.grid(linestyle=':')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum of the second source (TEMPO)')
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold');

# display the projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
im_hdl = plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)')

# display the field gradient vector associated to the measured projections
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), backend.to_numpy(fgrad[2]))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
ax.set_aspect('equal', 'box')
ax.set_xlabel("X axis (G/cm)")
ax.set_ylabel("Y axis (G/cm)")
ax.set_zlabel("Z axis (G/cm)")
_ = plt.title('magnetic field gradient samples')

# %%
#
# .. _dataset_tam-insert-in-tempo-20230929:
#
# TAM solution insert into a TEMPO solution (3D)
# ++++++++++++++++++++++++++++++++++++++++++++++
#
# The sample is made of a small eppendorf filled with a 12.5 mM
# solution of TAM placed inside a larger eppendorf filled with a 14 mM
# solution of TEMPO.
#
#
# .. figure:: ../../../_static/tam-insert-in-tempo-20230929-pic.png
#   :width: 48%
#   :align: center
#   :alt: TAM insert in TEMPO solution
#
# |
#
# The dataset ``tam-insert-in-tempo-20230929`` is three dimensional
# and was acquired at `SFR ICAT University of Angers
# <https://sfricat.univ-angers.fr/fr/index.html>`_ using an L-band
# Bruker spectrometer, with the kind help of `Dr Raffaella Soleti
# <https://www.univ-nantes.fr/raphaella-soleti>`_.

#--------------#
# Load dataset #
#--------------#
dtype = 'float32'
datasetname = 'tam-insert-in-tempo-20230929'
path_proj = datasets.get_path(datasetname + '-proj.npy')
path_B = datasets.get_path(datasetname + '-B.npy')
path_htam = datasets.get_path(datasetname + '-htam.npy')
path_htempo = datasets.get_path(datasetname + '-htempo.npy')
path_fgrad = datasets.get_path(datasetname + '-fgrad.npy')
proj = backend.from_numpy(np.load(path_proj).astype(dtype))
B = backend.from_numpy(np.load(path_B).astype(dtype))
htam = backend.from_numpy(np.load(path_htam).astype(dtype))
htempo = backend.from_numpy(np.load(path_htempo).astype(dtype))
fgrad = backend.from_numpy(np.load(path_fgrad).astype(dtype))

# ------------------------------------------------------------ #
# Compute and display several important acquisition parameters #
# ------------------------------------------------------------ #
print("Sweep-width = %g G" % (B[-1] - B[0]))
print("Number of projections = %d" % proj.shape[0])
print("Number of point per projection = %d" % proj.shape[1])
mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2 + fgrad[2]**2) # magnitudes of the field gradient vectors (constant for this dataset)
print("Field gradient magnitude = %g G/cm" % mu[0])

#-----------------#
# Display dataset #
#-----------------#

# display the reference spectrum of the first source (TAM)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(backend.to_numpy(B), backend.to_numpy(htam))
plt.grid(linestyle=':')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum of the first source (TAM)')

# display the reference spectrum of the first source (TEMPO)
plt.subplot(1, 2, 2)
plt.plot(backend.to_numpy(B), backend.to_numpy(htempo))
plt.grid(linestyle=':')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('measurements (arb. units)')
plt.title('reference spectrum of the second source (TEMPO)')
plt.suptitle("Dataset '" + datasetname + "'",  weight='demibold');

# display the projections
plt.figure(figsize=(10, 5))
extent = (B[0].item(), B[-1].item(), proj.shape[0] - 1, 0)
im_hdl = plt.imshow(backend.to_numpy(proj), extent=extent, aspect='auto')
cbar = plt.colorbar()
cbar.set_label('measurements (arb. units)')
plt.xlabel('B: homogeneous magnetic field intensity (G)')
plt.ylabel('projection index')
_ = plt.title('projections (proj)')

# display the field gradient vector associated to the measured projections
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter(backend.to_numpy(fgrad[0]), backend.to_numpy(fgrad[1]), backend.to_numpy(fgrad[2]))
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(-20, 20)
ax.set_aspect('equal', 'box')
ax.set_xlabel("X axis (G/cm)")
ax.set_ylabel("Y axis (G/cm)")
ax.set_zlabel("Z axis (G/cm)")
_ = plt.title('magnetic field gradient samples')
