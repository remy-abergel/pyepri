"""
.. _tutorial_backend:

CPU & GPU backends
==================

Details about the CPU & GPU backend of the package.

PyEPRI was designed in a modular fashion. The functions of this
package rely on standard datatypes (mostly arrays) and not on specific
classes. The only nonstandard object involved in this package is a
backend system whose role is to facilitate CPU & GPU compatible
scripting and development. More precisely, the PyEPRI backend system
allows the use of the libraries {``numpy``, ``cupy``, ``torch``} in a
unified framework. Deep understanding of this tutorial is **not
mandatory** but taking a quick look at this tutorial will facilitate
the understanding of all upcoming EPR imaging demo examples.


"""

#%%
#
# What is a PyEPRI backend?
# -------------------------
#
# Although most common functions of the ``numpy`` libraries are usually
# also available in the ``cupy``, and ``torch`` libraries, those
# functions rely on library-dependent types (e.g., ``numpy.ndarray``,
# ``cupy.ndarray``, ``torch.Tensor``, ...) and library-dependent
# datatypes (e.g., ``numpy.float32``, ``cupy.float32``,
# ``torch.float32``, ...). Besides, parameter naming conventions or default
# values oftenly differ between those libraries. The role of the backend
# is to remap those library-dependent types, datatypes and common
# functions towards a standardized counterpart (examples are provided
# below).
# 
# Technically, a PyEPRI backend is an instance of the
# :class:`pyepri.backends.Backend` class and must be passed as input of
# most functions of this package. It is also recommended to prefer the
# usage of the backend commands (e.g., ``backend.abs``,
# ``backend.meshgrid``, ``backend.rand``, ...) to their
# library-dependent counterparts in your own script. Doing so, moving
# from a CPU based computation framework with numpy to a GPU based
# computation framework with cupy or PyTorch is possible by simply
# changing the backend instance (keeping the rest of the script
# unchanged) as we systematically do in all provided EPR imaging demo
# examples.
#
# **Remark**: The available backend functions (``backend.abs``,
# ``backend.meshgrid``, ``backend.rand``, ...) have been restricted to
# the functions needed in the internal submodules and in the tutorial
# and example scripts. The backend functionalities may be extended in
# future releases of the PyEPRI package, depending on the needs.
# 

# %%
#
# Numpy backend (CPU)
# -------------------
#
# Since ``numpy`` is systematically installed with ``pyepri`` (as a
# mandatory dependency of this package), you should be able to create
# a numpy backend using the following commands.
#

# sphinx_gallery_thumbnail_path = '_static/thumbnail_tutorial_backend.png'
# ---------------------- #
# Create a numpy backend #
# ---------------------- #
import pyepri.backends as backends
backend = backends.create_numpy_backend()

#%%
#
# Once the ``backend`` object is created, it can be passed to any
# PyEPRI function requiring a backend input parameter. Doing so, all
# operations within the functions will be performed on the CPU using
# the Numpy library.
#
# You can also access to a bunch of standard functions on arrays,
# e.g. ``backend.abs``, ``backend.cos``, ``backend.sin``,
# ``backend.linspace``, ``backend.meshgrid``, ``backend.rand``, and
# many more. Those functions are lambda functions that simply remap
# their inputs to the approriate library-dependent function (in this
# case, to the functions ``numpy.abs``, ``numpy.cos``, ``numpy.sin``,
# ``numpy.linspace``, ``numpy.meshgrid``, ``numpy.random.rand``, etc.)
#
# For instance the next commands call the ``backend.rand`` function to
# generate a random array with shape ``(3, 4)``:

a = backend.rand(3, 4, dtype='float32')
print('type(a) is: %s' % type(a))
print('a.dtype is: %s' % a.dtype)

# %%
#
# The type and datatype of the generated array will depend on the
# backend instance as summarized below.
#
# .. table::
#    :align: center
#
#    +----------------------+---------------+---------------+
#    | Backend library      | Type of a     | Datatype of a |
#    +======================+===============+===============+
#    | numpy                | numpy.ndarray | numpy.float32 |
#    +----------------------+---------------+---------------+
#    | cupy                 | cupy.ndarray  | cupy.float32  |
#    +----------------------+---------------+---------------+
#    | torch                | torch.Tensor  | torch.float32 |
#    +----------------------+---------------+---------------+
#
# Here the backend instance is a *numpy backend* so the type and
# datatype of the array ``a`` are respectively ``numpy.ndarray`` and
# ``numpy.float32``.
#
# In practice, the backend instance provides a way to avoid calling
# the library dependent functions ``numpy.random.rand``,
# ``cupy.random.rand`` and ``torch.rand`` as well as the library
# dependent datatypes ``numpy.float32``, ``cupy.float32`` and
# ``torch.float32`` when developing a script. When the backend
# instance is changed, the function mapping is changed but those
# change have no impact on the content of the script. Only the choice
# of the backend instance will determine how the execution is
# performed.
#
# Note that a mini-documentation is provided for each standardized
# function of a backend instance. The documentation can be displayed
# using the ``help()`` function, as illustrated below for the
# ``backend.rand`` function.

help(backend.rand)

# %%
#
# More information about the standardized naming conventions and
# remapping functionalities are available in the
# :mod:`pyepri.backends` submodule documentation.
#

# %%
#
# Cupy backend (GPU)
# ------------------
#
# If the Cupy library is installed on your system (see the
# :ref:`installation instructions <heading-installation>` for the
# installation of this library as an optional dependency of the PyEPRI
# package), you should be able to instantiate a *cupy backend* by
# uncommenting the next command (an error will be raised if Cupy is
# not installed).
# 

# --------------------------- #
# Create a cupy backend (GPU) #
# --------------------------- #
#backend = backends.create_cupy_backend() # uncomment here for a CUPY backend

# %%
#
# Then, you can execute again the ``backend.rand`` commands presented
# above and check that the computed array has now the type
# ``cupy.ndarray`` and datatype ``cupy.float32``.

# %%
#
# PyTorch backend (CPU or GPU)
# ----------------------------
#
# If the PyTorch library is installed on your system (see the
# :ref:`installation instructions <heading-installation>` for the
# installation of this library as an optional dependency of the PyEPRI
# package), you should be able to instantiate a *torch backend* using
# either a CPU or a GPU device by uncommenting the next commands (an
# error will be raised if PyTorch is not installed or if the requested
# device is not available).
# 

# -------------------------------- #
# Create a torch-cpu backend (CPU) #
# -------------------------------- #
#backend = backends.create_torch_backend('cpu') # uncomment here for a PyTorch-CPU backend

# --------------------------------- #
# Create a torch-cuda backend (GPU) #
# --------------------------------- #
device = 'cuda' # select GPU device (if several GPU are available, you
                # can explicitly select the device index by using
                # device = 'cuda:0', device = 'cuda:1', etc.)                
#backend = backends.create_torch_backend(device=device) # uncomment here for a PyTorch GPU backend 

# %%
#
# Now, by executing again the ``backend.rand`` commands presented
# above, you can check that the computed array has now the type
# ``torch.Tensor`` and datatype ``torch.float32``.


