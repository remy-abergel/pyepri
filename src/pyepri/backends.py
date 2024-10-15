"""This module defines a standardized Backend class for unified use of
numpy, cupy and torch libraries (see <class 'backends.Backend'>).

This module also provides constructors of ready-to-use backends, see 

+ :py:func:`create_numpy_backend()`: numpy backend constructor \
  (CPU only) 
+ :py:func:`create_cupy_backend()`: cupy backend constructor \
  (GPU only)
+ :py:func:`create_torch_backend()`: torch backend constructor \
  (CPU or GPU)

A ``backends.Backend`` instance provides attributes for data type
mapping (the standardized type is a str, e.g., ``'float32'``,
``'float64'``, ``'int32'``, ..., is mapped to the library dependent
corresponding data types).

A :class:`Backend` instance also provides lambda function attributes
that are mapped to several standard lib-dependent functions
(``lib.zeros``, ``lib.arange``, ``lib.sin``, ``lib.cos``,
``lib.meshgrid``, ..., for lib in {``numpy``, ``cupy``, ``torch``})
but with unified fashion (parameters naming, default settings, etc).

For more details about how backend system works, you may consult the
:class:`Backend` class documentation by typing:

>>> import pyepri.backends as backends
>>> help(backends.Backend)

Examples of backend instantiation and utilization are also described
in the constructor documentations: 

>>> import pyepri.backends as backends
>>> help(backends.create_numpy_backend) # numpy backend constructor
>>> help(backends.create_cupy_backend)  # cupy backend constructor
>>> help(backends.create_torch_backend) # torch backend constructor

"""
import functools
import numpy as np
import scipy as sp
import re

def create_numpy_backend():
    """Create a numpy backend.

    Returns
    -------

    backend : <class 'backends.Backend'> 
        Backend configured to perform operations using numpy library
        on CPU device.

    See also 
    --------

    backends.create_cupy_backend()
    backends.create_torch_backend()

    Examples
    --------
    
    The following example illustrate how one can call the
    ``numpy.zeros`` and ``numpy.arange`` functions from a backend
    output of this :py:func:`create_numpy_backend` function.
    
    >>> import pyepri.backends as backends
    >>> b = backends.create_numpy_backend()
    >>> x = b.zeros((10,),dtype='float64') 
    >>> print(x)
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    >>> print(type(x)) 
    <class 'numpy.ndarray'>
    >>> print(x.dtype)
    float64
    >>> y = b.arange(3,10,2,dtype='int32') # another example
    >>> print(y)
    [3 5 7 9]
    >>> z = b.arange(10) # b.arange behaves as its numpy counterpart 
    >>>                  # (`start` and `end` parameters are optional
    >>>                  # as for numpy.arange)
    >>> print(z)
    [0 1 2 3 4 5 6 7 8 9]

    """
    return Backend(lib=np, device='cpu')

def create_cupy_backend():
    """Create a cupy backend.

    Returns
    -------

    backend : <class 'backends.Backend'> 
        Backend configured to perform operations using cupy library
        on GPU device.

    See also 
    --------

    backends.create_numpy_backend()
    backends.create_torch_backend()

    Examples
    --------
    
    The following example illustrate how one can call the
    `cupy.zeros` and `cupy.arange` functions from a backend output
    of this backends.create_cupy_backend function.
    
    >>> import pyepri.backends as backends
    >>> b = backends.create_cupy_backend()
    >>> x = b.zeros((10,),dtype='float64') 
    >>> print(x)
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    >>> print(type(x)) 
    <class 'cupy.ndarray'>
    >>> print(x.dtype)
    float64
    >>> print(x.device) # should return a GPU device (e.g., 
    >>>                 # <CUDA Device 0>)
    <CUDA Device 0>
    >>> y = b.arange(3,10,2,dtype='int32') # another example
    >>> print(y)
    [3 5 7 9]
    >>> z = b.arange(10) # b.arange behaves as its cuppy counterpart 
    >>>                  # (`start` and `end` parameters are optional
    >>>                  # as for cupy.arange)
    >>> print(z)
    [0 1 2 3 4 5 6 7 8 9]

    """
    try:
        import cupy as cp
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Cannot create cupy backend because cupy module is not installed.\n"
            "Please install cupy and try again."
        ) from e
    return Backend(lib=cp, device='cuda')
    
def create_torch_backend(device):
    """Create a torch backend with specified device.

    Parameter
    ---------

    device : str
        One of {'cpu', 'cuda', 'cuda:X'} with X = device index.

    Return
    ------

    backend : <class 'backends.Backend'> 
        A backends.Backend instance
        configured to perform operations using `torch` library on the
        specified device.

    Notes 
    -----

    This function will raise an error if torch library is not
    installed or if the input device argument is not available on your
    system.

    See also 
    --------

    backends.create_numpy_backend()
    backends.create_cupy_backend()

    Examples
    --------
    
    The following example illustrate how one can call the
    ``torch.zeros`` and ``torch.arange`` functions from a backend
    output of this :py:func:`create_torch_backend` function.
    
    Example 1: create and use a torch backend on 'cpu' device
    ---------------------------------------------------------

    To run the following example torch library should be installed on
    your system (otherwise the second line of this example will raise
    an error).

    >>> import pyepri.backends as backends
    >>> b = backends.create_torch_backend('cpu')
    >>> x = b.zeros((10,),dtype='float64') # type is specified using a
    >>>                                    # str (here 'float64')
    >>> print(x)
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
           dtype=torch.float64)
    >>> print(x.device) # device is 'cpu' because 'cpu' was used as
    >>>                 # input in line 2 of this example
    cpu
    >>> y = b.arange(3,10,2,dtype='int32') # another example
    >>> print(y)
    tensor([3, 5, 7, 9], dtype=torch.int32)
    >>> print(y.device)
    cpu
    >>> z = b.arange(10) # b.arange behaves as its torch counterpart 
    >>>                  # (`start` and `end` parameters are optional
    >>>                  # as for torch.arange)
    >>> print(z)
    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Example 2: create and use a torch backend on 'cuda' device
    ----------------------------------------------------------

    To run the following example torch library should be installed on
    your system and 'cuda' device should be available (otherwise the
    second line of this example will raise an error)

    >>> import pyepri.backends as backends
    >>> b = backends.create_torch_backend('cuda')
    >>> x = b.zeros((10,),dtype='float64') # type is specified using a
    >>>                                    # str (here 'float64')
    >>> print(x) # device is now cuda:0 (can be cuda:X with X = one 
    >>>          # available device index on your own system)
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0',
           dtype=torch.float64)
    >>> y = b.arange(3,10,2,dtype='int32') # another example
    >>> print(y)
    tensor([3, 5, 7, 9], device='cuda:0', dtype=torch.int32)
    >>> z = b.arange(10)
    >>> print(z)
    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device='cuda:0')

    """

    # check torch library availability
    try:
        import torch
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Cannot create torch backend because torch module is not installed.\n"
            "Please install torch and try again."
        ) from e

    
    # check device validity
    if not isinstance(device,str) or re.match('cpu$|cuda$|(cuda:[0-9]+)$',device) is None:
        raise ValueError(
            "Torch device must be one of {'cpu', 'cuda', 'cuda:X'} with X = device index\n"
            "(e.g. 'cuda:0'). Other kinds of Torch devices are not supported yet. Please use a\n"
            "supported Torch device or ask support to the developper of this package."
        )
    
    # check device availability
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Cannot create torch backend on device '" + device + "' because it seems that your system\n"
                "does not support CUDA (torch.cuda.is_available() returned False).\n"
                "Please fix the CUDA support on your system and try again."
            )
        else:
            try:
                torch.cuda.get_device_properties(device)
            except Exception as e:
                raise RuntimeError(
                    "Cannot create torch backend on device '" + device + "' because \n"
                    "torch.cuda.get_device_properties(" + device + ") raised an error.\n"
                    "This error is likely caused by an invalid device choice. Please fix\n"
                    "this issue and try again."
                ) from e
    
    # instanciate and return Backend object
    return Backend(lib=torch, device=device)


class Backend:
    """Class for mapping our standardized data types and methods to the specified library (numpy, torch, cupy).

   
    Type correspondences
    --------------------

    This package relies on data types specified with a str from which
    the library dependent data types will be inferred. The
    correspondence between the different data types is provided in the
    following Table.
    
    +------------------+-----------------------------+----------------------+
    | str              | lib = numpy or cupy         | lib = torch          |
    +==================+=============================+======================+
    | ``'int8'``       | ``lib.dtype('int8')``       | ``torch.int8``       |
    +------------------+-----------------------------+----------------------+
    | ``'int16'``      | ``lib.dtype('int16')``      | ``torch.int16``      |
    +------------------+-----------------------------+----------------------+
    | ``'int32'``      | ``lib.dtype('int32')``      | ``torch.int32``      |
    +------------------+-----------------------------+----------------------+
    | ``'int64'``      | ``lib.dtype('int64')``      | ``torch.int64``      |
    +------------------+-----------------------------+----------------------+
    | ``'float32'``    | ``lib.dtype('float32')``    | ``torch.float32``    |
    +------------------+-----------------------------+----------------------+
    | ``'float64'``    | ``lib.dtype('float64')``    | ``torch.float64``    |
    +------------------+-----------------------------+----------------------+
    | ``'complex64'``  | ``lib.dtype('complex64')``  | ``torch.complex64``  |
    +------------------+-----------------------------+----------------------+
    | ``'complex128'`` | ``lib.dtype('complex128')`` | ``torch.complex128`` |
    +------------------+-----------------------------+----------------------+
    | ``None``         | ``lib.dtype(None)``         | ``None``             | 
    +------------------+-----------------------------+----------------------+    
    
    The mapping between the data types in str format and those of the
    targeted library can be done using the dtypes and invdtypes
    dictionary attributes described below.

    We also provide a str to str mapping towards complex data types
    (non invertible mapping) :

    +-------------------+---------------------------+
    | data type         | complex data type         |
    +===================+===========================+
    | ``'int8'``        | ``None``                  |
    +-------------------+---------------------------+
    | ``'int16'``       | ``None``                  |
    +-------------------+---------------------------+
    | ``'int32'``       | ``'complex64'``           |
    +-------------------+---------------------------+
    | ``'float32'``     | ``'complex64'``           |
    +-------------------+---------------------------+
    | ``'complex64'``   | ``'complex64'``           |
    +-------------------+---------------------------+
    | ``'int64'``       | ``'complex128'``          |
    +-------------------+---------------------------+
    | ``'float64'``     | ``'complex128'``          |
    +-------------------+---------------------------+
    | ``'complex128'``  | ``'complex128'``          |
    +-------------------+---------------------------+

    The above mapping can be used to infer the data type of a complex
    array computed from another (non necessarily complex) input array
    (e.g., infer the data type of the discrete Fourier transform of an
    input array).
    
    Contextual attributes 
    ---------------------

    lib : <class 'module'>
        One of {numpy, cupy, torch}.

    device : str 
        Device identifier (see constructor documentation below).

    cls : <class 'type'>
        Native array data type in the backend library, as
        described below.

        +-------+-------------------------+
        | lib   | cls                     |
        +=======+=========================+
        | numpy | <class 'numpy.ndarray'> |
        +-------+-------------------------+
        | cupy  | <class 'cupy.ndarray'>  |
        +-------+-------------------------+
        | torch | <class 'torch.Tensor'>  |
        +-------+-------------------------+

    Data type mapping attributes
    ----------------------------
    
    str_to_lib_dtypes : dict
        Mapping data types in standardized str format -> lib dependent
        data types (see above).

    lib_to_str_dtypes : dict
        Invert of the str_to_lib_dtypes mapping (lib dependent data
        types -> data types in standardized str format).

    mapping_to_complex_dtypes : dict 
        Mapping from data type in standardized str format -> complex
        data type in standardized str format (see above).

    Other attributes (library-dependent methods)
    --------------------------------------------

    An instance ``backend`` with class ``pyepri.backends.Backend``
    remaps library dependent methods to basically the same methods
    coming from ``backend.lib`` but with standardized usage
    (e.g. ``backend.meshgrid`` can remap to ``{numpy or cupy or
    torch}.meshgrid`` depending on ``backend.lib``).

    A mini documentation is provided for each standardized method and
    can be displayed using the ``help()`` function, as illustrated
    below.

    >>> import pyepri.backends as backends
    >>> backend = backends.create_numpy_backend()
    >>> help(backend.meshgrid)
    ...Help on function <lambda> in module pyepri.backends:
    ...
    ...<lambda> lambda *xi, indexing='xy'
    ...    return numpy.meshgrid(*xi, indexing=indexing)

    """

    def __init__(self, lib, device):
        """Backend object constructor from the specified library and device.
        
        Parameters
        ----------
        
        lib : str
            One of {'numpy','cupy','torch'} used to specify which
            backend library must be mapped.
        
        device : str
            Device identifier, possible values depends on the lib
            parameters, as described below
        
            +-------+------------------------------------+
            | lib   | possible values for device         |
            +=======+====================================+
            | numpy | ``'cpu'``                          |
            +-------+------------------------------------+
            | cupy  | ``'cuda'``                         |
            +-------+------------------------------------+
            | torch | ``'cuda'`` or ``'cuda:X'``         |
            |       | (where X = available device index) |
            +-------+------------------------------------+
        
        Returns 
        -------
        
        A Backend object
        
        WARNING
        -------
        
        This constructor does not implement any validity test for the
        device input argument.
        
        """
        
        # check lib input validity
        if not hasattr(lib, '__name__') or lib.__name__ not in ['numpy','cupy','torch'] :
            raise ValueError("lib must be one of {numpy, cupy, torch}")
        
        # set contextual attibutes
        self.lib = lib
        if 'numpy' == lib.__name__:
            self.device = 'cpu'
            self.cls = lib.ndarray
        elif 'torch' == lib.__name__:
            self.device = device
            self.cls = lib.Tensor
        else:
            self.device = 'cuda'
            self.cls = lib.ndarray

        # add backend compliance verification method
        self.is_backend_compliant = lambda *args : all([isinstance(arg, self.cls) for arg in args])

        # set mapping str data type -> lib data type
        if lib.__name__ in ['numpy','cupy']: # lib = numpy or cupy
            
            self.str_to_lib_dtypes = {
                'int8'      : lib.dtype('int8'),
                'int16'      : lib.dtype('int16'),
                'int32'      : lib.dtype('int32'),
                'int64'      : lib.dtype('int64'),
                'float32'    : lib.dtype('float32'),
                'float64'    : lib.dtype('float64'),
                'complex64'  : lib.dtype('complex64'),
                'complex128' : lib.dtype('complex128'),
                None         : None,
            }
        
        else: # lib = torch
            
            self.str_to_lib_dtypes = {
                'float32'    : lib.float32,
                'float64'    : lib.float64,
                'int8'       : lib.int8,
                'int16'      : lib.int16,
                'int32'      : lib.int32,
                'int64'      : lib.int64,
                'complex64'  : lib.complex64,
                'complex128' : lib.complex128,
                None         : None,
            }

        # mapping to complex datatype (non invertible mapping)
        self.mapping_to_complex_dtypes = {
            'int8'       : None,
            'int16'      : None,
            'int32'      : 'complex64',
            'float32'    : 'complex64',
            'complex64'  : 'complex64',
            'int64'      : 'complex128',
            'float64'    : 'complex128',
            'complex128' : 'complex128',
        }

        # set invert mapping : lib data type -> str data type
        self.lib_to_str_dtypes = {value: key for key, value in self.str_to_lib_dtypes.items()}
        
        # set backend's methods that are common to all lib
        self.sqrt = lib.sqrt
        self.sin = lib.sin
        self.cos = lib.cos
        self.arccos = lib.arccos
        self.arctan2 = lib.arctan2
        self.real = lib.real
        self.argwhere = lib.argwhere
        self.abs = lib.abs
        self.tile = lib.tile
        self.moveaxis = lib.moveaxis
        self.meshgrid = lambda *xi, indexing='xy' : lib.meshgrid(*xi, indexing=indexing)
        self.exp = lambda arr, out=None : lib.exp(arr, out=out)

        # set minimal doc for the above defined lambda functions
        self.meshgrid.__doc__ = "return " + lib.__name__ + ".meshgrid(*xi, indexing=indexing)\n"
        self.exp.__doc__ = "return " + lib.__name__ + ".exp(arr, out=out)"
        
        # set lib-dependent backends methods
        if lib.__name__ in ['numpy','cupy']: 
            
            # remap lib-dependant methods using lambda functions
            self.zeros = lambda *size, dtype=None : lib.zeros(*size, dtype=lib.dtype(dtype))
            self.fftshift = lambda u, dim=None : lib.fft.fftshift(u, axes=dim)
            self.ifftshift = lambda u, dim=None : lib.fft.ifftshift(u, axes=dim)
            self.arange = lambda *args, dtype=None : functools.partial(lib.arange, dtype=dtype)(*args)
            self.linspace = lambda *args, dtype=None : functools.partial(lib.linspace, dtype=dtype)(*args)
            self.cumsum = lambda u, dim : lib.cumsum(u, axis=dim)
            self.randperm = lambda n, dtype='int64' : lib.random.permutation(n).astype(dtype)
            self.rand = lambda *dims, dtype='float32' : lib.random.rand(*dims).astype(dtype)
            self.randn = lambda size, dtype='float32', mean=0., std=1. : lib.random.normal(size=size, loc=mean, scale=std).astype(dtype)
            self.erfc = lambda x, out=None : sp.special.erfc(x, out=out)
            self.is_complex = lambda x : lib.iscomplexobj(x)
            self.cast = lambda x, dtype : x.astype(self.str_to_lib_dtypes[dtype])
            self.transpose = lambda x : x.transpose()
            self.copy = lambda x : lib.copy(x)
            self.maximum = lambda x1, x2 : lib.maximum(x1, x2)
            self.stack = lambda arrays, dim=0, out=None : lib.stack(arrays, axis=dim, out=out)
            self.quantile = lambda u, q, dim=None, keepdim=False, out=None, interpolation='linear' : lib.quantile(u, q, axis=dim, keepdims=keepdim, out=out, method=interpolation)
            self.frombuffer = lambda buffer, dtype='float32', count=-1, offset=0 : lib.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
            if 'numpy' == lib.__name__:
                self.to_numpy = lambda x : x
                self.from_numpy = lambda x : x
            else :
                self.to_numpy = lambda x : lib.asnumpy(x)
                self.from_numpy = lambda x : lib.asarray(x)

            # set minimal doc for the above defined lambda functions
            self.zeros.__doc__ = "return " + lib.__name__ + ".zeros(*size, dtype=" + lib.__name__ + ".dtype(dtype))"
            self.fftshift.__doc__ = "return " + lib.__name__ + ".fft.fftshift(u, axes=dim)"
            self.ifftshift.__doc__ = "return " + lib.__name__ + ".fft.ifftshift(u, axes=dim)"
            self.arange.__doc__ = "return functools.partial(" + lib.__name__ + ".arange, dtype=dtype)(*args)"
            self.linspace.__doc__ = "return functools.partial(" + lib.__name__ + ".linspace, dtype=dtype)(*args)"
            self.cumsum.__doc__ = "return " + lib.__name__ + ".cumsum(u, axis=dim)"
            self.rand.__doc__ = "return " + lib.__name__ + ".random.rand(*dims).astype(dtype)"
            self.randn.__doc__ = "return " + lib.__name__ + ".random.normal(size=size, loc=mean, scale=std).astype(dtype)"
            self.randperm.__doc__ = "return " + lib.__name__ + ".random.permutation(n).astype(dtype)"
            self.transpose.__doc__ = "return x.transpose()"
            self.copy.__doc__ = "return " + lib.__name__ + ".copy(x)"
            self.maximum.__doc__ = "return " + lib.__name__ + ".maximum(x1, x2)"
            self.erfc.__doc__ = "return scipy.special.erfc(x, out=out)"
            self.is_complex.__doc__ = "return" + lib.__name__ + ".iscomplexobj(x)"
            self.to_numpy.__doc__ = "return " + ("x" if 'numpy' == lib.__name__ else "cupy.asnumpy(x)")
            self.from_numpy.__doc__ = "return " + ("x" if 'numpy' == lib.__name__ else "cupy.asarray(x)")
            self.is_backend_compliant.__doc__ = (
                "return all([isinstance(arg, " + lib.__name__ + ".ndarray) for arg in args])"
            )
            self.cast.__doc__ = (
                "return x.astype(self.str_to_lib_dtypes[dtype]), where `self` denotes the \n"
                "backends.Backend class instance from wich this lambda function belongs to."
            )
            self.stack.__doc__ = "return " + lib.__name__ + ".stack(arrays, axis=dim, out=out)"
            self.quantile.__doc__ = "return " + lib.__name__ + ".quantile(u, q, axis=dim, keepdims=keepdim, out=out, method=interpolation)"
            self.frombuffer.__doc__ = "return " + lib.__name__ + ".frombuffer(buffer, dtype=" + lib.__name__ + ".dtype(dtype), count=count, offset=offset)"

            # deal with FFT support (differences of output data type
            # inference exist between numpy and cupy)
            if lib.__name__ == "numpy":
                
                # direct transform: default behavior of direct FFT
                # functions provided in numpy.fft is to return a
                # numpy.complex128 output array whatever the data type
                # of the input array. We will cast the output with a
                # custom complex data type inferred from the input
                # (which is the default behavior for `cupy.fft` and
                # `torch.fft` functions.
                self.rfft = lambda u, n=None, dim=-1, norm=None : lib.fft.rfft(u, n=n, axis=dim, norm=norm).astype(self.mapping_to_complex_dtypes[self.lib_to_str_dtypes[u.dtype]])                
                self.fft = lambda u, n=None, dim=-1, norm=None : lib.fft.fft(u, n=n, axis=dim, norm=norm).astype(self.mapping_to_complex_dtypes[self.lib_to_str_dtypes[u.dtype]])
                self.rfft2 = lambda u, s=None, dim=(-2, -1), norm=None : lib.fft.rfft2(u, s=s, axes=dim, norm=norm).astype(self.mapping_to_complex_dtypes[self.lib_to_str_dtypes[u.dtype]])                
                self.fft2 = lambda u, s=None, dim=(-2, -1), norm=None : lib.fft.fft2(u, s=s, axes=dim, norm=norm).astype(self.mapping_to_complex_dtypes[self.lib_to_str_dtypes[u.dtype]])
                self.rfftn = lambda u, s=None, dim=None, norm=None : lib.fft.rfftn(u, s=s, axes=dim, norm=norm).astype(self.mapping_to_complex_dtypes[self.lib_to_str_dtypes[u.dtype]])                
                self.fftn = lambda u, s=None, dim=None, norm=None : lib.fft.fftn(u, s=s, axes=dim, norm=norm).astype(self.mapping_to_complex_dtypes[self.lib_to_str_dtypes[u.dtype]])
                
                # inverse transform: the default behavior of numpy.fft.irfft is to return a numpy.float64 array whaterver the data type of the input array. We provide below 
                mapping_to_real_dtypes = {
                    'int32'      : 'float32',
                    'float32'    : 'float32',
                    'complex64'  : 'float32',
                    'int64'      : 'float64',
                    'float64'    : 'float64',
                    'complex128' : 'float64',
                }        
                self.irfft = lambda u, n=None, dim=-1, norm=None : lib.fft.irfft(u, n=n, axis=dim, norm=norm).astype(mapping_to_real_dtypes[self.lib_to_str_dtypes[u.dtype]])
                self.ifft = lambda u, n=None, dim=-1, norm=None : lib.fft.ifft(u, n=n, axis=dim, norm=norm).astype(self.mapping_to_complex_dtypes[self.lib_to_str_dtypes[u.dtype]])
                self.irfft2 = lambda u, s=None, dim=(-2, -1), norm=None : lib.fft.irfft2(u, s=s, axes=dim, norm=norm).astype(mapping_to_real_dtypes[self.lib_to_str_dtypes[u.dtype]])
                self.ifft2 = lambda u, s=None, dim=(-2, -1), norm=None : lib.fft.ifft2(u, s=s, axes=dim, norm=norm).astype(self.mapping_to_complex_dtypes[self.lib_to_str_dtypes[u.dtype]])
                self.irfftn = lambda u, s=None, dim=None, norm=None : lib.fft.irfftn(u, s=s, axes=dim, norm=norm).astype(mapping_to_real_dtypes[self.lib_to_str_dtypes[u.dtype]])
                self.ifftn = lambda u, s=None, dim=None, norm=None : lib.fft.ifftn(u, s=s, axes=dim, norm=norm).astype(self.mapping_to_complex_dtypes[self.lib_to_str_dtypes[u.dtype]])
                
                # set minimal doc for the above defined lambda functions
                self.rfft.__doc__ = (
                    "return " + lib.__name__ + ".fft.rfft(u, n=n, axis=dim, norm=norm).astype(cdtype)\n"
                    "where `cdtype` is a complex data type inferred from `u`."
                )
                self.fft.__doc__ = (
                    "return " + lib.__name__ + ".fft.fft(u, n=n, axis=dim, norm=norm).astype(cdtype)\n"
                    "where `cdtype` is a complex data type inferred from `u`."
                )
                self.rfft2.__doc__ = (
                    "return " + lib.__name__ + ".fft.rfft2(u, n=n, axes=dim, norm=norm).astype(cdtype)\n"
                    "where `cdtype` is a complex data type inferred from `u`."
                )
                self.fft2.__doc__ = (
                    "return " + lib.__name__ + ".fft.fft2(u, n=n, axes=dim, norm=norm).astype(cdtype)\n"
                    "where `cdtype` is a complex data type inferred from `u`."
                )
                self.rfftn.__doc__ = (
                    "return " + lib.__name__ + ".fft.rfftn(u, n=n, axes=dim, norm=norm).astype(cdtype)\n"
                    "where `cdtype` is a complex data type inferred from `u`."
                )
                self.fftn.__doc__ = (
                    "return " + lib.__name__ + ".fft.fftn(u, n=n, axes=dim, norm=norm).astype(cdtype)\n"
                    "where `cdtype` is a complex data type inferred from `u`."
                )
                self.irfft.__doc__ = (
                    "return " + lib.__name__ + ".fft.irfft(u, n=n, axis=dim, norm=norm).astype(dtype)\n"
                    "where `dtype` is a real data type inferred from `u`."
                )
                self.ifft.__doc__ = (
                    "return " + lib.__name__ + ".fft.ifft(u, n=n, axis=dim, norm=norm).astype(cdtype)\n"
                    "where `cdtype` is a complex data type inferred from `u`."
                )
                self.irfft2.__doc__ = (
                    "return " + lib.__name__ + ".fft.irfft2(u, n=n, axes=dim, norm=norm).astype(dtype)\n"
                    "where `dtype` is a real data type inferred from `u`."
                )
                self.ifft2.__doc__ = (
                    "return " + lib.__name__ + ".fft.ifft2(u, n=n, axes=dim, norm=norm).astype(cdtype)\n"
                    "where `cdtype` is a complex data type inferred from `u`."
                )
                self.irfftn.__doc__ = (
                    "return " + lib.__name__ + ".fft.irfftn(u, n=n, axes=dim, norm=norm).astype(dtype)\n"
                    "where `dtype` is a real data type inferred from `u`."
                )
                self.ifftn.__doc__ = (
                    "return " + lib.__name__ + ".fft.ifftn(u, n=n, axes=dim, norm=norm).astype(cdtype)\n"
                    "where `cdtype` is a complex data type inferred from `u`."
                )
                
            else: # lib == "cupy"
                # default output data type inference provided by
                # `cupy.fft` functions are kept unchanged
                self.rfft = lambda u, n=None, dim=-1, norm=None : lib.fft.rfft(u, n=n, axis=dim, norm=norm)
                self.irfft = lambda u, n=None, dim=-1, norm=None : lib.fft.irfft(u, n=n, axis=dim, norm=norm)
                self.fft = lambda u, n=None, dim=-1, norm=None : lib.fft.fft(u, n=n, axis=dim, norm=norm)
                self.ifft = lambda u, n=None, dim=-1, norm=None : lib.fft.ifft(u, n=n, axis=dim, norm=norm)
                self.rfft2 = lambda u, s=None, dim=(-2, -1), norm=None : lib.fft.rfft2(u, s=s, axes=dim, norm=norm)
                self.irfft2 = lambda u, s=None, dim=(-2, -1), norm=None : lib.fft.irfft2(u, s=s, axes=dim, norm=norm)
                self.fft2 = lambda u, s=None, dim=(-2, -1), norm=None : lib.fft.fft2(u, s=s, axes=dim, norm=norm)
                self.ifft2 = lambda u, s=None, dim=(-2, -1), norm=None : lib.fft.ifft2(u, s=s, axes=dim, norm=norm)
                self.rfftn = lambda u, s=None, dim=None, norm=None : lib.fft.rfftn(u, s=s, axes=dim, norm=norm)
                self.irfftn = lambda u, s=None, dim=None, norm=None : lib.fft.irfftn(u, s=s, axes=dim, norm=norm)
                self.fftn = lambda u, s=None, dim=None, norm=None : lib.fft.fftn(u, s=s, axes=dim, norm=norm)
                self.ifftn = lambda u, s=None, dim=None, norm=None : lib.fft.ifftn(u, s=s, axes=dim, norm=norm)
                self.rfft.__doc__ = "return " + lib.__name__ + ".fft.rfft(u, n=n, axis=dim, norm=norm)"
                self.irfft.__doc__ = "return " + lib.__name__ + ".fft.irfft(u, n=n, axis=dim, norm=norm)"
                self.fft.__doc__ = "return " + lib.__name__ + ".fft.fft(u, n=n, axis=dim, norm=norm)"
                self.ifft.__doc__ = "return " + lib.__name__ + ".fft.ifft(u, n=n, axis=dim, norm=norm)"
                self.rfft2.__doc__ = "return " + lib.__name__ + ".fft.rfft2(u, n=n, axes=dim, norm=norm)"
                self.irfft2.__doc__ = "return " + lib.__name__ + ".fft.irfft2(u, n=n, axes=dim, norm=norm)"
                self.fft2.__doc__ = "return " + lib.__name__ + ".fft.fft2(u, n=n, axes=dim, norm=norm)"
                self.ifft2.__doc__ = "return " + lib.__name__ + ".fft.ifft2(u, n=n, axes=dim, norm=norm)"
                self.rfftn.__doc__ = "return " + lib.__name__ + ".fft.rfftn(u, n=n, axes=dim, norm=norm)"
                self.irfftn.__doc__ = "return " + lib.__name__ + ".fft.irfftn(u, n=n, axes=dim, norm=norm)"
                self.fftn.__doc__ = "return " + lib.__name__ + ".fft.fftn(u, n=n, axes=dim, norm=norm)"
                self.ifftn.__doc__ = "return " + lib.__name__ + ".fft.ifftn(u, n=n, axes=dim, norm=norm)"
            
            # nufft support (use finufft for CPU device and cufinufft
            # for GPU device)
            if lib.__name__ == "numpy":
                import finufft
                self.nufft2d = finufft.nufft2d2
                self.nufft3d = finufft.nufft3d2
                self.nufft2d_adjoint = finufft.nufft2d1
                self.nufft3d_adjoint = finufft.nufft3d1
            else:
                import cufinufft
                self.nufft2d = cufinufft.nufft2d2
                self.nufft3d = cufinufft.nufft3d2
                self.nufft2d_adjoint = cufinufft.nufft2d1
                self.nufft3d_adjoint = cufinufft.nufft3d1
                        
        else: # lib == torch
            
            # remap some lib-dependant methods using lambda functions
            self.zeros = lambda *size, dtype=None : lib.zeros(*size, dtype=self.str_to_lib_dtypes[dtype], device=device)
            self.arange = lambda *args, dtype=None : functools.partial(lib.arange, dtype=self.str_to_lib_dtypes[dtype], device=device)(*args)
            self.linspace = lambda *args, dtype=None : functools.partial(lib.linspace, dtype=self.str_to_lib_dtypes[dtype], device=device)(*args)
            self.rand = lambda *dims, dtype='float32' : lib.rand(*dims, dtype=self.str_to_lib_dtypes[dtype], device=device)
            self.randperm = lambda n, dtype='int64' : lib.randperm(n, dtype=self.str_to_lib_dtypes[dtype], device=device)
            self.randn = lambda *size, dtype='float32', mean=0., std=1. : mean + std * lib.randn(*size, dtype=self.str_to_lib_dtypes[dtype], device=device)
            self.erfc = lambda x, out=None : lib.erfc(x, out=out)
            self.is_complex = lambda x : x.is_complex()
            self.to_numpy = lambda x : x.detach().cpu().numpy()
            #self.from_numpy = lambda x : lib.from_numpy(x).to(device, copy=True) # confilct between pytorch and numpy 2.0.0
            self.from_numpy = lambda x : lib.Tensor(x).to(device, copy=True)
            self.cast = lambda x, dtype : x.type(self.str_to_lib_dtypes[dtype])
            self.transpose = lambda x : x.moveaxis((0,1),(1,0))
            self.copy = lambda x : lib.clone(x).detach()
            self.maximum = lambda x1, x2 : lib.maximum(lib.as_tensor(x1), lib.as_tensor(x2))
            self.stack = lambda arrays, dim=0, out=None : lib.stack(arrays, dim=dim, out=out)
            self.quantile = lambda u, q, dim=None, keepdim=False, out=None, interpolation='linear' : lib.quantile(u, q, dim=dim, keepdim=keepdim, out=out, interpolation=interpolation)
            self.frombuffer = lambda buffer, dtype='float32', count=-1, offset=0 : lib.frombuffer(buffer, dtype=self.str_to_lib_dtypes[dtype], count=count, offset=offset)

            # remap some other lib-dependent methods using direct
            # mappings
            self.rfft = lib.fft.rfft
            self.irfft = lib.fft.irfft
            self.fft = lib.fft.fft
            self.ifft = lib.fft.ifft
            self.rfft2 = lib.fft.rfft2
            self.irfft2 = lib.fft.irfft2
            self.fft2 = lib.fft.fft2
            self.ifft2 = lib.fft.ifft2
            self.rfftn = lib.fft.rfftn
            self.irfftn = lib.fft.irfftn
            self.fftn = lib.fft.fftn
            self.ifftn = lib.fft.ifftn
            self.fftshift = lib.fft.fftshift
            self.ifftshift = lib.fft.ifftshift
            self.cumsum = lib.cumsum

            # set minimal doc for the above defined lambda functions
            self.zeros.__doc__ = (
                "return torch.zeros(*size, dtype=self.str_to_lib_dtypes[dtype], device='" + self.device + "')\n"
                "where `self` denotes the backends.Backend class instance from wich this lambda\n"
                "function belongs to."
            )
            self.arange.__doc__ = (
                "return functools.partial(torch.arange, dtype=self.str_to_lib_dtypes[dtype], device='" + self.device + "')(*args)\n"
                "where `self` denotes the backends.Backend class instance from wich this lambda\n"
                "function belongs to."
            )
            self.linspace.__doc__ = (
                "return functools.partial(torch.linspace, dtype=self.str_to_lib_dtypes[dtype], device='" + self.device + "')(*args)\n"
                "where `self` denotes the backends.Backend class instance from wich this lambda\n"
                "function belongs to."
            )
            self.rand.__doc__ = (
                "return torch.rand(*dims, dtype=self.str_to_lib_dtypes[dtype], device='" + self.device + "')"
                "where `self` denotes the backends.Backend class instance from wich this lambda\n"
                "function belongs to."
            )
            self.randperm.__doc__ = (
                "return torch.randperm(n, dtype=self.str_to_lib_dtypes[dtype], device='" + self.device + "')"
                "where `self` denotes the backends.Backend class instance from wich this lambda\n"
                "function belongs to."
            )
            self.randn.__doc__ = (
                "return mean + std * torch.randn(*size, dtype=self.str_to_lib_dtypes[dtype], device='" + self.device + "')"
                "where `self` denotes the backends.Backend class instance from wich this lambda\n"
                "function belongs to."
            )
            self.cast.__doc__ = (
                "return x.type(self.str_to_lib_dtypes[dtype]), where `self` denotes the backends.Backend\n"                
                "class instance from wich this lambda function belongs to."
            )
            self.transpose.__doc__ = "return x.moveaxis((0,1),(1,0))"
            self.copy.__doc__ = "return "+ lib.__name__ + ".clone(x).detach()"
            self.maximum.__doc__ = "return "+ lib.__name__ + ".maximum(" + lib.__name__ + ".as_tensor(x1), " + lib.__name__ + ".as_tensor(x2))"
            self.stack.__doc__ = "return " + lib.__name__ + ".stack(arrays, dim=dim, out=out)"
            self.erfc.__doc__ = "return torch.erfc(x, out=out)"
            self.is_complex.__doc__ = "return x.is_complex()"
            self.to_numpy.__doc__ = "return x.detach().cpu().numpy()"
            #self.from_numpy.__doc__ = "return torch.from_numpy(x).to('" + self.device + "', copy=True)"
            self.from_numpy.__doc__ = "return torch.Tensor(x).to('" + self.device + "', copy=True)"
            self.is_backend_compliant.__doc__ = "return all([isinstance(arg, torch.Tensor) for arg in args])"
            self.quantile.__doc__ = "return "+ lib.__name__ + ".quantile(u, q, dim=dim, keepdim=keepdim, out=out, interpolation=interpolation)"
            self.frombuffer.__doc__ = (
                "return " + lib.__name__ + ".frombuffer(buffer, dtype=self.str_to_lib_dtypes[dtype], count=count, offset=offset)\n"
                "where `self` denotes the backends.Backend class instance from wich this lambda\n"
                "function belongs to."
                )
            
            # nufft support (use finufft for CPU device and cufinufft
            # for GPU device)
            if device == "cpu" :
                import finufft

                # define decorator for finufft functions (those
                # function do not natively accept torch.Tensor input
                # arrays, such input need to be cast into
                # numpy.ndarray.
                def numpyfy(func):
                    '''Decorator to cast torch.Tensor inputs of func into numpy.ndarray
                    and the numpy.ndarray output of func into
                    torch.Tensor.

                    '''
                    def numpyfied_func(*args, **kwargs):
                        args2 = (a.numpy() if isinstance(a, lib.Tensor) else a for a in args)
                        kwargs2 = {key: val.numpy() if isinstance(val, lib.Tensor) else val for key, val in kwargs.items()}
                        return lib.from_numpy(func(*args2, **kwargs2))
                    return numpyfied_func

                # decorate finufft functions 
                self.nufft2d = numpyfy(finufft.nufft2d2)
                self.nufft3d = numpyfy(finufft.nufft3d2)
                self.nufft2d_adjoint = numpyfy(finufft.nufft2d1)
                self.nufft3d_adjoint = numpyfy(finufft.nufft3d1)

                # add short documentation
                self.nufft2d.__doc__ = (
                    "same as finufft.nufft2d2 but torch.Tensor inputs are cast into numpy.ndarray\n"                    
                    "and output is cast into torch.Tensor. Type `help(finufft.nufft2d2)`\n"
                    "for more details."                    
                )
                self.nufft3d.__doc__ = (
                    "same as finufft.nufft3d2 but torch.Tensor inputs are cast into numpy.ndarray\n"                    
                    "and output is cast into torch.Tensor. Type `help(finufft.nufft3d2)`\n"
                    "for more details."                    
                )
                self.nufft2d_adjoint.__doc__ = (
                    "same as finufft.nufft2d1 but torch.Tensor inputs are cast into numpy.ndarray\n"                    
                    "and output is cast into torch.Tensor. Type `help(finufft.nufft2d1)`\n"
                    "for more details."                    
                )
                self.nufft3d_adjoint.__doc__ = (
                    "same as finufft.nufft3d1 but torch.Tensor inputs are cast into numpy.ndarray\n"                    
                    "and output is cast into torch.Tensor. Type `help(finufft.nufft3d1)`\n"
                    "for more details."                    
                )
                
            else:
                import cufinufft
                self.nufft2d = cufinufft.nufft2d2
                self.nufft3d = cufinufft.nufft3d2
                self.nufft2d_adjoint = cufinufft.nufft2d1
                self.nufft3d_adjoint = cufinufft.nufft3d1
            
