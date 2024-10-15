"""This module contains functions for computing frequency attenuation
(a.k.a., apodization) profiles and performing frequency attenuation
operations by circular convolution.

"""
import math
import pyepri.checks as checks

def rfftconvol(u, rfft_filter, backend=None, dim=-1, notest=False):
    """1D Fourier convolution (a.k.a. Circular convolution) for real input signal.

    Parameters 
    ----------

    u : array_like (with type `backends.cls`)
        Input signal, can be multidimensional, the filtering operation
        will be applied along a single dimension (broadcasting).

    rfft_filter : array_like (with type `backend.cls`)
        Input convolution filter in Fourier domain, can be either
        mono-dimensional with length u.shape[dim]//2+1 OR
        multidimensional with shape satisfying 
                              
        + rfft_filter.shape[k] = u.shape[k]//2+1 if k == dim;
        + rfft_filter.shape[k] = u.shape[k] otherwise.

    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input arrays ``(u, rfft_filter)``.

    dim : int, optional
        Dimension along which the one-dimensional direct/inverse real
        FFTs (see how output is computed below) will be computed.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------

    out : array_like (with type `backend.cls`)
        Filtered signal computed using 

        `v = backend.irfft(backend.rfft(u, dim=dim) * rfft_filter,
        n=u.shape[dim], dim=dim)`.

    Example
    -------
    
    >>> ##################
    >>> # import modules #
    >>> ##################
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyepri.apodization as apodization
    >>> import pyepri.backends as backends
    >>> 
    >>> ######################################################
    >>> # create a numpy backend (you can try other choices) #
    >>> ######################################################
    >>> backend = backends.create_numpy_backend()
    >>>
    >>> #################################
    >>> # compute a random input signal #
    >>> #################################
    >>> u = backend.rand(100,) 
    >>>
    >>> ###########################################################
    >>> # compute filter in the frequency domain over half of the #
    >>> # full frequency domain (here a smoothed step profile)    #
    >>> ###########################################################
    >>> t = backend.arange(len(u)//2+1, dtype='float32') 
    >>> m, sig = 40, 2.
    >>> rfft_filter = apodization.smoothstep(t, m, sig, backend=backend)
    >>> 
    >>> ###########################
    >>> # compute filtered signal #
    >>> ###########################
    >>> v = apodization.rfftconvol(u, rfft_filter, backend=backend)
    >>> 
    >>> ###################
    >>> # display signals #
    >>> ###################
    >>> plt.figure(figsize=(10,5))
    >>> plt.subplot(1,2,1)
    >>> plt.plot(backend.to_numpy(u))
    >>> plt.plot(backend.to_numpy(v))
    >>> plt.title("input & filtered signals")
    >>> plt.legend(["input signal", "filtered signal"])
    >>> plt.subplot(1,2,2)
    >>> plt.plot(backend.to_numpy(t),backend.to_numpy(rfft_filter))
    >>> plt.xlabel("frequency indexes (half of the full domain)")
    >>> plt.title("filter in Fourier domain (frequency attenuation profile)")
    >>> plt.show()

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u)

    # consistency checks
    if not notest:
        _check_nd_inputs_("rfftconvol", backend, u=u,
                          rfft_filter=rfft_filter, dim=dim)

    # compute and return the filtered signal
    return backend.irfft(backend.rfft(u, dim=dim) * rfft_filter,
                         n=u.shape[dim], dim=dim)


def smoothstep(t, m, sig, backend=None, notest=False):
    r"""Apodization profile computed as the convolution between a rectangular step function and a Gaussian function.

    The computed profile roughly (but not exactly) satisfies:
    
    + h(t) = 1 for abs(t) < m/2 - 5*sig
    + h(t) = 0 for abs(t) > m/2 + 5*sig
    
    leading to the following "transition intervals" (where the profile
    increases from 0 to 1 or decreases from 1 to 0):  
    
    + -m/2 + [-5*sig, 5*sig] : the profile increases from roughly 0 to
      roughly 1;
    
    + m/2 + [-5*sig, 5*sig] : the profile decreases from roughly 1 to
      roughly 0.
    
    Parameters
    ----------
    
    t : array_like (with type `backend.cls`)
        Input sampling points for the apodization profile.
    
    m : float
        Support length for the step function (function f in the above
        mathematical description).
    
    sig : float
        Smoothing parameter of the apodization profile (= standard
        deviation of the Gaussian profile g in the above mathematical
        description).
    
    backend : <class 'pyepri.backends.Backend'> 
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input array ``t``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    h : array_like (with type `backend.cls`)
        Computed values of h(t) (same shape as input t). 

    Notes
    -----
    
    This module computes
    
    .. math:: h(t) = (f * g) (t)
    
    where :math:`*` denotes the convolution product, :math:`f` is the
    rectangular step function with support length `m` defined by

    .. math:: \forall x\in\mathbb{R}\,, \quad f(x) = \left\{\begin{array}{cl}x&\text{if } |x| \leq \frac{m}{2}\\0&\text{otherwise,}\end{array}\right.
    
    and :math:`g` is the Gaussian function with standard deviation
    :math:`\sigma` (corresponding the the input parameter `sig`)
    defined by
    
    .. math:: \forall x\in\mathbb{R}\,,\quad g(x) = \frac{1}{\sigma \sqrt{2\pi}} \cdot \exp{\left(-\frac{x^2}{2\sigma^2}\right)}\,.
    
    The practical evaluation of the output smooth step apodization profile `h`
    is done using
    
    .. math:: \forall t\in \mathbb{R}\,,\quad h(t) = \frac{1}{2} \cdot \left(\mathrm{erfc}\left(\frac{t-m/2}{\sigma \sqrt{2}}\right) - \mathrm{erfc}\left(\frac{t+m/2}{\sigma \sqrt{2}}\right)\right)
    
    where 

    .. math:: \forall t \in \mathbb{R}\,,\quad \mathrm{erfc}(t) = \frac{2}{\sqrt{\pi}} \int_{t}^{+\infty} e^{-x^2} \, dx

    is the complementary error function.    
    
    Example
    -------
    
    >>> ##################
    >>> # import modules #
    >>> ##################
    >>> import numpy as np
    >>> import pyepri.backends as backends
    >>> import pyepri.apodization as apodization
    >>> import matplotlib.pyplot as plt
    >>>
    >>> ##########################
    >>> # prepare figure display #
    >>> ##########################
    >>> plt.rcParams.update({'axes.xmargin' : 0, 'axes.ymargin' : 0})
    >>> plt.figure(figsize=(9.5, 5))
    >>> plt.xlabel("t")
    >>> plt.ylabel("h(t): smooth step profile")
    >>> 
    >>> ###############################
    >>> # compute apodization profile #
    >>> ###############################
    >>> backend = backends.create_numpy_backend()
    >>> t = np.linspace(-500, 500, 10000)
    >>> m = 350
    >>> sig = 10
    >>> h = apodization.smoothstep(t, m, sig, backend=backend)
    >>>
    >>> #############################
    >>> # draw the computed profile #
    >>> #############################
    >>> plt.plot(backend.to_numpy(t), backend.to_numpy(h))
    >>>
    >>> #########################################################
    >>> # draw boundaries on the right-side transition interval #
    >>> #########################################################
    >>> plt.plot((m/2-5*sig)*np.array([1, 1]), np.array([-.1, 1.1]), 'k:')
    >>> plt.plot((m/2+5*sig)*np.array([1, 1]), np.array([-.1, 1.1]), 'k:')
    >>>
    >>> #########################################################
    >>> # draw boundaries on the right-side transition interval #
    >>> #########################################################
    >>> plt.plot((-m/2+5*sig)*np.array([1, 1]), np.array([-.1, 1.1]), 'k:')
    >>> plt.plot((-m/2-5*sig)*np.array([1, 1]), np.array([-.1, 1.1]), 'k:')
    >>> plt.legend(('smooth step profile', 'boundaries of the' + '\n' + 
    ... 'transition intervals'), loc='upper right')
    >>> plt.show()
    
    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(t=t)
        
    # consistency checks
    if not notest:
        _check_nd_inputs_("smoothstep", backend, t=t, m=m, sig=sig)
    
    return .5*(backend.erfc((t-.5*m)/(sig*math.sqrt(2.))) -
               backend.erfc((t+.5*m)/(sig*math.sqrt(2.))))

def _check_nd_inputs_(caller, backend, u=None, rfft_filter=None,
                      dim=None, t=None, m=None, sig=None):
    """Factorized consistency checks for functions in the :py:mod:`pyepri.apodization` submodule."""

    ##################
    # general checks #
    ##################

    # check backend consistency 
    checks._check_backend_(backend, rfft_filter=rfft_filter, t=t)

    # other checks
    checks._check_type_(int, m=m, dim=dim)
    
    #################
    # custom checks #
    #################
    if "rfftconvol" == caller:
        
        # retrieve dtype
        dtype = u.dtype
        str_dtype = backend.lib_to_str_dtypes[dtype]
        str_cdtype = backend.mapping_to_complex_dtypes[str_dtype]
        cdtype = backend.str_to_lib_dtypes[str_cdtype]

        # retrieve dimensions
        s = u.shape
        ndim = u.ndim
        
        # check dim
        if dim not in (-1, *range(ndim)):
            raise RuntimeError(
                "Parameter ``dim`` must be -1 or an integer in the range [0, u.ndim-1]"
            )
        
        # check rfft_filter
        if rfft_filter.dtype not in (dtype, cdtype):
            raise RuntimeError(
                "Inconsistent data type for parameter `rfft_filter` (expected %s or %s)" % (dtype.name, cdtype.name)
            )
        if rfft_filter.ndim == 1:
            ok = len(rfft_filter) == u.shape[dim]//2 + 1
        else:
            ok = True
            for k, s in enumerate(rfft_filter.shape):
                sref = u.shape[k]//2 + 1 if k  == dim else u.shape[k]
                ok = ok and s == sref
        if not ok:
            raise RuntimeError(
                "Inconsistent shape for parameter ``rfft_filter`` (check function documentation)"
            )
        
    elif "smoothstep" == caller:

        # retrieve dtype
        dtype = t.dtype
        str_dtype = backend.lib_to_str_dtypes[dtype]
        str_cdtype = backend.mapping_to_complex_dtypes[str_dtype]
        cdtype = backend.str_to_lib_dtypes[str_cdtype]

        # check m
        if not isinstance(m, (float, int)):
            raise RuntimeError(
                "Parameter ``m`` must be a int or a float"
            )
        
        # check sig
        if not isinstance(sig, (float, int)) or sig <= 0:
            raise RuntimeError(
                "Parameter ``m`` must be a positive float (int is also tolerated)"
            )
    
    return True
