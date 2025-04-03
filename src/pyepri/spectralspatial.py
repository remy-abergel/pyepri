"""This module contains low-level operators related to **spectral-spatial
EPR imaging** (projection, backprojection, projection-backprojection
using Toeplitz kernels). Detailed mathematical definitions of the
operators are provided in the :ref:`mathematical_definitions` section
of the PyEPRI documentation.

"""

import math
import pyepri.checks as checks
from pyepri.monosrc import compute_3d_frequency_nodes

def compute_4d_frequency_nodes(B, delta, fgrad, backend=None,
                               rfft_mode=True, notest=False):
    """Compute 4D irregular frequency nodes involved in 4D projection & backprojection operations.
    
    Parameters
    ----------
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.

    delta : float 
        Pixel size given in a length unit denoted below as
        `[length-unit]` (can be `centimeter (cm)`, `millimeter (mm)`,
        ...).
    
    fgrad : array_like (with type `backend.cls`)
        Two-dimensional array with shape ``(3, fgrad.shape[1])``
        such that ``fgrad[:,k]`` corresponds to the (X,Y,Z)
        coordinates of the field gradient vector associated to the
        k-th EPR projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(B, fgrad)``.
    
    rfft_mode : bool, optional
        Set ``rfft_mode=True`` to compute only half of the frequency
        nodes (to be combined with the use of real FFT functions in
        further processing). Otherwise, set ``rfft_mode=False`` to
        compute all frequency nodes (to be combined with the use of
        complex FFT functions in further processing).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    nodes : dict 
        
        A dictionary with content ``{'x': x, 'y': y, 'z': z, 'xi': xi,
        't', t, 'lt': lt 'indexes': indexes, 'idt': idt, 'rfft_mode':
        rfft_mode}`` where
        
        + ``x, y, z`` are the 3D frequency nodes computed using
          :py:func:`pyepri.monosrc.compute_3d_frequency_nodes`
        
        + ``indexes`` is a one dimensional array, with same length as
          ``x``, ``y`` and ``z``, corresponding to the indexes where
          should be dispatched the computed Fourier coefficients in
          the (r)fft of the 4D projections.
        
        + ``rfft_mode`` a bool specifying whether the frequency nodes
          cover half of the frequency domain (``rfft_mode=True``) or the
          full frequency domain (``rfft_mode=False``).
        
        + ``t`` and ``idt`` are one dimensional arrays such that
          ``t[idt]`` has the same length as ``x``, ``y`` and ``z`` and
          represents the frequency nodes along the B-axis of the 4D
          image (the ``t`` array has no duplicate values). Said
          differently, the 4D frequency nodes involved in the 4D
          projection and backprojection operations are the ``(x, y, z,
          t[idt])`` nodes.
        
        + ``lt`` is a 2D array such that ``lt[l,k] = l * t[k]`` (those
          values are involved in the computation of weights for 4D
          projection and backprojection functions).
    
    
    See also
    --------
    
    pyepri.monosrc.compute_3d_frequency_nodes
    proj4d
    backproj4d
    
    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B, fgrad=fgrad)
    
    # consistency checks
    #if not notest:
    #    _check_nd_inputs_(2, B, delta, fgrad, backend,
    #                      rfft_mode=rfft_mode)    
    
    # retrieve datatype in str format and number of points along the B
    # axis
    dtype = backend.lib_to_str_dtypes[B.dtype]
    Nb = len(B)
    
    # compute the irregularly spaced 3D frequency nodes
    nodes = compute_3d_frequency_nodes(B, delta, fgrad,
                                       backend=backend,
                                       rfft_mode=rfft_mode,
                                       notest=True)
    
    # add the corresponding (regularly sampled) frequency nodes along
    # the spectral axis
    if rfft_mode: 
        alf = backend.arange(1 + Nb//2, dtype=dtype)
    else:
        alf = backend.ifftshift(-(Nb//2) + backend.arange(Nb, dtype=dtype))
    indexes = nodes['indexes']
    xi = (2. * math.pi / float(Nb)) * alf
    t = - backend.copy(xi).reshape((1, -1))
    t = (t * backend.ones((fgrad.shape[1], 1), dtype=dtype)) # todo : remplacer par un repmat
    t = t.reshape((-1,))[indexes]
    t, idt = backend.unique(t, return_inverse=True)
    lt = backend.arange(Nb, dtype=dtype).reshape((-1, 1)) * t.reshape((1, -1))
    nodes.update({'xi': xi, 't': t, 'idt': idt, 'lt': lt})
    
    return nodes


def compute_4d_weights(nodes, backend=None, nrm=(1 + 0j), isign=1,
                       make_contiguous=True, notest=False):
    """Precompute weights to accelerate further proj4d & backproj4d calls.
    
    Precomputing the weights is useful to save computation time when
    multiple evaluations of :py:func:`proj4d` or :py:func:`backproj4d`
    with option ``memory_usage=0`` are needed.
    
    Parameters
    ----------
    
    nodes : dict, optional 
        Frequency nodes computed using
        :py:func:`compute_4d_frequency_nodes`.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays stored into ``nodes``.
        
    nrm : complex, optional
        Normalization parameter involved in the weights definition
        (see below).
    
    isign : float, optional
        Must be equal to +1 or -1, used as sign in the complex
        exponential defining the weights (see below). The user must
        use ``isign = 1`` to compute the weigths involved in
        :py:func:`proj4d` (also :py:func:`proj4d_fft` and
        :py:func:`proj4d_rfft`) and ``isign = -1`` to compute those
        involved in :py:func:`backproj4d` (also
        :py:func:`backproj4d_fft` and :py:func:`backproj4d_rfft`).
    
    make_contiguous : bool, optional
        Set ``make_contiguous = True`` to make the output array
        contiguous and ordered in row-major order using ``weights =
        weigths.ravel().reshape(weights.shape)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    weights : complex array_like (with type `backend.cls`)
        A two-dimensional array equal to ``w = (nrm *
        (backend.exp(isign * 1j * nodes[`lt`]))[:, nodes['idt']]``.
    
    See also
    --------
    
    proj4d
    proj4d_fft
    proj4d_rfft
    backproj4d
    backproj4d_fft
    backproj4d_rfft

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(lt=nodes['lt'], idt=nodes['idt'])
    
    # consistency checks    
    #if not notest:
    #    _check_nd_inputs_(TODO)
    
    # retrieve real & complex datatypes
    lt, idt = nodes['lt'], nodes['idt']
    dtype = backend.lib_to_str_dtypes[lt.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # compute weights (the use of backend.cos and backend.sin is
    # faster and less memory demanding than the use of backend.exp,
    # because lt input is real valued)
    w = nrm * (backend.cos(lt) + isign * 1j * backend.sin(lt))
    if make_contiguous:
        w = w[:, idt].ravel().reshape((w.shape[0], len(idt)))
    else:
        w = w[:, idt]
    
    return w

def proj4d(u, delta, B, fgrad, backend=None, weights=None, eps=1e-06,
           rfft_mode=True, nodes=None, memory_usage=1, notest=False):
    """Compute EPR projections of a 4D image (adjoint of the backproj4d operation).
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Four-dimensional array corresponding to the input
        spectral-spatial 4D image to be projected (axis 0 is the
        spectral axis, axes 1, 2 and 3 correspond to the Y, X and Z
        spatial axes).
        
    delta : float 
        Pixel size given in a length unit denoted below as
        `[length-unit]` (can be `centimeter (cm)`, `millimeter (mm)`,
        ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.
        
        **WARNING**: this function assumes that the range covered by
        `B` is large enough so that the computed EPR projections are
        fully supported by `B`. Using a too small range for `B` will
        result in unrealistic projections due to B-domain aliasing
        phenomena.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(3, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y,Z) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(u, B, fgrad)``.
    
    weights : complex array_like (with type `backend.cls`), optional
        A two dimensional array with shape ``(len(B),
        len(nodes['idt']))`` of weights precomputed using
        :py:mod:`compute_4D_weights` (with option ``isign=1``).
        
        Note that those weights are only used when ``memory_usage=0``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    rfft_mode : bool, optional 
        The EPR projections are evaluated in the frequency domain
        (through their discrete Fourier coefficients) before being
        transformed back to the B-domain. Set ``rfft_mode=True`` to
        enable real FFT mode (only half of the Fourier coefficients
        will be computed to speed-up computation and reduce memory
        usage). Otherwise, use ``rfft_mode=False``, to enable standard
        (complex) FFT mode and compute all the Fourier coefficients.
    
    nodes : dict, optional 
        Precomputed frequency nodes used to evaluate the output
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`compute_4d_frequency_nodes`.
    
    memory_usage : int, optional
        Specify the computation strategy (depending on your available
        memory budget).

        + ``memory_usage = 0``: fast computation but memory demanding
    
        + ``memory_usage = 1``: (recommended) pretty good tradeoff
          between speed and reduced memory usage
        
        + ``memory_usage = 2``: slow computation but very light memory
          usage
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.

    
    Return
    ------
    
    out : array_like (with type `backend.cls`) 
        Output array with shape ``(Nproj, len(B))`` (where ``Nproj =
        fgrad.shape[1]`` corresponds to the number of computed
        projections) such that ``out[k,:]`` corresponds the EPR
        projection of u with field gradient ``fgrad[:,k]`` sampled
        over the grid `B`.    

    
    See also
    --------
    
    compute_4d_frequency_nodes
    compute_4d_weights
    backproj4d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, B=B, fgrad=fgrad)
    
    # consistency checks
    #if not notest:
    #    _check_nd_inputs_(2, B, delta, fgrad, backend, u=u, h=h,
    #                      eps=eps, nodes=nodes, rfft_mode=rfft_mode)
    
    # compute EPR projections in Fourier domain and apply inverse DFT
    # to get the projections in B-domain
    if rfft_mode:
        proj_rfft = proj4d_rfft(u, delta, B, fgrad, backend=backend,
                                weights=weights, eps=eps, nodes=nodes,
                                memory_usage=memory_usage,
                                notest=True)
        out = backend.irfft(proj_rfft, n=len(B), dim=-1)
    else:
        proj_fft = proj4d_fft(u, delta, B, fgrad, backend=backend,
                              weights=weights, eps=eps, nodes=nodes,
                              memory_usage=memory_usage, notest=True)
        out = backend.ifft(proj_fft, n=len(B), dim=-1).real
    
    return out


def proj4d_fft(u, delta, B, fgrad, backend=None, weights=None,
               eps=1e-06, out=None, nodes=None, memory_usage=1,
               notest=False):
    """Compute EPR projections of a 4D image (output in Fourier domain).
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Four-dimensional array corresponding to the input
        spectral-spatial 4D image to be projected (axis 0 is the
        spectral axis, axes 1, 2 and 3 correspond to the Y, X and Z
        spatial axes).
        
    delta : float 
        Pixel size given in a length unit denoted below as
        `[length-unit]` (can be `centimeter (cm)`, `millimeter (mm)`,
        ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.
        
        **WARNING**: this function assumes that the range covered by `B`
        is large enough so that the computed EPR projections are fully
        supported by `B`. Using a too small range for `B` will result
        in unrealistic projections due to B-domain aliasing phenomena.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(3, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y,Z) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(u, B, fgrad)``.
    
    weights : complex array_like (with type `backend.cls`), optional
        A two dimensional array with shape ``(len(B),
        len(nodes['idt']))`` of weights precomputed using
        :py:mod:`compute_4D_weights` (with option ``isign=1``).
    
        Note that those weights are only used when ``memory_usage=0``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    out : array_like (with type `backend.cls`), optional 
        Preallocated output complex array with shape
        ``(fgrad.shape[1], len(B))``.
    
    nodes : dict, optional 
        Precomputed frequency nodes used to evaluate the output
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`compute_4d_frequency_nodes`.
    
    memory_usage : int, optional
        Specify the computation strategy (depending on your available
        memory budget).

        + ``memory_usage = 0``: fast computation but memory demanding
    
        + ``memory_usage = 1``: (recommended) pretty good tradeoff
          between speed and reduced memory usage
        
        + ``memory_usage = 2``: slow computation but very light memory
          usage
        
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.

    
    Return
    ------
    
    out : complex array_like (with type `backend.cls`) 
        Output array with shape ``(Nproj, len(B))`` (where ``Nproj =
        fgrad.shape[1]`` corresponds to the number of computed
        projections) such that ``out[k,:]`` contains the discrete
        Fourier coefficients of the EPR projection of `u` with field
        gradient ``fgrad[:,k]``.    

    
    See also
    --------
    
    compute_4d_frequency_nodes
    compute_4d_weights
    proj4d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, B=B, fgrad=fgrad)
    
    # consistency checks
    #if not notest:
    #    _check_nd_inputs_(2, B, delta, fgrad, backend, u=u,
    #                      nodes=nodes, eps=eps, rfft_mode=False,
    #                      out_proj=out, fft_h=fft_h)
    
    # retrieve signals dimensions
    Nb = len(B) # number of points per projection
    Nproj = fgrad.shape[1] # number of projections
    _, Ny, Nx, Nz = u.shape # spatial dimensions
    
    # retrieve complex data type in str format and cast u to this
    # complex datatype
    dtype = backend.lib_to_str_dtypes[u.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    u_cplx = backend.cast(u, cdtype)
    
    # memory allocation
    if out is None:
        out = backend.zeros([Nproj, Nb], dtype=cdtype)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_4d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=False,
                                           notest=True)
    
    # retrieve frequency nodes
    x, y, z, indexes = nodes['x'], nodes['y'], nodes['z'], nodes['indexes']
    t, idt, lt = nodes['t'], nodes['idt'], nodes['lt']
    
    # fill output's non-zero discrete Fourier coefficients
    if 0 == memory_usage:
        plan = backend.nufft_plan(2, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        if weights is None:
            nrm = complex(delta**3)
            weights = compute_4d_weights(nodes, backend=backend,
                                         nrm=nrm, conj=False,
                                         notest=True)
        out.reshape((-1,))[indexes] = (backend.nufft_execute(plan, u_cplx) * weights).sum(0)
    elif 1 == memory_usage:
        plan = backend.nufft_plan(2, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        uhat = backend.nufft_execute(plan, u_cplx)
        nrm = delta**3
        for l in range(Nb):
            #w = nrm * backend.exp(1j * l * t) # slow and memory consuming
            w = delta**3 * (backend.cos(t * l) + 1j * backend.sin(t * l)) 
            out.reshape((-1,))[indexes] += uhat[l, :] * w[idt]
    else:
        for l in range(Nb):
            #w = delta**3 * backend.exp(1j * l * t) # slow and memory consuming
            w = delta**3 * (backend.cos(t * l) + 1j * backend.sin(t * l))
            out.reshape((-1,))[indexes] += backend.nufft3d(y, x, z, u_cplx[l, :, :, :], eps=eps) * w[idt]
    
    return out


def proj4d_rfft(u, delta, B, fgrad, backend=None, weights=None,
                eps=1e-06, out=None, nodes=None, memory_usage=1,
                notest=False):
    """Compute EPR projections of a 4D image (output in Fourier domain, half of the full spectrum).
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Four-dimensional array corresponding to the input
        spectral-spatial 4D image to be projected (axis 0 is the
        spectral axis, axes 1, 2 and 3 correspond to the Y, X and Z
        spatial axes).
    
    delta : float 
        Pixel size given in a length unit denoted below as
        `[length-unit]` (can be `centimeter (cm)`, `millimeter (mm)`,
        ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.
        
        **WARNING**: this function assumes that the range covered by `B`
        is large enough so that the computed EPR projections are fully
        supported by `B`. Using a too small range for `B` will result
        in unrealistic projections due to B-domain aliasing phenomena.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(3, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y,Z) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(u, B, fgrad)``.
        
    weights : complex array_like (with type `backend.cls`), optional
        A two dimensional array with shape ``(len(B),
        len(nodes['idt']))`` of weights precomputed using
        :py:mod:`compute_4D_weights` (with option ``isign=1``).
        
        Note that those weights are only used when ``memory_usage=0``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    out : array_like (with type `backend.cls`), optional
        Preallocated output array with shape ``(fgrad.shape[1], 1 +
        len(B)//2)``.
    
    nodes : dict, optional 
        Precomputed frequency nodes used to evaluate the output
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`compute_4d_frequency_nodes`.
    
    memory_usage : int, optional
        Specify the computation strategy (depending on your available
        memory budget).

        + ``memory_usage = 0``: fast computation but memory demanding
    
        + ``memory_usage = 1``: (recommended) pretty good tradeoff
          between speed and reduced memory usage
        
        + ``memory_usage = 2``: slow computation but very light memory
          usage
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.

    
    Return
    ------
    
    out : complex array_like (with type `backend.cls`)
        Output array with shape ``(Nproj, 1+len(B)//2)`` (where
        ``Nproj = fgrad.shape[1]`` corresponds to the number of
        computed projections) such that ``out[k,:]`` corresponds to
        half of the discrete Fourier coefficients of the EPR
        projection of `u` with field gradient ``fgrad[:,k]``.

    
    See also
    --------
    
    compute_4d_frequency_nodes
    compute_4d_weights
    proj4d
    
    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, B=B, fgrad=fgrad)
    
    # consistency checks
    #if not notest:
    #    _check_nd_inputs_(2, B, delta, fgrad, backend, u=u,
    #                      nodes=nodes, eps=eps, rfft_mode=True,
    #                      out_proj=out, fft_h=fft_h)
    
    # retrieve signals dimensions
    Nb = len(B) # number of points per projection
    Nproj = fgrad.shape[1] # number of projections
    _, Ny, Nx, Nz = u.shape # spatial dimensions
    
    # retrieve complex data type in str format and cast u to this
    # complex datatype
    dtype = backend.lib_to_str_dtypes[u.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    u_cplx = backend.cast(u, cdtype)
    
    # memory allocation
    if out is None:
        out = backend.zeros([Nproj, 1 + Nb//2], dtype=cdtype)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_4d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=True,
                                           notest=True)
    
    # retrieve frequency nodes
    x, y, z, indexes = nodes['x'], nodes['y'], nodes['z'], nodes['indexes']
    t, idt, lt = nodes['t'], nodes['idt'], nodes['lt']
    
    # fill output's non-zero discrete Fourier coefficients
    if 0 == memory_usage:
        plan = backend.nufft_plan(2, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        if weights is None:
            nrm = complex(delta**3)
            weights = compute_4d_weights(nodes, backend=backend,
                                         nrm=nrm, conj=False,
                                         notest=True)
        out.reshape((-1,))[indexes] = (backend.nufft_execute(plan, u_cplx) * weights).sum(0)
    elif 1 == memory_usage:
        plan = backend.nufft_plan(2, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        uhat = backend.nufft_execute(plan, u_cplx)
        nrm = delta**3
        for l in range(Nb):
            #w = nrm * backend.exp(1j * l * t) # slow & memory consuming
            w = nrm * (backend.cos(t * l) + 1j * backend.sin(t * l))
            out.reshape((-1,))[indexes] += uhat[l, :] * w[idt]
    else:
        nrm = delta**3
        for l in range(Nb):
            #w = nrm * backend.exp(1j * l * t) # slow and memory consuming
            w = delta**3 * (backend.cos(t * l) + 1j * backend.sin(t * l))
            out.reshape((-1,))[indexes] += backend.nufft3d(y, x, z, u_cplx[l, :, :, :], eps=eps) * w[idt]
    
    return out


def backproj4d(proj, delta, B, fgrad, out_shape, backend=None,
               weights=None, eps=1e-06, rfft_mode=True, nodes=None,
               memory_usage=1, notest=False):
    """Perform EPR backprojection from 4D EPR projections (adjoint of the proj4d operation).
    
    Parameters
    ----------
    
    proj : array_like (with type `backend.cls`)
        Two-dimensional array with shape ``(Nproj, len(B))`` (where
        ``Nproj = fgrad.shape[1]``) such that ``proj[k,:]``
        corresponds to the k-th EPR projection (acquired with field
        gradient ``fgrad[:,k]`` and sampled over the grid `B`).
    
    delta : float 
        Pixel size given in a length unit denoted below as
        `[length-unit]` (can be `centimeter (cm)`, `millimeter (mm)`,
        ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(3, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y,Z) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    out_shape : integer or integer tuple of length 4
        Shape of the output image `out_shape = out.shape = (N0, N1,
        N2, N3)`.
        
        Note: `N0` should be equal to `len(B)`.
    
    weights : complex array_like (with type `backend.cls`), optional
        A two dimensional array with shape ``(len(B),
        len(nodes['idt']))`` of weights precomputed using
        :py:mod:`compute_4D_weights` (with option ``isign=-1``).
        
        Note that those weights are only used when ``memory_usage=0``.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(proj, B, h, fgrad)``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    rfft_mode : bool, optional 
        The backprojection process involves the computation of
        discrete Fourier coefficients of the input projections. Set
        ``rfft_mode=True`` to enable real FFT mode (only half of the
        Fourier coefficients will be computed to speed-up computation
        and reduce memory usage). Otherwise, use ``rfft_mode=False``,
        to enable standard (complex) FFT mode and compute all the
        Fourier coefficients.
    
    nodes : dict, optional 
        Precomputed frequency nodes associated to the input
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`compute_4d_frequency_nodes`.
    
    memory_usage : int, optional
        Specify the computation strategy (depending on your available
        memory budget).

        + ``memory_usage = 0``: fast computation but memory demanding
    
        + ``memory_usage = 1``: (recommended) pretty good tradeoff
          between speed and reduced memory usage
        
        + ``memory_usage = 2``: slow computation but very light memory
          usage
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.

    
    Return 
    ------
    
    out : array_like (with type `backend.cls`)
        A four-dimensional array with specified shape corresponding
        to the backprojected image.

    
    See also
    --------
    
    compute_4d_frequency_nodes
    compute_4d_weights
    proj4d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(proj=proj, B=B,
                                             fgrad=fgrad)
    
    # consistency checks
    #if not notest:
    #    _check_nd_inputs_(3, B, delta, fgrad, backend, h=h, proj=proj,
    #                      eps=eps, nodes=nodes, rfft_mode=rfft_mode,
    #                      out_shape=out_shape)
    
    # perform backprojection
    if rfft_mode:
        rfft_proj = backend.rfft(proj)
        out = backproj4d_rfft(rfft_proj, delta, B, fgrad,
                              backend=backend, weights=weights,
                              eps=eps, out_shape=out_shape,
                              nodes=nodes, preserve_input=False,
                              memory_usage=memory_usage, notest=True).real
    else:
        fft_proj = backend.fft(proj)
        out = backproj4d_fft(fft_proj, delta, B, fgrad,
                             backend=backend, weights=weights,
                             eps=eps, out_shape=out_shape,
                             nodes=nodes, memory_usage=memory_usage,
                             preserve_input=False, notest=True).real
    
    return out


def backproj4d_fft(fft_proj, delta, B, fgrad, backend=None,
                   out_shape=None, out=None, weights=None, eps=1e-06,
                   nodes=None, preserve_input=False, memory_usage=1,
                   notest=False):
    """Perform EPR backprojection from 4D EPR projections provided in Fourier domain.
    
    Parameters
    ----------
    
    fft_proj : complex array_like (with type `backend.cls`)
        Two-dimensional array with shape ``(Nproj, len(B))`` (where
        ``Nproj = fgrad.shape[1]``) containing the EPR projections in
        Fourier domain.
        
        More precisely, ``fft_proj[k,:]`` corresponds to the FFT of
        the k-th EPR projection (acquired with field gradient
        ``fgrad[:,k]`` and sampled over the grid `B`).
    
    delta : float 
        Pixel size given in a length unit denoted below as
        `[length-unit]` (can be `centimeter (cm)`, `millimeter (mm)`,
        ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(3, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y,Z) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    weights : complex array_like (with type `backend.cls`), optional
        A two dimensional array with shape ``(len(B),
        len(nodes['idt']))`` of weights precomputed using
        :py:mod:`compute_4D_weights` (with option ``isign=-1``).
        
        Note that those weights are only used when ``memory_usage=0``.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(fft_proj, B, fgrad)``.
    
    out_shape : integer or integer tuple of length 4, optional 
        Shape of the output image `out_shape = out.shape = (N0, N2,
        N3)`. This optional input is in fact mandatory when no
        preallocated array is given (i.e., when ``out=None``).
        
        Note: `N0` should be equal to `len(B)`.
        
    out : complex array_like (with type `backend.cls`), optional
        Preallocated output array with shape ``(N0, N1, N2, N3)`` and
        **complex** data type. If `out_shape` is specifed, the shape
        must match (i.e., we must have ``out.shape == out_shape``),
        otherwise, `out_shape` is inferred from `out`.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : dict, optional 
        Precomputed frequency nodes associated to the input
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`compute_4d_frequency_nodes`.
    
    memory_usage : int, optional
        Specify the computation strategy (depending on your available
        memory budget).

        + ``memory_usage = 0``: fast computation but memory demanding
    
        + ``memory_usage = 1``: (recommended) pretty good tradeoff
          between speed and reduced memory usage
        
        + ``memory_usage = 2``: slow computation but very light memory
          usage
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : complex array_like (with type `backend.cls`)
        Backprojected 4D image in complex format (imaginary part
        should be close to zero and can be thrown away).
    
    
    See also
    --------
    
    compute_4d_frequency_nodes
    compute_4d_weights
    backproj4d
    
    """    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(fft_proj=fft_proj, B=B,
                                             fgrad=fgrad)
    
    # consistency checks
    #if not notest:
    #    _check_nd_inputs_(3, B, delta, fgrad, backend, nodes=nodes,
    #                      eps=eps, rfft_mode=False, out_im=out,
    #                      out_shape=out_shape, fft_h_conj=fft_h_conj,
    #                      fft_proj=fft_proj)
    
    # retrieve signals dimensions
    Nb = len(B) # number of points per projection
    Nproj = fgrad.shape[1] # number of projections
    out_shape = out.shape if out is not None else out_shape
    _, Ny, Nx, Nz = out_shape # spatial dimensions
    
    # retrieve complex data type in str format
    dtype = backend.lib_to_str_dtypes[B.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_4d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=False,
                                           notest=True)
    
    # retrieve frequency nodes
    x, y, z, indexes = nodes['x'], nodes['y'], nodes['z'], nodes['indexes']
    t, idt, lt = nodes['t'], nodes['idt'], nodes['lt']
    
    # compute adjoint nufft
    if 0 == memory_usage:
        plan = backend.nufft_plan(1, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        nrm = delta**3 / float(Nb)
        if weights is None:
            nrm = complex(delta**3 / float(Nb))
            weights = compute_4d_weights(nodes, backend=backend,
                                         nrm=nrm, conj=True,
                                         notest=True)
        out = backend.nufft_execute(plan, weights * fft_proj.reshape((-1,))[indexes], out=out)
    elif 1 == memory_usage:
        plan = backend.nufft_plan(1, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        phat = backend.empty((Nb, len(indexes)), dtype=cdtype)
        alf = backend.ifftshift(-(Nb//2) + backend.arange(Nb, dtype=dtype))
        xi = nodes['xi'] #(2. * math.pi / float(Nb)) * alf
        nrm = delta**3 / float(Nb)
        for l in range(Nb):
            #w = nrm * backend.exp(1j * l * xi) # slow and memory consuming
            xil = xi * l
            w = nrm * (backend.cos(xil) + 1j * backend.sin(xil)).reshape((-1,))
            phat[l, :] = (fft_proj * w).reshape((-1,))[indexes]
        out = backend.nufft_execute(plan, phat, out=out)
    else:
        out = backend.zeros(out_shape, dtype=cdtype)
        nrm = delta**3 / float(Nb)
        for l in range(Nb):
            #w = nrm * backend.exp(-1j * l * t[idt]) # slow and memory consuming
            tl = t[idt] * l
            w = nrm * (backend.cos(tl) - 1j * backend.sin(tl)).reshape((-1,))
            out[l, :, :, :] = backend.nufft3d_adjoint(y, x, z, fft_proj.reshape((-1))[indexes] * w, n_modes=(Ny, Nx, Nz), eps=eps, out=out[l, :, :, :])
    
    return out


def backproj4d_rfft(rfft_proj, delta, B, fgrad, backend=None,
                    out_shape=None, out=None, weights=None, eps=1e-06,
                    nodes=None, preserve_input=False, memory_usage=1,
                    notest=False):
    """Perform EPR backprojection from 4D EPR projections provided in Fourier domain (half of the full spectrum).
    
    Parameters
    ----------
    
    rfft_proj : complex array_like (with type `backend.cls`)
        Two-dimensional array with shape ``(Nproj, 1+len(B)//2)``
        (where ``Nproj = fgrad.shape[1]``) containing the EPR
        projections in Fourier domain (half of the spectrum).
        
        More precisely, ``rfft_proj[k,:]`` corresponds to the real FFT
        (rfft) of the k-th EPR projection (acquired with field
        gradient ``fgrad[:,k]`` and sampled over the grid `B`).
        
    delta : float
        Pixel size given in a length unit denoted below as
        `[length-unit]` (can be `centimeter (cm)`, `millimeter (mm)`,
        ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(3, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y,Z) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    weights : complex array_like (with type `backend.cls`), optional
        A two dimensional array with shape ``(len(B),
        len(nodes['idt']))`` of weights precomputed using
        :py:mod:`compute_4D_weights` (with option ``isign=-1``).
        
        Note that those weights are only used when ``memory_usage=0``.
        
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(rfft_proj, B, fgrad)``.
    
    out_shape : integer or integer tuple of length 4, optional Shape
        of the output image `out_shape = out.shape = (N0, N1, N2,
        N3)`. This optional input is in fact mandatory when no
        preallocated array is given (i.e., when ``out=None``).
        
        Note: `N0` should be equal to `len(B)`.
            
    out : complex array_like (with type `backend.cls`), optional
        Preallocated output array with shape ``(N0, N1, N2, N3)`` and
        **complex** data type. If `out_shape` is specifed, the shape
        must match (i.e., we must have ``out.shape == out_shape``),
        otherwise, `out_shape` is inferred from `out`.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : dict, optional 
        Precomputed frequency nodes associated to the input
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`compute_4d_frequency_nodes`.
    
    memory_usage : int, optional
        Specify the computation strategy (depending on your available
        memory budget).

        + ``memory_usage = 0``: fast computation but memory demanding
    
        + ``memory_usage = 1``: (recommended) pretty good tradeoff
          between speed and reduced memory usage
        
        + ``memory_usage = 2``: slow computation but very light memory
          usage
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : complex array_like (with type `backend.cls`)
        Backprojected 4D image in complex format (imaginary part
        should be close to zero and can be thrown away).
    
    
    See also
    --------
    
    compute_4d_frequency_nodes
    compute_4d_weights
    backproj4d
    
    """    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(rfft_proj=rfft_proj, B=B,
                                             fgrad=fgrad)
    
    # consistency checks
    #if not notest:
    #    _check_nd_inputs_(3, B, delta, fgrad, backend, nodes=nodes,
    #                      eps=eps, rfft_mode=False, out_im=out,
    #                      out_shape=out_shape, fft_h_conj=fft_h_conj,
    #                      fft_proj=fft_proj)
    
    # retrieve signals dimensions
    Nb = len(B) # number of points per projection
    Nproj = fgrad.shape[1] # number of projections
    out_shape = out.shape if out is not None else out_shape
    _, Ny, Nx, Nz = out_shape # spatial dimensions
    
    # retrieve complex data type in str format
    dtype = backend.lib_to_str_dtypes[B.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_4d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=True,
                                           notest=True)
    
    # retrieve frequency nodes
    x, y, z, indexes = nodes['x'], nodes['y'], nodes['z'], nodes['indexes']
    t, idt, lt = nodes['t'], nodes['idt'], nodes['lt']
    
    # rescale non zero DFT coefficients (this tricks allow to put back
    # the missing half-frequency-domain coefficient into the
    # backprojected signal)
    rfft_proj[:, 1::] *= 2.
    
    # compute adjoint nufft
    if 0 == memory_usage:
        plan = backend.nufft_plan(1, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        if weights is None:
            nrm = complex(delta**3 / float(Nb))
            weights = compute_4d_weights(nodes, backend=backend,
                                         nrm=nrm, conj=True,
                                         notest=True)
        out = backend.nufft_execute(plan, weights * rfft_proj.reshape((-1,))[indexes], out=out)
    elif 1 == memory_usage:
        plan = backend.nufft_plan(1, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        phat = backend.empty((Nb, len(indexes)), dtype=cdtype)
        xi = nodes['xi']
        nrm = delta**3 / float(Nb)
        for l in range(Nb):
            #w = nrm * backend.exp(1j * l * xi) # slow and memory consuming
            xil = xi * l
            w = nrm * (backend.cos(xil) + 1j * backend.sin(xil)).reshape((-1,))
            phat[l, :] = (rfft_proj * w).reshape((-1,))[indexes]
        out = backend.nufft_execute(plan, phat, out=out)
    else:
        out = backend.zeros(out_shape, dtype=cdtype)
        nrm = delta**3 / float(Nb)
        for l in range(Nb):
            #w = nrm * backend.exp(-1j * l * t[idt]) # slow and memory consuming
            tl = t[idt] * l
            w = nrm * (backend.cos(tl) - 1j * backend.sin(tl)).reshape((-1,))
            out[l, :, :, :] = backend.nufft3d_adjoint(y, x, z, rfft_proj.reshape((-1))[indexes] * w, n_modes=(Ny, Nx, Nz), eps=eps, out=out[l, :, :, :])
    
    if preserve_input:
        rfft_proj[:, 1::] /= 2.
    
    return out

# TODO: deal with memory_usage optional input (not used yet)
def compute_4d_toeplitz_kernel(B, delta, fgrad, out_shape,
                               backend=None, eps=1e-06,
                               rfft_mode=True, nodes=None,
                               return_rfft4=False, notest=False,
                               memory_usage=1):
    """Compute 4D Toeplitz kernel allowing fast computation of a ``proj4d`` followed by a ``backproj4d`` operation.
    
    Parameters
    ----------
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.
    
    delta : float 
        Pixel size given in a length unit denoted below as
        `[length-unit]` (can be `centimeter (cm)`, `millimeter (mm)`,
        ...).
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(3, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y,Z) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    out_shape : integer or integer tuple of length 4
        Shape of the output kernel ``out_shape = phi.shape = (M0, M1,
        M2, M3)``. The kernel shape should be twice the EPR image
        shape (i.e., denoting by `(N0, N1, N2, N3)` the shape of the
        4D EPR image, we should have ``(M0, M1, M2, M3) = (2*N0, 2*N1,
        2*N2, 2*N3)``).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(B, fgrad)``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    rfft_mode : bool, optional 
        The computation of the Toeplitz kernel involves the
        computation of discrete Fourier coefficients of real-valued
        signals. Set ``rfft_mode=True`` to enable real FFT mode
        (speed-up the computation and reduce memory usage). Otherwise,
        use ``rfft_mode=False``, to enable standard (complex) FFT
        mode.
    
    nodes : dict, optional 
        Precomputed frequency nodes used to evaluate the output
        kernel. If not given, `nodes` will be automatically inferred
        from `B`, `delta` and `fgrad` using
        :py:func:`compute_4d_frequency_nodes`.
    
    return_rfft4: bool, optional
        Set ``return_rfft4`` to return the real input FFT (rfft4) of
        the computed four-dimensional kernel (instead of the kernel
        itself).
    
    memory_usage : int, optional
        Specify the computation strategy (depending on your available
        memory budget).

        + ``memory_usage = 0``: fast computation but memory demanding
    
        + ``memory_usage = 1``: (recommended) pretty good tradeoff
          between speed and reduced memory usage
        
        + ``memory_usage = 2``: slow computation but very light memory
          usage
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.    

    
    Return
    ------
    
    phi : array_like (with type `backend.cls`)
        Computed Toeplitz kernel (or its four-dimensional real input
        FFT when ``return_rfft4 is True``).
    
    
    See also
    --------
    
    compute_4d_frequency_nodes
    proj4d
    backproj4d
    apply_4d_toeplitz_kernel

    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B, fgrad=fgrad)
    
    # consistency checks
    #if not notest:
    #    _check_nd_inputs_(3, B, delta, fgrad, backend, h1=h1, h2=h2,
    #                      nodes=nodes, eps=eps, out_shape=out_shape,
    #                      rfft_mode=rfft_mode,
    #                      return_rfft3=return_rfft3)
    
    # retrieve signals dimensions
    Nb = len(B) # number of points per projection
    Nproj = fgrad.shape[1] # number of projections
    _, Ny, Nx, Nz = out_shape # spatial dimensions
    
    # retrieve complex data type in str format
    dtype = backend.lib_to_str_dtypes[B.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_4d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=rfft_mode,
                                           notest=True)
    
    # retrieve frequency nodes
    x, y, z, indexes = nodes['x'], nodes['y'], nodes['z'], nodes['indexes']
    t, idt = nodes['t'], nodes['idt']
    
    # compute kernel
    lt2 = backend.arange(2 * Nb, dtype=dtype).reshape((-1, 1)) * t.reshape((1, -1))
    cof = (delta**6 / float(Nb)) * (backend.cos(lt2) + 1j * backend.sin(lt2))
    if rfft_mode:
        cof[:, 1::] *= 2.
    COFS = cof[:, idt].ravel().reshape((2 * Nb, len(idt)))   
    plan = backend.nufft_plan(1, (2*Ny, 2*Nx, 2*Nz), n_trans=2*Nb, dtype=cdtype, eps=eps)
    backend.nufft_setpts(plan, y, x, z)
    phi = backend.nufft_execute(plan, COFS).real
    
    return backend.rfftn(phi) if return_rfft4 else phi


def apply_4d_toeplitz_kernel(u, rfft4_phi, backend=None, notest=False):
    """Perform a ``proj4d`` followed by a ``backproj4d`` operation using a precomputed Toeplitz kernel provided in Fourier domain.
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Four-dimensional array corresponding to the input 4D image to
        be projected-backprojected.
    
    rfft4_phi : complex array_like (with type `backend.cls`)
        real input FFT of the 4D Toeplitz kernel computed using
        :py:func:`compute_4d_toeplitz_kernel`.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(u, rfft4_phi)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return 
    ------
    
    out : array_like (with type `backend.cls`) 
        output projected-backprojected image.
    
    
    See also
    --------
    
    compute_4d_toeplitz_kernel
    proj4d
    backproj4d
    
    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, rfft4_phi=rfft4_phi)
    
    # consistency checks
    if not notest:
        checks._check_backend_(backend, u=u, rfft4_phi=rfft4_phi)
        checks._check_ndim_(4, u=u, rfft4_phi=rfft4_phi)
        dtype = backend.lib_to_str_dtypes[u.dtype]
        cdtype = backend.mapping_to_complex_dtypes[dtype]
        libcdtype = backend.str_to_lib_dtypes[cdtype]
        checks._check_dtype_(libcdtype, rfft4_phi=rfft4_phi)
    
    # compute shape of the extended domain
    Nb, Ny, Nx, Nz = u.shape
    s = (2 * Nb, 2 * Ny, 2 * Nx, 2 * Nz)
    
    # compute & return output image
    return backend.irfftn(rfft4_phi * backend.rfftn(u, s=s),
                          s=s)[Nb::, Ny::, Nx::, Nz::]


# TODO
def _check_nd_inputs_():
    """Factorized consistency checks for functions in the :py:mod:`pyepri.spectralspatial` submodule."""
    pass
