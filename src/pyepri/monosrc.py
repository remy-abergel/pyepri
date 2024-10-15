"""This module contains low-level operators related to **single source
EPR imaging** (projection, backprojection, projection-backprojection
using Toeplitz kernels). Detailed mathematical definitions of the
operators are provided in the :ref:`mathematical_definitions` section
of the PyEPRI documentation.

"""
import math
import pyepri.checks as checks


def compute_2d_frequency_nodes(B, delta, fgrad, backend=None,
                               rfft_mode=True, notest=False):
    """Compute 2D irregular frequency nodes involved in 2D projection & backprojection operations.
    
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
        Two dimensional array with shape ``(2, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
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
    
        A dictionary with content ``{'x': x, 'y': y, 'indexes':
        indexes, 'rfft_mode': rfft_mode}`` where
    
        + ``x`` is a one dimensional array containing the frequency
          nodes along the horizontal axis;
    
        + ``y`` is a one dimensional array, with same length as ``x``,
          containing the frequency nodes along the vertical axis;
    
        + ``indexes`` is a one dimensional array, with same length as
          ``x``, corresponding to the indexes where should be dispatched
          the computed Fourier coefficients in the rfft of the 2D
          projections.
    
        + ``rfft_mode`` a bool specifying whether the frequency nodes
          cover half of the frequency domain (``rfft_mode=True``) or the
          full frequency domain (``rfft_mode=False``).

    See also
    --------

    proj2d
    backproj2d
    compute_2d_toeplitz_kernel

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B, fgrad=fgrad)

    # consistency checks
    if not notest:
        _check_nd_inputs_(2, B, delta, fgrad, backend,
                          rfft_mode=rfft_mode)
    
    # retrieve several constant (`dB` = sampling step of the
    # homogeneous magnetic grid, `mu` = field gradient amplitudes)
    Nb = len(B) 
    dB = B[1] - B[0] 
    mu = backend.sqrt(fgrad[0]**2 + fgrad[1]**2).reshape((-1,1))

    # retrieve standardized data type in str format
    dtype = backend.lib_to_str_dtypes[B.dtype]

    # compute regular frequency nodes & find indexes of nonzero output
    # frequencies    
    if rfft_mode: 
        alf = backend.arange(1 + Nb//2, dtype=dtype)
        T = (mu * alf < .5 * Nb * dB / delta) & (alf < .5 * Nb)
    else:
        alf = backend.ifftshift(-(Nb//2) + backend.arange(Nb,
                                                          dtype=dtype))
        T = (mu * backend.abs(alf) < .5 * Nb * dB / delta) & \
            (backend.abs(alf) < .5 * Nb)
    indexes = backend.argwhere(T.reshape((-1,))).reshape((-1,))
    xi = ((2. * math.pi * alf) / (Nb * dB)).reshape((1,-1))

    # compute irregular frequency nodes
    x = -((delta * fgrad[0]).reshape((-1,1)) * xi).reshape((-1,))[indexes]
    y = -((delta * fgrad[1]).reshape((-1,1)) * xi).reshape((-1,))[indexes]

    # compute output dictionary and return
    nodes = {'x': x, 'y': y, 'indexes': indexes, 'rfft_mode': rfft_mode}

    return nodes


def proj2d(u, delta, B, h, fgrad, backend=None, eps=1e-06,
           rfft_mode=True, nodes=None, notest=False):
    """Compute EPR projections of a 2D image (adjoint of the backproj2d operation).
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Two-dimensional array corresponding to the input 2D image to
        be projected.
        
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
    
    h : array_like (with type `backend.cls`) 
        One dimensional array with same length as `B` corresponding to
        the reference spectrum sampled over the grid `B`.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(2, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).

    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input arrays ``(u, B, h, fgrad)``.
        
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
        :py:func:`pyepri.monosrc.compute_2d_frequency_nodes`.
    
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
    
    compute_2d_frequency_nodes
    backproj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, B=B, h=h, fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, B, delta, fgrad, backend, u=u, h=h,
                          eps=eps, nodes=nodes, rfft_mode=rfft_mode)

    # compute EPR projections in Fourier domain and apply inverse DFT
    # to get the projections in B-domain
    if rfft_mode:
        rfft_h = backend.rfft(h)
        proj_rfft = proj2d_rfft(u, delta, B, rfft_h, fgrad,
                                backend=backend, eps=eps, nodes=nodes,
                                notest=True)
        out = backend.irfft(proj_rfft, n=len(B), dim=-1)
    else:
        fft_h = backend.fft(h)
        proj_fft = proj2d_fft(u, delta, B, fft_h, fgrad,
                              backend=backend, eps=eps, nodes=nodes,
                              notest=True)
        out = backend.ifft(proj_fft, n=len(B), dim=-1).real
    
    return out


def proj2d_fft(u, delta, B, fft_h, fgrad, backend=None, eps=1e-06,
               out=None, nodes=None, notest=False):
    """Compute EPR projections of a 2D image (output in Fourier domain).
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Two-dimensional array corresponding to the input 2D image to
        be projected.
        
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
    
    fft_h : complex array_like (with type `backend.cls`)
        One dimensional array with length ``len(B)`` containing the
        discrete Fourier coefficients of the reference spectrum
        sampled over `B`.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(2, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).

    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input arrays ``(u, B, rfft_h, fgrad)``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    out : array_like (with type `backend.cls`), optional 
        Preallocated output complex array with shape
        ``(fgrad.shape[1], len(B))``.
    
    nodes : dict, optional 
        Precomputed frequency nodes used to evaluate the output
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`pyepri.monosrc.compute_2d_frequency_nodes`.
    
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
    
    compute_2d_frequency_nodes
    proj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, B=B, fft_h=fft_h,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, B, delta, fgrad, backend, u=u,
                          nodes=nodes, eps=eps, rfft_mode=False,
                          out_proj=out, fft_h=fft_h)
    
    # retrieve signals dimensions and datatype
    Nb = len(B) # number of points per projection
    Nproj = fgrad.shape[1] # number of projections
    
    # retrieve complex data type in str format
    cdtype = backend.lib_to_str_dtypes[fft_h.dtype]
    
    # memory allocation
    if out is None:
        out = backend.zeros([Nproj, Nb], dtype=cdtype)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_2d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=False,
                                           notest=True)
    
    # fill output's non-zero discrete Fourier coefficients (notice
    # that the switch between x and y axis in the nufft2d function
    # below is made on purpose for compliance with standard image
    # processing axes ordering (axis 0 is the vertical axis (y) and
    # axis 1 is the horizontal one (x)))
    u_cplx = backend.cast(u, cdtype)
    x, y, indexes = nodes['x'], nodes['y'], nodes['indexes']
    out.reshape((-1,))[indexes] = backend.nufft2d(y, x, u_cplx, eps=eps)
    out *= (delta**2 * fft_h)

    return out


def proj2d_rfft(u, delta, B, rfft_h, fgrad, backend=None, eps=1e-06,
                out=None, nodes=None, notest=False):
    """Compute EPR projections of a 2D image (output in Fourier domain, half of the full spectrum).
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Two-dimensional array corresponding to the input 2D image to
        be projected.
        
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
    
    rfft_h : complex array_like (with type `backend.cls`)
        One dimensional array with length ``1+len(B)//2`` containing
        half of the discrete Fourier coefficients of the reference
        spectrum sampled over `B` (corresponds to the signal computed
        using the real input fft (rfft) of the reference spectrum).
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(2, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).

    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input arrays ``(u, B, rfft_h, fgrad)``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    out : array_like (with type `backend.cls`), optional
        Preallocated output array with shape ``(fgrad.shape[1], 1 +
        len(B)//2)``.
    
    nodes : dict, optional 
        Precomputed frequency nodes used to evaluate the output
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`pyepri.monosrc.compute_2d_frequency_nodes`.
    
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
    
    compute_2d_frequency_nodes
    proj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, B=B, rfft_h=rfft_h,
                                             fgrad=fgrad)
        
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, B, delta, fgrad, backend, u=u,
                          nodes=nodes, eps=eps, rfft_mode=True,
                          out_proj=out, rfft_h=rfft_h)
    
    # retrieve signals dimensions and datatype
    Nb = len(B) # number of points per projection
    Nproj = fgrad.shape[1] # number of projections
    
    # retrieve complex data type in str format
    cdtype = backend.lib_to_str_dtypes[rfft_h.dtype]
    
    # memory allocation
    if out is None:
        out = backend.zeros([Nproj, 1 + Nb//2], dtype=cdtype)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_2d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=True,
                                           notest=True)
    
    # fill output's non-zero discrete Fourier coefficients (notice
    # that the switch between x and y axis in the nufft2d function
    # below is made on purpose for compliance with standard image
    # processing axes ordering (axis 0 is the vertical axis (y) and
    # axis 1 is the horizontal one (x)))
    u_cplx = backend.cast(u, cdtype)
    x, y, indexes = nodes['x'], nodes['y'], nodes['indexes']
    out.reshape((-1,))[indexes] = backend.nufft2d(y, x, u_cplx, eps=eps)
    out *= (delta**2 * rfft_h)
    
    return out


def backproj2d(proj, delta, B, h, fgrad, out_shape, backend=None,
               eps=1e-06, rfft_mode=True, nodes=None, notest=False):
    """Perform EPR backprojection from 2D EPR projections (adjoint of the proj2d operation).
    
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
    
    h : array_like (with type `backend.cls`) 
        One dimensional array with same length as `B` corresponding to
        the reference spectrum sampled over the grid `B`.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(2, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    out_shape : integer or integer tuple of length 2
        Shape of the output image `out_shape = out.shape = (N1, N2)`
        (or `out_shape = N` when N1 = N2 = N). 
    
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
        :py:func:`pyepri.monosrc.compute_2d_frequency_nodes`.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    Return 
    ------
    
    out : array_like (with type `backend.cls`)
        A two-dimensional array with specified shape corresponding to
        the backprojected image.
    
    See also
    --------
    
    compute_2d_frequency_nodes
    proj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(proj=proj, B=B, h=h,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, B, delta, fgrad, backend, h=h, proj=proj,
                          eps=eps, nodes=nodes, rfft_mode=rfft_mode,
                          out_shape=out_shape)
    
    # perform backprojection
    if rfft_mode: 
        rfft_proj = backend.rfft(proj)
        rfft_h_conj = backend.rfft(h).conj()
        out = backproj2d_rfft(rfft_proj, delta, B, rfft_h_conj, fgrad,
                              backend=backend, eps=eps,
                              out_shape=out_shape, nodes=nodes,
                              notest=True)
    else:
        fft_proj = backend.fft(proj)
        fft_h_conj = backend.fft(h).conj()
        out = backproj2d_fft(fft_proj, delta, B, fft_h_conj, fgrad,
                             backend=backend, eps=eps,
                             out_shape=out_shape, nodes=nodes,
                             notest=True)
    
    return out.real


def backproj2d_fft(fft_proj, delta, B, fft_h_conj, fgrad,
                   backend=None, out_shape=None, out=None, eps=1e-06,
                   nodes=None, notest=False):
    """Perform EPR backprojection from 2D EPR projections provided in Fourier domain.
    
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
    
    fft_h_conj : complex array_like (with type `backend.cls`)
        One dimensional array with length ``len(B)`` containing
        the conjugate of half of the discrete Fourier coefficients of
        the reference spectrum sampled over `B`.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(2, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).

    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(fft_proj, B, fft_h_conj, fgrad)``.

    out_shape : integer or integer tuple of length 2, optional 
        Shape of the output image `out_shape = out.shape = (N1, N2)`
        (or `out_shape = N` when N1 = N2 = N). This optional input is
        in fact mandatory when no preallocated array is given (i.e.,
        when ``out=None``).
    
    out : complex array_like (with type `backend.cls`), optional
        Preallocated output array with shape ``(N1, N2)`` and
        **complex** data type. If `out_shape` is specifed, the shape
        must match (i.e., we must have ``out.shape == out_shape``),
        otherwise, `out_shape` is inferred from `out`.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : dict, optional 
        Precomputed frequency nodes associated to the input
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`pyepri.monosrc.compute_2d_frequency_nodes`.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : complex array_like (with type `backend.cls`)
        Backprojected image in complex format (imaginary part should
        be close to zero and can be thrown away).
    
    See also
    --------
    
    compute_2d_frequency_nodes
    backproj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(fft_proj=fft_proj, B=B,
                                             fft_h_conj=fft_h_conj,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, B, delta, fgrad, backend, nodes=nodes,
                          eps=eps, rfft_mode=False, out_im=out,
                          out_shape=out_shape, fft_h_conj=fft_h_conj,
                          fft_proj=fft_proj)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_2d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=False,
                                           notest=True)
    
    # compute adjoint nufft (notice that the switch between x and y
    # axis in the nufft2d_adjoint function below is made on purpose
    # for compliance with standard image processing axes ordering
    # (axis 0 is the vertical axis (y) and axis 1 is the horizontal
    # one (x)))
    x, y, indexes = nodes['x'], nodes['y'], nodes['indexes']
    c = (fft_proj * fft_h_conj).reshape((-1,))[indexes]
    out = backend.nufft2d_adjoint(y, x, c, n_modes=out_shape, out=out,
                                  eps=eps)
    out *= delta**2 / float(len(B))
    
    return out


def backproj2d_rfft(rfft_proj, delta, B, rfft_h_conj, fgrad,
                    backend=None, out_shape=None, out=None, eps=1e-06,
                    nodes=None, notest=False):
    """Perform EPR backprojection from 2D EPR projections provided in Fourier domain (half of the full spectrum).
    
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
    
    rfft_h_conj : complex array_like (with type `backend.cls`)
        One dimensional array with length ``1+len(B)//2`` containing
        the conjugate of half of the discrete Fourier coefficients of
        the reference spectrum sampled over `B` (corresponds to the
        conjugate of the signal computed using the real input FFT
        (rfft) of the reference spectrum).
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(2, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).

    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input arrays ``(rfft_proj, B, rfft_h_conj, fgrad)``.
    
    out_shape : integer or integer tuple of length 2, optional 
        Shape of the output image `out_shape = out.shape = (N1, N2)`
        (or `out_shape = N` when N1 = N2 = N). This optional input is
        in fact mandatory when no preallocated array is given (i.e.,
        when ``out=None``).
    
    out : complex array_like (with type `backend.cls`), optional
        Preallocated output array with shape ``(N1, N2)`` and
        **complex** data type. If `out_shape` is specifed, the shape
        must match (i.e., we must have ``out.shape == out_shape``),
        otherwise, `out_shape` is inferred from `out`.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : dict, optional 
        Precomputed frequency nodes associated to the input
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`pyepri.monosrc.compute_2d_frequency_nodes`.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : complex array_like (with type `backend.cls`)
        Backprojected image in complex format (imaginary part should
        be close to zero and can be thrown away).
    
    See also
    --------
    
    compute_2d_frequency_nodes
    backproj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(rfft_proj=rfft_proj, B=B,
                                             rfft_h_conj=rfft_h_conj,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, B, delta, fgrad, backend, nodes=nodes,
                          eps=eps, rfft_mode=True, out_im=out,
                          out_shape=out_shape,
                          rfft_h_conj=rfft_h_conj,
                          rfft_proj=rfft_proj)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_2d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=True,
                                           notest=True)
    
    # compute adjoint nufft (notice that the switch between x and y
    # axis in the nufft2d_adjoint function below is made on purpose
    # for compliance with standard image processing axes ordering
    # (axis 0 is the vertical axis (y) and axis 1 is the horizontal
    # one (x)))
    x, y, indexes = nodes['x'], nodes['y'], nodes['indexes']
    c = (rfft_proj * rfft_h_conj)
    c[:,0] *= .5 # avoid counting two times the zero-frequency
                 # coefficients when completing the sum below
    c = c.reshape((-1,))[indexes]
    out = backend.nufft2d_adjoint(y, x, c, n_modes=out_shape, out=out, eps=eps)
    out += out.conj() # complete the sum (the missing terms are the
                      # conjugate of the already computed terms)
    out *= delta**2 / float(len(B))
    
    return out


def compute_2d_toeplitz_kernel(B, h1, h2, delta, fgrad, out_shape,
                               backend=None, eps=1e-06,
                               rfft_mode=True, nodes=None,
                               return_rfft2=False, notest=False):
    """Compute 2D Toeplitz kernel allowing fast computation of a ``proj2d`` followed by a ``backproj2d`` operation.
    
    Parameters
    ----------
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.
    
    h1 : array_like (with type `backend.cls`) 
        One dimensional array with same length as `B` corresponding to
        the reference spectrum involved in the forward (proj2d)
        operation (and sampled over the grid `B`).

    h2 : array_like (with type `backend.cls`)
        One dimensional array with same length as `B` corresponding to
        the reference spectrum involved in the backward (backproj2d)
        operation (and sampled over the grid `B`).
    
    delta : float 
        Pixel size given in a length unit denoted below as
        `[length-unit]` (can be `centimeter (cm)`, `millimeter (mm)`,
        ...).
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(2, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).

    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``(B, h1, h2, fgrad)``.

    out_shape : integer or integer tuple of length 2
        Shape of the output kernel ``out_shape = phi.shape = (M1,
        M2)``. The kernel shape should be twice the EPR image shape
        (i.e., denoting by `(N1, N2)` the shape of the EPR image, we
        should have ``(M1, M2) = (2*N1, 2*N2)``).
    
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
        :py:func:`pyepri.monosrc.compute_2d_frequency_nodes`.

    return_rfft2: bool, optional
        Set ``return_rfft2`` to return the real input FFT (rfft2) of
        the computed two-dimensional kernel (instead of the kernel
        itself).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    phi : array_like (with type `backend.cls`)
        Computed Toeplitz kernel (or its two-dimensional real input
        FFT when ``return_rfft2 is True``).
    
    See also
    --------
    
    compute_2d_frequency_nodes
    proj2d
    backproj2d
    apply_2d_toeplitz_kernel

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B, h1=h1, h2=h2,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, B, delta, fgrad, backend, h1=h1, h2=h2,
                          nodes=nodes, eps=eps, out_shape=out_shape,
                          rfft_mode=rfft_mode,
                          return_rfft2=return_rfft2)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_2d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=rfft_mode,
                                           notest=True)

    # retrieve complex data type
    dtype = backend.lib_to_str_dtypes[B.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # compute kernel (notice that the switch between x and y axis in
    # the nufft2d_adjoint function below is made on purpose for
    # compliance with standard image processing axes ordering (axis 0
    # is the vertical axis (y) and axis 1 is the horizontal one (x)))
    x, y, indexes = nodes['x'], nodes['y'], nodes['indexes']
    if rfft_mode:
        g = backend.rfft(h1)
        g *= backend.rfft(h2).conj()
        c = backend.tile(g.reshape((1,-1)), (fgrad.shape[1], 1))
        c[:,0] *= .5 # avoid counting two times the zero-frequency
                     # coefficients when completing the sum below
        c = c.reshape((-1,))[nodes['indexes']]
        phi = backend.nufft2d_adjoint(y, x, c, n_modes=out_shape,
                                      eps=eps)        
        phi += phi.conj() # complete the sum (the missing terms are
                          # the conjugate of those computed above)
    else:
        g = backend.fft(h1).reshape((1,-1))
        g *= backend.fft(h2).conj().reshape((1,-1))
        c = backend.tile(g, (fgrad.shape[1], 1)).reshape((-1,))[indexes]
        phi = backend.nufft2d_adjoint(y, x, c, n_modes=out_shape,
                                      eps=eps)
    phi *= delta**4 / float(len(B))
    
    return backend.rfft2(phi.real) if return_rfft2 else phi.real


def apply_2d_toeplitz_kernel(u, rfft2_phi, backend=None, notest=False):
    """Perform a ``proj2d`` followed by a ``backproj2d`` operation using a precomputed Toeplitz kernel provided in Fourier domain.
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Two-dimensional array corresponding to the input 2D image to
        be projected-backprojected.
    
    rfft2_phi : complex array_like (with type `backend.cls`)
        real input FFT of the 2D Toeplitz kernel computed using
        :py:func:`pyepri.monosrc.compute_2d_toeplitz_kernel`.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(u, rfft2_phi)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return 
    ------
    
    out : array_like (with type `backend.cls`) 
        output projected-backprojected image.
    
    
    See also
    --------
    
    compute_2d_toeplitz_kernel
    proj2d
    backproj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, rfft2_phi=rfft2_phi)
    
    # consistency checks
    if not notest:
        checks._check_backend_(backend, u=u, rfft2_phi=rfft2_phi)
        checks._check_ndim_(2, u=u, rfft2_phi=rfft2_phi)
        cdtype = backend.str_to_lib_dtypes[backend.mapping_to_complex_dtypes[backend.lib_to_str_dtypes[u.dtype]]]
        checks._check_dtype_(cdtype, rfft2_phi=rfft2_phi)
    
    # compute shape of the extended domain
    Ny, Nx = u.shape
    s = (2 * Ny, 2 * Nx)

    # compute & return output image
    return backend.irfft2(rfft2_phi * backend.rfft2(u, s=s),
                          s=s)[Ny::, Nx::]


def compute_3d_frequency_nodes(B, delta, fgrad, backend=None,
                               rfft_mode=True, notest=False):
    """Compute 3D irregular frequency nodes involved in 3D projection & backprojection operations.
    
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
        Three dimensional array with shape ``(3, fgrad.shape[1])``
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
    
        A dictionary with content ``{'x': x, 'y': y, 'z': z,
        'indexes': indexes, 'rfft_mode': rfft_mode}`` where
    
        + ``x`` is a one dimensional array containing the frequency
          nodes along the X-axis;
    
        + ``y`` is a one dimensional array, with same length as ``x``,
          containing the frequency nodes along the Y-axis;

        + ``z`` is a one dimensional array, with same length as ``x``,
          containing the frequency nodes along the Z-axis;
    
        + ``indexes`` is a one dimensional array, with same length as
          ``x``, corresponding to the indexes where should be dispatched
          the computed Fourier coefficients in the rfft of the 3D
          projections.
    
        + ``rfft_mode`` a bool specifying whether the frequency nodes
          cover half of the frequency domain (``rfft_mode=True``) or the
          full frequency domain (``rfft_mode=False``).
    
    See also
    --------
    
    proj3d
    backproj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B, fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, B, delta, fgrad, backend,
                          rfft_mode=rfft_mode)
    
    # retrieve several constant (`dB` = sampling step of the
    # homogeneous magnetic grid, `mu` = field gradient amplitudes)
    Nb = len(B) 
    dB = B[1] - B[0] 
    mu = backend.sqrt((fgrad**2).sum(0)).reshape((-1,1))

    # retrieve standardized data type in str format
    dtype = backend.lib_to_str_dtypes[B.dtype]

    # compute regular frequency nodes & find indexes of nonzero output
    # frequencies    
    if rfft_mode: 
        alf = backend.arange(1 + Nb//2, dtype=dtype)
        T = (mu * alf < .5 * Nb * dB / delta) & (alf < .5 * Nb)
    else:
        alf = backend.ifftshift(-(Nb//2) + backend.arange(Nb,
                                                          dtype=dtype))
        T = (mu * backend.abs(alf) < .5 * Nb * dB / delta) & \
            (backend.abs(alf) < .5 * Nb)
    indexes = backend.argwhere(T.reshape((-1,))).reshape((-1,))
    xi = ((2. * math.pi * alf) / (Nb * dB)).reshape((1,-1))

    # compute irregular frequency nodes
    x = -((delta * fgrad[0]).reshape((-1,1)) * xi).reshape((-1,))[indexes]
    y = -((delta * fgrad[1]).reshape((-1,1)) * xi).reshape((-1,))[indexes]
    z = -((delta * fgrad[2]).reshape((-1,1)) * xi).reshape((-1,))[indexes]

    # compute output dictionary and return
    nodes = {'x': x, 'y': y, 'z': z, 'indexes': indexes, 'rfft_mode':
             rfft_mode}
    
    return nodes


def proj3d(u, delta, B, h, fgrad, backend=None, eps=1e-06,
           rfft_mode=True, nodes=None, notest=False):
    """Compute EPR projections of a 3D image (adjoint of the backproj3d operation).
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Three-dimensional array corresponding to the input 3D image to
        be projected.
        
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
    
    h : array_like (with type `backend.cls`) 
        One dimensional array with same length as `B` corresponding to
        the reference spectrum sampled over the grid `B`.
    
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
        input arrays ``(u, B, h, fgrad)``.
    
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
        :py:func:`pyepri.monosrc.compute_3d_frequency_nodes`.
    
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
    
    compute_3d_frequency_nodes
    backproj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, B=B, fgrad=fgrad,
                                             h=h)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, B, delta, fgrad, backend, u=u, h=h,
                          eps=eps, nodes=nodes, rfft_mode=rfft_mode)

    # compute EPR projections in Fourier domain and apply inverse DFT
    # to get the projections in B-domain
    if rfft_mode:
        rfft_h = backend.rfft(h)
        proj_rfft = proj3d_rfft(u, delta, B, rfft_h, fgrad,
                                backend=backend, eps=eps, nodes=nodes,
                                notest=True)
        out = backend.irfft(proj_rfft, n=len(B), dim=-1)
    else:
        fft_h = backend.fft(h)
        proj_fft = proj3d_fft(u, delta, B, fft_h, fgrad,
                              backend=backend, eps=eps, nodes=nodes,
                              notest=True)
        out = backend.ifft(proj_fft, n=len(B), dim=-1).real
    
    return out


def proj3d_fft(u, delta, B, fft_h, fgrad, backend=None, eps=1e-06,
               out=None, nodes=None, notest=False):
    """Compute EPR projections of a 3D image (output in Fourier domain).
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Three-dimensional array corresponding to the input 2D image to
        be projected.
        
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
    
    fft_h : complex array_like (with type `backend.cls`)
        One dimensional array with length ``len(B)`` containing the
        discrete Fourier coefficients of the reference spectrum
        sampled over `B`.
    
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
        input arrays ``(u, B, fft_h, fgrad)``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    out : array_like (with type `backend.cls`), optional 
        Preallocated output complex array with shape
        ``(fgrad.shape[1], len(B))``.
    
    nodes : dict, optional 
        Precomputed frequency nodes used to evaluate the output
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`pyepri.monosrc.compute_3d_frequency_nodes`.
    
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
    
    compute_2d_frequency_nodes
    proj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, B=B, fft_h=fft_h,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, B, delta, fgrad, backend, u=u,
                          nodes=nodes, eps=eps, rfft_mode=False,
                          out_proj=out, fft_h=fft_h)
    
    # retrieve signals dimensions and datatype
    Nb = len(B) # number of points per projection
    Nproj = fgrad.shape[1] # number of projections
    
    # retrieve complex data type in str format
    cdtype = backend.lib_to_str_dtypes[fft_h.dtype]
    
    # memory allocation
    if out is None:
        out = backend.zeros([Nproj, Nb], dtype=cdtype)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_3d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=False,
                                           notest=True)
    
    # fill output's non-zero discrete Fourier coefficients (notice
    # that the switch between x and y axis in the nufft3d function
    # below is made on purpose for compliance with standard image
    # processing axes ordering (axis 0 is the Y-axis, axis 1 is the
    # X-axis and axis 2 is the Z axis)    
    u_cplx = backend.cast(u, cdtype)
    x, y, z = nodes['x'], nodes['y'], nodes['z']
    indexes = nodes['indexes']
    out.reshape((-1,))[indexes] = backend.nufft3d(y, x, z, u_cplx, eps=eps)
    out *= (delta**3 * fft_h)

    return out


def proj3d_rfft(u, delta, B, rfft_h, fgrad, backend=None, eps=1e-06,
                out=None, nodes=None, notest=False):
    """Compute EPR projections of a 3D image (output in Fourier domain, half of the full spectrum).
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Three-dimensional array corresponding to the input 2D image to
        be projected.
        
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
    
    rfft_h : complex array_like (with type `backend.cls`)
        One dimensional array with length ``1+len(B)//2`` containing
        half of the discrete Fourier coefficients of the reference
        spectrum sampled over `B` (corresponds to the signal computed
        using the real input fft (rfft) of the reference spectrum).
    
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
        input arrays ``(u, B, rfft_h, fgrad)``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    out : array_like (with type `backend.cls`), optional
        Preallocated output array with shape ``(fgrad.shape[1], 1 +
        len(B)//2)``.
    
    nodes : dict, optional 
        Precomputed frequency nodes used to evaluate the output
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`pyepri.monosrc.compute_3d_frequency_nodes`.
    
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
    
    compute_2d_frequency_nodes
    proj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, B=B, rfft_h=rfft_h,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, B, delta, fgrad, backend, u=u,
                          nodes=nodes, eps=eps, rfft_mode=True,
                          out_proj=out, rfft_h=rfft_h)
    
    # retrieve signals dimensions and datatype
    Nb = len(B) # number of points per projection
    Nproj = fgrad.shape[1] # number of projections
    
    # retrieve complex data type in str format
    cdtype = backend.lib_to_str_dtypes[rfft_h.dtype]
    
    # memory allocation
    if out is None:
        out = backend.zeros([Nproj, 1 + Nb//2], dtype=cdtype)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_3d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=True,
                                           notest=True)
    
    # fill output's non-zero discrete Fourier coefficients (notice
    # that the switch between x and y axis in the nufft3d function
    # below is made on purpose for compliance with standard image
    # processing axes ordering (axis 0 is the Y-axis, axis 1 is the
    # X-axis and axis 2 is the Z axis)
    u_cplx = backend.cast(u, cdtype)
    x, y, z = nodes['x'], nodes['y'], nodes['z']
    indexes = nodes['indexes']
    out.reshape((-1,))[indexes] = backend.nufft3d(y, x, z, u_cplx, eps=eps)
    out *= (delta**3 * rfft_h)
    
    return out


def backproj3d(proj, delta, B, h, fgrad, out_shape, backend=None,
               eps=1e-06, rfft_mode=True, nodes=None, notest=False):
    """Perform EPR backprojection from 3D EPR projections (adjoint of the proj3d operation).
    
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
    
    h : array_like (with type `backend.cls`) 
        One dimensional array with same length as `B` corresponding to
        the reference spectrum sampled over the grid `B`.
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(3, fgrad.shape[1])`` such
        that ``fgrad[:,k]`` corresponds to the (X,Y,Z) coordinates of
        the field gradient vector associated to the k-th EPR
        projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
    
    out_shape : integer or integer tuple of length 3
        Shape of the output image `out_shape = out.shape = (N1, N2,
        N3)` (or `out_shape = N` when `N1 = N2 = N3 = N`).
    
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
        :py:func:`pyepri.monosrc.compute_3d_frequency_nodes`.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    Return 
    ------
    
    out : array_like (with type `backend.cls`)
        A three-dimensional array with specified shape corresponding
        to the backprojected image.
    
    See also
    --------
    
    compute_3d_frequency_nodes
    proj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(proj=proj, B=B, h=h,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, B, delta, fgrad, backend, h=h, proj=proj,
                          eps=eps, nodes=nodes, rfft_mode=rfft_mode,
                          out_shape=out_shape)
    
    # perform backprojection
    if rfft_mode: 
        rfft_proj = backend.rfft(proj)
        rfft_h_conj = backend.rfft(h).conj()
        out = backproj3d_rfft(rfft_proj, delta, B, rfft_h_conj, fgrad,
                              backend=backend, eps=eps,
                              out_shape=out_shape, nodes=nodes,
                              notest=True)
    else:
        fft_proj = backend.fft(proj)
        fft_h_conj = backend.fft(h).conj()
        out = backproj3d_fft(fft_proj, delta, B, fft_h_conj, fgrad,
                             backend=backend, eps=eps,
                             out_shape=out_shape, nodes=nodes,
                             notest=True)
    
    return out.real

def backproj3d_fft(fft_proj, delta, B, fft_h_conj, fgrad,
                   backend=None, out_shape=None, out=None, eps=1e-06,
                   nodes=None, notest=False):
    """Perform EPR backprojection from 3D EPR projections provided in \
    Fourier domain.
    
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
    
    fft_h_conj : complex array_like (with type `backend.cls`)
        One dimensional array with length ``len(B)`` containing
        the conjugate of half of the discrete Fourier coefficients of
        the reference spectrum sampled over `B`.
    
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
        input arrays ``(fft_proj, B, fft_h_conj, fgrad)``.
    
    out_shape : integer or integer tuple of length 3, optional 
        Shape of the output image `out_shape = out.shape = (N1, N2,
        N3)` (or `out_shape = N` when N1 = N2 = N3 = N). This optional
        input is in fact mandatory when no preallocated array is given
        (i.e., when ``out=None``).
    
    out : complex array_like (with type `backend.cls`), optional
        Preallocated output array with shape ``(N1, N2, N3)`` and
        **complex** data type. If `out_shape` is specifed, the shape
        must match (i.e., we must have ``out.shape == out_shape``),
        otherwise, `out_shape` is inferred from `out`.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : dict, optional 
        Precomputed frequency nodes associated to the input
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`pyepri.monosrc.compute_3d_frequency_nodes`.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : complex array_like (with type `backend.cls`)
        Backprojected 3D image in complex format (imaginary part
        should be close to zero and can be thrown away).
    
    See also
    --------
    
    compute_3d_frequency_nodes
    backproj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(fft_proj=fft_proj, B=B,
                                             fft_h_conj=fft_h_conj,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, B, delta, fgrad, backend, nodes=nodes,
                          eps=eps, rfft_mode=False, out_im=out,
                          out_shape=out_shape, fft_h_conj=fft_h_conj,
                          fft_proj=fft_proj)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_3d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=False,
                                           notest=True)
    
    # compute adjoint nufft (notice that the switch between x and y
    # axis in the nufft3d_adjoint function below is made on purpose
    # for compliance with standard image processing axes ordering
    # (axis 0 is Y-axis, axis 1 is the X-axis and axis 2 is the
    # Z-axis))
    x, y, z = nodes['x'], nodes['y'], nodes['z']
    indexes = nodes['indexes']
    c = (fft_proj * fft_h_conj).reshape((-1,))[indexes]
    out = backend.nufft3d_adjoint(y, x, z, c, n_modes=out_shape,
                                  out=out, eps=eps)
    out *= delta**3 / float(len(B))
    
    return out

def backproj3d_rfft(rfft_proj, delta, B, rfft_h_conj, fgrad,
                    backend=None, out_shape=None, out=None, eps=1e-06,
                    nodes=None, notest=False):
    """Perform EPR backprojection from 3D EPR projections provided in Fourier domain (half of the full spectrum).
    
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
    
    rfft_h_conj : complex array_like (with type `backend.cls`)
        One dimensional array with length ``1+len(B)//2`` containing
        the conjugate of half of the discrete Fourier coefficients of
        the reference spectrum sampled over `B` (corresponds to the
        conjugate of the signal computed using the real input FFT
        (rfft) of the reference spectrum).
    
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
        input arrays ``(rfft_proj, B, rfft_h_conj, fgrad)``.
    
    out_shape : integer or integer tuple of length 2, optional 
        Shape of the output image `out_shape = out.shape = (N1, N2,
        N3)` (or `out_shape = N` when N1 = N2 = N3 = N). This optional
        input is in fact mandatory when no preallocated array is given
        (i.e., when ``out=None``).
    
    out : complex array_like (with type `backend.cls`), optional
        Preallocated output array with shape ``(N1, N2, N3)`` and
        **complex** data type. If `out_shape` is specifed, the shape
        must match (i.e., we must have ``out.shape == out_shape``),
        otherwise, `out_shape` is inferred from `out`.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : dict, optional 
        Precomputed frequency nodes associated to the input
        projections. If not given, `nodes` will be automatically
        inferred from `B`, `delta` and `fgrad` using
        :py:func:`pyepri.monosrc.compute_3d_frequency_nodes`.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : complex array_like (with type `backend.cls`)
        Backprojected 3D image in complex format (imaginary part
        should be close to zero and can be thrown away).
    
    See also
    --------
    
    compute_3d_frequency_nodes
    backproj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(rfft_proj=rfft_proj, B=B,
                                             rfft_h_conj=rfft_h_conj,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, B, delta, fgrad, backend, nodes=nodes,
                          eps=eps, rfft_mode=True, out_im=out,
                          out_shape=out_shape,
                          rfft_h_conj=rfft_h_conj,
                          rfft_proj=rfft_proj)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_3d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=True,
                                           notest=True)
    
    # compute adjoint nufft (notice that the switch between x and y
    # axis in the nufft3d_adjoint function below is made on purpose
    # for compliance with standard image processing axes ordering
    # (axis 0 is Y-axis, axis 1 is the X-axis and axis 2 is the
    # Z-axis))
    x, y, z = nodes['x'], nodes['y'], nodes['z']
    indexes = nodes['indexes']
    c = (rfft_proj * rfft_h_conj)
    c[:,0] *= .5 # avoid counting two times the zero-frequency
                 # coefficients when completing the sum below
    c = c.reshape((-1,))[indexes]
    out = backend.nufft3d_adjoint(y, x, z, c, n_modes=out_shape,
                                  out=out, eps=eps)
    out += out.conj() # complete the sum (the missing terms are the
                      # conjugate of the already computed terms)
    out *= delta**3 / float(len(B))
    
    return out


def compute_3d_toeplitz_kernel(B, h1, h2, delta, fgrad, out_shape,
                               backend=None, eps=1e-06,
                               rfft_mode=True, nodes=None,
                               return_rfft3=False, notest=False):
    """Compute 3D Toeplitz kernel allowing fast computation of a ``proj3d`` followed by a ``backproj3d`` operation.
    
    Parameters
    ----------
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...), to
        use to compute the projections.
    
    h1 : array_like (with type `backend.cls`) 
        One dimensional array with same length as `B` corresponding to
        the reference spectrum involved in the forward (proj3d)
        operation (and sampled over the grid `B`).

    h2 : array_like (with type `backend.cls`)
        One dimensional array with same length as `B` corresponding to
        the reference spectrum involved in the backward (backproj3d)
        operation (and sampled over the grid `B`).
    
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

    out_shape : integer or integer tuple of length 3
        Shape of the output kernel ``out_shape = phi.shape = (M1, M2,
        M3)``. The kernel shape should be twice the EPR image shape
        (i.e., denoting by `(N1, N2, N3)` the shape of the EPR image,
        we should have ``(M1, M2, M3) = (2*N1, 2*N2, 2*N3)``).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input arrays ``(B, h1, h2, fgrad)``.
    
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
        :py:func:`pyepri.monosrc.compute_3d_frequency_nodes`.
    
    return_rfft3: bool, optional
        Set ``return_rfft3`` to return the real input FFT (rfft3) of
        the computed three-dimensional kernel (instead of the kernel
        itself).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    phi : array_like (with type `backend.cls`)
        Computed Toeplitz kernel (or its three-dimensional real input
        FFT when ``return_rfft3 is True``).
    
    
    See also
    --------
    
    compute_3d_frequency_nodes
    proj3d
    backproj3d
    apply_3d_toeplitz_kernel

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B, h1=h1, h2=h2,
                                             fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, B, delta, fgrad, backend, h1=h1, h2=h2,
                          nodes=nodes, eps=eps, out_shape=out_shape,
                          rfft_mode=rfft_mode,
                          return_rfft3=return_rfft3)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = compute_3d_frequency_nodes(B, delta, fgrad,
                                           backend=backend,
                                           rfft_mode=rfft_mode,
                                           notest=True)

    # retrieve complex data type
    dtype = backend.lib_to_str_dtypes[B.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # compute kernel (notice that the switch between x and y axis in
    # the nufft3d_adjoint function below is made on purpose for
    # compliance with standard image processing axes ordering (axis 0
    # is the Y-axis, axis 1 is X-axis, and axis 2 is the Z-axis))
    x, y, z = nodes['x'], nodes['y'], nodes['z']
    indexes = nodes['indexes']
    if rfft_mode:
        g = backend.rfft(h1)
        g *= backend.rfft(h2).conj()
        c = backend.tile(g.reshape((1,-1)), (fgrad.shape[1], 1))
        c[:,0] *= .5 # avoid counting two times the zero-frequency
                     # coefficients when completing the sum below
        c = c.reshape((-1,))[nodes['indexes']]
        phi = backend.nufft3d_adjoint(y, x, z, c, n_modes=out_shape,
                                      eps=eps)        
        phi += phi.conj() # complete the sum (the missing terms are
                          # the conjugate of those computed above)
    else:
        g = backend.fft(h1).reshape((1,-1))
        g *= backend.fft(h2).conj().reshape((1,-1))
        c = backend.tile(g, (fgrad.shape[1], 1)).reshape((-1,))[indexes]
        phi = backend.nufft3d_adjoint(y, x, z, c, n_modes=out_shape,
                                      eps=eps)
    phi *= delta**6 / float(len(B))
    
    return backend.rfftn(phi.real) if return_rfft3 else phi.real


def apply_3d_toeplitz_kernel(u, rfft3_phi, backend=None, notest=False):
    """Perform a ``proj3d`` followed by a ``backproj3d`` operation using a precomputed Toeplitz kernel provided in Fourier domain.
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Three-dimensional array corresponding to the input 3D image to
        be projected-backprojected.
    
    rfft3_phi : complex array_like (with type `backend.cls`)
        real input FFT of the 3D Toeplitz kernel computed using
        :py:func:`pyepri.monosrc.compute_3d_toeplitz_kernel`.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
    
        When backend is None, a default backend is inferred from the
        input arrays ``(u, rfft3_phi)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return 
    ------
    
    out : array_like (with type `backend.cls`) 
        output projected-backprojected image.
    
    
    See also
    --------
    
    compute_3d_toeplitz_kernel
    proj3d
    backproj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, rfft3_phi=rfft3_phi)
    
    # consistency checks
    if not notest:
        checks._check_backend_(backend, u=u, rfft3_phi=rfft3_phi)
        checks._check_ndim_(3, u=u, rfft3_phi=rfft3_phi)
        cdtype = backend.str_to_lib_dtypes[backend.mapping_to_complex_dtypes[backend.lib_to_str_dtypes[u.dtype]]]
        checks._check_dtype_(cdtype, rfft3_phi=rfft3_phi)
    
    # compute shape of the extended domain
    Ny, Nx, Nz = u.shape
    s = (2 * Ny, 2 * Nx, 2 * Nz)

    # compute & return output image
    return backend.irfftn(rfft3_phi * backend.rfftn(u, s=s),
                          s=s)[Ny::, Nx::, Nz::]


def _check_nd_inputs_(ndims, B, delta, fgrad, backend, u=None, h=None,
                      h1=None, h2=None, proj=None, eps=None,
                      nodes=None, rfft_mode=None, out_proj=None,
                      out_im=None, out_shape=None, fft_h=None,
                      fft_h_conj=None, rfft_h=None, rfft_h_conj=None,
                      fft_proj=None, rfft_proj=None,
                      return_rfft2=None, return_rfft3=None):
    """Factorized consistency checks for functions in the :py:mod:`pyepri.monosrc` submodule."""

    ##################
    # general checks #
    ##################
    
    # check backend consistency 
    checks._check_backend_(backend, u=u, B=B, fgrad=fgrad, h=h,
                           proj=proj, fft_h=fft_h, rfft_h=rfft_h,
                           fft_proj=fft_proj, rfft_proj=rfft_proj)
    
    # retrieve data types
    dtype = B.dtype
    str_dtype = backend.lib_to_str_dtypes[dtype]
    str_cdtype = backend.mapping_to_complex_dtypes[str_dtype]
    cdtype = backend.str_to_lib_dtypes[str_cdtype]
    
    # run other generic checks
    checks._check_same_dtype_(u=u, B=B, h=h, h1=h1, h2=h2, proj=proj)
    checks._check_dtype_(cdtype, fft_h=fft_h, rfft_h=rfft_h,
                         fft_h_conj=fft_h_conj,
                         rfft_h_conj=rfft_h_conj, fft_proj=fft_proj,
                         rfft_proj=rfft_proj)
    checks._check_ndim_(1, B=B, h=h, h1=h1, h2=h2, fft_h=fft_h,
                        rfft_h=rfft_h, fft_h_conj=fft_h_conj,
                        rfft_h_conj=rfft_h_conj)
    checks._check_ndim_(2, fgrad=fgrad, proj=proj, fft_proj=fft_proj)
    checks._check_ndim_(ndims, u=u)
    
    #################
    # custom checks #
    #################
    
    # retrieve characteristic dimensions
    Nb = len(B)
    Nproj = fgrad.shape[1]    
    
    # delta: must be a float
    if not isinstance(delta, (float, int)):
        raise RuntimeError(            
            "Parameter `delta` must be a float scalar number (int is also tolerated)."
        )
    
    # eps: must be None or a float
    if eps is not None and not isinstance(eps, float):
        raise RuntimeError(
            "Parameter `eps` must be a float scalar number."
        )
    
    # fgrad: must have ndims rows
    if fgrad.shape[0] != ndims:
        raise RuntimeError (            
            "Input parameter `fgrad` must contain %d rows (fgrad.shape[0] = %d)." % (ndims, ndims)
        )
    
    # h: must have same length as B
    if h is not None and len(h) != len(B):
        raise RuntimeError (            
            "Input parameter `h` must have the same length as `B` (here len(B) = %d)." % len(B)
        )
    
    # h1: must have same length as B
    if h1 is not None and len(h1) != len(B):
        raise RuntimeError (            
            "Input parameter `h1` must have the same length as `B` (here len(B) = %d)." % len(B)
        )
    
    # h2: must have same length as B
    if h2 is not None and len(h2) != len(B):
        raise RuntimeError (            
            "Input parameter `h2` must have the same length as `B` (here len(B) = %d)." % len(B)
        )
    
    # proj: must have shape (Nproj, Nb)
    if proj is not None and proj.shape != (Nproj, Nb):
        raise RuntimeError(
            "Input parameter `proj` must have shape (fgrad.shape[0], len(B)) = (%d, %d)." % (Nproj, Nb)
        )
    
    # out_shape: must be an integer or a tuple containing ndims integers
    err = False
    if isinstance(out_shape, tuple):
        err = (len(out_shape) != ndims) or not all([isinstance(x, int) for x in out_shape])
    elif out_shape is not None: 
        err = not isinstance(out_shape, int)
    if err: 
        raise RuntimeError(
            "Input parameter `out_shape` must be either an integer or a tuple of %d integers." % ndims
        )
    
    # out_im: is a ndims-dimensional signal stored in a complex array -> must have
    # consistent complex type (=cdtype) and shape
    checks._check_backend_(backend, out=out_im)
    checks._check_dtype_(cdtype, out=out_im)
    checks._check_ndim_(ndims, out=out_im)
    if out_shape is not None and out_im is not None and out_im.shape != out_shape:
        raise RuntimeError(
            "When both `out_shape` and `out` are specified, we must have out.shape == out_shape."
        )
    
    # out_proj: is a complex 2D array containing DFT coefficients of
    # projections -> must have consistent complex type (=cdtype) and
    # shape
    checks._check_backend_(backend, out=out_proj)
    checks._check_dtype_(cdtype, out=out_proj)
    checks._check_ndim_(2, out=out_proj)
    if out_proj is not None:
        if rfft_mode and out_proj.shape != (Nproj, 1+Nb//2):
            raise RuntimeError(
                "Preallocated array `out` has inconsistent shape (expected \n"
                "out.shape = (fgrad.shape[1], 1+len(B)//2) = (%d, %d))." % (Nproj, 1+Nb//2)
            )
        elif not rfft_mode and out_proj.shape != (Nproj, Nb):
            raise RuntimeError(
                "Preallocated array `out` has inconsistent shape (expected \n"
                "out.shape = (fgrad.shape[1], len(B)) = (%d, %d))." % (Nproj, Nb)
            )
    
    # fft_h: must have same length as B
    if fft_h is not None and len(fft_h) != Nb:
        raise RuntimeError(
            "Input parameter `fft_h` must have the same length as `B`."
        )
    
    # fft_h_conj: must have same length as B
    if fft_h_conj is not None and len(fft_h_conj) != Nb:
        raise RuntimeError(
            "Input parameter `fft_h_conj` must have the same length as `B`."
        )

    # rfft_h: must have length 1+Nb//2
    if rfft_h is not None and len(rfft_h) != 1+Nb//2:
        raise RuntimeError(
            "Input parameter `rfft_h` must have length 1+len(B)//2 (= %d)." % (1+Nb//2)
        )
    
    # rfft_h_conj: must have length 1+Nb//2
    if rfft_h_conj is not None and len(rfft_h_conj) != 1+Nb//2:
        raise RuntimeError(
            "Input parameter `rfft_h_conj` must have length 1+len(B)//2 (= %d)." % (1+Nb//2)
        )
    
    # fft_proj: must have shape (Nproj, Nb)
    if fft_proj is not None and fft_proj.shape != (Nproj, Nb):
        raise RuntimeError(
            "Input parameter `fft_proj` has inconsistent shape (expected \n"
            "fft_proj.shape = (fgrad.shape[0], len(B)) = (%d, %d))." % (Nproj, Nb)
        )
    
    # rfft_proj: must have shape (Nproj, 1+Nb//2)
    if rfft_proj is not None and rfft_proj.shape != (Nproj, 1+Nb//2):
        raise RuntimeError(
            "Input parameter `rfft_proj` has inconsistent shape (expected \n"
            "rfft_proj.shape = (fgrad.shape[0], 1+len(B)//2) = (%d, %d))." % (Nproj, 1+Nb//2)
        )
    
    # if one of rfft_proj or fft_proj is not None, out_shape and out_im cannot be both None
    if ((fft_proj is not None) or (rfft_proj is not None)) and (out_shape is None and out_im is None):
        raise RuntimeError(
            "Input parameter `out_shape` is mandatory when preallocated output \n"
            "array `out` is not specified."            
        )
    
    # nodes: many things to check
    if not nodes is None:
        if not isinstance(nodes, dict):
            raise RuntimeError("Input parameter `nodes` must be a dictionary (dict).")
        if ndims == 2:
            if not all({t in nodes.keys() for t in {'x', 'y', 'indexes', 'rfft_mode'}}):
                raise RuntimeError(
                    "Input parameter `nodes` must contain the keys 'x', 'y', 'indexes', 'rfft_mode'"
                )
            if not backend.is_backend_compliant(nodes['x'], nodes['y'], nodes['indexes']):
                raise RuntimeError(
                    "The content of the `nodes` parameter is not consistent with the provided backend.\n"
                    "Since `backend.lib` is `" + backend.lib.__name__ + "`, `nodes['x']`, `nodes['y']` and "
                    "`nodes['indexes']` must all be\n"
                    "" + str(backend.cls) + " instances."
                )
            checks._check_ndim_(1, **{"nodes['x']": nodes['x'],
                                      "nodes['y']": nodes['y'],
                                      "nodes['indexes']": nodes['indexes']})
            if not len(nodes['x']) == len(nodes['y']) == len(nodes['indexes']):
                raise RuntimeError(                    
                    "nodes['x'], nodes['y'] and nodes['indexes'] must have the same length.\n"
                )
            checks._check_dtype_(dtype, **{"nodes['x']": nodes['x'],
                                           "nodes['y']": nodes['y']})
        if ndims == 3:
            if not all({t in nodes.keys() for t in {'x', 'y', 'z', 'indexes', 'rfft_mode'}}):
                raise RuntimeError(
                    "Input parameter `nodes` must contain the keys 'x', 'y', 'z', 'indexes', 'rfft_mode'"
                )
            if not backend.is_backend_compliant(nodes['x'], nodes['y'], nodes['z'], nodes['indexes']):
                raise RuntimeError(
                    "The content of the `nodes` parameter is not consistent with the provided backend.\n"
                    "Since `backend.lib` is `" + backend.lib.__name__ + "`, `nodes['x']`, `nodes['y']` "
                    "`nodes['z']` and `nodes['indexes']` must all be\n"
                    "" + str(backend.cls) + " instances."
                )        
            if not len(nodes['x']) == len(nodes['y']) == len(nodes['z']) == len(nodes['indexes']):
                raise RuntimeError(                    
                    "nodes['x'], nodes['y'], nodes['z'] and nodes['indexes'] must have the same length.\n"
                )
            checks._check_dtype_(dtype, **{"nodes['x']": nodes['x'],
                                           "nodes['y']": nodes['y'],
                                           "nodes['z']": nodes['z']})
        if rfft_mode is not None and rfft_mode != nodes['rfft_mode']:
            raise RuntimeError(                
                "Cannot use ``rfft_mode=%s`` with ``nodes['rfft_mode']=%s`` (both must be the same)."
                % (rfft_mode, nodes['rfft_mode'])
            )
        cof = (Nb//2+1 if rfft_mode else Nb)
        numel = Nproj * cof
        if nodes['indexes'].min() < 0 or nodes['indexes'].max() >= numel:
            raise RuntimeError(
                "The values in nodes['indexes'] must be in the range [0,M)\n"
                "with M = fgrad.shape[1] * %s (= %d * %d = %d)."                
                % ("(1 + len(B)//2)" if rfft_mode else "len(B)",
                  fgrad.shape[1] , cof, numel)
            )
    
    return True
