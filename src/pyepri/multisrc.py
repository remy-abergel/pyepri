"""This module contains low-level operators related to **multisources
EPR imaging** (projection, backprojection, projection-backprojection
using Toeplitz kernels). Detailed mathematical definitions of the
operators are provided in the :ref:`mathematical_definitions` section
of the PyEPRI documentation.

"""
import pyepri.checks as checks
import pyepri.monosrc as monosrc

def proj2d(u, delta, B, h, fgrad, backend=None, eps=1e-06,
           rfft_mode=True, nodes=None, notest=False):
    """Compute EPR projections of from a sequence of 2D source images.
    
    This function can be used to simulate EPR projections from a
    mixture of 2D sources, in different experimental conditions (e.g.,
    different microwave power).
    
    In the following, the index `j` shall refer to the `j-th` source
    image, while the index `i` shall refer to the `i-th` experimental
    setup. We shall denote by `K` the number of sources and by `L` the
    number of different experimental setup (those numbers are computed
    using ``K = len(u)`` and ``L = max(len(h), len(fgrad))``.
    
    Parameters
    ----------
    
    u : sequence of array_like (with type `backend.cls`)
        A sequence with length `K` containing the source images. More
        precisely, ``u[j]`` must be a 2D array corresponding to the
        `j-th` source image of the mixture to be projected.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    h : sequence of sequence of array_like (with type `backend.cls`)
        Contains the reference spectra (sampled over the grid ``B``)
        of each individual source (`j`) for each experimental setup
        (`i`). More precisely, ``h[i][j]`` is a monodimensional array
        corresponding to the sampling over ``B`` of the reference
        spectrum of the `j-th` source in the `i-th` experimental
        setting.
        
        When ``L > 1`` and ``len(h) = 1``, we assume that ``h[i][j] =
        h[0][j]``` for all ``i in range(L)`` and all ``j in
        range(K)``.
        
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    rfft_mode : bool, optional 
        The EPR projections are evaluated in the frequency domain
        (through their dicrete Fourier coefficients) before being
        transformed back to the B-domain. Set ``rfft_mode=True`` to
        enable real FFT mode (only half of the Fourier coefficients
        will be computed to speed-up computation and reduce memory
        usage). Otherwise, use ``rfft_mode=False``, to enable standard
        (complex) FFT mode and compute all the Fourier coefficients.
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes used to evaluate the output projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
            
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    proj : sequence of array_like (with type `backend.cls`) 
        A sequence with length `L` containing the EPR projections
        synthesized for each experimental setting. More precisely,
        ``proj[i]`` is an array with shape ``(fgrad[i].shape[1],
        len(B))`` and ``proj[i][k,:]`` corresponds the EPR projection
        of the mixture `u` with field gradient ``fgrad[i][:,k]``
        sampled over the grid `B`.
    
    
    See also
    --------
    
    proj2d_fft
    proj2d_rfft
    backproj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:     
        _check_nd_inputs_(2, backend, B=B, delta=delta, fgrad=fgrad,
                          u=u, h=h, eps=eps, rfft_mode=rfft_mode,
                          nodes=nodes)
    
    # compute EPR projections in Fourier domain and apply inverse DFT
    # to get the projections in B-domain
    if rfft_mode:
        rfft_h = [[backend.rfft(hij) for hij in hi] for hi in h]
        rfft_proj = proj2d_rfft(u, delta, B, rfft_h, fgrad, backend=backend, eps=eps, nodes=nodes, notest=True)
        proj = [backend.irfft(rfft_sino, n=len(B)) for rfft_sino in rfft_proj]
    else:
        fft_h = [[backend.fft(hij) for hij in hi] for hi in h]
        fft_proj = proj2d_fft(u, delta, B, fft_h, fgrad, backend=backend, eps=eps, nodes=nodes, notest=True)
        proj = [backend.ifft(fft_sino, n=len(B)).real for fft_sino in fft_proj]
    
    return proj

def proj2d_fft(u, delta, B, fft_h, fgrad, backend=None, eps=1e-06,
               nodes=None, notest=False):
    """Compute EPR projections of from a sequence of 2D source images (output in Fourier domain).
    
    Parameters
    ----------
    
    u : sequence of array_like (with type `backend.cls`)
        A sequence with length `K` containing the source images. More
        precisely, ``u[j]`` must be a 2D array corresponding to the
        `j-th` source image of the mixture to be projected.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    fft_h : sequence of sequence of complex array_like (with type
        `backend.cls`) Contains the discrete Fourier coefficients of
        the reference spectra (sampled over the grid ``B``) of each
        individual source (`j`) for each experimental setup
        (`i`). More precisely, ``fft_h[i][j]`` is a monodimensional
        array corresponding to the FFT of ``h[i][j]`` (the reference
        spectrum of the `j-th` source in the `i-th` experimental
        setting).
        
        When ``L > 1`` and ``len(fft_h) = 1``, we assume that
        ``fft_h[i][j] = fft_h[0][j]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
        
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
        
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes used to evaluate the output projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    fft_proj : sequence of complex array_like (with type
        `backend.cls`) A sequence with length `L` containing the
        discrete Fourier coefficients of the EPR projections
        synthesized for each experimental setting. More precisely,
        ``fft_proj[i]`` is an array with shape ``(fgrad[i].shape[1],
        len(B))`` and ``fft_proj[i][k,:]`` corresponds the FFT of the
        EPR projection of the mixture `u` with field gradient
        ``fgrad[i][:,k]`` sampled over the grid `B`.
    
    
    See also
    --------
    
    proj2d_rfft
    proj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, backend, B=B, delta=delta, fgrad=fgrad,
                          u=u, fft_h=fft_h, eps=eps, nodes=nodes)
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    L = max(len(fgrad), len(fft_h))
    K = len(u)
    
    # retrieve real & complex data types in str format
    dtype = backend.lib_to_str_dtypes[u[0].dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # memory allocation
    if len(fgrad) == 1:
        Nproj = fgrad[0].shape[1]
        fft_s = [backend.zeros([Nproj, Nb], dtype=cdtype) for _ in range(L)]
    else:
        fft_s = [backend.zeros([fg.shape[1], Nb], dtype=cdtype) for fg in fgrad]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_2d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=False,
                                                    notest=True) for
                 fgi in fgrad]
    
    # compute EPR projections in Fourier domain
    for i in range(L):
        fft_hi = fft_h[i] if len(fft_h) > 1 else fft_h[0]
        fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
        ni = nodes[i] if len(nodes) > 1 else nodes[0]
        for j in range(K):
            fft_s[i] += monosrc.proj2d_fft(u[j], delta, B, fft_hi[j],
                                           fgi, backend=backend,
                                           eps=eps, nodes=ni,
                                           notest=True)
    
    return fft_s

def proj2d_rfft(u, delta, B, rfft_h, fgrad, backend=None, eps=1e-06,
                nodes=None, notest=False):
    """Compute EPR projections of from a sequence of 2D source images (output in Fourier domain, half of the full spectrum).
    
    Parameters
    ----------
    
    u : sequence of array_like (with type `backend.cls`)
        A sequence with length `K` containing the source images. More
        precisely, ``u[j]`` must be a 2D array corresponding to the
        `j-th` source image of the mixture to be projected.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    rfft_h : sequence of sequence of complex array_like (with type
        `backend.cls`) Contains half of the discrete Fourier
        coefficients of the reference spectra (sampled over the grid
        ``B``) of each individual source (`j`) for each experimental
        setup (`i`). More precisely, ``rfft_h[i][j]`` is a
        monodimensional array corresponding to the real input FFT
        (rfft) of ``h[i][j]`` (the reference spectrum of the `j-th`
        source in the `i-th` experimental setting).
        
        When ``L > 1`` and ``len(rfft_h) = 1``, we assume that
        ``rfft_h[i][j] = rfft_h[0][j]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
        
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes used to evaluate the output projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    rfft_proj : sequence of complex array_like (with type
        `backend.cls`) A sequence with length `L` containing half of
        the discrete Fourier coefficients of the EPR projections
        synthesized for each experimental setting. More precisely,
        ``rfft_proj[i]`` is an array with shape ``(fgrad[i].shape[1],
        1+len(B)//2)`` and ``rfft_proj[i][k,:]`` corresponds the real
        input FFT (rfft) of the EPR projection of the mixture `u` with
        field gradient ``fgrad[i][:,k]`` sampled over the grid `B`.
    
    
    See also
    --------
    
    proj2d_fft
    proj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest: 
        _check_nd_inputs_(2, backend, B=B, delta=delta, fgrad=fgrad,
                          u=u, rfft_h=rfft_h, eps=eps, nodes=nodes)
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    L = max(len(fgrad), len(rfft_h))
    K = len(u)
    
    # retrieve real & complex data types in str format
    dtype = backend.lib_to_str_dtypes[u[0].dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # memory allocation
    if len(fgrad) == 1:
        Nproj = fgrad[0].shape[1]
        rfft_s = [backend.zeros([Nproj, 1+Nb//2], dtype=cdtype) for _ in range(L)]
    else:
        rfft_s = [backend.zeros([fg.shape[1], 1+Nb//2], dtype=cdtype) for fg in fgrad]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_2d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=True,
                                                    notest=True) for
                 fgi in fgrad]
        
    # compute EPR projections in (half) Fourier domain
    for i in range(L):
        rfft_hi = rfft_h[i] if len(rfft_h) > 1 else rfft_h[0]
        fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
        ni = nodes[i] if len(nodes) > 1 else nodes[0]
        for j in range(K):
            rfft_s[i] += monosrc.proj2d_rfft(u[j], delta, B,
                                             rfft_hi[j], fgi,
                                             backend=backend, eps=eps,
                                             nodes=ni, notest=True)
    
    return rfft_s

def backproj2d(proj, delta, B, h, fgrad, out_shape, backend=None,
               eps=1e-06, rfft_mode=True, nodes=None, notest=False):
    """Perform EPR backprojection from 2D multisources EPR projections (adjoint of the ``proj2d`` operation).
    
    In the following, and as we did in the documentation of the
    :py:func:`pyepri.multisrc.proj2d` function, the index `j` shall
    refer to the `j-th` source image, while the index `i` shall refer
    to the `i-th` experimental setup. We shall denote by `K` the
    number of sources and by `L` the number of different experimental
    setup (those numbers are computed using ``K = len(u)`` and ``L =
    len(proj)``.
    
    Parameters
    ----------
    
    proj : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the EPR projections
        associated to each experimental setup. More precisely,
        ``proj[i][k,:]`` corresponds to the EPR projection of the
        multisources mixture acquired with field gradient
        ``fgrad[:,k]`` and `i-th` experimental setup.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    h : sequence of sequence of array_like (with type `backend.cls`)
        Contains the reference spectra (sampled over the grid ``B``)
        of each individual source (`j`) for each experimental setup
        (`i`). More precisely, ``h[i][j]`` is a monodimensional array
        corresponding to the sampling over ``B`` of the reference
        spectrum of the `j-th` source in the `i-th` experimental
        setting.
        
        When ``L > 1`` and ``len(h) = 1``, we assume that ``h[i][j] =
        h[0][j]``` for all ``i in range(L)`` and all ``j in
        range(K)``.
    
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    out_shape : sequence of sequence of int
        Sequence made of each source shape. More precisely,
        ``out_shape[j]`` corresponds the the shape of the `j-th`
        source image.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    rfft_mode : bool, optional
        The backprojection process involves the computation of the
        discrete Fourier coefficients of the input projections. Set
        ``rfft_mode=True`` to enable real FFT mode (only half of the
        Fourier coefficients will be computed to speed-up computation
        and reduce memory usage). Otherwise, use ``rfft_mode=False``,
        to enable standard (complex) FFT mode and compute all the
        Fourier coefficients.
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes associated to the input projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
        
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : sequence of array_like (with type `backend.cls`)
        A sequence with lenght `K` containing the backprojected source
        images (with shape ``out[j].shape = out_shape[j]`` for ``j in
        range(K)``).
    
    
    See also
    --------
   
    backproj2d_fft
    backproj2d_rfft
    proj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest: 
        _check_nd_inputs_(2, backend, B=B, delta=delta, fgrad=fgrad,
                          proj=proj, h=h, out_shape=out_shape,
                          eps=eps, rfft_mode=rfft_mode, nodes=nodes)
    
    # perform backprojection
    if rfft_mode:
        rfft_proj = [backend.rfft(pi) for pi in proj]
        rfft_h_conj = [[backend.rfft(hij).conj() for hij in hi] for hi in h]
        out = backproj2d_rfft(rfft_proj, delta, B, rfft_h_conj, fgrad,
                              backend=backend, out_shape=out_shape,
                              eps=eps, nodes=nodes, notest=True)
    else:
        fft_proj = [backend.fft(pi) for pi in proj]
        fft_h_conj = [[backend.fft(hij).conj() for hij in hi] for hi in h]
        out = backproj2d_fft(fft_proj, delta, B, fft_h_conj, fgrad,
                             backend=backend, out_shape=out_shape,
                             eps=eps, nodes=nodes, notest=True)
    
    return out

def backproj2d_fft(fft_proj, delta, B, fft_h_conj, fgrad,
                   backend=None, out_shape=None, out=None, eps=1e-6, nodes=None,
                   notest=False):
    """Perform EPR backprojection from 2D multisources EPR projections provided in Fourier domain.
    
    Parameters
    ----------
    
    fft_proj : sequence of complex array_like (with type
        `backend.cls`) A sequence with length `L` containing the
        discrete Fourier coefficients of the EPR projections
        associated to each experimental setup. More precisely,
        ``fft_proj[i][k,:]`` corresponds to the FFT of the EPR
        projection of the multisources mixture acquired with field
        gradient ``fgrad[:,k]`` and `i-th` experimental setup.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
    
    fft_h_conj : sequence of sequence of complex array_like (with
        type `backend.cls`) Contains the conjugated discrete Fourier
        coefficients of the reference spectra (sampled over the grid
        ``B``) of each individual source (`j`) for each experimental
        setup (`i`). More precisely, ``fft_h_conj[i][j]`` is a
        monodimensional array corresponding to the conjugate of the
        FFT of ``h[i][j]`` (the reference spectrum of the `j-th`
        source in the `i-th` experimental setting).
        
        When ``L > 1`` and ``len(fft_h_conj) = 1``, we assume that
        ``fft_h_conj[i][j] = fft_h_conj[0][j]``` for all ``i in
        range(L)`` and all ``j in range(K)``.
    
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    out_shape : sequence of sequence of int
        Sequence made of each source shape. More precisely,
        ``out_shape[j]`` corresponds to the shape of the `j-th`
        source image. optional input is in fact mandatory when no
        preallocated array is given (i.e., when ``out=None``).
    
    out : sequence of array_like (with type `backend.cls`), optional
        Preallocated sequence of output arrays (with shape
        ``out[j].shape = out_shape[j]`` for ``j in range(K)``).
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes associated to the input projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
    
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : sequence of array_like (with type `backend.cls`)
        A sequence with lenght `K` containing the backprojected source
        images (with shape ``out[j].shape = out_shape[j]`` for ``j in
        range(K)``).
    
    
    See also
    --------
    
    backproj2d
    backproj2d_rfft

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest: 
        _check_nd_inputs_(2, backend, B=B, delta=delta, fgrad=fgrad,
                          fft_proj=fft_proj, fft_h_conj=fft_h_conj,
                          out_shape=out_shape, out=out, eps=eps,
                          nodes=nodes)
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    #L = max(len(fgrad), len(fft_h_conj))
    L = len(fft_proj)
    K = len(out_shape)
    
    # retrieve real data type in str format
    dtype = backend.lib_to_str_dtypes[B.dtype]
    
    # memory allocation
    if out is None: 
        out = [backend.zeros(out_shape[j], dtype=dtype) for j in
               range(K)]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_2d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=False,
                                                    notest=True) for
                 fgi in fgrad]
    
    # main loop
    for i in range(L):
        fft_hi_conj = fft_h_conj[i] if len(fft_h_conj) > 1 else fft_h_conj[0]
        fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
        ni = nodes[i] if len(nodes) > 1 else nodes[0]
        for j in range(K):
            out[j] += monosrc.backproj2d_fft(fft_proj[i], delta, B,
                                             fft_hi_conj[j], fgi,
                                             backend=backend, eps=eps,
                                             out_shape=out[j].shape,
                                             nodes=ni).real
    
    return out

def backproj2d_rfft(rfft_proj, delta, B, rfft_h_conj, fgrad,
                    backend=None, out_shape=None, out=None, eps=1e-6,
                    nodes=None, notest=False):
    """Perform EPR backprojection from 2D multisources EPR projections provided in Fourier domain.
    
    Parameters
    ----------
    
    rfft_proj : sequence of complex array_like (with type
        `backend.cls`) A sequence with length `L` containing half of
        the discrete Fourier coefficients of the EPR projections
        associated to each experimental setup. More precisely,
        ``rfft_proj[i][k,:]`` corresponds to the real input FFT (rfft)
        of the EPR projection of the multisources mixture acquired
        with field gradient ``fgrad[:,k]`` and `i-th` experimental
        setup.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
    
    rfft_h_conj : sequence of sequence of complex array_like (with
        type `backend.cls`) Contains half of the conjugated discrete
        Fourier coefficients of the reference spectra (sampled over
        the grid ``B``) of each individual source (`j`) for each
        experimental setup (`i`). More precisely,
        ``rfft_h_conj[i][j]`` is a monodimensional array corresponding
        to the conjugate of the real input FFT of ``h[i][j]`` (the
        reference spectrum of the `j-th` source in the `i-th`
        experimental setting).
        
        When ``L > 1`` and ``len(rfft_h_conj) = 1``, we assume that
        ``rfft_h_conj[i][j] = rfft_h_conj[0][j]``` for all ``i in
        range(L)`` and all ``j in range(K)``.
        
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    out_shape : sequence of sequence of int
        Sequence made of each source shape. More precisely,
        ``out_shape[j]`` corresponds to the shape of the `j-th`
        source image. optional input is in fact mandatory when no
        preallocated array is given (i.e., when ``out=None``).
    
    out : sequence of array_like (with type `backend.cls`), optional
        Preallocated sequence of output arrays (with shape
        ``out[j].shape = out_shape[j]`` for ``j in range(K)``).
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes associated to the input projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : sequence of array_like (with type `backend.cls`)
        A sequence with lenght `K` containing the backprojected source
        images (with shape ``out[j].shape = out_shape[j]`` for ``j in
        range(K)``).
    
    
    See also
    --------
    
    backproj2d
    backproj2d_fft
    
    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, backend, B=B, delta=delta, fgrad=fgrad,
                          rfft_proj=rfft_proj,
                          rfft_h_conj=rfft_h_conj,
                          out_shape=out_shape, out=out, eps=eps,
                          nodes=nodes)
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    #L = max(len(fgrad), len(rfft_h_conj))
    L = len(rfft_proj)
    K = len(out_shape)
    
    # retrieve real data type in str format
    dtype = backend.lib_to_str_dtypes[B.dtype]
    
    # memory allocation
    if out is None:
        out = [backend.zeros(out_shape[j], dtype=dtype) for j in
               range(K)]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_2d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=True,
                                                    notest=True) for
                 fgi in fgrad]
    
    # main loop
    for i in range(L):
        rfft_hi_conj = rfft_h_conj[i] if len(rfft_h_conj) > 1 else rfft_h_conj[0]
        fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
        ni = nodes[i] if len(nodes) > 1 else nodes[0]
        for j in range(K):
            out[j] += monosrc.backproj2d_rfft(rfft_proj[i], delta, B,
                                              rfft_hi_conj[j], fgi,
                                              backend=backend,
                                              eps=eps,
                                              out_shape=out[j].shape,
                                              nodes=ni).real
    
    return out
    
def compute_2d_toeplitz_kernels(B, h, delta, fgrad, src_shape,
                                backend=None, eps=1e-06,
                                rfft_mode=True, nodes=None,
                                notest=False):
    """Compute 2D Toeplitz kernels allowing fast computation of a ``proj2d`` followed by a ``backproj2d`` operation.
    
    In the following, and as we did in the documentation of the
    :py:func:`pyepri.multisrc.proj2d` function, the index `j` shall
    refer to the `j-th` source image, while the index `i` shall refer
    to the `i-th` experimental setup. We shall denote by `K` the
    number of sources and by `L` the number of different experimental
    setup (those numbers are computed using ``K = len(src_shape)`` and
    ``L = max(len(h), len(fgrad))``.
    
    Parameters
    ----------
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    h : sequence of sequence of array_like (with type `backend.cls`)
        Contains the reference spectra (sampled over the grid ``B``)
        of each individual source (`j`) for each experimental setup
        (`i`). More precisely, ``h[i][j]`` is a monodimensional array
        corresponding to the sampling over ``B`` of the reference
        spectrum of the `j-th` source in the `i-th` experimental
        setting.
        
        When ``L > 1`` and ``len(h) = 1``, we assume that ``h[i][j] =
        h[0][j]``` for all ``i in range(L)`` and all ``j in
        range(K)``.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    src_shape : sequence of sequence of int
        Sequence made of each source shape. More precisely,
        ``src_shape[j]`` corresponds the the shape of the `j-th`
        source image.
        
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    rfft_mode : bool, optional 
        The evaluation of the sequence of output Toeplitz kernels
        involves the computation Fourier coefficients of real-valued
        signals. Set ``rfft_mode=True`` to enable real FFT mode (only
        half of the Fourier coefficients will be computed to speed-up
        computation and reduce memory usage). Otherwise, use
        ``rfft_mode=False``, to enable standard (complex) FFT mode and
        compute all the Fourier coefficients.
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes used to evaluate the output projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    phi : sequence of sequence of array_like (with type `backend.cls`)
        Output sequence of 2D cross source kernels (``phi[k][j]`` is
        the cross source kernel associated to source `k` (backward)
        and source `j` (forward)).        
    
    
    See also
    --------
    
    proj2d
    backproj2d
    apply_2d_toeplitz_kernels

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, backend, B=B, delta=delta, fgrad=fgrad,
                          h=h, src_shape=src_shape, eps=eps,
                          rfft_mode=rfft_mode, nodes=nodes)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_2d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=rfft_mode,
                                                    notest=True) for
                 fgi in fgrad]
    
    # retrieve complex data type
    dtype = backend.lib_to_str_dtypes[B.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    L = max(len(fgrad), len(h))
    K = len(src_shape)
    
    # memory allocation
    phi = [[backend.zeros([sk[0] + sj[0], sk[1] + sj[1]], dtype=dtype)
            for sj in src_shape] for sk in src_shape]
    
    # main loop: compute kernels for cross sources (k,j)
    for k in range(K):
        for j in range(K):
            s = (src_shape[k][0] + src_shape[j][0], src_shape[k][1] +
                 src_shape[j][1])
            for i in range(L):
                fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
                ni = nodes[i] if len(nodes) > 1 else nodes[0]
                phi[k][j] += monosrc.compute_2d_toeplitz_kernel(B,
                                                                h[i][j],
                                                                h[i][k],
                                                                delta,
                                                                fgi,
                                                                s,
                                                                backend=backend,
                                                                eps=eps,
                                                                rfft_mode=rfft_mode,
                                                                nodes=ni,
                                                                notest=True).real
    
    return phi

def apply_2d_toeplitz_kernels(u, rfft2_phi, backend=None, notest=False):
    """Perform a ``proj2d`` followed by a ``backproj2d`` operation using precomputed Toeplitz kernels provided in Fourier domain.
    
    Parameters
    ----------
    
    u : sequence of array_like (with type `backend.cls`)
        A sequence with length `K` containing the source images. More
        precisely, ``u[j]`` must be a 2D array corresponding to the
        `j-th` source image of the mixture to be projected.
    
    rfft2_phi : sequence of sequence of complex array_like (with type
        `backend.cls`) Sequence of real input FFT of the 2D cross
        sources Toeplitz kernels computed using
        :py:func:`pyepri.multisrc.compute_2d_toeplitz_kernels`.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
    
        When backend is None, a default backend is inferred from the
        input array ``u[0]``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return 
    ------
    
    out : sequence of array_like (with type `backend.cls`) 
        Sequence of output sources (stored in the same order as in
        ``u``).
    
    
    See also
    --------
    
    compute_2d_toeplitz_kernels
    proj2d
    backproj2d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u[0])
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, backend, u=u, rfft2_phi=rfft2_phi)
    
    # retrieve sources (real) data type in str format
    dtype = backend.lib_to_str_dtypes[u[0].dtype]
    
    # memory allocation
    out = [backend.zeros(im.shape, dtype=dtype) for im in u]
    
    # main loop (store sources real FFTs into a dictionary to avoid
    # unnecessary recomputations)
    rfft2_u = {}
    for j in range(len(u)):
        sj = u[j].shape
        for k in range(len(u)):
            sk = u[k].shape
            s = (sk[0] + sj[0], sk[1] + sj[1])
            # compute offsets (d0, d1) needed for sources with
            # different shapes (not well understood)
            d0 = s[0] % 2 if sj[0] % 2 == 1 else 0 
            d1 = s[1] % 2 if sj[1] % 2 == 1 else 0 
            r0, r1 = sj[0] - d0, s[0] - d0
            c0, c1 = sj[1] - d1, s[1] - d1
            # retrieve real FFT of source u[j] zero-padded to shape s
            if (j, s) not in rfft2_u.keys():
                rfft2_u[(j, s)] = backend.rfft2(u[j], s=s)
            # accumulate source contribution
            out[k] += backend.irfft2(rfft2_u[(j, s)] *
                                     rfft2_phi[k][j], s=s)[r0:r1,
                                                           c0:c1]
    
    return out
    
def proj3d(u, delta, B, h, fgrad, backend=None, eps=1e-06,
           rfft_mode=True, nodes=None, notest=False):
    """Compute EPR projections of from a sequence of 3D source images.
    
    This function can be used to simulate EPR projections from a
    mixture of 3D sources, in different experimental conditions (e.g.,
    different microwave power).
    
    In the following, the index `j` shall refer to the `j-th` source
    image, while the index `i` shall refer to the `i-th` experimental
    setup. We shall denote by `K` the number of sources and by `L` the
    number of different experimental setup (those numbers are computed
    using ``K = len(u)`` and ``L = max(len(h), len(fgrad))``.
    
    Parameters
    ----------
    
    u : sequence of array_like (with type `backend.cls`)
        A sequence with length `K` containing the source images. More
        precisely, ``u[j]`` must be a 3D array corresponding to the
        `j-th` source image of the mixture to be projected.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    h : sequence of sequence of array_like (with type `backend.cls`)
        Contains the reference spectra (sampled over the grid ``B``)
        of each individual source (`j`) for each experimental setup
        (`i`). More precisely, ``h[i][j]`` is a monodimensional array
        corresponding to the sampling over ``B`` of the reference
        spectrum of the `j-th` source in the `i-th` experimental
        setting.
        
        When ``L > 1`` and ``len(h) = 1``, we assume that ``h[i][j] =
        h[0][j]``` for all ``i in range(L)`` and all ``j in
        range(K)``.
        
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y,Z)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    rfft_mode : bool, optional 
        The EPR projections are evaluated in the frequency domain
        (through their dicrete Fourier coefficients) before being
        transformed back to the B-domain. Set ``rfft_mode=True`` to
        enable real FFT mode (only half of the Fourier coefficients
        will be computed to speed-up computation and reduce memory
        usage). Otherwise, use ``rfft_mode=False``, to enable standard
        (complex) FFT mode and compute all the Fourier coefficients.
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes used to evaluate the output projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
            
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    proj : sequence of array_like (with type `backend.cls`) 
        A sequence with length `L` containing the EPR projections
        synthesized for each experimental setting. More precisely,
        ``proj[i]`` is an array with shape ``(fgrad[i].shape[1],
        len(B))`` and ``proj[i][k,:]`` corresponds the EPR projection
        of the mixture `u` with field gradient ``fgrad[i][:,k]``
        sampled over the grid `B`.
    
    
    See also
    --------
    
    proj3d_fft
    proj3d_rfft
    backproj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)

    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, u=u, delta=delta, B=B, h=h,
                          fgrad=fgrad, eps=eps, rfft_mode=rfft_mode,
                          nodes=nodes)
    
    # compute EPR projections in Fourier domain and apply inverse DFT
    # to get the projections in B-domain
    if rfft_mode:
        rfft_h = [[backend.rfft(hij) for hij in hi] for hi in h]
        rfft_proj = proj3d_rfft(u, delta, B, rfft_h, fgrad,
                                backend=backend, eps=eps, nodes=nodes,
                                notest=True)
        proj = [backend.irfft(rfft_sino, n=len(B)) for rfft_sino in
                rfft_proj]
    else:
        fft_h = [[backend.fft(hij) for hij in hi] for hi in h]
        fft_proj = proj3d_fft(u, delta, B, fft_h, fgrad,
                              backend=backend, eps=eps, nodes=nodes,
                              notest=True)
        proj = [backend.ifft(fft_sino, n=len(B)).real for fft_sino in
                fft_proj]
    
    return proj

def proj3d_fft(u, delta, B, fft_h, fgrad, backend=None, eps=1e-06,
               nodes=None, notest=False):
    """Compute EPR projections of from a sequence of 3D source images (output in Fourier domain).
    
    Parameters
    ----------
    
    u : sequence of array_like (with type `backend.cls`)
        A sequence with length `K` containing the source images. More
        precisely, ``u[j]`` must be a 3D array corresponding to the
        `j-th` source image of the mixture to be projected.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    fft_h : sequence of sequence of complex array_like (with type
        `backend.cls`) Contains the discrete Fourier coefficients of
        the reference spectra (sampled over the grid ``B``) of each
        individual source (`j`) for each experimental setup
        (`i`). More precisely, ``fft_h[i][j]`` is a monodimensional
        array corresponding to the FFT of ``h[i][j]`` (the reference
        spectrum of the `j-th` source in the `i-th` experimental
        setting).
        
        When ``L > 1`` and ``len(fft_h) = 1``, we assume that
        ``fft_h[i][j] = fft_h[0][j]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
        
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y,Z)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
        
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes used to evaluate the output projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    fft_proj : sequence of complex array_like (with type
        `backend.cls`) A sequence with length `L` containing the
        discrete Fourier coefficients of the EPR projections
        synthesized for each experimental setting. More precisely,
        ``fft_proj[i]`` is an array with shape ``(fgrad[i].shape[1],
        len(B))`` and ``fft_proj[i][k,:]`` corresponds the FFT of the
        EPR projection of the mixture `u` with field gradient
        ``fgrad[i][:,k]`` sampled over the grid `B`.
    
    
    See also
    --------
    
    proj3d_rfft
    proj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, u=u, delta=delta, B=B,
                          fft_h=fft_h, fgrad=fgrad, eps=eps,
                          nodes=nodes)
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    L = max(len(fgrad), len(fft_h))
    K = len(u)
    
    # retrieve real & complex data types in str format
    dtype = backend.lib_to_str_dtypes[u[0].dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # memory allocation
    if len(fgrad) == 1:
        Nproj = fgrad[0].shape[1]
        fft_s = [backend.zeros([Nproj, Nb], dtype=cdtype) for _ in range(L)]
    else:
        fft_s = [backend.zeros([fg.shape[1], Nb], dtype=cdtype) for fg in fgrad]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_3d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=False,
                                                    notest=True) for
                 fgi in fgrad]
    
    # compute EPR projections in (half) Fourier domain
    for i in range(L):
        fft_hi = fft_h[i] if len(fft_h) > 1 else fft_h[0]
        fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
        ni = nodes[i] if len(nodes) > 1 else nodes[0]
        for j in range(K):
            fft_s[i] += monosrc.proj3d_fft(u[j], delta, B, fft_hi[j],
                                           fgi, backend=backend,
                                           eps=eps, nodes=ni,
                                           notest=True)
    
    return fft_s

def proj3d_rfft(u, delta, B, rfft_h, fgrad, backend=None, eps=1e-06,
                nodes=None, notest=False):
    """Compute EPR projections of from a sequence of 3D source images (output in Fourier domain, half of the full spectrum).
    
    Parameters
    ----------
    
    u : sequence of array_like (with type `backend.cls`)
        A sequence with length `K` containing the source images. More
        precisely, ``u[j]`` must be a 3D array corresponding to the
        `j-th` source image of the mixture to be projected.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    rfft_h : sequence of sequence of complex array_like (with type
        `backend.cls`) Contains half of the discrete Fourier
        coefficients of the reference spectra (sampled over the grid
        ``B``) of each individual source (`j`) for each experimental
        setup (`i`). More precisely, ``rfft_h[i][j]`` is a
        monodimensional array corresponding to the real input FFT
        (rfft) of ``h[i][j]`` (the reference spectrum of the `j-th`
        source in the `i-th` experimental setting).
        
        When ``L > 1`` and ``len(rfft_h) = 1``, we assume that
        ``rfft_h[i][j] = rfft_h[0][j]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y,Z)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
        
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes used to evaluate the output projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    rfft_proj : sequence of complex array_like (with type
        `backend.cls`) A sequence with length `L` containing half of
        the discrete Fourier coefficients of the EPR projections
        synthesized for each experimental setting. More precisely,
        ``rfft_proj[i]`` is an array with shape ``(fgrad[i].shape[1],
        1+len(B)//2)`` and ``rfft_proj[i][k,:]`` corresponds the real
        input FFT (rfft) of the EPR projection of the mixture `u` with
        field gradient ``fgrad[i][:,k]`` sampled over the grid `B`.
    
    
    See also
    --------
    
    proj3d_fft
    proj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, u=u, delta=delta, B=B,
                          rfft_h=rfft_h, fgrad=fgrad, eps=eps,
                          nodes=nodes)
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    L = max(len(fgrad), len(rfft_h))
    K = len(u)
    
    # retrieve real & complex data types in str format
    dtype = backend.lib_to_str_dtypes[u[0].dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # memory allocation
    if len(fgrad) == 1:
        Nproj = fgrad[0].shape[1]
        rfft_s = [backend.zeros([Nproj, 1+Nb//2], dtype=cdtype) for _ in range(L)]
    else:
        rfft_s = [backend.zeros([fg.shape[1], 1+Nb//2], dtype=cdtype) for fg in fgrad]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_3d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=True,
                                                    notest=True) for
                 fgi in fgrad]
        
    # compute EPR projections in (half) Fourier domain
    for i in range(L):
        rfft_hi = rfft_h[i] if len(rfft_h) > 1 else rfft_h[0]
        fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
        ni = nodes[i] if len(nodes) > 1 else nodes[0]
        for j in range(K):
            rfft_s[i] += monosrc.proj3d_rfft(u[j], delta, B,
                                             rfft_hi[j], fgi,
                                             backend=backend, eps=eps,
                                             nodes=ni, notest=True)
    
    return rfft_s

def backproj3d(proj, delta, B, h, fgrad, out_shape, backend=None,
               eps=1e-06, rfft_mode=True, nodes=None, notest=False):
    """Perform EPR backprojection from 3D multisources EPR projections (adjoint of the ``proj3d`` operation).
    
    In the following, and as we did in the documentation of the
    :py:func:`pyepri.multisrc.proj2d` function, the index `j` shall
    refer to the `j-th` source image, while the index `i` shall refer
    to the `i-th` experimental setup. We shall denote by `K` the
    number of sources and by `L` the number of different experimental
    setup (those numbers are computed using ``K = len(u)`` and ``L =
    len(proj)``.
    
    Parameters
    ----------
    
    proj : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the EPR projections
        associated to each experimental setup. More precisely,
        ``proj[i][k,:]`` corresponds to the EPR projection of the
        multisources mixture acquired with field gradient
        ``fgrad[:,k]`` and `i-th` experimental setup.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    h : sequence of sequence of array_like (with type `backend.cls`)
        Contains the reference spectra (sampled over the grid ``B``)
        of each individual source (`j`) for each experimental setup
        (`i`). More precisely, ``h[i][j]`` is a monodimensional array
        corresponding to the sampling over ``B`` of the reference
        spectrum of the `j-th` source in the `i-th` experimental
        setting.
        
        When ``L > 1`` and ``len(h) = 1``, we assume that ``h[i][j] =
        h[0][j]``` for all ``i in range(L)`` and all ``j in
        range(K)``.
    
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y,Z)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    out_shape : sequence of sequence of int
        Sequence made of each source shape. More precisely,
        ``out_shape[j]`` corresponds the the shape of the `j-th`
        source image.
        
    backend : <class 'pyepri.backends.Backend'>  or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    eps : float, optional
        Precision requested (>1e-16).
    
    rfft_mode : bool, optional
        The backprojection process involves the computation of the
        discrete Fourier coefficients of the input projections. Set
        ``rfft_mode=True`` to enable real FFT mode (only half of the
        Fourier coefficients will be computed to speed-up computation
        and reduce memory usage). Otherwise, use ``rfft_mode=False``,
        to enable standard (complex) FFT mode and compute all the
        Fourier coefficients.
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the frequency nodes
        associated to the input projections. If not given, the `nodes`
        sequence is automatically inferred from `B`, `h`, `delta` and
        `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : sequence of array_like (with type `backend.cls`)
        A sequence with lenght `K` containing the backprojected source
        images (with shape ``out[j].shape = out_shape[j]`` for ``j in
        range(K)``).
    
    
    See also
    --------
    
    backproj3d_fft
    backproj3d_rfft
    proj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, proj=proj, delta=delta, B=B,
                          h=h, fgrad=fgrad, out_shape=out_shape,
                          eps=eps, rfft_mode=rfft_mode, nodes=nodes)
    
    # perform backprojection
    if rfft_mode:
        rfft_proj = [backend.rfft(pi) for pi in proj]
        rfft_h_conj = [[backend.rfft(hij).conj() for hij in hi] for hi in h]
        out = backproj3d_rfft(rfft_proj, delta, B, rfft_h_conj, fgrad,
                              backend=backend, out_shape=out_shape,
                              eps=eps, nodes=nodes, notest=True)
    else:
        fft_proj = [backend.fft(pi) for pi in proj]
        fft_h_conj = [[backend.fft(hij).conj() for hij in hi] for hi in h]
        out = backproj3d_fft(fft_proj, delta, B, fft_h_conj, fgrad,
                             backend=backend, out_shape=out_shape,
                             eps=eps, nodes=nodes, notest=True)
    
    return out

def backproj3d_fft(fft_proj, delta, B, fft_h_conj, fgrad,
                   backend=None, out_shape=None, out=None, eps=1e-6,
                   nodes=None, notest=False):
    """Perform EPR backprojection from 3D multisources EPR projections provided in Fourier domain.
    
    Parameters
    ----------
    
    fft_proj : sequence of complex array_like (with type
        `backend.cls`) A sequence with length `L` containing the
        discrete Fourier coefficients of the EPR projections
        associated to each experimental setup. More precisely,
        ``fft_proj[i][k,:]`` corresponds to the FFT of the EPR
        projection of the multisources mixture acquired with field
        gradient ``fgrad[:,k]`` and `i-th` experimental setup.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
    
    fft_h_conj : sequence of sequence of complex array_like (with
        type `backend.cls`) Contains the conjugated discrete Fourier
        coefficients of the reference spectra (sampled over the grid
        ``B``) of each individual source (`j`) for each experimental
        setup (`i`). More precisely, ``fft_h_conj[i][j]`` is a
        monodimensional array corresponding to the conjugate of the
        FFT of ``h[i][j]`` (the reference spectrum of the `j-th`
        source in the `i-th` experimental setting).
        
        When ``L > 1`` and ``len(fft_h_conj) = 1``, we assume that
        ``fft_h_conj[i][j] = fft_h_conj[0][j]``` for all ``i in
        range(L)`` and all ``j in range(K)``.
    
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y,Z)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    out_shape : sequence of sequence of int
        Sequence made of each source shape. More precisely,
        ``out_shape[j]`` corresponds to the shape of the `j-th`
        source image. optional input is in fact mandatory when no
        preallocated array is given (i.e., when ``out=None``).
    
    out : sequence of array_like (with type `backend.cls`), optional
        Preallocated sequence of output arrays (with shape
        ``out[j].shape = out_shape[j]`` for ``j in range(K)``).
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes associated to the input projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
    
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : sequence of array_like (with type `backend.cls`)
        A sequence with lenght `K` containing the backprojected source
        images (with shape ``out[j].shape = out_shape[j]`` for ``j in
        range(K)``).
    
    
    See also
    --------
    
    backproj2d
    backproj2d_rfft

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, fft_proj=fft_proj, delta=delta,
                          B=B, fft_h_conj=fft_h_conj, fgrad=fgrad,
                          out_shape=out_shape, out=out, eps=eps,
                          nodes=nodes)
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    L = max(len(fgrad), len(fft_h_conj))
    K = len(out_shape)
    
    # retrieve real data type in str format
    dtype = backend.lib_to_str_dtypes[B.dtype]
    
    # memory allocation
    if out is None: 
        out = [backend.zeros(out_shape[j], dtype=dtype) for j in
               range(K)]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_3d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=False,
                                                    notest=True) for
                 fgi in fgrad]
    
    # main loop
    for i in range(L):
        fft_hi_conj = fft_h_conj[i] if len(fft_h_conj) > 1 else fft_h_conj[0]
        fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
        ni = nodes[i] if len(nodes) > 1 else nodes[0]
        for j in range(K):
            out[j] += monosrc.backproj3d_fft(fft_proj[i], delta, B,
                                             fft_hi_conj[j], fgi,
                                             backend=backend, eps=eps,
                                             out_shape=out[j].shape,
                                             nodes=ni).real
    
    return out

def backproj3d_rfft(rfft_proj, delta, B, rfft_h_conj, fgrad,
                    backend=None, out_shape=None, out=None, eps=1e-6,
                    nodes=None, notest=False):
    """Perform EPR backprojection from 3D multisources EPR projections provided in Fourier domain.
    
    Parameters
    ----------
    
    rfft_proj : sequence of complex array_like (with type
        `backend.cls`) A sequence with length `L` containing half of
        the discrete Fourier coefficients of the EPR projections
        associated to each experimental setup. More precisely,
        ``rfft_proj[i][k,:]`` corresponds to the real input FFT (rfft)
        of the EPR projection of the multisources mixture acquired
        with field gradient ``fgrad[:,k]`` and `i-th` experimental
        setup.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
    
    rfft_h_conj : sequence of sequence of complex array_like (with
        type `backend.cls`) Contains half of the conjugated discrete
        Fourier coefficients of the reference spectra (sampled over
        the grid ``B``) of each individual source (`j`) for each
        experimental setup (`i`). More precisely,
        ``rfft_h_conj[i][j]`` is a monodimensional array corresponding
        to the conjugate of the real input FFT of ``h[i][j]`` (the
        reference spectrum of the `j-th` source in the `i-th`
        experimental setting).
        
        When ``L > 1`` and ``len(rfft_h_conj) = 1``, we assume that
        ``rfft_h_conj[i][j] = rfft_h_conj[0][j]``` for all ``i in
        range(L)`` and all ``j in range(K)``.
        
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y,Z)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
    
    out_shape : sequence of sequence of int
        Sequence made of each source shape. More precisely,
        ``out_shape[j]`` corresponds to the shape of the `j-th`
        source image. optional input is in fact mandatory when no
        preallocated array is given (i.e., when ``out=None``).
    
    out : sequence of array_like (with type `backend.cls`), optional
        Preallocated sequence of output arrays (with shape
        ``out[j].shape = out_shape[j]`` for ``j in range(K)``).
    
    eps : float, optional
        Precision requested (>1e-16).
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes associated to the input projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : sequence of array_like (with type `backend.cls`)
        A sequence with lenght `K` containing the backprojected source
        images (with shape ``out[j].shape = out_shape[j]`` for ``j in
        range(K)``).
    
    
    See also
    --------
    
    backproj3d
    backproj3d_fft
    
    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, rfft_proj=rfft_proj,
                          delta=delta, B=B, rfft_h_conj=rfft_h_conj,
                          fgrad=fgrad, out_shape=out_shape, out=out,
                          eps=eps, nodes=nodes)
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    L = max(len(fgrad), len(rfft_h_conj))
    K = len(out_shape)
    
    # retrieve real data type in str format
    dtype = backend.lib_to_str_dtypes[B.dtype]
    
    # memory allocation
    if out is None:
        out = [backend.zeros(out_shape[j], dtype=dtype) for j in
               range(K)]
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_3d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=True,
                                                    notest=True) for
                 fgi in fgrad]
    
    # main loop
    for i in range(L):
        rfft_hi_conj = rfft_h_conj[i] if len(rfft_h_conj) > 1 else rfft_h_conj[0]
        fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
        ni = nodes[i] if len(nodes) > 1 else nodes[0]
        for j in range(K):
            out[j] += monosrc.backproj3d_rfft(rfft_proj[i], delta, B,
                                              rfft_hi_conj[j], fgi,
                                              backend=backend,
                                              eps=eps,
                                              out_shape=out[j].shape,
                                              nodes=ni).real
    
    return out
    
def compute_3d_toeplitz_kernels(B, h, delta, fgrad, src_shape,
                                backend=None, eps=1e-06,
                                rfft_mode=True, nodes=None,
                                notest=False):
    """Compute 3D Toeplitz kernels allowing fast computation of a ``proj3d`` followed by a ``backproj3d`` operation.
    
    In the following, and as we did in the documentation of the
    :py:func:`pyepri.multisrc.proj3d` function, the index `j` shall
    refer to the `j-th` source image, while the index `i` shall refer
    to the `i-th` experimental setup. We shall denote by `K` the
    number of sources and by `L` the number of different experimental
    setup (those numbers are computed using ``K = len(src_shape)`` and
    ``L = max(len(h), len(fgrad))``.
    
    Parameters
    ----------
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        [B-unit] (can be Gauss (G), millitesla (mT), ...), to use to
        compute the projections.
        
    h : sequence of sequence of array_like (with type `backend.cls`)
        Contains the reference spectra (sampled over the grid ``B``)
        of each individual source (`j`) for each experimental setup
        (`i`). More precisely, ``h[i][j]`` is a monodimensional array
        corresponding to the sampling over ``B`` of the reference
        spectrum of the `j-th` source in the `i-th` experimental
        setting.
        
        When ``L > 1`` and ``len(h) = 1``, we assume that ``h[i][j] =
        h[0][j]``` for all ``i in range(L)`` and all ``j in
        range(K)``.
    
    delta : float
        Pixel size given in a length unit denoted below as
        [length-unit] (can be centimeter (cm), millimeter (mm), ...).
    
    fgrad : sequence of array_like (with type `backend.cls`)
        A sequence with length `L` containing the coordinates of the
        field gradient vector used for each experimental setting. More
        precisely, ``fgrad[i][:,k]`` corresponds to the (X,Y,Z)
        coordinates of the field gradient vector associated to the
        `k-th` EPR projection in the `i-th` experimental setting.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    src_shape : sequence of sequence of int
        Sequence made of each source shape. More precisely,
        ``src_shape[j]`` corresponds the the shape of the `j-th`
        source image.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``B``.
        
    eps : float, optional
        Precision requested (>1e-16).
    
    rfft_mode : bool, optional 
        The evaluation of the sequence of output Toeplitz kernels
        involves the computation Fourier coefficients of real-valued
        signals. Set ``rfft_mode=True`` to enable real FFT mode (only
        half of the Fourier coefficients will be computed to speed-up
        computation and reduce memory usage). Otherwise, use
        ``rfft_mode=False``, to enable standard (complex) FFT mode and
        compute all the Fourier coefficients.
    
    nodes : sequence of dict, optional
        A sequence of length `L` containing the precomputed frequency
        nodes used to evaluate the output projections for each
        experimental setting. If not given, the `nodes` sequence is
        automatically inferred from `B`, `delta` and `fgrad`.
        
        When ``L > 1`` and ``len(nodes) = 1``, we assume that
        ``nodes[i] = nodes[0]``` for all ``i in range(L)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    phi : sequence of sequence of array_like (with type `backend.cls`)
        Output sequence of 3D cross source kernels (``phi[k][j]`` is
        the cross source kernel associated to source `k` (backward)
        and source `j` (forward)).        
    
    
    See also
    --------
    
    proj3d
    backproj3d
    apply_3d_toeplitz_kernels

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(B=B)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, B=B, h=h, delta=delta,
                          fgrad=fgrad, src_shape=src_shape, eps=eps,
                          rfft_mode=rfft_mode, nodes=nodes)
    
    # compute irregular frequency nodes (if not provided as input)
    if nodes is None:
        nodes = [monosrc.compute_3d_frequency_nodes(B, delta, fgi,
                                                    backend=backend,
                                                    rfft_mode=rfft_mode,
                                                    notest=True) for
                 fgi in fgrad]
    
    # retrieve complex data type
    dtype = backend.lib_to_str_dtypes[B.dtype]
    cdtype = backend.mapping_to_complex_dtypes[dtype]
    
    # retrieve signals dimensions (Nb = number or sample per
    # projection, L = number of sinograms, K = number of sources)
    Nb = len(B)
    L = max(len(fgrad), len(h))
    K = len(src_shape)
    
    # memory allocation
    phi = [[backend.zeros([sk[m] + sj[m] for m in range(3)],
                          dtype=dtype) for sj in src_shape] for sk in
           src_shape]
    
    # main loop: compute kernels for cross sources (k,j)
    for k in range(K):
        for j in range(K):
            s = tuple((src_shape[k][m] + src_shape[j][m] for m in range(3)))
            for i in range(L):
                fgi = fgrad[i] if len(fgrad) > 1 else fgrad[0]
                ni = nodes[i] if len(nodes) > 1 else nodes[0]
                phi[k][j] += monosrc.compute_3d_toeplitz_kernel(B,
                                                                h[i][j],
                                                                h[i][k],
                                                                delta,
                                                                fgi,
                                                                s,
                                                                backend=backend,
                                                                eps=eps,
                                                                rfft_mode=rfft_mode,
                                                                nodes=ni,
                                                                notest=True).real
    
    return phi

def apply_3d_toeplitz_kernels(u, rfft3_phi, backend=None, notest=False):
    """Perform a ``proj3d`` followed by a ``backproj3d`` operation using precomputed Toeplitz kernels provided in Fourier domain.
    
    Parameters
    ----------
    
    u : sequence of array_like (with type `backend.cls`)
        A sequence with length `K` containing the source images. More
        precisely, ``u[j]`` must be a 3D array corresponding to the
        `j-th` source image of the mixture to be projected.
    
    rfft3_phi : sequence of sequence of complex array_like (with type
        `backend.cls`) Sequence of real input FFT of the 3D cross
        sources Toeplitz kernels computed using
        :py:func:`pyepri.multisrc.compute_3d_toeplitz_kernels`.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``u[0]``.    
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return 
    ------
    
    out : sequence of array_like (with type `backend.cls`) 
        Sequence of output sources (stored in the same order as in
        ``u``).
    
    
    See also
    --------
    
    compute_3d_toeplitz_kernels
    proj3d
    backproj3d

    """
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u[0])
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, u=u, rfft3_phi=rfft3_phi)
    
    # retrieve sources (real) data type in str format
    dtype = backend.lib_to_str_dtypes[u[0].dtype]
    
    # memory allocation
    out = [backend.zeros(im.shape, dtype=dtype) for im in u]
    
    # main loop (store sources real FFTs into a dictionary to avoid
    # unnecessary recomputations)
    rfft3_u = {}
    for j in range(len(u)):
        sj = u[j].shape
        for k in range(len(u)):
            sk = u[k].shape
            s = tuple((sk[m] + sj[m] for m in range(3)))
            # compute offsets (d0, d1) needed for sources with
            # different shapes (not well understood)
            d0 = s[0] % 2 if sj[0] % 2 == 1 else 0 
            d1 = s[1] % 2 if sj[1] % 2 == 1 else 0 
            d2 = s[2] % 2 if sj[2] % 2 == 1 else 0 
            r0, r1 = sj[0] - d0, s[0] - d0
            c0, c1 = sj[1] - d1, s[1] - d1
            z0, z1 = sj[2] - d2, s[2] - d2
            # retrieve real FFT of source u[j] zero-padded to shape s
            if (j, s) not in rfft3_u.keys():
                rfft3_u[(j, s)] = backend.rfftn(u[j], s=s)
            # accumulate source contribution
            out[k] += backend.irfftn(rfft3_u[(j, s)] *
                                     rfft3_phi[k][j], s=s)[r0:r1,
                                                           c0:c1,
                                                           z0:z1]
    
    return out

def _check_nd_inputs_(ndims, backend, B=None, delta=None, fgrad=None,
                      u=None, h=None, proj=None, eps=None, nodes=None,
                      rfft_mode=None, out_shape=None, fft_h=None,
                      fft_h_conj=None, rfft_h=None, rfft_h_conj=None,
                      fft_proj=None, rfft_proj=None, src_shape=None,
                      rfft2_phi=None, rfft3_phi=None, out=None):
    """Factorized consistency checks for functions in the :py:mod:`pyepri.multisrc` submodule."""
    
    ######################################################
    # retrieve number of sources & number of experiments #
    ######################################################
    K = checks._max_len_(u=u, out_shape=out_shape, src_shape=src_shape)
    L = checks._max_len_(proj=proj, fft_proj=fft_proj, rfft_proj=rfft_proj,
                  h=h, fft_h=fft_h, fft_h_conj=fft_h_conj,
                  rfft_h=rfft_h, rfft_h_conj=rfft_h_conj)
    
    ################
    # start checks #
    ################
    if B is not None: # every function except apply_2d_toeplitz_kernels
                      # and apply_3d_toeplitz_kernels
        
        # retrieve number of magnetic field samples
        Nb = len(B)
        
        # retrieve datatypes
        dtype = B.dtype
        str_dtype = backend.lib_to_str_dtypes[dtype]
        str_cdtype = backend.mapping_to_complex_dtypes[str_dtype]
        cdtype = backend.str_to_lib_dtypes[str_cdtype]
        
        # check B
        checks._check_backend_(backend, B=B)
        checks._check_ndim_(1, B=B)
        
        # delta: must be None or a float
        if delta is not None and not isinstance(delta, float):
            raise RuntimeError(
                "Parameter `delta` must be a float scalar number."
            )
        
        # eps: must be None or a float
        if eps is not None and not isinstance(eps, float):
            raise RuntimeError(
                "Parameter `eps` must be a float scalar number."
            )        
        
        # fgrad: a sequence of 1 or L array_like with ndims row each
        checks._check_seq_(t=backend.cls, dtype=dtype, ndim=2, fgrad=fgrad)
        if len(fgrad) not in (1, L):
            raise RuntimeError(
                "Incorrect length for parameter ``fgrad``"
            )
        if not all([arr.shape[0] == ndims for arr in fgrad]):
            raise RuntimeError(
                "The array elements in `fgrad` must have %d rows each" % ndims
            )
        
        # (u, out): sequences made of K array_like with dtype==dtype
        checks._check_seq_(t=backend.cls, dtype=dtype, n=K,
                           ndim=ndims, u=u, out=out)
        
        # out_shape & src_shape: sequences of sequences of ndim int
        checks._check_seq_of_seq_(t=int, len0=K, len1=ndims,
                                  out_shape=out_shape,
                                  src_shape=src_shape)
                      
        # check h: (check dtype == dtype, made of L or 1 sequence(s)
        # containing K element each, all leaves have ndim=1 & len =
        # Nb)
        checks._check_seq_of_seq_(t=backend.cls, dtype=dtype, ndim=1,
                                  tlen0=(1,L), len1=K, len2=Nb)
        
        # check fft_h, fft_h_conj, rfft_h, rfft_h_conj: check dtype ==
        # cdtype, number of elements, & all leaves have dim = 1 & len
        # = Nb or 1+Nb//2
        checks._check_seq_of_seq_(t=backend.cls, dtype=cdtype, ndim=1,
                                  tlen0=(1,L), len1=K, len2=Nb,
                                  fft_h=fft_h, fft_h_conj=fft_h_conj)
        checks._check_seq_of_seq_(t=backend.cls, dtype=cdtype, ndim=1,
                                  tlen0=(1,L), len1=K, len2=1+Nb//2,
                                  rfft_h=rfft_h,
                                  rfft_h_conj=rfft_h_conj)
        
        # check fft_proj
        checks._check_seq_(t=backend.cls, dtype=cdtype, n=L, ndim=2,
                           fft_proj=fft_proj)
        if fft_proj is not None:
            for i, p in enumerate(fft_proj):
                fg = fgrad[i] if len(fgrad) > 1 else fgrad[0]
                Nproj = fg.shape[1]
                if p.shape[0] != Nproj or p.shape[1] != Nb:
                    raise RuntimeError(                    
                        "Inconsistent shape for %d-th element of `fft_proj` (expected shape = (%d, %d))" % (i, Nproj, Nb)
                    )
                
        # check rfft_proj
        checks._check_seq_(t=backend.cls, dtype=cdtype, n=L, ndim=2,
                           rfft_proj=rfft_proj)
        if fft_proj is not None:
            for i, p in enumerate(rfft_proj):
                fg = fgrad[i] if len(fgrad) > 1 else fgrad[0]
                Nproj = fg.shape[1]
                if p.shape[0] != Nproj or p.shape[1] != 1 + Nb//2:
                    raise RuntimeError(                    
                        "Inconsistent shape for %d-th element of `rfft_proj` (expected shape = (%d, %d))" % (i, Nproj, 1 + Nb//2)
                    )
        
        # check proj
        checks._check_seq_(t=backend.cls, dtype=dtype, n=L, ndim=2,
                           proj=proj)
        if proj is not None:
            for i, p in enumerate(proj):
                fg = fgrad[i] if len(fgrad) > 1 else fgrad[0]
                Nproj = fg.shape[1]
                if p.shape[0] != Nproj or p.shape[1] != Nb:
                    raise RuntimeError(                    
                        "Inconsistent shape for %d-th element of `proj` (expected shape = (%d, %d))" % (i, Nproj, Nb)
                    )

        # check out or out_shape is provided in backproj?d_* functions
        if fft_proj is not None or rfft_proj is not None and out == out_shape == None:
            raise RuntimeError(
                "At least one of {`out_shape`, `out`} parameters must be given"
            )
        
        # check nodes
        checks._check_seq_(t=dict, nodes=nodes)
        if nodes is not None:
            for k, _nodes in enumerate(nodes):
                strnodes = "nodes[%d]" % k
                if ndims == 2:
                    if not all({t in _nodes.keys() for t in {'x', 'y', 'indexes', 'rfft_mode'}}):
                        raise RuntimeError(
                            "Input ``%s`` must contain the keys 'x', 'y', 'indexes', 'rfft_mode'" % strnodes
                        )
                    if not backend.is_backend_compliant(_nodes['x'], _nodes['y'], _nodes['indexes']):
                        raise RuntimeError(
                            "The content of the ``%s`` parameter is not consistent with the provided backend.\n"
                            "Since `backend.lib` is `" + backend.lib.__name__ + "`, ``%s['x']``, ``%s['y']`` and "
                            "``%s['indexes']`` must all be\n"
                            "" + str(backend.cls) + " instances." % (strnodes,)*4
                        )
                    checks._check_ndim_(1, **{"%s['x']" % strnodes : _nodes['x'],
                                              "%s['y']" % strnodes : _nodes['y'],
                                              "%s['indexes']" % strnodes: _nodes['indexes']})
                    if not len(_nodes['x']) == len(_nodes['y']) == len(_nodes['indexes']):
                        raise RuntimeError(
                            "%s['x'], %s['y'] and %s['indexes'] must have the same length.\n" % (strnodes,)*3
                        )
                    checks._check_dtype_(dtype, **{"%s['x']" % strnodes : _nodes['x'],
                                                   "%s['y']" % strnodes : _nodes['y']})
                elif ndims == 3:
                    if not all({t in _nodes.keys() for t in {'x', 'y', 'z', 'indexes', 'rfft_mode'}}):
                        raise RuntimeError(
                            "Input parameter ``%s`` must contain the keys 'x', 'y', 'z', 'indexes', 'rfft_mode'" % strnodes
                        )
                    if not backend.is_backend_compliant(_nodes['x'], _nodes['y'], _nodes['z'], _nodes['indexes']):
                        raise RuntimeError(
                            "The content of the ``%s`` parameter is not consistent with the provided backend.\n"
                            "Since `backend.lib` is `" + backend.lib.__name__ + "`, ``%s['x']``, ``%s['y']`` "
                            "``%s['z']`` and ``%s['indexes']`` must all be\n"
                            "" + str(backend.cls) + " instances." % (strnodes,)*5
                        )
                    if not len(_nodes['x']) == len(_nodes['y']) == len(_nodes['z']) == len(_nodes['indexes']):
                        raise RuntimeError(                    
                            "%s['x'], %s['y'], %s['z'] and %s['indexes'] must have the same length.\n" % (strnodes,)*4
                        )
                    checks._check_dtype_(dtype, **{"%s['x']" % strnodes: nodes['x'],
                                                   "%s['y']" % strnodes: nodes['y'],
                                                   "%s['z']" % strnodes: nodes['z']})
                if rfft_mode is not None and rfft_mode != _nodes['rfft_mode']:
                    raise RuntimeError(
                        "Cannot use ``rfft_mode=%s`` with ``%s['rfft_mode']=%s`` (both must be the same)."
                        % (rfft_mode, strnodes, _nodes['rfft_mode'])
                    )
                cof = (Nb//2+1 if rfft_mode else Nb)
                numel = Nproj * cof
                if _nodes['indexes'].min() < 0 or _nodes['indexes'].max() >= numel:
                    raise RuntimeError(
                        "The values in %s['indexes'] must be in the range [0,M)\n"
                        "with M = fgrad.shape[1] * %s (= %d * %d = %d)."                
                        % (strnodes, "(1 + len(B)//2)" if rfft_mode else "len(B)",
                           fgrad.shape[1] , cof, numel)
                    )
        
    else:
        
        # check u 
        checks._check_seq_(t=backend.cls, n=K, ndim=ndims, u=u)
        dtype = u[0].dtype
        str_dtype = backend.lib_to_str_dtypes[dtype]
        str_cdtype = backend.mapping_to_complex_dtypes[str_dtype]
        cdtype = backend.str_to_lib_dtypes[str_cdtype]
        
        # check rfft2_phi
        checks._check_seq_of_seq_(t=backend.cls, dtype=cdtype, len0=K, len1=K, ndim=2, rfft2_phi=rfft2_phi)
        
        # check rfft3_phi
        checks._check_seq_of_seq_(t=backend.cls, dtype=cdtype, len0=K, len1=K, ndim=3, rfft3_phi=rfft3_phi)
        
    return True
