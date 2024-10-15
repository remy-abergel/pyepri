"""This module provides high-level tools for EPR image reconstruction
(filtered-backprojection, TV-regularized least-squares, ...).

"""
import math
import matplotlib.pyplot as plt
import types
import pyepri.checks as checks
import pyepri.multisrc as multisrc
import pyepri.monosrc as monosrc
import pyepri.optimization as optimization
import pyepri.utils as utils

def tv_monosrc(proj, B, fgrad, delta, h, lbda, out_shape,
               backend=None, init=None, tol=1e-5, nitermax=None,
               eval_energy=False, verbose=False, video=False,
               displayer=None, Ndisplay=20, eps=1e-6,
               disable_toeplitz_kernel=False, notest=False):
    r"""EPR single source image reconstruction using TV-regularized least-squares.
    
    This functions performs EPR image reconstruction by minimization
    of the TV-regularized least-squares energy
    
    .. math :: 
       
       E(u) := \frac{1}{2} \|A(u) - s\|_2^2 + \lambda_{\text{unrm}}
       \mathrm{TV}(u)
    
    Where :math:`u` is a 2D or 3D image, :math:`A` denotes the EPR
    projection operator (that changes an image `u` into EPR
    projections), :math:`s` denotes the measured projections (input
    parameter ``proj`` below), :math:`TV` denotes a discrete total
    variation regularizer, and :math:`\lambda_{\text{unrm}} > 0` is a
    regularity parameter (or TV weight) whose value is set
    proportional to the ``lbda`` input parameter (see below).
    
    This functions relies on the numerical optimization scheme
    :py:func:`pyepri.optimization.tvsolver_cp2016` to perform the
    minimization of the TV-regularized least-squares energy :math:`E`
    (see :cite:p:`Abergel_2023` for more details).
    
    Parameters
    ----------
    
    proj : array_like (with type `backend.cls`)
        Two-dimensional array with shape ``(Nproj, len(B))`` (where
        ``Nproj = fgrad.shape[1]``) such that ``proj[k,:]``
        corresponds to the k-th EPR projection (acquired with field
        gradient ``fgrad[:,k]`` and sampled over the grid ``B``).
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...),
        associated to the input projections ``proj``.
    
    fgrad : array_like (with type `backend.cls`)
        Two or three dimensional array such that ``fgrad[:,k]``
        corresponds to the coordinates of the field gradient vector
        associated to the k-th EPR projection to be computed.
        
        The physical unit of the field gradient should be consistent
        with that of `B` and delta, i.e., `fgrad` must be provided in
        `[B-unit] / [length-unit]` (e.g., `G/cm`, `mT/cm`, ...).
        
    delta : float 
        Pixel size for the reconstruction, given in a length unit
        denoted as `[length-unit]` (can be `centimeter (cm)`,
        `millimeter (mm)`, ...).
    
    h : array_like (with type `backend.cls`) 
        One dimensional array with same length as `B` corresponding to
        the reference spectrum sampled over the grid `B`.
    
    lbda : float 
        Normalized regularization parameter, the actual TV weight
        :math:`\lambda_{\text{unrm}}` (denoted below by ``lbda_unrm``)
        will be computed using
        
        >>> Bsw = (B[-1] - B[0]).item() 
        >>> ndim = fgrad.shape[0]
        >>> cof1 = (h.max() - h.min()).item() 
        >>> cof2 = fgrad.shape[1] / (2.**(ndim - 1) * math.pi) 
        >>> cof3 = (len(B) / Bsw) 
        >>> lbda_unrm = cof1 * cof2 * cof3 * delta**(ndim-1) * lbda
        
    out_shape : tuple or list of int with length 2 or 3
        Shape of the output image.    
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(proj, B, fgrad, h)``.
    
    init : array_like (with type `backend.cls`)
        Initializer for the numerical optimization scheme
        (:py:func:`pyepri.optimization.tvsolver_cp2016`).
    
    tol : float, optional
        A tolerance parameter used to stop the iterations of the
        numerical optimization scheme.
    
    nitermax : int, optional
        Maximal number of iterations for the numerical optimization
        scheme.
    
    eval_energy : bool, optional 
        Enable / disable energy evaluation during the iterations of
        the numerical optimization scheme.
    
        When ``eval_energy is True``, the TV-regularized least-squares
        energy to be minimized will be computed each ``Ndisplay``
        iteration of the scheme. The computed energy values are
        displayed when ``verbose`` or ``video`` modes are enabled (see
        below). 
    
        The computed energy values are not returned by this function
        (to retrieve the energy value, the user can directly use the
        :py:func:`pyepri.optimization.tvsolver_cp2016` function).
    
        Enabling energy computation will increase computation time.
    
    verbose : bool, optional 
        Enable / disable verbose mode. Set ``verbose = True`` to print
        each enable verbose mode of the numerical optimization scheme.
    
    video : bool, optional
        Enable / disable video mode (display and refresh the latent
        image each ``Ndisplay`` iteration).
    
    displayer : <class 'pyepri.displayers.Displayer'>, optional 
        Image displayer used to display the latent image during the
        scheme iteration. When not given (``displayer is None``), a
        default displayer is instantiated.
    
        Enabling video is definitely helpful but also slows-down the
        execution (it is recommended to use the ``Ndisplay`` parameter
        to control the image refreshing rate).
        
    Ndisplay : int, optional
        Can be used to limit energy evaluation, image display, and
        verbose mode to the iteration indexes ``k`` that are multiples
        of Ndisplay (useful to speed-up the process).
    
    eps : float, optional
        Precision requested (>1e-16) for the EPR related monosrc
        involved in the reconstruction model (see functions in the
        :py:mod:`pyepri.monosrc` submodule).
    
    disable_toeplitz_kernel : bool, optional
        The numerical optimization scheme performs at each iteration a
        projection followed by a backprojection operation. By default
        (i.e., when ``disable_toeplitz_kernel is False``), this
        operation is done by means of a circular convolution using a
        Toeplitz kernel (which can be evaluated efficiently using FFT).
    
        When ``disable_toeplitz_kernel is True``, the Toeplitz kernel is
        not used and the evaluation is down sequentially (compute
        projection and then backprojection).
    
        The setting of this parameter should not affect the returned
        image, but only the computation time (the default setting
        should be the faster).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : array_like (with type `backend.cls`) 
        Output image (when convergence of the numerical optimization
        scheme, this image is a minimizer of the TV-regularized
        least-squares energy).
    
    
    See also
    --------
    
    pyepri.optimization.tvsolver_cp2016
    tv_multisrc

    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(proj=proj, B=B, fgrad=fgrad, h=h)
    
    # consistency checks
    if not notest:
        _check_inputs_("tv_monosrc", backend, proj=proj, B=B,
                       fgrad=fgrad, delta=delta, h=h, lbda=lbda,
                       out_shape=out_shape, init=init, tol=tol,
                       nitermax=nitermax, eval_energy=eval_energy,
                       disable_toeplitz_kernel=disable_toeplitz_kernel,
                       verbose=verbose, video=video,
                       displayer=displayer, eps=eps,
                       Ndisplay=Ndisplay)
    
    # retrieve number of dimensions (2D/3D)
    ndim = fgrad.shape[0]
    
    # define the direct operator (A) and its adjoint (adjA)
    if 2 == ndim:
        gen_nodes = monosrc.compute_2d_frequency_nodes
        project = monosrc.proj2d
        backproject = monosrc.backproj2d
    else:
        gen_nodes = monosrc.compute_3d_frequency_nodes
        project = monosrc.proj3d
        backproject = monosrc.backproj3d
    nodes = gen_nodes(B, delta, fgrad, backend=backend,
                      rfft_mode=True)
    A = lambda u : project(u, delta, B, h, fgrad, eps=eps,
                           backend=backend, rfft_mode=True,
                           nodes=nodes, notest=True)
    adjA = lambda v : backproject(v, delta, B, h, fgrad, out_shape,
                                  backend=backend, eps=eps,
                                  nodes=nodes, rfft_mode=True,
                                  notest=True)
    
    # compute v = adjA(proj), define the mapping gradf corresponding
    # to the gradient of the quadratic data-fidelity term (gradf = u
    # -> adjA(A(u)) - v), and compute a Lipschitz constant (Lf) of
    # gradf
    v = adjA(proj)
    s = tuple(2*s for s in out_shape)
    if 2 == ndim:
        M1, M2 = out_shape
        gen_kernel = monosrc.compute_2d_toeplitz_kernel
        rfft2_phi = gen_kernel(B, h, h, delta, fgrad, s,
                               backend=backend, eps=eps,
                               rfft_mode=True, nodes=nodes,
                               return_rfft2=True)
        Lf = backend.abs(rfft2_phi).max()
        rfft2 = backend.rfft2 # improve readability in gradf
        irfft2 = backend.irfft2 # improve readability in gradf
        gradf = lambda u : irfft2(rfft2(u, s=s) * rfft2_phi,
                                  s=s)[M1:, M2:] - v
    else:
        M1, M2, M3 = out_shape
        gen_kernel = monosrc.compute_3d_toeplitz_kernel
        rfft3_phi = gen_kernel(B, h, h, delta, fgrad, s,
                               backend=backend, eps=eps,
                               rfft_mode=True, nodes=nodes,
                               return_rfft3=True)
        Lf = backend.abs(rfft3_phi).max()
        rfftn = backend.rfftn # improve readability in gradf
        irfftn = backend.irfftn # improve readability in gradf
        gradf = lambda u : irfftn(rfftn(u, s=s) * rfft3_phi,
                                      s=s)[M1:, M2:, M3:] - v
    
    # deal with disable_toeplitz_kernel option
    if disable_toeplitz_kernel:
        gradf = lambda u : adjA(A(u)) - v
    
    # compute unnormalized regularity parameter
    Bsw = (B[-1] - B[0]).item()
    cof1 = (h.max() - h.min()).item()
    cof2 = fgrad.shape[1] / (2.**(ndim - 1) * math.pi)
    cof3 = (len(B) / Bsw)
    lbda_unrm = cof1 * cof2 * cof3 * delta**(ndim-1) * lbda
    
    # prepare generic tv solver configuration
    if 2 == ndim:
        grad = lambda u : utils.grad2d(u, backend=backend, notest=True)
        div = lambda g : utils.div2d(g, backend=backend, notest=True)
        Lgrad = math.sqrt(8.)
    else:
        grad = lambda u : utils.grad3d(u, backend=backend, notest=True)
        div = lambda g : utils.div3d(g, backend=backend, notest=True)
        Lgrad = math.sqrt(12.)
    if eval_energy:
        tv = lambda u : (backend.sqrt((grad(u)**2).sum(axis=0))).sum()
        evalE = lambda u : .5*((A(u)-proj)**2).sum() + lbda_unrm*tv(u)
    else:
        evalE = None
    if init is None:
        init = v / (delta**ndim * v.sum())
    
    # run generic TV solver
    out = optimization.tvsolver_cp2016(init, gradf, Lf, lbda_unrm,
                                       grad, div, Lgrad, tol=tol,
                                       backend=backend,
                                       nitermax=nitermax, evalE=evalE,
                                       verbose=verbose, video=video,
                                       Ndisplay=Ndisplay,
                                       displayer=displayer)
    
    return out['u']

def tv_multisrc(proj, B, fgrad, delta, h, lbda, out_shape,
                backend=None, init=None, tol=1e-5, nitermax=None,
                eval_energy=False, disable_toeplitz_kernel=False,
                verbose=False, video=False, Ndisplay=20,
                displayer=None, eps=1e-6, notest=False):
    r"""EPR source separation TV-regularized least-squares.
    
    This function implements the multi-sources EPR image
    reconstruction method presented in :cite:p:`Boussaa_2023`. This
    function can be viewed as a generalization of
    :py:func:`tv_monosrc` which has be extended to address either one
    or both of the following situations:
    
    + the sample is made of K >= 1 EPR source, leading to the problem
      of reconstructing K image(s) (one image per EPR source);
    
    + projections acquisitions are done in different experimental
      setups (for instance by changing the microwave power from one
      experiment to another), so that the reconstruction is done
      according to L >= 1 sinogram acquisitions.
    
    Parameters
    ----------
    
    proj : sequence of array_like (with type `backend.cls`)
        A sequence with length L such that ``proj[i]`` is a
        two-dimensional array containing the EPR projections acquired
        in the `i-th` acquisition setup. 
    
        In particular, each ``proj[i]`` must contain ``len(B)``
        columns (see parameter ``B`` below) and each row ``pi in
        proj[i]`` corresponds to an EPR projection of the multisources
        sample in the `i-th` acquisition setup.
    
    B : array_like (with type `backend.cls`)
        One dimensional array corresponding to the homogeneous
        magnetic field sampling grid, with unit denoted below as
        `[B-unit]` (can be `Gauss (G)`, `millitesla (mT)`, ...),
        associated to the input projections ``proj``.
    
    fgrad : sequence of array_like (with type `backend.cls`) 
        A sequence with length L, such that each ``frad[i]`` is a two
        or three dimensional array with the following properties:
        
        + ``fgrad[i].shape[0]`` is 2 or 3 (depending on the dimension
          of the images to be reconstructed);
        + ``fgrad[i].shape[1] == proj[i].shape[0]``; 
        + ``fgrad[i][:,k]`` corresponds to the coordinates of the
          field gradient vector used to acquire ``proj[i][k,:]``.
        
        The physical unit of the provided field gradients `[B-unit] /
        [length-unit]` (e.g., `G/cm`, `mT/cm`, ...) and must be
        consistent with that units of the input parameters ``B`` and
        ``delta``.
        
        When ``L > 1`` and ``len(fgrad) = 1``, we assume that
        ``fgrad[i][j] = fgrad[0][i]``` for all ``i in range(L)`` and
        all ``j in range(K)``.
    
    delta : float 
        Pixel size for the reconstruction, given in `[length-unit]`
        consistent with that used in ``fgrad`` (can be `centimeter
        (cm)`, `millimeter (mm)`, ...).
    
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
    
    lbda : float
        Normalized regularization parameter, the actual TV weight
        (denoted below by ``lbda_unrm``) that will be passed to the
        numerical optimization scheme
        (:py:func:`pyepri.optimization.tvsolver_cp2016_multisrc`) will
        be computed using
    
        >>> Bsw = (B[-1] - B[0]).item()    
        >>> cof1 = max(tuple((h_ij.max() - h_ij.min()).item() for h_i
        ... in h for h_ij in h_i))
        >>> cof2 = float(Nproj) / (2.**(ndim - 1) * math.pi * len(fgrad))
        >>> cof3 = (len(B) / Bsw)
        >>> lbda_unrm = cof1 * cof2 * cof3 * delta**(ndim-1) * lbda
        
    out_shape : sequence of sequence of int
        Sequence made of each source shape. More precisely,
        ``out_shape[j]`` corresponds the the shape of the `j-th`
        source image.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``(proj, B, fgrad)``.
    
    init : sequence of array_like (with type `backend.cls`)
        A sequence of length L used as initializer for the numerical
        optimization scheme
        (:py:func:`pyepri.optimization.tvsolver_cp2016_multisrc`).
    
    tol : float, optional
        A tolerance parameter used to stop the iterations of the
        numerical optimization scheme.
    
    nitermax : int, optional
        Maximal number of iterations for the numerical optimization
        scheme.
    
    eval_energy : bool, optional 
        Enable / disable energy evaluation during the iterations of
        the numerical optimization scheme.
    
        When ``eval_energy is True``, the TV-regularized least-squares
        energy to be minimized will be computed each ``Ndisplay``
        iteration of the scheme. The computed energy values are
        displayed when ``verbose`` or ``video`` modes are enabled (see
        below). 
    
        The computed energy values are not returned by this function
        (to retrieve the energy value, the user can directly use the
        :py:func:`pyepri.optimization.tvsolver_cp2016_multisrc`
        function).
    
        Enabling energy computation will increase computation time.
    
    verbose : bool, optional 
        Enable / disable verbose mode. Set ``verbose = True`` to print
        each enable verbose mode of the numerical optimization scheme.
    
    video : bool, optional
        Enable / disable video mode (display and refresh the latent
        image each ``Ndisplay`` iteration).
    
    displayer : <class 'pyepri.displayers.Displayer'>, optional 
        Image displayer used to display the latent multi-source image
        during the scheme iteration. When not given (``displayer is
        None``), a default displayer is instantiated.
    
        Enabling video is definitely helpful but also slows-down the
        execution (it is recommended to use the ``Ndisplay`` parameter
        to control the image refreshing rate).
        
    Ndisplay : int, optional
        Can be used to limit energy evaluation, image display, and
        verbose mode to the iteration indexes ``k`` that are multiples
        of Ndisplay (useful to speed-up the process).
    
    eps : float, optional
        Precision requested (>1e-16) for the EPR related monosrc
        involved in the reconstruction model (see functions in the
        :py:mod:`pyepri.monosrc` submodule).
    
    disable_toeplitz_kernel : bool, optional
        The numerical optimization scheme performs at each iteration a
        projection followed by a backprojection operation. By default
        (i.e., when ``disable_toeplitz_kernel is False``), this
        operation is done by means of a circular convolutions between
        some Toeplitz kernels and the sources images. Those
        convolutions can be evaluated efficiently using FFT.
    
        When ``disable_toeplitz_kernel is True``, the Toeplitz kernels
        are not used and the evaluation is down sequentially (compute
        projection and then backprojection).
    
        The setting of this parameter should not affect the returned
        images, but only the computation time (the default setting
        should be the faster).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    out : array_like (with type `backend.cls`) 
        Output image (when convergence of the numerical optimization
        scheme, this image is a minimizer of the TV-regularized
        least-squares energy).
    
    
    See also
    --------
    
    pyepri.optimization.tvsolver_cp2016
    tv_monosrc
    
    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(proj=proj, B=B, fgrad=fgrad)
    
    # consistency checks
    if not notest:
        _check_inputs_("tv_multisrc", backend, proj=proj, B=B,
                       fgrad=fgrad, delta=delta, h=h, lbda=lbda,
                       out_shape=out_shape, init=init, tol=tol,
                       nitermax=nitermax, eval_energy=eval_energy,
                       disable_toeplitz_kernel=disable_toeplitz_kernel,
                       verbose=verbose, video=video,
                       displayer=displayer, eps=eps,
                       Ndisplay=Ndisplay)
    
    # retrieve number of dimensions (2D/3D)
    ndim = fgrad[0].shape[0]
    
    # define the direct operator (A) and its adjoint (adjA)
    if 2 == ndim:
        gen_nodes = monosrc.compute_2d_frequency_nodes
        project = multisrc.proj2d
        backproject = multisrc.backproj2d
    else:
        gen_nodes = monosrc.compute_3d_frequency_nodes
        project = multisrc.proj3d
        backproject = multisrc.backproj3d
    nodes = [gen_nodes(B, delta, fgi, backend=backend, rfft_mode=True,
                       notest=True) for fgi in fgrad]
    A = lambda u : project(u, delta, B, h, fgrad, backend=backend,
                           eps=eps, rfft_mode=True, nodes=nodes,
                           notest=True)
    adjA = lambda v : backproject(v, delta, B, h, fgrad, out_shape,
                                  backend=backend, eps=eps,
                                  nodes=nodes, rfft_mode=True,
                                  notest=True)
    
    # compute v = adjA(proj), define the mapping gradf corresponding
    # to the gradient of the quadratic data-fidelity term (gradf = u
    # -> adjA(A(u)) - v), and compute a Lipschitz constant (Lf) of
    # gradf
    v = adjA(proj)

    # compute Toeplitz kernels
    if 2 == ndim:
        gen_kernel = multisrc.compute_2d_toeplitz_kernels
        apply_kernel = multisrc.apply_2d_toeplitz_kernels
        rfftn = backend.rfft2
    else:
        gen_kernel = multisrc.compute_3d_toeplitz_kernels
        apply_kernel = multisrc.apply_3d_toeplitz_kernels
        rfftn = backend.rfftn
    rfftn_phi = [[rfftn(phi_kj) for phi_kj in phi_k] for phi_k in
                 gen_kernel(B, h, delta, fgrad, out_shape,
                            backend=backend, eps=eps, nodes=nodes,
                            rfft_mode=True, notest=True)]
    
    # compute Lf
    K = len(rfftn_phi)
    Lf = 0.
    for k in range(K):
        s = 0.
        for j in range(K):
            s += backend.abs(rfftn_phi[k][j]).max().item()**2
        Lf = max([Lf, s])        
    Lf = math.sqrt(Lf)
    
    # define gradf
    def gradf(u):
        adjAAu = apply_kernel(u, rfftn_phi, backend=backend, notest=True)
        return tuple(adjAAu[j] - v[j] for j in range(len(u)))
    
    # deal with disable_toeplitz_kernel option
    if disable_toeplitz_kernel:
        def gradf(u):
            adjAAu = adjA(A(u))
            return tuple(adjAAu[j] - v[j] for j in range(len(u)))
    
    # compute total number of projections
    Nproj = 0.
    for fg in fgrad:
        Nproj += fg.shape[1]
    
    # compute unnormalized regularity parameter
    Bsw = (B[-1] - B[0]).item()    
    cof1 = max(tuple((h_ij.max() - h_ij.min()).item() for h_i in h for h_ij in h_i))
    cof2 = float(Nproj) / (2.**(ndim - 1) * math.pi * len(fgrad))
    cof3 = (len(B) / Bsw)
    lbda_unrm = cof1 * cof2 * cof3 * delta**(ndim-1) * lbda
    
    # prepare generic tv solver configuration
    if 2 == ndim:
        grad = lambda u : utils.grad2d(u, backend=backend, notest=True)
        div = lambda g : utils.div2d(g, backend=backend, notest=True)
        Lgrad = math.sqrt(8.)
    else:
        grad = lambda u : utils.grad3d(u, backend=backend, notest=True)
        div = lambda g : utils.div3d(g, backend=backend, notest=True)
        Lgrad = math.sqrt(12.)
    if eval_energy:
        def evalE(u):
            Au = A(u)
            s = 0.
            for i in range(len(proj)):
                s += ((Au[i] - proj[i])**2).sum().item()
            t = 0.
            for j in range(len(u)):
                t += backend.sqrt((grad(u[j])**2).sum(axis=0)).sum().item()
            return .5 * s + lbda_unrm * t
    else:
        evalE = None
    if init is None:
        init = [im / (delta**ndim * im.sum()) for im in v]
    
    # run generic TV solver
    solver = optimization.tvsolver_cp2016_multisrc
    out = solver(init, gradf, Lf, lbda_unrm, grad, div, Lgrad,
                 backend=backend, tol=tol, nitermax=nitermax,
                 evalE=evalE, verbose=verbose, video=video,
                 Ndisplay=Ndisplay, displayer=displayer)
    
    return out['u']

def eprfbp2d(proj, fgrad, h, B, xgrid, ygrid, interp1, backend=None,
             frequency_cutoff=1., verbose=False, video=False,
             displayer=None, Ndisplay=1, shuffle=False, notest=False):
    """Filtered back-projection for 2D image reconstruction from EPR measurements.
    
    Parameters
    ----------
    
    proj : array_like (with type `backend.cls`)
        Two-dimensional array containing the measured EPR projections
        (each row represents a measured projection).
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(2, proj.shape[0])`` such
        that ``fgrad[:,k,]`` corresponds to the (X,Y) coordinates of
        the field gradient vector associated to ``proj[k,:]``.
        
        We shall denote below by [B-unit]/[length-unit] the physical
        unit of the field gradient, where [B-unit] stands for the
        magnetic B-field stength unit (e.g., Gauss (G), millitesla
        (mT), ...) and [length-unit] stends for the lenght unit (e.g.,
        centimeter (cm), millimeter (mm), ...).
    
    h : array_like (with type `backend.cls`)
        One dimensional array with length ``proj.shape[1]``
        corresponding to the reference (or zero-gradient) spectrum
        sampled over the same sampling grid as the measured
        projections.
    
    B : array_like (with type `backend.cls`)
        One dimensional array with length ``proj.shape[1]``
        corresponding to the homogeneous magnetic field sampling grid
        (with unit [B-unit]) used to measure the projections
        (``proj``) and the reference spectrum (``h``).
    
    xgrid : array_like (with type `backend.cls`)
        Sampling nodes (with unit [length-unit]) along the X-axis of
        the 2D space (axis with index 1 of the reconstructed image).
    
    ygrid : array_like (with type `backend.cls`)
        Sampling nodes (with unit [length-unit]) along the Y-axis of
        the 2D space (axis with index 0 of the reconstructed image).
    
    interp1 : <class 'function'> 
        One-dimensional interpolation function with prototype ``y =
        interp1(xp,fp,x)``, where ``xp``, ``fp`` and ``x`` are
        one-dimensional arrays (with type `backend.cls`) corresponding
        respectively to
        
        - the coordinate nodes of the input data points,
        - the input data points,
        - the coordinate query nodes at which to evaluate the
          interpolated values, 
        - the interpolated values.
        
        Some examples of linear interpolations in numpy, cupy and
        torch are given below :
        
        >>> lambda xp, fp, x : numpy.interp(x, xp, fp, left=0., right=0.)
        >>> lambda xp, fp, x : cupy.interp(x, xp, fp, left=0., right=0.)
        >>> lambda xp, fp, x : torchinterp1d.interp1d(xp, fp, x)
    
        The last example above, relies on the `torchinterp1d` package
        (available from `Github
        <https://github.com/aliutkus/torchinterp1d>`).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input arrays ``(proj, fgrad, h, B, xgrid, ygrid)``.
    
    frequency_cutoff : float, optional
        Apply a frequency cutoff to the projections during the
        filtering process (to avoid noise amplification caused by the
        Ram-Lak filter).
    
        The input `frequency_cutoff` value corresponds to the
        proportion of Fourier coefficients to preserve (e.g., use
        ``frequency_cutoff = 0.1`` to preserve 10% of the
        (lowest-frequency) Fourier coefficients and set to zero the
        90% remaining (high-frequency) Fourier coefficients).
    
        Since frequency cutoff causes ringing artifact (i.e.,
        generates spurious oscillation patterns), we recommend to
        avoid using this parameter (i.e., to keep ``frequency_cutoff =
        1.``) and perform instead apodization of the input projections
        using a smoother apodization profile as a preprocessing before
        calling this function (see Example section below).
    
    verbose : bool, optional 
        Enable / disable verbose mode (display a message each time a
        projection is processed).
    
    video : bool, optional
        Enable / disable video mode. When video mode is enable
        (``video=True``), show the latent image and refresh display
        each time a chunk of ``Ndisplay`` projections has been
        processed.
    
        Please note that enabling video mode will slow down the
        computation (especially when a GPU device is used because the
        the displayed images are transferred to the CPU device).
    
    displayer : <class 'pyepri.displayers.Displayer'>, optional
        When video is True, the attribute ``displayer.init_display``
        and ``displayer.update_display`` will be used to display the
        latent image ``u`` along the iteration of the numerical
        scheme.
    
        When not given (``displayer=None``), a default displayer will
        be instantiated.
    
    Ndisplay : int, optional 
        Refresh the displayed image each time a chunk of Ndisplay
        projections has been processed (this parameter is not used
        when ``video is False``).
        
    shuffle : bool, optional
        Enable / disable shuffling of the projections order (shuffling
        has no effect on the final returned volume, but may lead to
        faster shape appearance when video mode is enabled).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return   
    ------    
    
    out : array_like (with type `backend.cls`)
        Two dimensional array corresponding to the reconstructed
        image. 
        
        The dimensions (0,1) correspond to the spatial axes (Y,X) of
        the image (this is the classical convention in image
        processing).
        
        The reconstruction is performed over the rectangular grid
        generated from the input grids along each coordinate axis
        (xgrid, ygrid), i.e.,
        
        ``out[i,j]`` 
        
        corresponds to the reconstructed image at the spatial location
        ``X, Y = xgrid[j], ygrid[i]``.
    
    
    See also
    --------
    
    eprfbp3d

    """

    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(proj=proj, fgrad=fgrad,
                                             h=h, B=B, xgrid=xgrid,
                                             ygrid=ygrid)
    
    # consistency checks
    if not notest:
        _check_inputs_("eprfbp2d", backend, proj=proj, fgrad=fgrad,
                       h=h, B=B, xgrid=xgrid, ygrid=ygrid,
                       interp1=interp1,
                       frequency_cutoff=frequency_cutoff,
                       verbose=verbose, video=video,
                       displayer=displayer, Ndisplay=Ndisplay,
                       shuffle=shuffle)
    
    # retrieve field gradient coordinates & amplitude 
    gx, gy = fgrad 
    mu = backend.sqrt(gx**2 + gy**2)
    
    # compute volumetric sampling nodes
    x, y = backend.meshgrid(xgrid, ygrid)
    x = x.ravel()
    y = y.ravel()
    
    # memory allocation for output volume
    Nx, Ny = len(xgrid), len(ygrid)
    im = backend.zeros([1, Nx * Ny],
                       dtype=backend.lib_to_str_dtypes[proj.dtype])
    
    # compute absorption profile (in Fourier domain)
    dB = B[1] - B[0]
    rfft_g = backend.rfft(backend.ifftshift(backend.cumsum(h, dim=0) *
                                            dB, dim=0))
    
    # precompute filter in Fourier domain
    Nb = len(B)
    alf = backend.arange(Nb//2+1, dtype='int32') # low frequency
                                                 # indexes (half of
                                                 # the low-frequency
                                                 # domain)
    #xi = 2. * math.pi * alf / (Nb * dB) # corresponding pulsations (rad / [B-unit])
    rfft_w = mu.reshape((-1, 1)) * (-1j / rfft_g).reshape((1, -1))
    
    # apply frequency cutoff
    M = max(int(1), int(round(frequency_cutoff * Nb)))
    rfft_w[:, (M//2 + 1)::] = 0.
        
    # compute filtered projections 
    fproj = backend.real(backend.fftshift(backend.irfft(backend.rfft(
        backend.ifftshift(proj, dim=1), dim=1) * rfft_w, n=Nb, dim=1), dim=1))
    rnodes = (-(Nb//2) + backend.arange(Nb, dtype='int32')) * dB
    
    # deal with video mode
    if video :
        if displayer is None:
            displayer = displayers.create_3d_displayer(nsrc=1,
                                                       figsize=None,
                                                       extents=extents,
                                                       adjust_dynamic=True,
                                                       display_labels=True,
                                                       grids=(ygrid, xgrid),
                                                       origin='lower',
                                                       boundaries='same')
        im_np = backend.to_numpy(im.reshape([Ny, Nx]))
        fg = displayer.init_display(im_np)
        fgnum = displayer.get_number(fg)
    
    # avoid zero-gradient projections and deal with shuffle option
    idproj = backend.argwhere(mu != 0)
    if shuffle :
        idproj = idproj[backend.randperm(len(idproj))]

    #############
    # main loop #
    #############
    cnt = 0
    for id in idproj:
        
        # resample and accumulate filtered projections
        r = - (x * gx[id] + y * gy[id])
        im += interp1(rnodes, fproj[id,:], r)
        cnt += 1
        
        # deal with verbose & video modes
        if verbose or video:
            msg = "processed projection %d/%d" % (cnt, len(idproj))
        if verbose:
            print(msg)
        if video and 0 == cnt % Ndisplay:
            if video and plt.fignum_exists(fgnum):
                im_np = backend.to_numpy(im.reshape([Ny, Nx]))
                displayer.update_display(im_np, fg)
                displayer.title(msg)
                displayer.pause()
        
        # deal with quit event (when video mode is enabled)
        if video and not plt.fignum_exists(fgnum):
            break
    
    # display final result before returning (when video mode is enabled)
    if video and plt.fignum_exists(fgnum) :
        im_np = backend.to_numpy(im.reshape([Ny, Nx]))
        displayer.update_display(im_np, fg)
        msg = "processed projection %d/%d" % (cnt, len(idproj))
        displayer.title(msg)
    
    # close figure when code is running on interactive notebook
    if video and displayer.notebook:
        displayer.clear_output()

    return im.reshape([len(ygrid), len(xgrid)])


def eprfbp3d(proj, fgrad, h, B, xgrid, ygrid, zgrid, interp1,
             backend=None, frequency_cutoff=1., verbose=False,
             video=False, displayer=None, Ndisplay=1, shuffle=False,
             notest=False):
    """Filtered back-projection for 3D image reconstruction from EPR measurements.
    
    Parameters
    ----------
    
    proj : array_like (with type `backend.cls`)
        Two-dimensional array containing the measured EPR projections
        (each row represents a measured projection).
    
    fgrad : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(3, proj.shape[0])`` such
        that ``fgrad[:,k,]`` corresponds to the (X,Y,Z) coordinates of
        the field gradient vector associated to ``proj[k,:]``.
        
        We shall denote below by [B-unit]/[length-unit] the physical
        unit of the field gradient, where [B-unit] stands for the
        magnetic B-field stength unit (e.g., Gauss (G), millitesla
        (mT), ...) and [length-unit] stends for the lenght unit (e.g.,
        centimeter (cm), millimeter (mm), ...).
    
    h : array_like (with type `backend.cls`)
        One dimensional array with length ``proj.shape[1]``
        corresponding to the reference (or zero-gradient) spectrum
        sampled over the same sampling grid as the measured
        projections.
    
    B : array_like (with type `backend.cls`)
        One dimensional array with length ``proj.shape[1]``
        corresponding to the homogeneous magnetic field sampling grid
        (with unit [B-unit]) used to measure the projections
        (``proj``) and the reference spectrum (``h``).
    
    xgrid : array_like (with type `backend.cls`)
        Sampling nodes (with unit [length-unit]) along the X-axis of
        the 3D space (axis with index 1 of the reconstructed volume).
    
    ygrid : array_like (with type `backend.cls`)
        Sampling nodes (with unit [length-unit]) along the Y-axis of
        the 3D space (axis with index 0 of the reconstructed volume).
    
    zgrid : array_like (with type `backend.cls`)
        Sampling nodes (with unit [length-unit]) along the Z-axis of
        the 3D space (axis with index 2 of the reconstructed volume).
    
    interp1 : <class 'function'> 
        One-dimensional interpolation function with prototype ``y =
        interp1(xp,fp,x)``, where ``xp``, ``fp`` and ``x`` are
        one-dimensional arrays (with type backend.cls) corresponding
        respectively to
        
        - the coordinate nodes of the input data points,
        - the input data points,
        - the coordinate query nodes at which to evaluate the
          interpolated values, 
        - the interpolated values.
        
        Some examples of linear interpolations in numpy, cupy and
        torch are given below :
        
        >>> lambda xp, fp, x : numpy.interp(x, xp, fp, left=0., right=0.)
        >>> lambda xp, fp, x : cupy.interp(x, xp, fp, left=0., right=0.)
        >>> lambda xp, fp, x : torchinterp1d.interp1d(xp, fp, x)
    
        The last example above, relies on the `torchinterp1d` package
        (available from `Github
        <https://github.com/aliutkus/torchinterp1d>`).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).

        When backend is None, a default backend is inferred from the
        input arrays ``(proj, fgrad, h, B, xgrid, ygrid, zgrid)``.

    frequency_cutoff : float, optional
        Apply a frequency cutoff to the projections during the
        filtering process (to avoid noise amplification caused by the
        Ram-Lak filter).
    
        The input `frequency_cutoff` value corresponds to the
        proportion of Fourier coefficients to preserve (e.g., use
        `frequency_cutoff = 0.1` to preserve 10% of the
        (lowest-frequency) Fourier coefficients and set to zero the
        90% remaining (high-frequency) Fourier coefficients).
    
        Since frequency cutoff causes ringing artifact (i.e.,
        generates spurious oscillation patterns), we recommend to
        avoid using this parameter (i.e., to keep ``frequency_cutoff =
        1.``) and perform instead apodization of the input projections
        using a smoother apodization profile as a preprocessing before
        calling this function (see Example section below).
    
    verbose : bool, optional 
        Enable / disable verbose mode (display a message each time a
        projection is processed).
    
    video : bool, optional
        Enable / disable video mode. When video mode is enable
        (`video=True`), show three slices of the latent volume and
        refresh the display each time a chunk of Ndisplay projections
        has been processed.
    
        Please note that enabling video mode will slow down the
        computation (especially when a GPU device is used because the
        the displayed images are transferred to the CPU device).
    
    displayer : <class 'pyepri.displayers.Displayer'>, optional
        When video is True, the attribute ``displayer.init_display``
        and ``displayer.update_display`` will be used to display the
        latent image ``u`` along the iteration of the numerical
        scheme.
    
        When not given (``displayer=None``), a default displayer will
        be instantiated.
    
    Ndisplay : int, optional 
        Refresh the displayed images each time a chunk of Ndisplay
        projections has been processed (this parameter is not used
        when ``video is False``).
        
    shuffle : bool, optional
        Enable / disable shuffling of the projections order (shuffling
        has no effect on the final returned volume, but may lead to
        faster shape appearance when video mode is enabled).
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return   
    ------    
    
    vol : array_like (with type `backend.cls`)
        Three dimensional array corresponding to the reconstructed
        volume. 
        
        The dimensions (0,1,2) correspond to the spatial axes (Y,X,Z)
        (we used this convention for compliance with data ordering in
        the 2D setting).
        
        The reconstruction is performed over the rectangular grid
        generated from the input grids along each coordinate axis
        (xgrid, ygrid, zgrid), i.e.,
        
        ``vol[i,j,k]`` 
        
        corresponds to the reconstructed volume at the spatial
        location ``X, Y, Z = xgrid[j], ygrid[i], zgrid[k]``.
    
    See also
    --------

    eprfbp2d

    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(proj=proj, fgrad=fgrad,
                                             h=h, B=B, xgrid=xgrid,
                                             ygrid=ygrid, zgrid=zgrid)
    
    # consistency checks
    if not notest:
        _check_inputs_("eprfbp3d", backend, proj=proj, fgrad=fgrad,
                       h=h, B=B, xgrid=xgrid, ygrid=ygrid,
                       zgrid=zgrid, interp1=interp1,
                       frequency_cutoff=frequency_cutoff,
                       verbose=verbose, video=video,
                       displayer=displayer, Ndisplay=Ndisplay,
                       shuffle=shuffle)
    
    # retrieve field gradient coordinates & amplitude 
    gx, gy, gz = fgrad 
    mu = backend.sqrt(gx**2 + gy**2 + gz**2)
    
    # compute volumetric sampling nodes
    x, y, z = backend.meshgrid(xgrid, ygrid, zgrid)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    
    # memory allocation for output volume
    Nx, Ny, Nz = len(xgrid), len(ygrid), len(zgrid)
    vol = backend.zeros([1, Nx * Ny * Nz],
                        dtype=backend.lib_to_str_dtypes[proj.dtype])
    
    # compute absorption profile (in Fourier domain)
    dB = B[1] - B[0]
    rfft_g = backend.rfft(backend.ifftshift(backend.cumsum(h, dim=0) *
                                            dB, dim=0))
    
    # precompute filter in Fourier domain
    Nb = len(B)
    alf = backend.arange(Nb//2+1, dtype='int32') # low frequency
                                                 # indexes (half of
                                                 # the low-frequency
                                                 # domain)
    xi = 2. * math.pi * alf / (Nb * dB) # corresponding pulsations
                                        # (rad / [B-unit])
    rfft_w = (mu * backend.sin(backend.arccos(gz / mu))).reshape((-1, 1)) * (-1j * xi / rfft_g).reshape((1, -1))
    
    # apply frequency cutoff
    M = max(int(1), int(round(frequency_cutoff * Nb)))
    rfft_w[:, (M//2 + 1)::] = 0.
        
    # compute filtered projections 
    fproj = backend.real(backend.fftshift(backend.irfft(backend.rfft(
        backend.ifftshift(proj, dim=1), dim=1) * rfft_w, n=Nb, dim=1), dim=1))
    rnodes = (-(Nb//2) + backend.arange(Nb, dtype='int32')) * dB
    
    # deal with video mode
    if video :
        if displayer is None:
            extent_01 = [t.item() for t in (xgrid[0], xgrid[-1],
                                            ygrid[-1], ygrid[0])]          
            extent_02 = [t.item() for t in (zgrid[0], zgrid[-1],
                                            ygrid[-1], ygrid[0])]
            extent_12 = [t.item() for t in (zgrid[0], zgrid[-1],
                                            xgrid[-1], xgrid[0])]
            extents = (extent_01, extent_02, extent_12)
            grids = (ygrid, xgrid, zgrid)
            displayer = displayers.create_3d_displayer(nsrc=1,
                                                       figsize=None,
                                                       extents=extents,
                                                       adjust_dynamic=True,
                                                       display_labels=True,
                                                       grids=grids,
                                                       origin='lower',
                                                       boundaries='same')
        vol_np = backend.to_numpy(vol.reshape([Ny, Nx, Nz]))
        fg = displayer.init_display(vol_np)
        fgnum = displayer.get_number(fg)
    
    # avoid zero-gradient projections and deal with shuffle option
    idproj = backend.argwhere(mu != 0)
    if shuffle :
        idproj = idproj[backend.randperm(len(idproj))]

    #############
    # main loop #
    #############
    cnt = 0
    for id in idproj:
        
        # resample and accumulate filtered projections
        r = - (x * gx[id] + y * gy[id] + z * gz[id])
        vol += interp1(rnodes, fproj[id,:], r)
        cnt += 1
        
        # deal with verbose & video modes
        if verbose or video:
            msg = "processed projection %d/%d" % (cnt, len(idproj))
        if verbose:
            print(msg)
        if video and 0 == cnt % Ndisplay:
            if video and plt.fignum_exists(fgnum):
                vol_np = backend.to_numpy(vol.reshape([Ny, Nx, Nz]))
                displayer.update_display(vol_np, fg)
                displayer.title(msg)
                displayer.pause()
        
        # deal with quit event (when video mode is enabled)
        if video and not plt.fignum_exists(fgnum):
            break
    
    # display final result before returning (when video mode is
    # enabled)
    if video and plt.fignum_exists(fgnum) :
        vol_np = backend.to_numpy(vol.reshape([Ny, Nx, Nz]))
        displayer.update_display(vol_np, fg)
        msg = "processed projection %d/%d" % (cnt, len(idproj))
        displayer.title(msg)
    
    # close figure when code is running on interactive notebook
    if video and displayer.notebook:
        displayer.clear_output()
    
    return vol.reshape([len(ygrid), len(xgrid), len(zgrid)])

def _check_inputs_(caller, backend, proj=None, B=None, fgrad=None,
                   delta=None, h=None, lbda=None, out_shape=None,
                   init=None, tol=None, nitermax=None,
                   eval_energy=None, disable_toeplitz_kernel=None,
                   verbose=None, video=None, displayer=None, eps=None,
                   Ndisplay=None, xgrid=None, ygrid=None, zgrid=None,
                   interp1=None, frequency_cutoff=None, shuffle=None):
    """Factorized consistency checks for functions in the :py:mod:`pyepri.processing` submodule."""
    
    ##################
    # General checks #
    ##################
    
    # check backend consistency 
    checks._check_backend_(backend, B=B, init=init, xgrid=xgrid,
                           ygrid=ygrid, zgrid=zgrid)
    
    # retrieve dtype & Nb
    dtype = B.dtype
    
    # custom type checks
    checks._check_dtype_(dtype, xgrid=xgrid, ygrid=ygrid, zgrid=zgrid)
    checks._check_type_(bool, eval_energy=eval_energy,
                        disable_toeplitz_kernel=disable_toeplitz_kernel,
                        verbose=verbose, video=video, shuffle=shuffle)
    checks._check_type_(float, delta=delta, tol=tol, eps=eps)
    checks._check_type_(int, Ndisplay=Ndisplay, nitermax=nitermax)
    checks._check_ndim_(1, B=B, xgrid=xgrid, ygrid=ygrid, zgrid=zgrid)
    Nb = len(B)
    
    # check lbda > 0 (if not None)
    if lbda is not None and (not isinstance(lbda, (float, int)) or lbda < 0):
        raise RuntimeError(
            "Parameter `lbda` must be a nonnegative float scalar number (int is also tolerated)."
        )
    
    # check eps > 0 (if not None)
    if eps is not None and eps < 0:
        raise RuntimeError("Parameter ``eps`` must be positive")
    
    # check Ndisplay >= 1 (if not None)
    if Ndisplay is not None and Ndisplay < 1:
        raise RuntimeError("Parameter ``Ndisplay`` must be >= 1")
    
    # check nitermax >= 0 (if not None)
    if nitermax is not None and nitermax < 1:
        raise RuntimeError("Parameter ``Ndisplay`` must be >= 1")

    # check interp1 (if not None)
    if (interp1 is not None) and (not isinstance(interp1, (types.FunctionType, types.MethodType))):
        raise RuntimeError(
            "Parameter ``interp1`` must have type %s or %s." % (types.FunctionType, types.MethodType)
        )

    ###########################
    # Caller dependent checks #
    ###########################
    
    # check B, h, fgrad (common for tv_monosrc, eprfbp2d and eprfbp3d)
    if caller in ("tv_monosrc", "eprfbp2d", "eprfbp3d"):
        
        # generic checks
        checks._check_backend_(backend, proj=proj, h=h, fgrad=fgrad)
        checks._check_ndim_(1, h=h)
        checks._check_ndim_(2, fgrad=fgrad, proj=proj)
        checks._check_dtype_(dtype, proj=proj, fgrad=fgrad, h=h)
        
        # retrieve dims from fgrad
        dim, Nproj = fgrad.shape
        
        # check projection shape
        checks._check_ndim_(2, proj=proj)
        if proj.shape[0] != Nproj:
            raise RuntimeError(
                "Parameters ``proj`` and ``fgrad`` have inconsistent shapes (proj.shape[0] represents the number\n" + 
                "of measured projections and should be equal to fgrad.shape[1])"
            )
        if proj.shape[1] != Nb:
            raise RuntimeError(
                "Parameters ``proj`` and ``B`` have inconsistent shapes (proj.shape[1] represents the number\n" + 
                "of measured sample per projection and should be equal to len(B))"
            )
        
        # check number of elements of h
        if len(h) != Nb:
            raise RuntimeError(
                "Parameters ``h`` and ``B`` must have the same number of elements"
            )
    
    # function specific tests
    if "tv_monosrc" == caller:
        
        # check out_shape consistency
        checks._check_seq_(t=int, out_shape=out_shape)
        if dim != len(out_shape):
            raise RuntimeError(
                "Parameters ``out_shape`` and ``fgrad`` are inconsistents (len(out_shape) == fgrad.shape[0] is excpected)"
            )
        for n in out_shape:
            if n < 1:
                raise RuntimeError(
                    "Parameter ``out_shape`` contains invalid items (all items must be >= 1)"
                )
        
        # check init shape (if provided)
        if init is not None and (dim != init.ndim or init.shape != tuple(out_shape)):
            raise RuntimeError(
                "Parameter ``init`` shape must be ``out_shape``."
            )
    
    elif "eprfbp2d" == caller:
        
        # check 2D framework
        if 2 != dim:
            raise RuntimeError(
                "Parameter ``fgrad`` has inconsistent shape (expected ``fgrad.shape[0] = 2`` for 2D image reconstruction)"
            )
    
    elif "eprfbp3d" == caller:
        
        # check 3D framework
        if 3 != dim:
            raise RuntimeError(
                "Parameter ``fgrad`` has inconsistent shape (expected ``fgrad.shape[0] = 3`` for 3D image reconstruction)"
            )
    
    elif "tv_multisrc" == caller:
        
        # check outshape and retrieve problem dimensions
        checks._check_seq_of_seq_(t=int, tlen0=(2, 3), out_shape=out_shape)
        K = len(out_shape)
        dim = len(out_shape[0])
        checks._check_seq_of_seq_(len0=K, len1=dim, out_shape=out_shape)
        
        # check proj
        checks._check_seq_(t=backend.cls, dtype=dtype, ndim=2,
                           proj=proj)
        L = len(proj)
        
        # check fgrad
        checks._check_seq_(t=backend.cls, dtype=dtype, fgrad=fgrad)
        if len(fgrad) not in (1, L):
            raise RuntimeError(
                "Parameter ``fgrad`` must have length equal to either one or equal to the length of ``proj``."
            )
        for i, p in enumerate(proj):
            fg = fgrad[i] if len(fgrad) > 1 else fgrad[0]
            Nproj = fg.shape[1]
            if p.shape[0] != Nproj or p.shape[1] != Nb:
                raise RuntimeError(                    
                    "Inconsistent shape for %d-th element of `proj` (expected shape = (%d, %d))" % (i, Nproj, Nb)
                )
        
        # check h: (check dtype == dtype, made of L or 1 sequence(s)
        # containing K element each, all leaves have ndim=1 & len =
        # Nb)
        checks._check_seq_of_seq_(t=backend.cls, dtype=dtype, ndim=1,
                                  tlen0=(1,L), len1=K, len2=Nb)
        
        # check init
        checks._check_seq_(t=backend.cls, dtype=dtype, n=K, ndim=dim,
                           init=init)
        if init is not None: 
            for j, im in enumerate(init):
                if im.shape != tuple(out_shape[j]):
                    raise RuntimeError(
                        "Inconsistent shape of the %d-th element of ``init`` (expected ``init[%d].shape == out_shape[%d]``" % (j, j, j)
                    )
    
    return True
