"""This module contains simple and relatively standard functions, like
operators usually involved in signal or image processing.

"""
import pyepri.checks as checks

# ------------------------- #
# Monodimensional operators #
# ------------------------- #

def grad1d(u, backend=None, notest=False):
    """Gradient (= forward finite differences) of a mono-dimensional array with Neumann boundary condition.
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Mono-dimensional array.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``u``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    G : array_like (with type `backend.cls`)
        Output array same shape as ``u`` corresponding to the forward
        finite differences of ``u``.
    
    
    See also
    --------
    
    div1d
    
    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(1, backend, u=u)
    
    # retrieve dimensions & data type and allocate memory of the
    # output    
    dtype = backend.lib_to_str_dtypes[u.dtype]
    G = backend.zeros(u.shape, dtype=dtype)
    
    # fill output array    
    G[:-1] = u[1::] - u[:-1]
    
    return G

def div1d(P, backend=None, notest=False):
    """Discrete divergence of a mono-dimensional array (opposite adjoint of grad1d).
    
    Parameters
    ----------
    
    P : array_like (with type `backend.cls`)
        Mono-dimensional input array.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``P``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    div : array_like (with type `backend.cls`)
        Mono-dimensional array with same shape as ``u`` corresponding
        to the discrete divergence (or opposite adjoint of the
        ``grad1d`` operator) of the input array ``P``.
    
    See also
    --------
    
    grad1d
    
    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(P=P)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(1, backend, P=P)
    
    # retrieve dimensions & data type and allocate memory of the
    # output
    K = len(P)
    dtype = backend.lib_to_str_dtypes[P.dtype]
    div = backend.zeros(P.shape, dtype=dtype)
    
    # fill output array
    div[1:K-1] =  P[1:K-1] - P[0:K-2]
    div[0]     =  P[0]
    div[K-1]   = -P[K-2]
    
    return div


# ------------------------- #
# Two-dimensional operators #
# ------------------------- #

def compute_2d_gradient_masks(mask, backend=None, notest=False):
    """Compute 2D gradient masks from a 2D binary mask.
    
    Parameters
    ----------
    
    mask: array_like (with type `backend.cls`)
        A two-dimensional array (expected to contains booleans or
        values in [0, 1]).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``mask``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    masks : array_like (with type `backend.cls`)    
        Output array with shape ``(2,) + mask.shape`` and same
        datatype as ``mask``. This array is such that each masks[j]
        (for j=0 or j=1) takes the value True (or 1) at locations
        where the the j-th output of the :py:func:`grad2d` (evaluated
        on any signal ``u`` with same shape as ``mask``) involves
        differences of two pixels both falling inside the input
        ``mask`` (meaning that ``mask`` is True (or 1) for those two
        pixels).
    
    
    Note
    ----
    
    When input mask is None, the output is also None.
    
    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(mask=mask)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, backend, mask=mask)
    
    # when mask is None, we return None
    if mask is None:
        return None
    
    # retrieve dimensions & data type and allocate memory of the
    # output        
    M = backend.zeros((2, *mask.shape))
    
    # fill the output array (masks) with values that indicate whether    
    # or not the forward neighboring pixel (along the X, Y or Z axis)
    # falls inside the input mask    
    M[0][:-1,   :] = mask[1::,   :]
    M[1][  :, :-1] = mask[  :, 1::]
    
    # restrict the mask components to pixels that fall inside the
    # input mask
    M[0] *= mask 
    M[1] *= mask 
    
    return M

def grad2d(u, masks=None, backend=None, notest=False):
    """Gradient (= forward finite differences) of a 2-dimensional array with Neumann boundary condition.
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Two-dimensional array.
    
    masks : array_like (with type `backend.cls`) or None, optional
        When given, the output array ``G`` is multiplied by masks on
        return.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``u``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    G : array_like (with type `backend.cls`)
        Output array with shape ``(2,) + u.shape`` such that ``G[j]``
        correspond to the forward finite differences of ``u`` along
        its `j-th` dimension (for ``j in range(1)``).
    
    
    See also
    --------
    
    div2d
    compute_2d_gradient_masks
    
    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(2, backend, u=u, masks=masks)
    
    # retrieve data type and allocate memory for the output
    dtype = backend.lib_to_str_dtypes[u.dtype]
    G = backend.zeros((2, *u.shape), dtype=dtype)
    
    # fill output array
    G[0][:-1,   :] = u[1::,   :] - u[:-1,   :]
    G[1][  :, :-1] = u[  :, 1::] - u[  :, :-1]
    
    # deal with masks option
    if masks is not None:
        G *= masks
    
    return G

def div2d(P, masks=None, backend=None, notest=False):
    """Discrete divergence of a 2D field vector (opposite adjoint of grad2d).
    
    Parameters
    ----------
    
    P : array_like (with type `backend.cls`)
        Two-dimensional vector field array with shape ``(2, Ny, Nx)``.
    
    masks : array_like (with type `backend.cls`) or None, optional
        When given, the divergence of ``masks * P`` is returned.
    
    backend : <class 'pyepri.backends.Backend'>or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``P``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    div : array_like (with type `backend.cls`)
        Two dimensional array with shape ``(Ny, Nx)`` corresponding to
        the discrete divergence (or opposite adjoint of the ``grad2d``
        operator) of the input field vector array ``P``.

    
    See also
    --------
    
    grad2d
    compute_2d_gradient_masks
    
    """
    
    if masks is None:
        
        # backend inference (if necessary)
        if backend is None:
            backend = checks._backend_inference_(P=P)
        
        # consistency checks
        if not notest:
            _check_nd_inputs_(2, backend, P=P, masks=masks)
        
        # retrieve dimensions & data type and allocate memory of the
        # output
        ny, nx = P[0].shape
        dtype = backend.lib_to_str_dtypes[P.dtype]
        div = backend.zeros((ny, nx), dtype=dtype)
        
        # process the first component of the input field (column axis)
        div[1:ny-1, :] =  P[0][1:ny-1, :] - P[0][0:ny-2, :]
        div[     0, :] =  P[0][     0, :]
        div[  ny-1, :] = -P[0][  ny-2, :]
        
        # process the second component of the input field (row axis)
        div[:, 1:nx-1] += P[1][:, 1:nx-1] - P[1][:, 0:nx-2]
        div[:,      0] += P[1][:,      0]
        div[:,   nx-1] -= P[1][:,   nx-2]
    
    else:
        
        div = div2d(masks * P, masks=None, backend=backend, notest=notest)
    
    # return output divergence
    return div


# --------------------------- #
# Three-dimensional operators #
# --------------------------- #

def compute_3d_gradient_masks(mask, backend=None, notest=False):
    """Compute 3D gradient masks from a 3D binary mask.
    
    Parameters
    ----------
    
    mask: array_like (with type `backend.cls`)
        A three-dimensional array (expected to contains booleans or
        values in [0, 1]).
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``mask``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    masks : array_like (with type `backend.cls`)    
        Output array with shape ``(3,) + mask.shape`` and same
        datatype as ``mask``. This array is such that each masks[j]
        (for 0 <= j <= 2) takes the value True (or 1) at locations
        where the the j-th output of the :py:func:`grad3d` (evaluated
        on any signal ``u`` with same shape as ``mask``) involves
        differences of two pixels both falling inside the input
        ``mask`` (meaning that ``mask`` is True (or 1) for those two
        pixels).
    
    
    Note
    ----
    
    When input mask is None, the output is also None.

    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(mask=mask)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, mask=mask)
    
    # when mask is None, we return None
    if mask is None:
        return None
    
    # retrieve dimensions & data type and allocate memory of the
    # output        
    M = backend.zeros((3, *mask.shape))
    
    # fill the output array (masks) with values that indicate whether    
    # or not the forward neighboring pixel (along the X, Y or Z axis)
    # falls inside the input mask    
    M[0][:-1,   :,   :] = mask[1::,   :,   :] 
    M[1][  :, :-1,   :] = mask[  :, 1::,   :] 
    M[2][  :,   :, :-1] = mask[  :,   :, 1::]
    
    # restrict the mask components to pixels that fall inside the
    # input mask
    M[0] *= mask 
    M[1] *= mask 
    M[2] *= mask 
    
    return M

def grad3d(u, masks=None, backend=None, notest=False):
    """Gradient (= forward finite differences) of a 3-dimensional array with Neumann boundary condition.
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        Three-dimensional array.
    
    masks : array_like (with type `backend.cls`) or None, optional
        When given, the output array ``G`` is multiplied by masks on
        return.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``u``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    G : array_like (with type `backend.cls`) 
        Output array with shape ``(3,) + u.shape`` such that ``G[j]``
        correspond to the forward finite differences of ``u`` along
        its `j-th` dimension (for ``j in range(3)``).
    
    
    See also
    --------
    
    div3d
    compute_3d_gradient_masks
    
    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u)
    
    # consistency checks
    if not notest:
        _check_nd_inputs_(3, backend, u=u, masks=masks)
    
    # retrieve data type and allocate memory for the output
    dtype = backend.lib_to_str_dtypes[u.dtype]
    G = backend.zeros((3, *u.shape), dtype=dtype)
    
    # fill output array
    G[0][:-1,   :,   :] = u[1::,   :,   :] - u[:-1,   :,   :]
    G[1][  :, :-1,   :] = u[  :, 1::,   :] - u[  :, :-1,   :]
    G[2][  :,   :, :-1] = u[  :,   :, 1::] - u[  :,   :, :-1]
    
    # deal with masks option
    if masks is not None:
        G *= masks
    
    return G

def div3d(P, masks=None, backend=None, notest=False):
    """Discrete divergence of a 3D field vector (opposite adjoint of grad3d).
    
    Parameters
    ----------
    
    P : array_like (with type `backend.cls`)
        Three-dimensional vector field array with shape ``(3, Ny, Nx,
        Nz)``.
    
    masks : array_like (with type `backend.cls`) or None, optional
        When given, the divergence of ``masks * P`` is returned.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input array ``P``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
    
    
    Return
    ------
    
    div : array_like (with type `backend.cls`)
        Three dimensional array with shape ``(Ny, Nx, Nz)``
        corresponding to the discrete divergence (or opposite adjoint
        of the ``grad3d`` operator) of the input field vector array
        ``P``.
    
    See also
    --------
    
    grad3d
    compute_3d_gradient_masks
    
    """
    
    if masks is None: # compute discrete divergence of P
        
        # backend inference (if necessary)
        if backend is None:
            backend = checks._backend_inference_(P=P)
        
        # consistency checks
        if not notest:
            _check_nd_inputs_(3, backend, P=P, masks=masks)
        
        # retrieve dimensions & data type and allocate memory of the
        # output
        ny, nx, nz = P[0].shape
        dtype = backend.lib_to_str_dtypes[P.dtype]
        div = backend.zeros((ny, nx, nz), dtype=dtype)
        
        # process the first component of the input field (column axis)
        div[1:ny-1, :, :] =  P[0][1:ny-1, :, :] - P[0][0:ny-2, :, :]
        div[     0, :, :] =  P[0][     0, :, :]
        div[  ny-1, :, :] = -P[0][  ny-2, :, :]
        
        # process the second component of the input field (row axis)
        div[:, 1:nx-1, :] += P[1][:, 1:nx-1, :] - P[1][:, 0:nx-2, :]
        div[:,      0, :] += P[1][:,      0, :]
        div[:,   nx-1, :] -= P[1][:,   nx-2, :]
        
        # process the third component of the input field (depth axis)
        div[:, :, 1:nz-1] += P[2][: , :, 1:nz-1] - P[2][:, :, 0:nz-2]
        div[:, :,      0] += P[2][: , :,      0]
        div[:, :,   nz-1] -= P[2][: , :,   nz-2]
    
    else: # compute discrete divergence of masks * P
        
        div = div3d(P * masks, masks=None, backend=backend, notest=notest)
    
    # return output divergence
    return div


# ----- #
# Other #
# ----- #

def _relerr_(u, v, backend=None, nrm=None, notest=False):
    """Compute relative error between two arrays.    
    
    Parameters
    ----------
    
    u : array_like (with type `backend.cls`)
        First input array.
    
    v : array_like (with type `backend.cls`)
        Second input array.
    
    backend : <class 'pyepri.backends.Backend'> or None, optional
        A numpy, cupy or torch backend (see :py:mod:`pyepri.backends`
        module).
        
        When backend is None, a default backend is inferred from the
        input arrays ``u`` and ``v``.
    
    nrm : float, optional
        A normalization factor to be applied to both input array
        (useful to avoid underflow or overflow issues in the
        computation of the relative error). If not given, ``nrm`` will
        be taken equal to one over the infinite norm of ``u``, i.e.,
        ``nrm = 1. / max(||u||_inf, ||v||_inf)``.
    
    notest : bool, optional
        Set ``notest=True`` to disable consistency checks.
        
    Return
    ------
    
    rel : float
        The relative error ``||nrm * u - nrm * v|| / ||nrm * u||``
        (where ``||.||`` denotes the l2 norm), or 0 when u and v are
        full of 0.

    """
    
    # backend inference (if necessary)
    if backend is None:
        backend = checks._backend_inference_(u=u, v=v)
    
    # consistency checks
    if not notest:
        checks._check_backend_(backend, u=u, v=v)
        checks._check_dtype_(u.dtype, v=v)
        checks._check_ndim_(u.ndim, v=v)
    
    # deal with u and v full of zero case (return 0)
    uinf = backend.abs(u).max().item()
    vinf = backend.abs(v).max().item()
    denom = max(uinf, vinf)
    if denom == 0:
        return 0
    
    # deal with nrm default value
    if nrm is None:
        nrm = 1. / denom
    
    # compute & return relative error between nrm * u and nrm * v
    rel = backend.sqrt((backend.abs(nrm * u - nrm * v)**2).sum() / (backend.abs(nrm * u)**2).sum()).item()
    return rel


# --------------------------- #
# Internal consistency checks #
# --------------------------- #

def _check_nd_inputs_(ndims, backend, u=None, P=None, mask=None, masks=None):
    """Factorized consistency checks for functions in the :py:mod:`pyepri.utils` submodule."""
    
    # check backend consistency 
    checks._check_backend_(backend, u=u, P=P, mask=mask, masks=masks)
    
    # check number of dimensions
    checks._check_ndim_(ndims, u=u, mask=mask)
    checks._check_ndim_(ndims if ndims == 1 else ndims + 1, P=P, masks=masks)
    
    # other checks
    if P is not None and ndims > 1 and ndims != P.shape[0]:
        raise RuntimeError(
            "Input vector field ``P`` has inconsistent shape (``P.shape[0] = %d`` instead of %d)"
            % (P.shape[0], ndims)
        )
    if P is not None and masks is not None and P.shape != masks.shape:
        raise RuntimeError("Input ``masks`` musht have the same shape as the input vector field ``P``")
    
    return True
