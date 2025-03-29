"""TODO header"""

import math
import pyepri.checks as checks
from pyepri.monosrc import compute_3d_frequency_nodes

# TODO deal with out optional argument for backproj functions

def compute_4d_frequency_nodes(B, delta, fgrad, backend=None,
                               rfft_mode=True, notest=False):
    """TODO header"""
    
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
    t = -(2. * math.pi / float(Nb)) * alf.reshape((1, -1))
    t = (t * backend.ones((fgrad.shape[1], 1), dtype=dtype))
    t = t.reshape((-1,))[indexes]
    t, idt = backend.unique(t, return_inverse=True)
    lt = backend.arange(Nb, dtype=dtype).reshape((-1, 1)) * t.reshape((1, -1))
    nodes.update({'t': t, 'idt': idt, 'lt': lt})
    
    return nodes


def proj4d(u, delta, B, fgrad, backend=None, weights=None, eps=1e-06,
           rfft_mode=True, nodes=None, memory_usage=1, notest=False):
    """TODO header"""
    
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
    """TODO header"""
    
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
            w = delta**3 * (backend.cos(lt) + 1j * backend.sin(lt))[:, idt]
            out.reshape((-1,))[indexes] = (backend.nufft_execute(plan, u_cplx) * w).sum(0)
        else:
            out.reshape((-1,))[indexes] = (backend.nufft_execute(plan, u_cplx) * weights).sum(0)
    elif 1 == memory_usage:
        plan = backend.nufft_plan(2, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        uhat = backend.nufft_execute(plan, u_cplx)
        for l in range(Nb):
            w = delta**3 * (backend.cos(t * l) + 1j * backend.sin(t * l))
            out.reshape((-1,))[indexes] += uhat[l, :] * w[idt]
    else:
        for l in range(Nb):
            w = delta**3 * (backend.cos(t * l) + 1j * backend.sin(t * l))
            out.reshape((-1,))[indexes] += backend.nufft3d(y, x, z, u_cplx[l, :, :, :], eps=eps) * w[idt]
    
    return out


def proj4d_rfft(u, delta, B, fgrad, backend=None, weights=None,
                eps=1e-06, out=None, nodes=None, memory_usage=1,
                notest=False):
    """TODO header"""
    
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
            w = delta**3 * (backend.cos(lt) + 1j * backend.sin(lt))[:, idt]
            out.reshape((-1,))[indexes] = (backend.nufft_execute(plan, u_cplx) * w).sum(0)
        else:
            out.reshape((-1,))[indexes] = (backend.nufft_execute(plan, u_cplx) * weights).sum(0)
    elif 1 == memory_usage:
        plan = backend.nufft_plan(2, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        uhat = backend.nufft_execute(plan, u_cplx)
        nrm = delta**3
        for l in range(Nb):
            w = nrm * (backend.cos(t * l) + 1j * backend.sin(t * l))
            out.reshape((-1,))[indexes] += uhat[l, :] * w[idt]
    else:
        for l in range(Nb):
            w = delta**3 * (backend.cos(t * l) + 1j * backend.sin(t * l))
            out.reshape((-1,))[indexes] += backend.nufft3d(y, x, z, u_cplx[l, :, :, :], eps=eps) * w[idt]
    
    return out


def backproj4d(proj, delta, B, fgrad, out_shape, backend=None,
               weights=None, eps=1e-06, rfft_mode=True, nodes=None,
               memory_usage=1, notest=False):
    """TODO header"""
    
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
                              memory_usage=memory_usage, notest=True)
    else:
        fft_proj = backend.fft(proj)
        out = backproj4d_fft(fft_proj, delta, B, fgrad,
                             backend=backend, weights=weights,
                             eps=eps, out_shape=out_shape,
                             nodes=nodes, memory_usage=memory_usage,
                             preserve_input=False, notest=True)
    
    return out


def backproj4d_fft(fft_proj, delta, B, fgrad, backend=None,
                   out_shape=None, out=None, weights=None, eps=1e-06,
                   nodes=None, preserve_input=False, memory_usage=1,
                   notest=False):
    """TODO header"""
    
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
        if weights is None:
            w = (delta**3 / float(Nb)) * (backend.cos(lt) - 1j * backend.sin(lt))
            w = w[:, idt].ravel().reshape((w.shape[0], len(idt)))
        else:
            w = weights
        out = backend.nufft_execute(plan, w * fft_proj.reshape((-1,))[indexes]).real
    if 1 == memory_usage:
        plan = backend.nufft_plan(1, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        phat = backend.empty((Nb, len(indexes)), dtype=cdtype)
        alf = backend.ifftshift(-(Nb//2) + backend.arange(Nb, dtype=dtype))
        xi = (2. * math.pi / float(Nb)) * alf
        nrm = delta**3 / float(Nb)
        for l in range(Nb):
            xil = xi * l
            w = nrm * (backend.cos(xil) + 1j * backend.sin(xil)).reshape((-1,))
            phat[l, :] = (fft_proj * w).reshape((-1,))[indexes]
        out = backend.nufft_execute(plan, phat).real
    else:
        out = backend.zeros(out_shape, dtype=dtype)
        nrm = delta**3 / float(Nb)
        for l in range(Nb):
            tl = t[idt] * l
            w = nrm * (backend.cos(tl) - 1j * backend.sin(tl)).reshape((-1,))
            out[l, :, :, :] = backend.nufft3d_adjoint(y, x, z, fft_proj.reshape((-1))[indexes] * w, n_modes=(Ny, Nx, Nz), eps=eps).real
    
    return out


def backproj4d_rfft(rfft_proj, delta, B, fgrad, backend=None,
                    out_shape=None, out=None, weights=None, eps=1e-06,
                    nodes=None, preserve_input=False, memory_usage=1,
                    notest=False):
    """TODO header"""
    
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
            w = (delta**3 / float(Nb)) * (backend.cos(lt) - 1j * backend.sin(lt))
            w = w[:, idt].ravel().reshape((w.shape[0], len(idt)))
        else:
            w = weights
        out = backend.nufft_execute(plan, w * rfft_proj.reshape((-1,))[indexes]).real
    if 1 == memory_usage:
        plan = backend.nufft_plan(1, (Ny, Nx, Nz), n_trans=Nb, dtype=cdtype, eps=eps)
        backend.nufft_setpts(plan, y, x, z)
        phat = backend.empty((Nb, len(indexes)), dtype=cdtype)
        xi = (2. * math.pi / float(Nb)) * backend.arange(1 + Nb//2, dtype=dtype)
        nrm = delta**3 / float(Nb)
        for l in range(Nb):
            xil = xi * l
            w = nrm * (backend.cos(xil) + 1j * backend.sin(xil)).reshape((-1,))
            phat[l, :] = (rfft_proj * w).reshape((-1,))[indexes]
        out = backend.nufft_execute(plan, phat).real
    else:
        out = backend.zeros(out_shape, dtype=dtype)
        nrm = delta**3 / float(Nb)
        for l in range(Nb):
            tl = t[idt] * l
            w = nrm * (backend.cos(tl) - 1j * backend.sin(tl)).reshape((-1,))
            out[l, :, :, :] = backend.nufft3d_adjoint(y, x, z, rfft_proj.reshape((-1))[indexes] * w, n_modes=(Ny, Nx, Nz), eps=eps).real
    
    if preserve_input:
        rfft_proj[:, 1::] /= 2.
    
    return out


def compute_4d_toeplitz_kernel(B, delta, fgrad, out_shape,
                               backend=None, eps=1e-06,
                               rfft_mode=True, nodes=None,
                               return_rfft4=False, notest=False,
                               memory_usage=1):
    """TODO header"""
    
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
    
    return backend.rfftn(phi.real) if return_rfft4 else phi.real


def apply_4d_toeplitz_kernel(u, rfft4_phi, backend=None, notest=False):
    """TODO header"""
    
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
