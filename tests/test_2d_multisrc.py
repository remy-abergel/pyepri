import pyepri.backends as backends
import pyepri.multisrc as multisrc
import pyepri.utils as utils
import numpy as np

def test_proj2d_rfftmode(libname, dtype, nruns, tol):
    
    # create backend
    if libname == 'numpy':
        backend = backends.create_numpy_backend()
    elif libname == 'torch-cpu': 
        backend = backends.create_torch_backend('cpu')
    elif libname == 'torch-cuda': 
        backend = backends.create_torch_backend('cuda')
    elif libname == 'cupy': 
        backend = backends.create_cupy_backend()
    
    # retrieve machine epsilon
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # check that multisrc.proj2d returns the same results whenever
    # rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5 * backend.rand(1)[0])
        L = 1 + int(5 * backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(25 * backend.rand(1)[0])
        u_shape = [(1 + int(8 * backend.rand(1)[0]), 1 + int(8 * backend.rand(1)[0])) for j in range(K)]
        s_shape = [(1 + int(10 * backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(2, s[0], dtype=dtype) for s in s_shape]
        
        # sample random source images
        u = [backend.rand(s[0], s[1], dtype=dtype) for s in u_shape]
        
        # apply proj2d operator (with rfft_mode enabled or not)
        Bu1 = multisrc.proj2d(u, delta, B, h, fgrad, backend=backend, rfft_mode=True, eps=eps)
        Bu2 = multisrc.proj2d(u, delta, B, h, fgrad, backend=backend, rfft_mode=False, eps=eps)
        
        # compare results
        rel = [utils._relerr_(Bu1[i], Bu2[i], backend=backend, notest=True) for i in range(L)]
        assert max(rel) < tol * eps


def test_backproj2d_rfftmode(libname, dtype, nruns, tol):
    
    # create backend
    if libname == 'numpy':
        backend = backends.create_numpy_backend()
    elif libname == 'torch-cpu': 
        backend = backends.create_torch_backend('cpu')
    elif libname == 'torch-cuda': 
        backend = backends.create_torch_backend('cuda')
    elif libname == 'cupy': 
        backend = backends.create_cupy_backend()
    
    # retrieve machine epsilon
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # check that multisrc.backproj2d returns the same results whenever
    # rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5 * backend.rand(1)[0])
        L = 1 + int(5 * backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(25 * backend.rand(1)[0])
        u_shape = [(1 + int(8 * backend.rand(1)[0]), 1 + int(8 * backend.rand(1)[0])) for j in range(K)]
        s_shape = [(1 + int(10 * backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(2, s[0], dtype=dtype) for s in s_shape]
                
        # sample random 2D projections
        s = [backend.rand(s[0], Nb, dtype=dtype) for s in s_shape]
        
        # apply backproj2d operator (with rfft_mode enabled or not)
        adjBs1 = multisrc.backproj2d(s, delta, B, h, fgrad, u_shape, backend=backend, rfft_mode=True, eps=eps)
        adjBs2 = multisrc.backproj2d(s, delta, B, h, fgrad, u_shape, backend=backend, rfft_mode=False, eps=eps)
        
        # compare results
        rel = [utils._relerr_(adjBs1[j], adjBs2[j], backend=backend, notest=True) for j in range(K)]
        assert max(rel) < tol * eps


def test_2d_toeplitz_kernel_rfftmode(libname, dtype, nruns, tol):
    
    # create backend
    if libname == 'numpy':
        backend = backends.create_numpy_backend()
    elif libname == 'torch-cpu': 
        backend = backends.create_torch_backend('cpu')
    elif libname == 'torch-cuda': 
        backend = backends.create_torch_backend('cuda')
    elif libname == 'cupy': 
        backend = backends.create_cupy_backend()
    
    # retrieve machine epsilon
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # check that multisrc.compute_2d_toeplitz_kernels returns the same
    # results whenever rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5 * backend.rand(1)[0])
        L = 1 + int(5 * backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(25 * backend.rand(1)[0])
        u_shape = [(1 + int(8 * backend.rand(1)[0]), 1 + int(8 * backend.rand(1)[0])) for j in range(K)]
        s_shape = [(1 + int(10 * backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(2, s[0], dtype=dtype) for s in s_shape]

        # compute Toeplitz kernels (with rfft_mode enabled or not)
        phi1 = multisrc.compute_2d_toeplitz_kernels(B, h, delta, fgrad, u_shape, backend=backend, eps=eps, rfft_mode=True)
        phi2 = multisrc.compute_2d_toeplitz_kernels(B, h, delta, fgrad, u_shape, backend=backend, eps=eps, rfft_mode=False)
        
        # compare results
        rel = [utils._relerr_(phi1[k][j], phi2[k][j], backend=backend, notest=True) for k in range(K) for j in range(K)]
        assert max(rel) < tol * eps


def test_proj2d_and_backproj2d_adjointness(libname, dtype, nruns, tol):
    
    # create backend
    if libname == 'numpy':
        backend = backends.create_numpy_backend()
    elif libname == 'torch-cpu': 
        backend = backends.create_torch_backend('cpu')
    elif libname == 'torch-cuda': 
        backend = backends.create_torch_backend('cuda')
    elif libname == 'cupy': 
        backend = backends.create_cupy_backend()
    
    # retrieve machine epsilon
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # test adjoint identity <B(u), s> = <u, adjB(s)> for B =
    # multisrc.proj2d and adjB = multisrc.backproj2d
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5 * backend.rand(1)[0])
        L = 1 + int(5 * backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(25 * backend.rand(1)[0])
        u_shape = [(1 + int(8 * backend.rand(1)[0]), 1 + int(8 * backend.rand(1)[0])) for j in range(K)]
        s_shape = [(1 + int(10 * backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(2, s[0], dtype=dtype) for s in s_shape]
        
        # sample random signals (u = source images, s = list of 2D
        # projections)
        u = [backend.rand(s[0], s[1], dtype=dtype) for s in u_shape]
        s = [backend.rand(s[0], Nb, dtype=dtype) for s in s_shape]
        
        # apply operators (randomly select whether rfft_mode shall be
        # used or not
        rfft_mode = backend.rand(1)[0] > 0.5
        Bu = multisrc.proj2d(u, delta, B, h, fgrad, backend, rfft_mode=rfft_mode, eps=eps)
        rfft_mode = backend.rand(1)[0] > 0.5
        adjBs = multisrc.backproj2d(s, delta, B, h, fgrad, u_shape, backend=backend, rfft_mode=rfft_mode, eps=eps)
        
        # compute inner products
        inprod1 = sum([(Bu[i] * s[i]).sum() for i in range(L)])
        inprod2 = sum([(u[j] * adjBs[j]).sum() for j in range(K)])
        rel = abs(1 - inprod1 / inprod2)
        assert rel < tol * eps


def test_2d_toeplitz_kernel(libname, dtype, nruns, tol):
    
    # create backend
    if libname == 'numpy':
        backend = backends.create_numpy_backend()
    elif libname == 'torch-cpu': 
        backend = backends.create_torch_backend('cpu')
    elif libname == 'torch-cuda': 
        backend = backends.create_torch_backend('cuda')
    elif libname == 'cupy': 
        backend = backends.create_cupy_backend()
    
    # retrieve dtype precision (threshold to 1e-16)
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # denoting by B and adjB the operators associated to
    # multisrc.proj2d and multisrc.backproj2d, check that adjB(B(u))
    # is correctly computed by means of 2D circular convolution
    # between the sources images and cross source Toeplitz kernels
    # obtained using multisrc.compute_2d_toeplitz_kernels
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5 * backend.rand(1)[0])
        L = 1 + int(5 * backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(25 * backend.rand(1)[0])
        u_shape = [(1 + int(8 * backend.rand(1)[0]), 1 + int(8 * backend.rand(1)[0])) for j in range(K)]
        s_shape = [(1 + int(10 * backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype) * dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(2, s[0], dtype=dtype) for s in s_shape]
        
        # sample random sources images
        u = [backend.rand(s[0], s[1], dtype=dtype) for s in u_shape]
        
        # apply operators (randomly select whether rfft_mode shall be
        # enabled or not)
        rfft_mode = backend.rand(1).item() > 0.5
        Bu = multisrc.proj2d(u, delta, B, h, fgrad, backend=backend, eps=eps, rfft_mode=rfft_mode)
        rfft_mode = backend.rand(1).item() > 0.5
        adjBBu = multisrc.backproj2d(Bu, delta, B, h, fgrad, u_shape, backend=backend, eps=eps, rfft_mode=rfft_mode)

        # compute 2D Toeplitz kernel (directly apply real FFTs)
        rfft_mode = backend.rand(1).item() > 0.5
        rfft2_phi = [[backend.rfft2(phi_kj) for phi_kj in phi_k] for
                     phi_k in multisrc.compute_2d_toeplitz_kernels(B,
                                                                   h, delta, fgrad, u_shape, backend=backend,
                                                                   eps=eps, rfft_mode=rfft_mode)]
        
        # apply 2D convolutions
        out = multisrc.apply_2d_toeplitz_kernels(u, rfft2_phi, backend=backend)
        
        # check that `adjBBu` and `out` are close to each other
        rel = [utils._relerr_(adjBBu[j], out[j], backend=backend, notest=True) for j in range(K)]
        assert max(rel) < tol * eps
        
