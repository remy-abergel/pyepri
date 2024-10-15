import pytest
import pyepri.backends as backends
import pyepri.multisrc as multisrc
import importlib.util
import numpy as np

libname = ['numpy']

if importlib.util.find_spec('torch') is not None:
    import torch
    libname += ['torch-cpu']
    if torch.cuda.is_available():
        libname += ['torch-cuda']
if importlib.util.find_spec('cupy') is not None:
    import cupy
    libname += ['cupy']


@pytest.mark.parametrize("libname", libname)
@pytest.mark.parametrize("dtype", ['float32', 'float64'])
def test_proj3d_rfftmode(libname, dtype, nruns=100, tol=1000):
    
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
    
    # relative error computation macro
    relerr = lambda arr1, arr2 : backend.sqrt(((arr1-arr2)**2).sum() / ((arr1)**2).sum())    
    
    # check that multisrc.proj3d returns the same results whenever
    # rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5*backend.rand(1)[0])
        L = 1 + int(5*backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(40*backend.rand(1)[0])
        u_shape = [(1 + int(15*backend.rand(1)[0]), 1 +
                    int(15*backend.rand(1)[0]), 1 + int(15*backend.rand(1)[0]))
                   for j in range(K)]
        s_shape = [(1 + int(25*backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200+100*backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(3, s[0], dtype=dtype) for s in s_shape]
        
        # sample random source images
        u = [backend.rand(s[0], s[1], s[2], dtype=dtype) for s in u_shape]
        
        # apply proj3d operator (with rfft_mode enabled or not)
        Bu1 = multisrc.proj3d(u, delta, B, h, fgrad, backend=backend, rfft_mode=True, eps=eps)
        Bu2 = multisrc.proj3d(u, delta, B, h, fgrad, backend=backend, rfft_mode=False, eps=eps)
        
        # compare results
        rel = [relerr(Bu1[i], Bu2[i]) for i in range(L)]
        assert max(rel) < tol*eps
        print("simu %d /%d : done" % (id+1, nruns))

@pytest.mark.parametrize("libname", libname)
@pytest.mark.parametrize("dtype", ['float32', 'float64'])
def test_backproj3d_rfftmode(libname, dtype, nruns=100, tol=1000):
    
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
    
    # relative error computation macro
    relerr = lambda arr1, arr2 : backend.sqrt(((arr1-arr2)**2).sum() / ((arr1)**2).sum())    
    
    # check that multisrc.backproj3d returns the same results whenever
    # rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5*backend.rand(1)[0])
        L = 1 + int(5*backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(40*backend.rand(1)[0])
        u_shape = [(1 + int(15*backend.rand(1)[0]), 1 +
                    int(15*backend.rand(1)[0]), 1 + int(15*backend.rand(1)[0]))
                   for j in range(K)]
        s_shape = [(1 + int(25*backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200+100*backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * B0 + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(3, s[0], dtype=dtype) for s in s_shape]
                
        # sample random 3D projections
        s = [backend.rand(s[0], Nb, dtype=dtype) for s in s_shape]
        
        # apply backproj3d operator (with rfft_mode enabled or not)
        adjBs1 = multisrc.backproj3d(s, delta, B, h, fgrad, u_shape, backend=backend, rfft_mode=True, eps=eps)
        adjBs2 = multisrc.backproj3d(s, delta, B, h, fgrad, u_shape, backend=backend, rfft_mode=False, eps=eps)
        
        # compare results
        rel = [relerr(adjBs1[j], adjBs2[j]) for j in range(K)]
        assert max(rel) < tol*eps

@pytest.mark.parametrize("libname", libname)
@pytest.mark.parametrize("dtype", ['float32', 'float64'])
def test_3d_toeplitz_kernel_rfftmode(libname, dtype, nruns=100, tol=1000):
    
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

    # relative error computation macro
    relerr = lambda arr1, arr2 : backend.sqrt(((arr1-arr2)**2).sum() / ((arr1)**2).sum())    
    
    # check that multisrc.compute_3d_toeplitz_kernels returns the same
    # results whenever rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5*backend.rand(1)[0])
        L = 1 + int(5*backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(40*backend.rand(1)[0])
        u_shape = [(1 + int(15*backend.rand(1)[0]), 1 +
                    int(15*backend.rand(1)[0]), 1 + int(15*backend.rand(1)[0]))
                   for j in range(K)]
        s_shape = [(1 + int(25*backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200+100*backend.rand(1)[0], dtype)
        dB = 10 * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(3, s[0], dtype=dtype) for s in s_shape]

        # compute Toeplitz kernels (with rfft_mode enabled or not)
        phi1 = multisrc.compute_3d_toeplitz_kernels(B, h, delta, fgrad, u_shape, backend=backend, eps=eps, rfft_mode=True)
        phi2 = multisrc.compute_3d_toeplitz_kernels(B, h, delta, fgrad, u_shape, backend=backend, eps=eps, rfft_mode=False)
        
        # compare results
        rel = [relerr(phi1[k][j], phi2[k][j]) for k in range(K) for j in range(K)]
        assert max(rel) < tol*eps

@pytest.mark.parametrize("libname", libname)
@pytest.mark.parametrize("dtype", ['float32', 'float64'])
def test_proj3d_and_backproj3d_adjointness(libname, dtype, nruns=100, tol=1000):
    
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
    # multisrc.proj3d and adjB = multisrc.backproj3d
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5*backend.rand(1)[0])
        L = 1 + int(5*backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(50*backend.rand(1)[0])
        u_shape = [(1 + int(15*backend.rand(1)[0]), 1 +
                    int(15*backend.rand(1)[0]), 1 + int(15*backend.rand(1)[0]))
                   for j in range(K)]
        s_shape = [(1 + int(25*backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200+100*backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(3, s[0], dtype=dtype) for s in s_shape]
        
        # sample random signals (u = source images, s = list of 3D
        # projections)
        u = [backend.rand(s[0], s[1], s[2], dtype=dtype) for s in u_shape]
        s = [backend.rand(s[0], Nb, dtype=dtype) for s in s_shape]
        
        # apply operators (randomly select whether rfft_mode shall be
        # used or not
        rfft_mode = backend.rand(1)[0] > 0.5
        Bu = multisrc.proj3d(u, delta, B, h, fgrad, backend=backend, rfft_mode=rfft_mode, eps=eps)
        rfft_mode = backend.rand(1)[0] > 0.5
        adjBs = multisrc.backproj3d(s, delta, B, h, fgrad, u_shape, backend=backend, rfft_mode=rfft_mode, eps=eps)
        
        # compute inner products
        inprod1 = sum([(Bu[i] * s[i]).sum() for i in range(L)])
        inprod2 = sum([(u[j] * adjBs[j]).sum() for j in range(K)])
        rel = abs(1-inprod1/inprod2)
        assert rel < tol*eps

@pytest.mark.parametrize("libname", libname)
@pytest.mark.parametrize("dtype", ['float32', 'float64'])
def test_3d_toeplitz_kernel(libname, dtype, nruns=100, tol=1000):
    
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
    
    # relative error computation macro
    relerr = lambda arr1, arr2 : backend.sqrt(((arr1-arr2)**2).sum() / ((arr1)**2).sum())    
    
    # denoting by B and adjB the operators associated to
    # multisrc.proj3d and multisrc.backproj3d, check that adjB(B(u))
    # is correctly computed by means of 3D circular convolution
    # between the sources images and cross source Toeplitz kernels
    # obtained using multisrc.compute_3d_toeplitz_kernels
    for id in range(nruns):
        
        # sample random number of sources and random number of
        # experiment
        K = 1 + int(5*backend.rand(1)[0])
        L = 1 + int(5*backend.rand(1)[0])
        
        # sample random dimensions
        Nb = 2 + int(40*backend.rand(1)[0])
        u_shape = [(1 + int(15*backend.rand(1)[0]), 1 +
                    int(15*backend.rand(1)[0]), 1 + int(15*backend.rand(1)[0]))
                   for j in range(K)]
        s_shape = [(1 + int(25*backend.rand(1)[0]), Nb) for i in range(L)]
        
        # sample random inputs 
        B0 = backend.cast(200+100*backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        h = [[backend.rand(Nb, dtype=dtype) for j in range(K)] for i in range(L)]
        fgrad = [backend.rand(3, s[0], dtype=dtype) for s in s_shape]
        
        # sample random sources images
        u = [backend.rand(s[0], s[1], s[2], dtype=dtype) for s in u_shape]
        
        # apply operators (randomly select whether rfft_mode shall be
        # enabled or not)
        rfft_mode = backend.rand(1).item() > 0.5
        Bu = multisrc.proj3d(u, delta, B, h, fgrad, backend=backend, eps=eps, rfft_mode=rfft_mode)
        rfft_mode = backend.rand(1).item() > 0.5
        adjBBu = multisrc.backproj3d(Bu, delta, B, h, fgrad, u_shape, backend=backend, eps=eps, rfft_mode=rfft_mode)

        # compute 3D Toeplitz kernel (directly apply real FFTs)
        rfft_mode = backend.rand(1).item() > 0.5
        rfft3_phi = [[backend.rfftn(phi_kj) for phi_kj in phi_k] for
                     phi_k in multisrc.compute_3d_toeplitz_kernels(B,
                                                                   h, delta, fgrad, u_shape, backend=backend,
                                                                   eps=eps, rfft_mode=rfft_mode)]
        
        # apply 3D convolutions
        out = multisrc.apply_3d_toeplitz_kernels(u, rfft3_phi, backend=backend)
        
        # check that `adjBBu` and `out` are close to each other
        rel = [relerr(adjBBu[j], out[j]) for j in range(K)]
        assert max(rel) < tol*eps
        
