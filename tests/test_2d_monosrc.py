import pyepri.backends as backends
import pyepri.monosrc as monosrc
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
    #eps = backend.lib.finfo(backend.str_to_lib_dtypes[dtype]).eps
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # check that monosrc.proj2d returns the same results whenever
    # rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(16 * backend.rand(1)[0])
        N2 = 1 + int(16 * backend.rand(1)[0])
        Nproj = 1 + int(25 * backend.rand(1)[0])
        Nb = 2 + int(50 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype) * dB
        h = backend.rand(Nb, dtype=dtype)
        fgrad = backend.rand(2, Nproj, dtype=dtype)
        
        # sample random 2D image
        x = backend.rand(N1, N2, dtype=dtype)
        
        # apply proj2d operator (with rfft_mode enabled or not)
        Ax1 = monosrc.proj2d(x, delta, B, h, fgrad, backend=backend, rfft_mode=True, eps=eps)
        Ax2 = monosrc.proj2d(x, delta, B, h, fgrad, backend=backend, rfft_mode=False, eps=eps)
        
        # compare results
        rel = utils._relerr_(Ax1, Ax2, backend=backend, notest=True)
        assert rel < tol * eps
        

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
    #eps = backend.lib.finfo(backend.str_to_lib_dtypes[dtype]).eps
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # check that monosrc.backproj2d returns the same results whenever
    # rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(16 * backend.rand(1)[0])
        N2 = 1 + int(16 * backend.rand(1)[0])
        Nproj = 1 + int(25 * backend.rand(1)[0])
        Nb = 2 + int(50 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype) * dB
        h = backend.rand(Nb, dtype=dtype)
        fgrad = backend.rand(2, Nproj, dtype=dtype)
        
        # sample random 2D projections
        y = backend.rand(Nproj, Nb, dtype=dtype)
        
        # apply backproj2d operator (with rfft_mode enabled or not)
        out_shape=(N1, N2)
        adjAy1 = monosrc.backproj2d(y, delta, B, h, fgrad, out_shape, backend=backend, rfft_mode=True, eps=eps)
        adjAy2 = monosrc.backproj2d(y, delta, B, h, fgrad, out_shape, backend=backend, rfft_mode=False, eps=eps)
        
        # compare results
        rel = utils._relerr_(adjAy1, adjAy2, backend=backend, notest=True)
        assert rel < tol * eps
        

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
    #eps = backend.lib.finfo(backend.str_to_lib_dtypes[dtype]).eps
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # check that monosrc.compute_2d_toeplitz_kernel returns the same
    # results whenever rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(16 * backend.rand(1)[0])
        N2 = 1 + int(16 * backend.rand(1)[0])
        Nproj = 1 + int(25 * backend.rand(1)[0])
        Nb = 2 + int(50 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype) * dB
        h1 = backend.rand(Nb, dtype=dtype)
        h2 = backend.rand(Nb, dtype=dtype)
        fgrad = backend.rand(2, Nproj, dtype=dtype)
        
        # compute Toeplitz kernel (with rfft_mode enabled or not)
        phi1 = monosrc.compute_2d_toeplitz_kernel(B, h1, h2, delta, fgrad, (2*N1, 2*N2), backend=backend, eps=eps, rfft_mode=True)
        phi2 = monosrc.compute_2d_toeplitz_kernel(B, h1, h2, delta, fgrad, (2*N1, 2*N2), backend=backend, eps=eps, rfft_mode=False)
        
        # compare results
        rel = utils._relerr_(phi1, phi2, backend=backend, notest=True)
        assert rel < tol * eps
        

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
    #eps = backend.lib.finfo(backend.str_to_lib_dtypes[dtype]).eps
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # test adjoint identity <A(x), y> = <x, adjA(y)> for A =
    # monosrc.proj2d and adjA = monosrc.backproj2d
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(16 * backend.rand(1)[0])
        N2 = 1 + int(16 * backend.rand(1)[0])
        Nproj = 1 + int(25 * backend.rand(1)[0])
        Nb = 2 + int(50 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype) * dB
        h = backend.rand(Nb, dtype=dtype)
        fgrad = backend.rand(2, Nproj, dtype=dtype)
        
        # sample random signals (x=image, y=projections)
        x = backend.rand(N1, N2, dtype=dtype)
        y = backend.rand(Nproj, Nb, dtype=dtype)
        
        # apply operators (randomly select whether rfft_mode shall be
        # used or not
        rfft_mode = backend.rand(1)[0] > 0.5
        Ax = monosrc.proj2d(x, delta, B, h, fgrad, backend=backend, rfft_mode=rfft_mode, eps=eps)
        adjAy = monosrc.backproj2d(y, delta, B, h, fgrad, backend=backend, out_shape=(N1, N2), rfft_mode=rfft_mode, eps=eps)
        
        # compute inner products
        inprod1 = (Ax * y).sum()
        inprod2 = (x * adjAy).sum()
        rel = abs(1 - inprod1 / inprod2)
        assert rel < tol * eps


def test_proj2d_and_backproj2d_matrices(libname, dtype, nruns, tol):
    
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
    #eps = backend.lib.finfo(backend.str_to_lib_dtypes[dtype]).eps
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # check adjointness for the monosrc.proj2d and monosrc.backproj2d
    # via matrix representation over very small datasets
    for id in range(nruns):
        
        # compute a very small dataset
        N1 = 1 + int(4 * backend.rand(1)[0])
        N2 = 1 + int(4 * backend.rand(1)[0])
        Nproj = 1 + int(5 * backend.rand(1)[0])
        Nb = 2 + int(5 * backend.rand(1)[0])
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype) * dB
        fgrad = backend.rand(2, Nproj, dtype=dtype)
        h = backend.rand(Nb, dtype=dtype)
        y = backend.rand(Nproj, Nb, dtype=dtype)
        
        # evaluate adjA(y) using monosrc.backproj2d        
        rfft_mode = backend.rand(1)[0] > 0.5
        adjAy1 = monosrc.backproj2d(y, delta, B, h, fgrad, backend=backend, out_shape=(N1, N2), rfft_mode=rfft_mode, eps=eps)
        
        # compute matricial representation of A and its adjoint over this dataset
        A = lambda x : monosrc.proj2d(x, delta, B, h, fgrad, backend=backend, rfft_mode=rfft_mode, eps=eps)
        arr_to_vec = lambda arr : arr.reshape((-1,))
        vec_to_arr = lambda vec, shape: vec.reshape(shape)
        m = Nproj * Nb
        n = N1 * N2
        M = backend.zeros([m, n], dtype=dtype)
        v = backend.zeros((n,), dtype=dtype)
        for col in range(n):
            v[col-1] = 0
            v[col] = 1
            M[:,col] = arr_to_vec(A(vec_to_arr(v, (N1, N2))))
        
        # evaluate adjA(y) by matrix-vector multiplication
        Mt = backend.transpose(M)
        adjAy2 = vec_to_arr(Mt @ arr_to_vec(y), (N1, N2))
        rel = utils._relerr_(adjAy1, adjAy2, backend=backend, notest=True)
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
    #eps = float(max(1e-16, backend.lib.finfo(backend.str_to_lib_dtypes[dtype]).eps))
    eps = 1e-15 if dtype == 'float64' else 1e-6
    
    # denoting by A and adjA the operators associated to
    # monosrc.proj2d and monosrc.backproj2d, check that adjA(A(u)) is
    # the same as the convolution between u and phi, denoting by phi
    # the kernel obtained using monosrc.compute_2d_toeplitz_kernel
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(16 * backend.rand(1)[0])
        N2 = 1 + int(16 * backend.rand(1)[0])
        Nproj = 1 + int(25 * backend.rand(1)[0])
        Nb = 2 + int(50 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype) * dB
        h = backend.rand(Nb, dtype=dtype)
        fgrad = backend.rand(2, Nproj, dtype=dtype)
        
        # sample random image
        u = backend.rand(N1, N2, dtype=dtype)
        
        # apply operators (randomly select whether rfft_mode shall be
        # enabled or not, and whether nodes must be precomputed or
        # not)
        rfft_mode = backend.rand(1).item() > 0.5
        precompute_nodes = backend.rand(1).item() > 0.5
        nodes = monosrc.compute_2d_frequency_nodes(B, delta, fgrad, backend=backend, rfft_mode=rfft_mode) if precompute_nodes else None
        Au = monosrc.proj2d(u, delta, B, h, fgrad, backend=backend, rfft_mode=rfft_mode, nodes=nodes, eps=eps)
        adjAAu = monosrc.backproj2d(Au, delta, B, h, fgrad, (N1, N2), backend=backend, rfft_mode=rfft_mode, nodes=nodes, eps=eps)
        
        # compute 2D Toeplitz kernel
        phi = monosrc.compute_2d_toeplitz_kernel(B, h, h, delta, fgrad, (2*N1, 2*N2), backend=backend, eps=eps, rfft_mode=rfft_mode, nodes=nodes)
        
        # apply 2D convolution
        out = monosrc.apply_2d_toeplitz_kernel(u, backend.rfft2(phi), backend=backend)
        
        # check that `adjAAu` and `out` are close to each other
        rel = utils._relerr_(adjAAu, out, backend=backend, notest=True)
        assert rel < tol * eps
        
