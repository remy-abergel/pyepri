import pyepri.backends as backends
import pyepri.spectralspatial as ss
import pyepri.utils as utils
import numpy as np

def test_proj4d_rfftmode(libname, dtype, nruns, tol):
    
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
    
    # check that ss.proj4d returns the same results whenever rfft_mode
    # parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(10 * backend.rand(1)[0])
        N2 = 1 + int(10 * backend.rand(1)[0])
        N3 = 1 + int(10 * backend.rand(1)[0])
        Nproj = 1 + int(10 * backend.rand(1)[0])
        Nb = 2 + int(10 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        
        # sample random 4D image
        x = backend.rand(Nb, N1, N2, N3, dtype=dtype)
        
        # apply proj4d operator (with rfft_mode enabled or not)
        Ax1 = ss.proj4d(x, delta, B, fgrad, backend=backend, rfft_mode=True, eps=eps)
        Ax2 = ss.proj4d(x, delta, B, fgrad, backend=backend, rfft_mode=False, eps=eps)
        
        # compare results
        rel = utils._relerr_(Ax1, Ax2, backend=backend, notest=True)
        assert rel < tol * eps


def test_proj4d_memory_usage(libname, dtype, nruns, tol):
    
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
    
    # check that ss.proj4d returns the same results whenever rfft_mode
    # parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(10 * backend.rand(1)[0])
        N2 = 1 + int(10 * backend.rand(1)[0])
        N3 = 1 + int(10 * backend.rand(1)[0])
        Nproj = 1 + int(10 * backend.rand(1)[0])
        Nb = 2 + int(10 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        
        # sample random 4D image
        x = backend.rand(Nb, N1, N2, N3, dtype=dtype)
        
        # apply proj4d operator
        Ax0 = ss.proj4d(x, delta, B, fgrad, backend=backend, rfft_mode=(backend.rand(1)[0] > 0.5), eps=eps, memory_usage=0)
        Ax1 = ss.proj4d(x, delta, B, fgrad, backend=backend, rfft_mode=(backend.rand(1)[0] > 0.5), eps=eps, memory_usage=1)
        Ax2 = ss.proj4d(x, delta, B, fgrad, backend=backend, rfft_mode=(backend.rand(1)[0] > 0.5), eps=eps, memory_usage=2)
        
        # compare results
        nrm = 1. / backend.abs(Ax0).max()
        rel1 = utils._relerr_(Ax0, Ax1, backend=backend, notest=True)
        rel2 = utils._relerr_(Ax0, Ax2, backend=backend, notest=True)
        rel = max(rel1, rel2)
        assert rel < tol * eps


def test_backproj4d_rfftmode(libname, dtype, nruns, tol):
    
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
    
    # check that ss.backproj4d returns the same results whenever
    # rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(10 * backend.rand(1)[0])
        N2 = 1 + int(10 * backend.rand(1)[0])
        N3 = 1 + int(10 * backend.rand(1)[0])
        Nproj = 1 + int(10 * backend.rand(1)[0])
        Nb = 2 + int(10 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        
        # sample random 4D projections
        y = backend.rand(Nproj, Nb, dtype=dtype)
        
        # apply backproj4d operator (with rfft_mode enabled or not)
        out_shape=(Nb, N1, N2, N3)
        adjAy1 = ss.backproj4d(y, delta, B, fgrad, out_shape, backend=backend, rfft_mode=True, eps=eps)
        adjAy2 = ss.backproj4d(y, delta, B, fgrad, out_shape, backend=backend, rfft_mode=False, eps=eps)
        
        # compare results
        rel = utils._relerr_(adjAy1, adjAy2, backend=backend, notest=True)
        assert rel < tol * eps


def test_backproj4d_memory_usage(libname, dtype, nruns, tol):
    
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
    
    # check that ss.backproj4d returns the same results whenever
    # rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(10 * backend.rand(1)[0])
        N2 = 1 + int(10 * backend.rand(1)[0])
        N3 = 1 + int(10 * backend.rand(1)[0])
        Nproj = 1 + int(10 * backend.rand(1)[0])
        Nb = 2 + int(10 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        
        # sample random 4D projections
        y = backend.rand(Nproj, Nb, dtype=dtype)
        
        # apply backproj4d operator (with rfft_mode enabled or not)
        out_shape=(Nb, N1, N2, N3)
        adjAy0 = ss.backproj4d(y, delta, B, fgrad, out_shape, backend=backend, rfft_mode=(backend.rand(1)[0] > 0.5), eps=eps, memory_usage=0)
        adjAy1 = ss.backproj4d(y, delta, B, fgrad, out_shape, backend=backend, rfft_mode=(backend.rand(1)[0] > 0.5), eps=eps, memory_usage=1)
        adjAy2 = ss.backproj4d(y, delta, B, fgrad, out_shape, backend=backend, rfft_mode=(backend.rand(1)[0] > 0.5), eps=eps, memory_usage=2)
        
        # compare results
        rel1 = utils._relerr_(adjAy0, adjAy1, backend=backend, notest=True)
        rel2 = utils._relerr_(adjAy0, adjAy2, backend=backend, notest=True)
        rel = max(rel1, rel2)
        assert rel < tol * eps


def test_4d_toeplitz_kernel_rfftmode(libname, dtype, nruns, tol):
    
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
    
    # relative error computation macro (nrm input can be used to avoid
    # underflow/overflow for the square values)
    relerr = lambda arr1, arr2, nrm=1. : backend.sqrt((backend.abs(nrm * arr1 - nrm * arr2)**2).sum() / (backend.abs(nrm * arr1)**2).sum())
    
    # check that ss.compute_4d_toeplitz_kernel returns the same
    # results whenever rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(10 * backend.rand(1)[0])
        N2 = 1 + int(10 * backend.rand(1)[0])
        N3 = 1 + int(10 * backend.rand(1)[0])
        Nproj = 1 + int(10 * backend.rand(1)[0])
        Nb = 2 + int(10 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        
        # compute Toeplitz kernel (with rfft_mode enabled or not)
        phi1 = ss.compute_4d_toeplitz_kernel(B, delta, fgrad, (2*Nb, 2*N1, 2*N2, 2*N3), backend=backend, eps=eps, rfft_mode=True)
        phi2 = ss.compute_4d_toeplitz_kernel(B, delta, fgrad, (2*Nb, 2*N1, 2*N2, 2*N3), backend=backend, eps=eps, rfft_mode=False)
        
        # compare results
        rel = backend._relerr_(phi1, phi2, backend=backend, notest=True)
        assert rel < tol * eps


def test_proj4d_and_backproj4d_adjointness(libname, dtype, nruns, tol):
    
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
    
    # test adjoint identity <A(x), y> = <x, adjA(y)> for A = ss.proj4d
    # and adjA = ss.backproj4d
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(10 * backend.rand(1)[0])
        N2 = 1 + int(10 * backend.rand(1)[0])
        N3 = 1 + int(10 * backend.rand(1)[0])
        Nproj = 1 + int(10 * backend.rand(1)[0])
        Nb = 2 + int(10 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        
        # sample random signals (x=image, y=projections)
        x = backend.rand(Nb, N1, N2, N3, dtype=dtype)
        y = backend.rand(Nproj, Nb, dtype=dtype)
        
        # apply operators (randomly select whether rfft_mode shall be
        # used or not
        rfft_mode = backend.rand(1)[0] > 0.5
        Ax = ss.proj4d(x, delta, B, fgrad, backend=backend, rfft_mode=rfft_mode, eps=eps)
        adjAy = ss.backproj4d(y, delta, B, fgrad, (Nb, N1, N2, N3), backend=backend, rfft_mode=rfft_mode, eps=eps)
        
        # compute inner products
        inprod1 = (Ax * y).sum()
        inprod2 = (x * adjAy).sum()
        rel = abs(1 - inprod1 / inprod2)
        assert rel < tol * eps


def test_proj4d_and_backproj4d_matrices(libname, dtype, nruns, tol):

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
    
    # check adjointness for the ss.proj4d and ss.backproj4d via matrix
    # representation over very small datasets
    for id in range(nruns):
    
        # compute a very small dataset
        N1 = 1 + int(4 * backend.rand(1)[0])
        N2 = 1 + int(4 * backend.rand(1)[0])
        N3 = 1 + int(4 * backend.rand(1)[0])
        Nproj = 1 + int(5 * backend.rand(1)[0])
        Nb = 2 + int(5 * backend.rand(1)[0])
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        x = backend.rand(Nb, N1, N2, N3, dtype=dtype)
        y = backend.rand(Nproj, Nb, dtype=dtype)
        
        # compute matricial representation of A and its adjoint (use a very small dataset)
        A = lambda x : ss.proj4d(x, delta, B, fgrad, backend=backend, rfft_mode=True, eps=eps)
        arr_to_vec = lambda arr : arr.reshape((-1,))
        vec_to_arr = lambda vec, shape: vec.reshape(shape)
        m = Nproj * Nb
        n = Nb * N1 * N2 * N3
        M = backend.zeros([m, n], dtype=dtype)
        v = backend.zeros((n,), dtype=dtype)
        for col in range(n):
            v[col-1] = 0
            v[col] = 1
            M[:,col] = arr_to_vec(A(vec_to_arr(v, (Nb, N1, N2, N3))))
        
        # evaluate adjA(y) by matrix-vector multiplication
        Mt = backend.transpose(M)
        adjAy = ss.backproj4d(y, delta, B, fgrad, (Nb, N1, N2, N3), backend=backend, rfft_mode=True, eps=eps)
        adjAy2 = vec_to_arr(Mt @ arr_to_vec(y), (Nb, N1, N2, N3))

        # check relative error
        rel = utils._relerr_(adjAy, adjAy2, backend=backend, notest=True)
        assert rel < tol * eps


def test_4d_toeplitz_kernel_rfftmode(libname, dtype, nruns, tol):
    
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
    
    # check that monosrc.compute_3d_toeplitz_kernel returns the same
    # results whenever rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(10 * backend.rand(1)[0])
        N2 = 1 + int(10 * backend.rand(1)[0])
        N3 = 1 + int(10 * backend.rand(1)[0])
        Nproj = 1 + int(10 * backend.rand(1)[0])
        Nb = 2 + int(10 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        
        # compute Toeplitz kernel (with rfft_mode enabled or not)
        phi1 = ss.compute_4d_toeplitz_kernel(B, delta, fgrad, (2*Nb, 2*N1, 2*N2, 2*N3), backend=backend, eps=eps, rfft_mode=True)
        phi2 = ss.compute_4d_toeplitz_kernel(B, delta, fgrad, (2*Nb, 2*N1, 2*N2, 2*N3), backend=backend, eps=eps, rfft_mode=False)
        
        # compare results
        rel = utils._relerr_(phi1, phi2, backend=backend, notest=True)
        assert rel < tol * eps


def test_4d_toeplitz_kernel_memory_usage(libname, dtype, nruns, tol):
    
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
    
    # check that monosrc.compute_3d_toeplitz_kernel returns the same
    # results whenever rfft_mode parameter is True or False
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(10 * backend.rand(1)[0])
        N2 = 1 + int(10 * backend.rand(1)[0])
        N3 = 1 + int(10 * backend.rand(1)[0])
        Nproj = 1 + int(10 * backend.rand(1)[0])
        Nb = 2 + int(10 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        
        # compute Toeplitz kernel (with rfft_mode enabled or not)
        phi0 = ss.compute_4d_toeplitz_kernel(B, delta, fgrad, (2*Nb, 2*N1, 2*N2, 2*N3), backend=backend, eps=eps, rfft_mode=(backend.rand(1)[0] > 0.5), memory_usage=0)
        phi2 = ss.compute_4d_toeplitz_kernel(B, delta, fgrad, (2*Nb, 2*N1, 2*N2, 2*N3), backend=backend, eps=eps, rfft_mode=(backend.rand(1)[0] > 0.5), memory_usage=2)
        
        # compare results
        rel = utils._relerr_(phi0, phi2, backend=backend, notest=True)
        assert rel < tol * eps


def test_4d_toeplitz_kernel(libname, dtype, nruns, tol):
    
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
    
    # denoting by A and adjA the operators associated to ss.proj4d and
    # ss.backproj4d, check that Adj(A(u)) is the same as the
    # convolution between u and phi, denoting by phi the kernel
    # obtained using ss.compute_4d_toeplitz_kernel
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 1 + int(10 * backend.rand(1)[0])
        N2 = 1 + int(10 * backend.rand(1)[0])
        N3 = 1 + int(10 * backend.rand(1)[0])
        Nproj = 1 + int(10 * backend.rand(1)[0])
        Nb = 2 + int(10 * backend.rand(1)[0])
        
        # sample random inputs 
        B0 = backend.cast(200 + 100 * backend.rand(1)[0], dtype)
        dB = 10. * B0 * eps + backend.rand(1, dtype=dtype)[0]
        delta = float(10. * eps + backend.rand(1)[0])
        B = B0 + backend.arange(Nb, dtype=dtype)*dB
        fgrad = backend.rand(3, Nproj, dtype=dtype)
        
        # sample random image
        u = backend.rand(Nb, N1, N2, N3, dtype=dtype)
        
        # apply operators (randomly select whether rfft_mode shall be
        # used or not, and whether nodes must be reused or not
        #rfft_mode = True
        rfft_mode = backend.rand(1).item() > 0.5
        precompute_nodes = backend.rand(1).item() > 0.5
        nodes = ss.compute_4d_frequency_nodes(B, delta, fgrad, backend=backend, rfft_mode=rfft_mode) if precompute_nodes else None
        Au = ss.proj4d(u, delta, B, fgrad, backend=backend, rfft_mode=rfft_mode, eps=eps)
        adjAAu = ss.backproj4d(Au, delta, B, fgrad, (Nb, N1, N2, N3), backend=backend, rfft_mode=rfft_mode, nodes=nodes, eps=eps)
        
        # compute 4D Toeplitz kernel
        npb = backends.create_numpy_backend()
        #phi = ss.compute_4d_toeplitz_kernel(B, delta, fgrad, (2*Nb, 2*N1, 2*N2, 2*N3), backend=backend, eps=eps, rfft_mode=rfft_mode, nodes=nodes) # not ready
        phi = backend.from_numpy(ss.compute_4d_toeplitz_kernel(backend.to_numpy(B), delta, backend.to_numpy(fgrad), [2*Nb, 2*N1, 2*N2, 2*N3], backend=npb, eps=eps, rfft_mode=True))
        
        # apply 4D convolution
        out = ss.apply_4d_toeplitz_kernel(u, backend.rfftn(phi), backend=backend)
        
        # check that `adjAAu` and `out` are close to each other
        rel = utils._relerr_(adjAAu, out, backend=backend, notest=True)
        assert rel < tol * eps
