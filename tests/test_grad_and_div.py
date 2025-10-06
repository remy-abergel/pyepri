import pyepri.backends as backends
import pyepri.utils as utils
import numpy as np

def test_grad1d_and_div1d_adjointness(libname, dtype, nruns, tol):
    
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
    
    # check adjointness of grad1d and -div1d
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 10 + int(10 * backend.rand(1)[0])
        
        # sample random signals
        u = backend.rand(N1, dtype=dtype)
        P = backend.rand(N1, dtype=dtype)
        
        # apply operators
        G = utils.grad1d(u, backend=backend)
        div = utils.div1d(P, backend=backend)
        
        # check adjointness
        inprod1 = (G * P).sum()
        inprod2 = - (u * div).sum()
        rel = abs(1 - inprod1 / inprod2)
        assert rel < tol * eps


def test_grad2d_and_div2d_adjointness(libname, dtype, nruns, tol):
    
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
    
    # check adjointness of grad1d and -div1d
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 10 + int(10 * backend.rand(1)[0])
        N2 = 10 + int(10 * backend.rand(1)[0])
        
        # randomly decide whether a mask should be used or not
        use_mask = backend.rand(1)[0].item() > .5
        mask = backend.rand(N1, N2, dtype=dtype) > .5 if use_mask else None
        masks = utils.compute_2d_gradient_masks(mask, backend=backend)
        
        # sample random signals
        u = backend.rand(N1, N2, dtype=dtype)
        P = backend.rand(2, N1, N2, dtype=dtype)
        
        # apply operators
        G = utils.grad2d(u, masks=masks, backend=backend)
        div = utils.div2d(P, masks=masks, backend=backend)
        
        # check adjointness
        inprod1 = (G * P).sum()
        inprod2 = - (u * div).sum()
        rel = abs(1 - inprod1 / inprod2)
        assert rel < tol * eps

        
def test_grad3d_and_div3d_adjointness(libname, dtype, nruns, tol):
    
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
    
    # check adjointness of grad1d and -div1d
    for id in range(nruns):
        
        # sample random dimensions 
        N1 = 3 + int(5 * backend.rand(1)[0])
        N2 = 3 + int(5 * backend.rand(1)[0])
        N3 = 3 + int(5 * backend.rand(1)[0])
        
        # randomly decide whether a mask should be used or not
        #use_mask = backend.rand(1)[0].item() > .5
        use_mask = False
        mask = backend.rand(N1, N2, N3, dtype=dtype) > .5 if use_mask else None
        masks = utils.compute_3d_gradient_masks(mask, backend=backend)
        
        # sample random signals
        u = backend.rand(N1, N2, N3, dtype=dtype)
        P = backend.rand(3, N1, N2, N3, dtype=dtype)
        
        # apply operators
        G = utils.grad3d(u, masks=masks, backend=backend)
        div = utils.div3d(P, masks=masks, backend=backend)
        
        # check adjointness
        inprod1 = (G * P).sum()
        inprod2 = - (u * div).sum()
        rel = abs(1 - inprod1 / inprod2)
        assert rel < tol * eps

        
