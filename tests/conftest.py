import importlib.util

def pytest_addoption(parser):
    parser.addoption("--nruns",
                     action="store",
                     type=int,
                     default=100,
                     help="number of runs (or passes) within each test")
    parser.addoption("--tol",
                     action="store",
                     type=float,
                     default=1000,
                     help="tolerance parameter (check relative error <= tol * eps)")
    parser.addoption("--libname",
                     action="append",
                     type=str,
                     choices=['numpy', 'torch-cpu', 'torch-cuda', 'cupy'],
                     default=[],
                     help="use a specific library (backend) (must be in ['numpy', 'torch-cpu', 'torch-cuda', 'cupy'])")
    parser.addoption("--dtype",
                     action="append",
                     type=str,
                     choices=['float32', 'float64'],
                     default=[],
                     help="use a specific datatype (must be in ['float32', 'float64'])")

def pytest_generate_tests(metafunc):
    
    # deal with nruns option
    if 'nruns' in metafunc.fixturenames:
        nruns = metafunc.config.getoption('nruns')
        metafunc.parametrize('nruns', [nruns])
    
    # deal with tol option
    if 'tol' in metafunc.fixturenames:
        tol = metafunc.config.getoption('tol')
        metafunc.parametrize('tol', [tol])
    
    # deal with dtype option
    if 'dtype' in metafunc.fixturenames:        
        dtype = metafunc.config.getoption('dtype')        
        metafunc.parametrize('dtype', ["float32", "float64"] if not dtype else dtype)
    
    if 'libname' in metafunc.fixturenames:
        
        # get requested libname(s)
        libname = metafunc.config.getoption('libname')
        if not libname:            
            # retrieve all available backends
            L = ['numpy']
            if importlib.util.find_spec('torch') is not None:
                L += ['torch-cpu']
                import torch
                if torch.cuda.is_available():
                    L += ['torch-cuda']
            if importlib.util.find_spec('cupy') is not None:
                L += ['cupy']
                import cupy
        else:
            # keep only the requested backend
            L = libname
        
        # generate tests
        metafunc.parametrize('libname', L)    
